import os
import pickle
import re
from typing import Optional, Type
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ---------------- MD FILE ----------------
MD_FILE = "DriveWatcher.md"

# ---------------- AUTH ----------------
def authenticate(scopes):
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return creds

# ---------------- HELPERS ----------------
def extract_folder_id(folder_input: str) -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
    return match.group(1) if match else folder_input

def load_seen_items():
    if os.path.exists("seen_items.pkl"):
        with open("seen_items.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def save_seen_items(seen_items):
    with open("seen_items.pkl", "wb") as f:
        pickle.dump(seen_items, f)

def list_all_items(drive_service, folder_id):
    """Recursively list all files and folders under a given folder_id."""
    items = []
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, createdTime, modifiedTime, parents, lastModifyingUser)",
        pageSize=1000
    ).execute()

    for item in results.get("files", []):
        items.append(item)
        # If it's a folder, recurse into it
        if item["mimeType"] == "application/vnd.google-apps.folder":
            items.extend(list_all_items(drive_service, item["id"]))

    return items

def get_full_path(drive_service, file_id, top_folder_id):
    path_parts = []
    def build_path(fid):
        if fid == top_folder_id:
            return
        try:
            file = drive_service.files().get(fileId=fid, fields="id, name, parents").execute()
            path_parts.insert(0, file["name"])
            parents = file.get("parents", [])
            if parents:
                build_path(parents[0])
        except Exception:
            path_parts.insert(0, "Unknown")
    build_path(file_id)
    return "/" + "/".join(path_parts)

# ---------------- MD LOGGING ----------------
def save_to_md(logs, md_file=MD_FILE):
    file_exists = os.path.exists(md_file)
    with open(md_file, "a", encoding="utf-8") as f:
        if not file_exists or os.stat(md_file).st_size == 0:
            f.write("| Type | Name | Path | Created | Modified | Committed By |\n")
            f.write("|------|------|------|---------|---------|--------------|\n")
        for log in logs:
            parts = log.split("\t")
            f.write("| " + " | ".join(parts) + " |\n")

# ---------------- CHECK FOR CHANGES ----------------
def check_for_changes(drive_service, folder_id):
    """Run once to detect changes compared to previous state (recursive)."""
    seen_items = load_seen_items()
    items = list_all_items(drive_service, folder_id)
    current_ids = set()
    logs = []

    for item in items:
        item_id = item["id"]
        current_ids.add(item_id)
        item_name = item["name"]
        created = item.get("createdTime", "Unknown")
        modified = item.get("modifiedTime", "Unknown")

        committed_by = "Unknown"
        if "lastModifyingUser" in item:
            user = item["lastModifyingUser"]
            committed_by = user.get("displayName", "Unknown")
            if "emailAddress" in user:
                committed_by += f" ({user['emailAddress']})"

        path = get_full_path(drive_service, item_id, folder_id)

        # --- New item ---
        if item_id not in seen_items:
            seen_items[item_id] = {
                "name": item_name,
                "path": path,
                "created": created,
                "modified": modified,
                "committed_by": committed_by
            }
            logs.append(f"Created/Uploaded\t{item_name}\t{path}\t{created}\t{modified}\t{committed_by}")

        # --- Modified item ---
        elif seen_items[item_id]["modified"] != modified:
            seen_items[item_id]["modified"] = modified
            logs.append(f"Edited\t{item_name}\t{path}\t{seen_items[item_id]['created']}\t{modified}\t{committed_by}")

    # --- Deleted items ---
    deleted_ids = set(seen_items.keys()) - current_ids
    for del_id in deleted_ids:
        deleted_item = seen_items[del_id]
        logs.append(
            f"Deleted\t{deleted_item['name']}\t{deleted_item['path']}\t"
            f"{deleted_item.get('created','-')}\t{deleted_item.get('modified','-')}\t{deleted_item.get('committed_by','-')}"
        )
        del seen_items[del_id]

    save_seen_items(seen_items)

    # Always ensure md is updated
    if not logs:
        logs = ["\tNo new changes detected\t-\t-\t-\t-"]

    save_to_md(logs)

# ---------------- CREWAI TOOL ----------------
class DriveWatcherInput(BaseModel):
    folder_input: str = Field(..., description="Google Drive folder ID or full URL")

class DriveWatcherTool(BaseTool):
    name: str = "drive_watcher"
    description: str = "Checks a Google Drive folder ONCE for changes compared to previous run (recursive), logs into DriveWatcher.md"
    args_schema: Optional[Type[BaseModel]] = DriveWatcherInput

    def _run(self, folder_input: str) -> str:
        SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
        creds = authenticate(SCOPES)
        drive_service = build("drive", "v3", credentials=creds)
        top_folder_id = extract_folder_id(folder_input)

        check_for_changes(drive_service, top_folder_id)

        return "Change check completed. Logs stored in DriveWatcher.md"
