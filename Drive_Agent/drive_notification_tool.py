import os, pickle, re, time, base64
from email.mime.text import MIMEText
from typing import Optional, Type
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ---------------- AUTHENTICATION ----------------
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

# ---------------- HELPER: Extract Folder ID ----------------
def extract_folder_id(folder_input: str) -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
    return match.group(1) if match else folder_input

# ---------------- HELPER: Full Path ----------------
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
        except:
            path_parts.insert(0, "Unknown")
    build_path(file_id)
    return "/" + "/".join(path_parts).replace("\\", "/")

# ---------------- EMAIL SENDER ----------------
def send_email(service, to, subject, body):
    message = MIMEText(body, "html")
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print(f"📧 Email sent to {to}: {subject}")

# ---------------- SAVE/LOAD SEEN ITEMS ----------------
def load_seen_items():
    if os.path.exists("seen_items.pkl"):
        with open("seen_items.pkl", "rb") as f:
            return pickle.load(f)
    return {}

def save_seen_items(seen_items):
    with open("seen_items.pkl", "wb") as f:
        pickle.dump(seen_items, f)

# ---------------- LIST ALL ITEMS RECURSIVELY ----------------
def list_all_items(drive_service, folder_id):
    items = []
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType, createdTime, modifiedTime, owners, lastModifyingUser(displayName,emailAddress))",
        pageSize=1000
    ).execute()
    for item in results.get("files", []):
        items.append(item)
        if item["mimeType"] == "application/vnd.google-apps.folder":
            items.extend(list_all_items(drive_service, item["id"]))
    return items

# ---------------- WATCH FOLDER ----------------
def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
    seen_items = load_seen_items()
    changes_batch = []
    last_email_time = time.time()
    print(f"👀 Watching folder {folder_id} every {interval}s...")

    while True:
        try:
            items = list_all_items(drive_service, folder_id)
            current_time = time.time()
            current_ids = set()

            for item in items:
                item_id = item["id"]
                current_ids.add(item_id)
                item_name = item["name"]
                item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
                created = item.get("createdTime")
                modified = item.get("modifiedTime")

                if item_type == "File":
                    user = item.get("lastModifyingUser", {})
                    user_info = f"{user.get('displayName','Unknown')} ({user.get('emailAddress','N/A')})"
                else:
                    owner = item.get("owners", [{}])[0]
                    user_info = f"{owner.get('displayName','Unknown')} ({owner.get('emailAddress','N/A')})"

                full_path = get_full_path(drive_service, item_id, folder_id)
                link = f"https://drive.google.com/drive/folders/{item_id}" if item_type=="Folder" else f"https://drive.google.com/file/d/{item_id}/view"

                if item_id not in seen_items:
                    changes_batch.append({
                        "type": f"🆕 {item_type}",
                        "name": item_name,
                        "path": full_path,
                        "created": created,
                        "modified": modified,
                        "user": user_info,
                        "link": link
                    })
                elif seen_items[item_id] != modified:
                    changes_batch.append({
                        "type": f"✏️ Updated {item_type}",
                        "name": item_name,
                        "path": full_path,
                        "created": created,
                        "modified": modified,
                        "user": user_info,
                        "link": link
                    })

                seen_items[item_id] = modified

            deleted_ids = set(seen_items.keys()) - current_ids
            for del_id in deleted_ids:
                changes_batch.append({
                    "type": "❌ Deleted",
                    "name": "item_name",
                    "path": "N/A",
                    "created": "N/A",
                    "modified": "N/A",
                    "user": "N/A",
                    "link": "#"
                })
                del seen_items[del_id]

            if current_time - last_email_time >= email_batch_interval and changes_batch:
                html_rows = ""
                for c in changes_batch:
                    color = "#ffffff"
                    if "🆕" in c["type"]: color="#d4edda"
                    elif "✏️" in c["type"]: color="#fff3cd"
                    elif "❌" in c["type"]: color="#f8d7da"
                    html_rows += f"<tr style='background-color:{color}'><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td><td>{c['user']}</td></tr>"

                html_body = f"""
                <h2>Google Drive Updates</h2>
                <table border='1' cellpadding='5' cellspacing='0'>
                <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th><th>Committed By</th></tr>
                {html_rows}
                </table>
                """
                send_email(gmail_service, recipient_email, f"📢 Google Drive Updates (last {email_batch_interval} sec)", html_body)
                changes_batch = []
                last_email_time = current_time
                save_seen_items(seen_items)

            time.sleep(interval)

        except Exception as e:
            print(f"⚠️ Error: {str(e)}")
            time.sleep(interval)

# ---------------- INPUT SCHEMA ----------------
class DriveWatcherInput(BaseModel):
    folder_input: str = Field(..., description="Google Drive folder ID or full URL")
    recipient_email: str = Field(..., description="Email address to notify")
    email_batch_interval: int = Field(60, description="Batch interval in seconds for sending emails")

# ---------------- CREWAI TOOL ----------------
class DriveWatcherTool(BaseTool):
    name: str = "drive_watcher"
    description: str = "Continuously watches a Google Drive folder and emails updates about changes."
    args_schema: Optional[Type[BaseModel]] = DriveWatcherInput

    def _run(self, folder_input: str, recipient_email: str, email_batch_interval: int = 60) -> str:
        SCOPES = [
            "https://www.googleapis.com/auth/drive.metadata.readonly",
            "https://www.googleapis.com/auth/gmail.send"
        ]
        creds = authenticate(SCOPES)
        drive_service = build("drive", "v3", credentials=creds)
        gmail_service = build("gmail", "v1", credentials=creds)

        top_folder_id = extract_folder_id(folder_input)
        print("🚀 Drive watcher tool started")
        watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, interval=15, email_batch_interval=email_batch_interval)
        
        return "✅ Drive watcher started. It will keep running until manually interrupted."
