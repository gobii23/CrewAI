import os, pickle, re
from typing import Optional, Type
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# ---------------- AUTH ----------------
def authenticate():
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("drive", "v3", credentials=creds)

# ---------------- HELPER: Extract Folder ID ----------------
def extract_folder_id(folder_input: str) -> str:
    match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
    if match:
        return match.group(1)
    return folder_input

# ---------------- LIST DIRECTORY TREE ----------------
def list_directory_tree(service, folder_id, indent=""):
    """Recursively list folder contents as a tree."""
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)"
        ).execute()
        items = results.get("files", [])

        tree_output = ""
        for item in items:
            if item["mimeType"] == "application/vnd.google-apps.folder":
                tree_output += f"{indent}📂 {item['name']}/\n"
                tree_output += list_directory_tree(service, item["id"], indent + "   ")
            else:
                tree_output += f"{indent}📄 {item['name']}\n"
        return tree_output
    except Exception as e:
        return f"Error: {str(e)}"

def get_folder_name(service, folder_id):
    """Fetch the folder's own name."""
    folder = service.files().get(fileId=folder_id, fields="name").execute()
    return folder.get("name", "Unknown Folder")

# ---------------- INPUT SCHEMA ----------------
class DriveTreeInput(BaseModel):
    folder_input: str = Field(..., description="Google Drive folder ID or full folder URL")

# ---------------- CREWAI TOOL ----------------
class DriveTreeTool(BaseTool):
    name: str = "drive_tree"
    description: str = "Lists the directory tree of a Google Drive folder, saves it as Markdown."
    args_schema: Optional[Type[BaseModel]] = DriveTreeInput

    def _run(self, folder_input: str) -> str:
        drive_service = authenticate()
        folder_id = extract_folder_id(folder_input)

        # Get folder name
        folder_name = get_folder_name(drive_service, folder_id)

        # Build tree with folder name on top
        tree_output = f"# 📂 {folder_name}\n\n"
        tree_output += "```\n"
        tree_output += list_directory_tree(drive_service, folder_id)
        tree_output += "```"

        # Save into file
        md_file = "Directory_Tree.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(tree_output)

        return f"✅ Drive tree saved to **{md_file}**"
