from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os, pickle, re

# ---------------- AUTHENTICATION ----------------
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
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
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
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="files(id, name, mimeType)"
        ).execute()
        items = results.get("files", [])

        if not items:
            return f"{indent}-- empty folder --\n"

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
    

# ---------------- MAIN ----------------
if __name__ == "__main__":
    drive_service = authenticate()
    
    # Replace with your folder URL or ID
    folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
    folder_id = extract_folder_id(folder_input)

    print("Listing Google Drive folder contents:\n")
    print(list_directory_tree(drive_service, folder_id))












# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)
#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)
#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     return match.group(1) if match else folder_input

# # ---------------- HELPER: Full Path ----------------
# def get_full_path(drive_service, file_id, top_folder_id):
#     path_parts = []
#     def build_path(fid):
#         if fid == top_folder_id:
#             return
#         try:
#             file = drive_service.files().get(fileId=fid, fields="id, name, parents").execute()
#             path_parts.insert(0, file["name"])
#             parents = file.get("parents", [])
#             if parents:
#                 build_path(parents[0])
#         except:
#             path_parts.insert(0, "Unknown")
#     build_path(file_id)
#     return "/" + "/".join(path_parts).replace("\\", "/")

# # ---------------- EMAIL SENDER ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body, "html")
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- SAVE/LOAD SEEN ITEMS ----------------
# def load_seen_items():
#     if os.path.exists("seen_items.pkl"):
#         with open("seen_items.pkl", "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open("seen_items.pkl", "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- LIST ALL ITEMS RECURSIVELY ----------------
# def list_all_items(drive_service, folder_id):
#     items = []
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime, owners, lastModifyingUser(displayName,emailAddress))",
#         pageSize=1000
#     ).execute()
#     for item in results.get("files", []):
#         items.append(item)
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             items.extend(list_all_items(drive_service, item["id"]))
#     return items

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()
#     changes_batch = []
#     last_email_time = time.time()
#     print(f"👀 Watching folder {folder_id} every {interval}s...")

#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)
#             current_time = time.time()
#             current_ids = set()

#             for item in items:
#                 item_id = item["id"]
#                 current_ids.add(item_id)
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")

#                 # Determine last modifying user
#                 if item_type == "File":
#                     user = item.get("lastModifyingUser", {})
#                     user_info = f"{user.get('displayName','Unknown')} ({user.get('emailAddress','N/A')})"
#                 else:
#                     # For folders, use first owner
#                     owner = item.get("owners", [{}])[0]
#                     user_info = f"{owner.get('displayName','Unknown')} ({owner.get('emailAddress','N/A')})"

#                 full_path = get_full_path(drive_service, item_id, folder_id)
#                 link = f"https://drive.google.com/drive/folders/{item_id}" if item_type=="Folder" else f"https://drive.google.com/file/d/{item_id}/view"

#                 if item_id not in seen_items:
#                     changes_batch.append({
#                         "type": f"🆕 {item_type}",
#                         "name": item_name,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified,
#                         "user": user_info,
#                         "link": link
#                     })
#                 elif seen_items[item_id] != modified:
#                     changes_batch.append({
#                         "type": f"✏️ Updated {item_type}",
#                         "name": item_name,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified,
#                         "user": user_info,
#                         "link": link
#                     })

#                 seen_items[item_id] = modified

#             # Handle deletions
#             deleted_ids = set(seen_items.keys()) - current_ids
#             for del_id in deleted_ids:
#                 deleted_item = seen_items[del_id] if isinstance(seen_items[del_id], dict) else {"name": "Unknown", "full_path": "Unknown"}
#                 changes_batch.append({
#                     "type": "❌ Deleted",
#                     "name": deleted_item.get("name","Unknown"),
#                     "path": deleted_item.get("full_path","N/A"),
#                     "created": "N/A",
#                     "modified": "N/A",
#                     "user": "N/A",
#                     "link": "#"
#                 })
#                 del seen_items[del_id]

#             # Send email
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 html_rows = ""
#                 for c in changes_batch:
#                     color = "#ffffff"
#                     if "🆕" in c["type"]: color="#d4edda"
#                     elif "✏️" in c["type"]: color="#fff3cd"
#                     elif "❌" in c["type"]: color="#f8d7da"
#                     html_rows += f"<tr style='background-color:{color}'><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td><td>{c['user']}</td></tr>"

#                 html_body = f"""
#                 <h2>Google Drive Updates</h2>
#                 <table border='1' cellpadding='5' cellspacing='0'>
#                 <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th><th>Committed By</th></tr>
#                 {html_rows}
#                 </table>
#                 """
#                 send_email(gmail_service, recipient_email, f"📢 Google Drive Updates (last {email_batch_interval} sec)", html_body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(interval)

#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = ["https://www.googleapis.com/auth/drive.metadata.readonly",
#               "https://www.googleapis.com/auth/gmail.send"]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"  # Your folder ID
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     top_folder_id = extract_folder_id(folder_input)
#     watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, interval=15, email_batch_interval=60)







































# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- SEND EMAIL ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body)
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- LIST ALL ITEMS (RECURSIVE) ----------------
# def list_all_items(drive_service, folder_id):
#     """Recursively fetch all files/folders under a folder."""
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime)"
#     ).execute()
#     items = results.get("files", [])
#     all_items = items.copy()

#     for item in items:
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             all_items.extend(list_all_items(drive_service, item["id"]))
#     return all_items

# # ---------------- WATCH FOLDER (RECURSIVE) ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15):
#     seen_items = set()

#     print(f"👀 Recursively watching folder {folder_id} every {interval}s...")
#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)

#             for item in items:
#                 item_signature = (item["id"], item.get("createdTime", item["modifiedTime"]))
#                 if item_signature not in seen_items:
#                     seen_items.add(item_signature)

#                     # Different subject if it's a folder or file
#                     if item["mimeType"] == "application/vnd.google-apps.folder":
#                         subject = f"📁 New Folder Added: {item['name']}"
#                         body = f"A new folder was created:\n\nName: {item['name']}\nID: {item['id']}\nCreated: {item.get('createdTime')}"
#                     else:
#                         subject = f"📄 New/Updated File: {item['name']}"
#                         body = f"A file was added or updated:\n\nName: {item['name']}\nID: {item['id']}\nType: {item['mimeType']}\nCreated: {item.get('createdTime')}"

#                     send_email(gmail_service, recipient_email, subject, body)

#             time.sleep(interval)
#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     # Replace with your folder URL/ID and recipient email
#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     folder_id = extract_folder_id(folder_input)

#     # Start recursive watching
#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15)





# ----------------------------------------------------------------- single mail -------------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- SEND EMAIL ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body)
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- PATH + PARENT ----------------
# def get_full_path(drive_service, file_id, root_folder_id):
#     """Return full path of item + its parent folder ID."""
#     path_parts = []
#     current_id = file_id
#     parent_id = root_folder_id

#     while True:
#         file = drive_service.files().get(
#             fileId=current_id,
#             fields="id, name, parents, mimeType"
#         ).execute()
#         path_parts.append(file["name"])

#         if "parents" in file and file["parents"][0] != root_folder_id:
#             parent_id = file["parents"][0]
#             current_id = parent_id
#         else:
#             if "parents" in file:
#                 parent_id = file["parents"][0]
#             break

#     return " / ".join(reversed(path_parts)), parent_id

# # ---------------- RECURSIVE LIST ----------------
# def list_all_items(drive_service, folder_id):
#     """Recursively fetch all files/folders under a folder."""
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime)"
#     ).execute()
#     items = results.get("files", [])
#     all_items = items.copy()

#     for item in items:
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             all_items.extend(list_all_items(drive_service, item["id"]))
#     return all_items

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15):
#     seen_items = {}

#     print(f"👀 Watching folder {folder_id} every {interval}s...")
#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)

#             for item in items:
#                 item_id = item["id"]
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")

#                 full_path, parent_id = get_full_path(drive_service, item_id, folder_id)

#                 # Decide link
#                 if item_type == "Folder":
#                     file_link = f"https://drive.google.com/drive/folders/{item_id}"
#                 else:
#                     file_link = f"https://drive.google.com/drive/folders/{parent_id}"

#                 if item_id not in seen_items:
#                     # NEW ITEM
#                     seen_items[item_id] = modified
#                     subject = f"🆕 New {item_type} Added: {item_name}"
#                     body = (
#                         f"A new {item_type.lower()} was added:\n\n"
#                         f"Name: {item_name}\n"
#                         f"Path: {full_path}\n"
#                         f"Folder Link: {file_link}\n"
#                         f"ID: {item_id}\n"
#                         f"Created: {created}\n"
#                         f"Modified: {modified}"
#                     )
#                     send_email(gmail_service, recipient_email, subject, body)

#                 elif seen_items[item_id] != modified:
#                     # UPDATED ITEM
#                     seen_items[item_id] = modified
#                     subject = f"✏️ {item_type} Updated: {item_name}"
#                     body = (
#                         f"A {item_type.lower()} was updated:\n\n"
#                         f"Name: {item_name}\n"
#                         f"Path: {full_path}\n"
#                         f"Folder Link: {file_link}\n"
#                         f"ID: {item_id}\n"
#                         f"Modified: {modified}"
#                     )
#                     send_email(gmail_service, recipient_email, subject, body)

#             time.sleep(interval)
#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)
#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     folder_id = extract_folder_id(folder_input)

#     # Start watching
#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15)





#----------------------batch mail------------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- SEND EMAIL ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body)
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- PATH + PARENT ----------------
# def get_full_path(drive_service, file_id, root_folder_id):
#     path_parts = []
#     current_id = file_id
#     parent_id = root_folder_id

#     while True:
#         file = drive_service.files().get(
#             fileId=current_id,
#             fields="id, name, parents, mimeType"
#         ).execute()
#         path_parts.append(file["name"])

#         if "parents" in file and file["parents"][0] != root_folder_id:
#             parent_id = file["parents"][0]
#             current_id = parent_id
#         else:
#             if "parents" in file:
#                 parent_id = file["parents"][0]
#             break

#     return " / ".join(reversed(path_parts)), parent_id

# # ---------------- RECURSIVE LIST ----------------
# def list_all_items(drive_service, folder_id):
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime)"
#     ).execute()
#     items = results.get("files", [])
#     all_items = items.copy()

#     for item in items:
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             all_items.extend(list_all_items(drive_service, item["id"]))
#     return all_items

# # ---------------- WATCH FOLDER (BATCHED EMAIL) ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
#     seen_items = {}
#     changes_batch = []

#     last_email_time = time.time()
#     print(f"👀 Watching folder {folder_id} every {interval}s...")

#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)
#             current_time = time.time()

#             for item in items:
#                 item_id = item["id"]
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")

#                 full_path, parent_id = get_full_path(drive_service, item_id, folder_id)

#                 # Decide link
#                 if item_type == "Folder":
#                     file_link = f"https://drive.google.com/drive/folders/{item_id}"
#                 else:
#                     file_link = f"https://drive.google.com/drive/folders/{parent_id}"

#                 if item_id not in seen_items:
#                     # NEW ITEM
#                     seen_items[item_id] = modified
#                     changes_batch.append(
#                         f"🆕 New {item_type}: {item_name}\nPath: {full_path}\nFolder Link: {file_link}\nCreated: {created}\nModified: {modified}\n"
#                     )

#                 elif seen_items[item_id] != modified:
#                     # UPDATED ITEM
#                     seen_items[item_id] = modified
#                     changes_batch.append(
#                         f"✏️ Updated {item_type}: {item_name}\nPath: {full_path}\nFolder Link: {file_link}\nModified: {modified}\n"
#                     )

#             # Send batch email if interval reached and there are changes
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 body = "\n\n".join(changes_batch)
#                 send_email(gmail_service, recipient_email, subject, body)
#                 changes_batch = []
#                 last_email_time = current_time

#             time.sleep(interval)

#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"
#     folder_id = extract_folder_id(folder_input)

#     # Watch folder, send batch email every 60 seconds
#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60)















#------------------------------pkl file------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- HELPER: Get Full Path ----------------
# def get_full_path(drive_service, item_id, root_id, path=""):
#     if item_id == root_id:
#         return f"/{path}", item_id

#     item = drive_service.files().get(fileId=item_id, fields="id, name, parents").execute()
#     parent_id = item.get("parents", [root_id])[0]
#     return get_full_path(drive_service, parent_id, root_id, f"{item['name']}/{path}" if path else item['name'])

# # ---------------- HELPER: List All Items Recursively ----------------
# def list_all_items(drive_service, folder_id):
#     items = []
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime)"
#     ).execute()
#     for item in results.get("files", []):
#         items.append(item)
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             items.extend(list_all_items(drive_service, item["id"]))
#     return items

# # ---------------- SEND EMAIL ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body)
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- STATE PERSISTENCE ----------------
# STATE_FILE = "seen_items.pickle"

# def load_seen_items():
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open(STATE_FILE, "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()
#     changes_batch = []
#     last_email_time = time.time()
#     print(f"👀 Watching folder {folder_id} every {interval}s...")

#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)
#             current_time = time.time()

#             for item in items:
#                 item_id = item["id"]
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")

#                 full_path, parent_id = get_full_path(drive_service, item_id, folder_id)

#                 # Folder link or file link
#                 if item_type == "Folder":
#                     link = f"https://drive.google.com/drive/folders/{item_id}"
#                 else:
#                     link = f"https://drive.google.com/drive/folders/{parent_id}"

#                 if item_id not in seen_items:
#                     seen_items[item_id] = modified
#                     changes_batch.append(f"🆕 New {item_type}: {item_name}\nPath: {full_path}\nLink: {link}\nCreated: {created}\nModified: {modified}\n")
#                 elif seen_items[item_id] != modified:
#                     seen_items[item_id] = modified
#                     changes_batch.append(f"✏️ Updated {item_type}: {item_name}\nPath: {full_path}\nLink: {link}\nModified: {modified}\n")

#             # Send batch email
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 body = "\n\n".join(changes_batch)
#                 send_email(gmail_service, recipient_email, subject, body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(interval)
#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     # Replace with your folder URL/ID and recipient email
#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     folder_id = extract_folder_id(folder_input)

#     # Start watching
#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60)






#-----------------------------TABLE----------------------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- HELPER: Get Full Path ----------------
# def get_full_path(drive_service, item_id, root_id, path=""):
#     if item_id == root_id:
#         return f"/{path}", item_id
#     item = drive_service.files().get(fileId=item_id, fields="id, name, parents").execute()
#     parent_id = item.get("parents", [root_id])[0]
#     return get_full_path(drive_service, parent_id, root_id, f"{item['name']}/{path}" if path else item['name'])

# # ---------------- HELPER: List All Items Recursively ----------------
# def list_all_items(drive_service, folder_id):
#     items = []
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime)"
#     ).execute()
#     for item in results.get("files", []):
#         items.append(item)
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             items.extend(list_all_items(drive_service, item["id"]))
#     return items

# # ---------------- SEND HTML EMAIL ----------------
# def send_email(service, to, subject, html_body):
#     message = MIMEText(html_body, "html")
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- STATE PERSISTENCE ----------------
# STATE_FILE = "seen_items.pickle"

# def load_seen_items():
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open(STATE_FILE, "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()
#     changes_batch = []
#     last_email_time = time.time()
#     print(f"👀 Watching folder {folder_id} every {interval}s...")

#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)
#             current_time = time.time()

#             for item in items:
#                 item_id = item["id"]
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")

#                 full_path, parent_id = get_full_path(drive_service, item_id, folder_id)

#                 if item_type == "Folder":
#                     link = f"https://drive.google.com/drive/folders/{item_id}"
#                 else:
#                     link = f"https://drive.google.com/file/d/{item_id}/view"

#                 if item_id not in seen_items:
#                     seen_items[item_id] = modified
#                     changes_batch.append({
#                         "type": f"🆕 {item_type}",
#                         "name": item_name,
#                         "link": link,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified
#                     })
#                 elif seen_items[item_id] != modified:
#                     seen_items[item_id] = modified
#                     changes_batch.append({
#                         "type": f"✏️ Updated {item_type}",
#                         "name": item_name,
#                         "link": link,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified
#                     })

#             # Send batch email
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 html_rows = ""
#                 for c in changes_batch:
#                     html_rows += f"<tr><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td></tr>"
#                 html_body = f"""
#                 <h2>Google Drive Updates</h2>
#                 <table border='1' cellpadding='5' cellspacing='0'>
#                 <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th></tr>
#                 {html_rows}
#                 </table>
#                 """
#                 send_email(gmail_service, recipient_email, subject, html_body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(interval)
#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"  # Replace with your folder ID
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     folder_id = extract_folder_id(folder_input)

#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60)


















#----------------------------name heheh----------------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- HELPER: Get Full Path ----------------
# def get_full_path(drive_service, item_id, root_id, path=""):
#     if item_id == root_id:
#         return f"/{path}", item_id
#     item = drive_service.files().get(fileId=item_id, fields="id, name, parents").execute()
#     parent_id = item.get("parents", [root_id])[0]
#     return get_full_path(drive_service, parent_id, root_id, f"{item['name']}/{path}" if path else item['name'])

# # ---------------- HELPER: List All Items Recursively ----------------
# def list_all_items(drive_service, folder_id):
#     items = []
#     results = drive_service.files().list(
#         q=f"'{folder_id}' in parents and trashed=false",
#         fields="files(id, name, mimeType, createdTime, modifiedTime, lastModifyingUser(displayName, emailAddress))"
#     ).execute()
#     for item in results.get("files", []):
#         items.append(item)
#         if item["mimeType"] == "application/vnd.google-apps.folder":
#             items.extend(list_all_items(drive_service, item["id"]))
#     return items

# # ---------------- SEND HTML EMAIL ----------------
# def send_email(service, to, subject, html_body):
#     message = MIMEText(html_body, "html")
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- STATE PERSISTENCE ----------------
# STATE_FILE = "seen_items.pickle"

# def load_seen_items():
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open(STATE_FILE, "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()
#     changes_batch = []
#     last_email_time = time.time()
#     print(f"👀 Watching folder {folder_id} every {interval}s...")

#     while True:
#         try:
#             items = list_all_items(drive_service, folder_id)
#             current_time = time.time()

#             for item in items:
#                 item_id = item["id"]
#                 item_name = item["name"]
#                 item_type = "Folder" if item["mimeType"] == "application/vnd.google-apps.folder" else "File"
#                 created = item.get("createdTime")
#                 modified = item.get("modifiedTime")
#                 user = item.get("lastModifyingUser", {})
#                 user_info = f"{user.get('displayName', 'Unknown')} ({user.get('emailAddress', 'N/A')})"

#                 full_path, parent_id = get_full_path(drive_service, item_id, folder_id)

#                 if item_type == "Folder":
#                     link = f"https://drive.google.com/drive/folders/{item_id}"
#                 else:
#                     link = f"https://drive.google.com/file/d/{item_id}/view"

#                 if item_id not in seen_items:
#                     seen_items[item_id] = modified
#                     changes_batch.append({
#                         "type": f"🆕 {item_type}",
#                         "name": item_name,
#                         "link": link,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified,
#                         "user": user_info
#                     })
#                 elif seen_items[item_id] != modified:
#                     seen_items[item_id] = modified
#                     changes_batch.append({
#                         "type": f"✏️ Updated {item_type}",
#                         "name": item_name,
#                         "link": link,
#                         "path": full_path,
#                         "created": created,
#                         "modified": modified,
#                         "user": user_info
#                     })

#             # Send batch email
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 html_rows = ""
#                 for c in changes_batch:
#                     html_rows += f"<tr><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td><td>{c['user']}</td></tr>"
#                 html_body = f"""
#                 <h2>Google Drive Updates</h2>
#                 <table border='1' cellpadding='5' cellspacing='0'>
#                 <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th><th>Committed By</th></tr>
#                 {html_rows}
#                 </table>
#                 """
#                 send_email(gmail_service, recipient_email, subject, html_body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(interval)
#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"  # Replace with your folder ID
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     folder_id = extract_folder_id(folder_input)

#     watch_folder(drive_service, gmail_service, folder_id, recipient_email, interval=15, email_batch_interval=60)
































#------------------------FINAL MAIL PENDING------------------------------

# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)

#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)

#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     if match:
#         return match.group(1)
#     return folder_input

# # ---------------- HELPER: Full Path ----------------
# def get_full_path(drive_service, file_id, top_folder_id):
#     path_parts = []

#     def build_path(fid):
#         if fid == top_folder_id:
#             return
#         try:
#             file = drive_service.files().get(fileId=fid, fields="id, name, parents").execute()
#             path_parts.insert(0, file["name"])
#             parents = file.get("parents", [])
#             if parents:
#                 build_path(parents[0])
#         except:
#             path_parts.insert(0, "Unknown")

#     build_path(file_id)
#     return "/" + "/".join(path_parts)

# # ---------------- EMAIL SENDER ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body, "html")
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- SAVE/LOAD SEEN ITEMS ----------------
# def load_seen_items():
#     if os.path.exists("seen_items.pkl"):
#         with open("seen_items.pkl", "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open("seen_items.pkl", "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, scan_interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()  # file_id: {"modified":..., "full_path":..., "name":...}
#     changes_batch = []
#     last_email_time = time.time()

#     print(f"👀 Watching folder {top_folder_id} every {scan_interval}s...")

#     while True:
#         try:
#             current_scan_ids = set()

#             def scan_folder(folder_id):
#                 results = drive_service.files().list(
#                     q=f"'{folder_id}' in parents and trashed=false",
#                     fields="files(id, name, mimeType, modifiedTime, createdTime, owners)",
#                     pageSize=1000
#                 ).execute()
#                 items = results.get("files", [])

#                 for item in items:
#                     file_id = item["id"]
#                     current_scan_ids.add(file_id)
#                     full_path = get_full_path(drive_service, file_id, top_folder_id)

#                     owner_info = item.get("owners", [{}])[0]
#                     user_name = owner_info.get("displayName", "Unknown")
#                     user_email = owner_info.get("emailAddress", "Unknown")
#                     user_display = f"{user_name} <{user_email}>"

#                     is_folder = item["mimeType"].startswith("application/vnd.google-apps.folder")
#                     previous_data = seen_items.get(file_id)

#                     if not previous_data:
#                         changes_batch.append({
#                             "type": "🆕 Folder" if is_folder else "🆕 File",
#                             "name": item["name"],
#                             "path": full_path,
#                             "created": item.get("createdTime"),
#                             "modified": item.get("modifiedTime"),
#                             "user": user_display,
#                             "link": f"https://drive.google.com/drive/folders/{file_id}" if is_folder else f"https://drive.google.com/file/d/{file_id}/view"
#                         })
#                     elif previous_data["modified"] != item["modifiedTime"]:
#                         changes_batch.append({
#                             "type": "⚡ Updated",
#                             "name": item["name"],
#                             "path": full_path,
#                             "created": item.get("createdTime"),
#                             "modified": item.get("modifiedTime"),
#                             "user": user_display,
#                             "link": f"https://drive.google.com/drive/folders/{file_id}" if is_folder else f"https://drive.google.com/file/d/{file_id}/view"
#                         })

#                     seen_items[file_id] = {"modified": item["modifiedTime"], "full_path": full_path, "name": item["name"]}

#                     if is_folder:
#                         scan_folder(file_id)

#             scan_folder(top_folder_id)

#             # Detect deletions
#             deleted_ids = set(seen_items.keys()) - current_scan_ids
#             for del_id in deleted_ids:
#                 deleted_item = seen_items[del_id]
#                 changes_batch.append({
#                     "type": "❌ Deleted",
#                     "name": deleted_item["name"],
#                     "path": deleted_item["full_path"],
#                     "created": "N/A",
#                     "modified": "N/A",
#                     "user": "N/A",
#                     "link": "#"
#                 })
#                 del seen_items[del_id]

#             # Send batch email if interval passed
#             current_time = time.time()
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 html_rows = ""
#                 for c in changes_batch:
#                     if "🆕" in c["type"]:
#                         color = "#d4edda"
#                     elif "⚡" in c["type"]:
#                         color = "#fff3cd"
#                     elif "❌" in c["type"]:
#                         color = "#f8d7da"
#                     else:
#                         color = "#ffffff"
#                     html_rows += f"<tr style='background-color:{color}'><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td><td>{c['user']}</td></tr>"

#                 html_body = f"""
#                 <h2>Google Drive Updates</h2>
#                 <table border='1' cellpadding='5' cellspacing='0'>
#                 <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th><th>Committed By</th></tr>
#                 {html_rows}
#                 </table>
#                 """
#                 send_email(gmail_service, recipient_email, subject, html_body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(scan_interval)

#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(scan_interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     top_folder_id = extract_folder_id(folder_input)
#     watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, scan_interval=15, email_batch_interval=60)
































#----------------------------------------------FINAL-----------------------------------------------------------


# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from email.mime.text import MIMEText
# import base64, os, pickle, re, time

# # ---------------- AUTHENTICATION ----------------
# def authenticate(scopes):
#     creds = None
#     if os.path.exists("token.pickle"):
#         with open("token.pickle", "rb") as token:
#             creds = pickle.load(token)

#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scopes)
#             creds = flow.run_local_server(port=0)
#         with open("token.pickle", "wb") as token:
#             pickle.dump(creds, token)
#     return creds

# # ---------------- HELPER: Extract Folder ID ----------------
# def extract_folder_id(folder_input: str) -> str:
#     match = re.search(r"folders/([a-zA-Z0-9_-]+)", folder_input)
#     return match.group(1) if match else folder_input

# # ---------------- HELPER: Full Path ----------------
# def get_full_path(drive_service, file_id, top_folder_id):
#     path_parts = []

#     def build_path(fid):
#         if fid == top_folder_id:
#             return
#         try:
#             file = drive_service.files().get(fileId=fid, fields="id, name, parents").execute()
#             path_parts.insert(0, file["name"])
#             parents = file.get("parents", [])
#             if parents:
#                 build_path(parents[0])
#         except:
#             path_parts.insert(0, "Unknown")

#     build_path(file_id)
#     return "/" + "/".join(path_parts)

# # ---------------- EMAIL SENDER ----------------
# def send_email(service, to, subject, body):
#     message = MIMEText(body, "html")
#     message["to"] = to
#     message["subject"] = subject
#     raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
#     service.users().messages().send(userId="me", body={"raw": raw}).execute()
#     print(f"📧 Email sent to {to}: {subject}")

# # ---------------- SAVE/LOAD SEEN ITEMS ----------------
# def load_seen_items():
#     if os.path.exists("seen_items.pkl"):
#         with open("seen_items.pkl", "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_seen_items(seen_items):
#     with open("seen_items.pkl", "wb") as f:
#         pickle.dump(seen_items, f)

# # ---------------- WATCH FOLDER ----------------
# def watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, scan_interval=15, email_batch_interval=60):
#     seen_items = load_seen_items()
#     changes_batch = []
#     last_email_time = time.time()
#     print(f"👀 Watching folder {top_folder_id} every {scan_interval}s...")

#     while True:
#         try:
#             current_scan_ids = set()

#             def scan_folder(folder_id):
#                 results = drive_service.files().list(
#                     q=f"'{folder_id}' in parents and trashed=false",
#                     fields="files(id, name, mimeType, modifiedTime, createdTime, owners)",
#                     pageSize=1000
#                 ).execute()
#                 items = results.get("files", [])

#                 for item in items:
#                     file_id = item["id"]
#                     current_scan_ids.add(file_id)
#                     full_path = get_full_path(drive_service, file_id, top_folder_id)

#                     is_folder = item["mimeType"].startswith("application/vnd.google-apps.folder")
#                     owner_info = item.get("owners", [{}])[0]
#                     user_name = owner_info.get("displayName", "Unknown")
#                     user_email = owner_info.get("emailAddress", "Unknown")
#                     user_display = f"{user_name} <{user_email}>"

#                     # For files, use Revisions API to get last modifying user
#                     if not is_folder:
#                         try:
#                             revisions = drive_service.revisions().list(
#                                 fileId=file_id,
#                                 fields="revisions(lastModifyingUser(displayName,emailAddress))"
#                             ).execute()
#                             revs = revisions.get("revisions", [])
#                             if revs:
#                                 last_user = revs[-1].get("lastModifyingUser", {})
#                                 user_name = last_user.get("displayName", user_name)
#                                 user_email = last_user.get("emailAddress", user_email)
#                                 user_display = f"{user_name} <{user_email}>"
#                         except:
#                             pass

#                     previous_data = seen_items.get(file_id)
#                     if not previous_data:
#                         changes_batch.append({
#                             "type": "🆕 Folder" if is_folder else "🆕 File",
#                             "name": item["name"],
#                             "path": full_path,
#                             "created": item.get("createdTime"),
#                             "modified": item.get("modifiedTime"),
#                             "user": user_display,
#                             "link": f"https://drive.google.com/drive/folders/{file_id}" if is_folder else f"https://drive.google.com/file/d/{file_id}/view"
#                         })
#                     elif previous_data["modified"] != item["modifiedTime"]:
#                         changes_batch.append({
#                             "type": "⚡ Updated",
#                             "name": item["name"],
#                             "path": full_path,
#                             "created": item.get("createdTime"),
#                             "modified": item.get("modifiedTime"),
#                             "user": user_display,
#                             "link": f"https://drive.google.com/drive/folders/{file_id}" if is_folder else f"https://drive.google.com/file/d/{file_id}/view"
#                         })

#                     seen_items[file_id] = {"modified": item["modifiedTime"], "full_path": full_path, "name": item["name"]}

#                     if is_folder:
#                         scan_folder(file_id)

#             scan_folder(top_folder_id)

#             # Detect deletions
#             deleted_ids = set(seen_items.keys()) - current_scan_ids
#             for del_id in deleted_ids:
#                 deleted_item = seen_items[del_id]
#                 changes_batch.append({
#                     "type": "❌ Deleted",
#                     "name": deleted_item["name"],
#                     "path": deleted_item["full_path"],
#                     "created": "N/A",
#                     "modified": "N/A",
#                     "user": "N/A",
#                     "link": "#"
#                 })
#                 del seen_items[del_id]

#             # Send batch email if interval passed
#             current_time = time.time()
#             if current_time - last_email_time >= email_batch_interval and changes_batch:
#                 subject = f"📢 Google Drive Updates (last {email_batch_interval} sec)"
#                 html_rows = ""
#                 for c in changes_batch:
#                     color = "#ffffff"
#                     if "🆕" in c["type"]:
#                         color = "#d4edda"
#                     elif "⚡" in c["type"]:
#                         color = "#fff3cd"
#                     elif "❌" in c["type"]:
#                         color = "#f8d7da"
#                     html_rows += f"<tr style='background-color:{color}'><td>{c['type']}</td><td><a href='{c['link']}' target='_blank'>{c['name']}</a></td><td>{c['path']}</td><td>{c['created']}</td><td>{c['modified']}</td><td>{c['user']}</td></tr>"

#                 html_body = f"""
#                 <h2>Google Drive Updates</h2>
#                 <table border='1' cellpadding='5' cellspacing='0'>
#                 <tr><th>Change Type</th><th>Name</th><th>Path</th><th>Created</th><th>Modified</th><th>Committed By</th></tr>
#                 {html_rows}
#                 </table>
#                 """
#                 send_email(gmail_service, recipient_email, subject, html_body)
#                 changes_batch = []
#                 last_email_time = current_time
#                 save_seen_items(seen_items)

#             time.sleep(scan_interval)

#         except Exception as e:
#             print(f"⚠️ Error: {str(e)}")
#             time.sleep(scan_interval)

# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     SCOPES = [
#         "https://www.googleapis.com/auth/drive.metadata.readonly",
#         "https://www.googleapis.com/auth/gmail.send",
#     ]

#     creds = authenticate(SCOPES)
#     drive_service = build("drive", "v3", credentials=creds)
#     gmail_service = build("gmail", "v1", credentials=creds)

#     folder_input = "1ICmbTGt4gJwY_Vs-9HDJBEIvpL2ZT7Xn"  # Replace with your folder ID
#     recipient_email = "gmindia.tn.ml.gthangavel@gmail.com"

#     top_folder_id = extract_folder_id(folder_input)
#     watch_folder(drive_service, gmail_service, top_folder_id, recipient_email, scan_interval=15, email_batch_interval=60)















