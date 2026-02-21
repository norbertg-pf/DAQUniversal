import os
import re
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal

class UploadThread(QThread):
    status_signal = pyqtSignal(str, str) 

    def __init__(self, file_path, gdrive_link, auth_method, is_export=False):
        super().__init__()
        self.file_path = file_path
        self.gdrive_link = gdrive_link
        self.auth_method = auth_method
        self.is_export = is_export

    def run(self):
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials as OAuthCredentials
            from google.oauth2.service_account import Credentials as ServiceAccountCredentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            auth_dir = os.path.join(os.getcwd(), 'auth')
            os.makedirs(auth_dir, exist_ok=True)
            base_folder_id = None
            match = re.search(r'folders/([a-zA-Z0-9_-]+)', self.gdrive_link)
            if match:
                base_folder_id = match.group(1)
            else:
                match = re.search(r'id=([a-zA-Z0-9_-]+)', self.gdrive_link)
                if match: base_folder_id = match.group(1)
                else:
                    self.status_signal.emit("Failed: Invalid Link", "orange")
                    return

            self.status_signal.emit("Authenticating...", "orange")
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            creds = None

            if self.auth_method == "OAuth 2.0 (User Login)":
                token_path = os.path.join(auth_dir, 'token.json')
                secret_path = os.path.join(auth_dir, 'client_secret.json')
                
                if os.path.exists(token_path):
                    creds = OAuthCredentials.from_authorized_user_file(token_path, SCOPES)
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        if not os.path.exists(secret_path):
                            self.status_signal.emit("Failed: auth/client_secret.json missing", "orange")
                            return
                        flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
                        creds = flow.run_local_server(port=0)
                    with open(token_path, 'w') as token:
                        token.write(creds.to_json())
            
            elif self.auth_method == "Service Account (Robot)":
                cred_path = os.path.join(auth_dir, 'credentials.json')
                if not os.path.exists(cred_path):
                    self.status_signal.emit("Failed: auth/credentials.json missing", "orange")
                    return
                creds = ServiceAccountCredentials.from_service_account_file(cred_path, scopes=SCOPES)
            else:
                self.status_signal.emit("Failed: Unknown Auth Method", "orange")
                return

            service = build('drive', 'v3', credentials=creds)
            date_str = datetime.now().strftime("%d_%m_%Y")
            
            def get_or_create_folder(folder_name, parent_id):
                query = f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                results = service.files().list(
                    q=query, spaces='drive', fields='nextPageToken, files(id, name)',
                    includeItemsFromAllDrives=True, supportsAllDrives=True, corpora='allDrives'
                ).execute()
                items = results.get('files', [])
                if not items:
                    file_metadata = {
                        'name': folder_name,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [parent_id]
                    }
                    folder = service.files().create(body=file_metadata, fields='id', supportsAllDrives=True).execute()
                    return folder.get('id')
                else:
                    return items[0]['id']

            self.status_signal.emit(f"Resolving Cloud Folder...", "orange")
            target_folder_id = get_or_create_folder(date_str, base_folder_id)
            if self.is_export:
                target_folder_id = get_or_create_folder("export", target_folder_id)

            self.status_signal.emit(f"Uploading file...", "orange")
            file_metadata = {'name': os.path.basename(self.file_path), 'parents': [target_folder_id]}
            media = MediaFileUpload(self.file_path, resumable=True)
            
            service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()
            self.status_signal.emit("Upload OK", "green")
            
        except Exception as e:
            self.status_signal.emit(f"Upload Failed", "red")
            print(f"[GDRIVE ERROR] {e}")
