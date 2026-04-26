"""
Google Sheets incident logger.

Setup (one time):
  1. Go to console.cloud.google.com → New project
  2. Enable: Google Sheets API + Google Drive API
  3. IAM & Admin → Service Accounts → Create → Download JSON key
  4. Rename the downloaded file to: credentials.json  (place in this folder)
  5. Run:  python sheets_logger.py
     It will print the Sheet URL and your service account email.
  6. Share the Google Sheet with that email (Editor access).

After setup, monitor.py and video_check.py will automatically log to the sheet.
If credentials.json is missing, incidents are printed to console only (no crash).
"""

import os
import cv2
from datetime import datetime

CREDENTIALS_FILE = "credentials.json"
SHEET_NAME       = "PPE Incidents"
SNAPSHOT_DIR     = "snapshots"
HEADERS          = ["Timestamp", "Camera Source", "Violation", "Person ID", "Detail", "Snapshot Path"]


class SheetsLogger:
    def __init__(self, snapshot_dir=SNAPSHOT_DIR):
        self.snapshot_dir = snapshot_dir
        self.sheet        = None
        os.makedirs(snapshot_dir, exist_ok=True)
        self._connect()

    def _connect(self):
        if not os.path.exists(CREDENTIALS_FILE):
            print(f"[WARN] {CREDENTIALS_FILE} not found — incidents printed to console only.")
            print("[WARN] See sheets_logger.py header for setup instructions.")
            return
        try:
            import gspread
            from google.oauth2.service_account import Credentials

            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ]
            creds  = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
            client = gspread.authorize(creds)

            try:
                wb = client.open(SHEET_NAME)
            except gspread.SpreadsheetNotFound:
                wb = client.create(SHEET_NAME)
                ws = wb.sheet1
                ws.append_row(HEADERS)
                ws.format("A1:F1", {"textFormat": {"bold": True}})

            self.sheet = wb.sheet1

            if not self.sheet.get_all_values():
                self.sheet.append_row(HEADERS)

            url = f"https://docs.google.com/spreadsheets/d/{wb.id}"
            print(f"[INFO] Google Sheets logging active:")
            print(f"[INFO]   {url}")

        except ImportError:
            print("[WARN] gspread not installed — run: pip install gspread google-auth")
        except Exception as e:
            print(f"[WARN] Google Sheets setup failed: {e}")
            print("[WARN] Falling back to console-only logging.")

    def log(self, source, violation, person_id, detail, frame):
        ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname     = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_P{person_id}_{violation}.jpg"
        snap_path = os.path.join(self.snapshot_dir, fname)
        cv2.imwrite(snap_path, frame)

        row = [ts, str(source), violation, str(person_id), detail, snap_path]

        if self.sheet is not None:
            try:
                self.sheet.append_row(row)
            except Exception as e:
                print(f"[WARN] Sheet append failed: {e}")

        print(f"[INCIDENT] {ts} | {violation} | P{person_id} | {detail}")

    def close(self):
        pass


# ── Quick setup / test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger = SheetsLogger()
    if logger.sheet is not None:
        print("\n[OK] Connection successful!")
        print(f"[OK] Service account email (share the sheet with this):")
        import json
        with open(CREDENTIALS_FILE) as f:
            print(f"     {json.load(f)['client_email']}")
    else:
        print("\n[INFO] Fix credentials.json and re-run to test.")
