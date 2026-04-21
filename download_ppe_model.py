import subprocess
import sys
import os

def install_roboflow():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])

def download_model(api_key):
    from roboflow import Roboflow

    print("[INFO] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    CANDIDATES = [
        ("roboflow-universe-projects", "construction-site-safety"),
        ("roboflow-universe-projects", "hard-hat-universe"),
        ("roboflow-universe-projects", "ppe-detection"),
        ("roboflow-universe-projects", "safety-ppe-detection"),
    ]

    downloaded = False
    for workspace_name, project_name in CANDIDATES:
        print(f"[INFO] Trying {workspace_name}/{project_name}...")
        try:
            project = rf.workspace(workspace_name).project(project_name)
        except Exception as e:
            print(f"  Skipping: {e}")
            continue

        for v in range(1, 15):
            try:
                version = project.version(v)
                print(f"[INFO] Found version {v}. Downloading...")
                version.download("yolov8", location="ppe_model")
                print(f"\n[DONE] Model saved to: ppe_model/")
                downloaded = True
                break
            except Exception:
                continue

        if downloaded:
            break

    if not downloaded:
        print("\n[ERROR] Could not download any PPE model automatically.")
        print("Please go to https://universe.roboflow.com and search 'PPE detection',")
        print("open any project, click Export > YOLOv8, and download manually to ppe_model/")

if __name__ == "__main__":
    print("=" * 55)
    print("  PPE Model Downloader — Roboflow Universe")
    print("=" * 55)
    print()
    print("You need a FREE Roboflow account to download the model.")
    print("Steps:")
    print("  1. Go to https://roboflow.com and sign up (free)")
    print("  2. Go to https://app.roboflow.com — click your profile (top right)")
    print("  3. Copy your API key")
    print()

    api_key = input("Paste your Roboflow API key here: ").strip()
    if not api_key:
        print("[ERROR] No API key entered. Exiting.")
        sys.exit(1)

    print("[INFO] Installing roboflow package...")
    install_roboflow()

    download_model(api_key)

    print()
    print("Next step: python ppe_detection.py --source 1")
