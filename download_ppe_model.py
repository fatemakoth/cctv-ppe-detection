import subprocess
import sys
import os
import glob
import shutil

def install_roboflow():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])

def find_images_root(base):
    """Return the directory that contains train/valid/test image folders, or None."""
    for root, dirs, files in os.walk(base):
        if "train" in dirs and "valid" in dirs:
            return root
    return None

def flatten_into_ppe_model(found_root):
    """Move train/valid/test from found_root into ppe_model/ directly."""
    dest = "ppe_model"
    for split in ("train", "valid", "test"):
        src = os.path.join(found_root, split)
        dst = os.path.join(dest, split)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"  Moved {src} → {dst}")

    # Copy over the downloaded data.yaml if ours is still the stub
    src_yaml = os.path.join(found_root, "data.yaml")
    if os.path.exists(src_yaml) and found_root != dest:
        shutil.copy2(src_yaml, os.path.join(dest, "data.yaml"))

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

                # SDK may nest files in a subdirectory — find and flatten
                found_root = find_images_root("ppe_model")
                if found_root and found_root != "ppe_model":
                    print(f"[INFO] Files landed in {found_root} — moving to ppe_model/...")
                    flatten_into_ppe_model(found_root)
                    # Clean up the now-empty subdirectory
                    try:
                        shutil.rmtree(found_root)
                    except Exception:
                        pass
                    found_root = "ppe_model"

                # Final check
                train_imgs = glob.glob("ppe_model/train/images/*")
                if train_imgs:
                    print(f"\n[DONE] Dataset ready in ppe_model/  ({len(train_imgs)} training images)")
                    downloaded = True
                    break
                else:
                    print(f"  [WARN] No training images found after download. Trying next version...")
            except Exception as e:
                print(f"  [WARN] version {v} failed: {e}")
                continue

        if downloaded:
            break

    if not downloaded:
        print("\n[ERROR] Could not download any PPE dataset automatically.")
        print("Manual steps:")
        print("  1. Go to universe.roboflow.com")
        print("  2. Search 'construction site safety' or 'PPE detection'")
        print("  3. Open a project > Versions > Export Dataset")
        print("  4. Format: YOLOv8, download as zip")
        print("  5. Extract into ppe_model/ so you have:")
        print("       ppe_model/train/images/")
        print("       ppe_model/valid/images/")

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
    print("Next step: python train_ppe.py")
