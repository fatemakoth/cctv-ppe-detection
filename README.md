# Smart CCTV Safety Monitor

Detects two workplace safety violations in real time from a CCTV or webcam feed:
- **Workers standing on machinery** (height anomaly)
- **Missing PPE** — no helmet or safety vest

Runs fully on a laptop. Live dashboard accessible from a phone browser on the same Wi-Fi.

---

## Requirements

- Python 3.10+
- A webcam **or** a CCTV camera with an RTSP URL

---

## Installation

```bash
git clone <repo-url>
cd cctv-ppe-detection
pip install -r requirements.txt
```

Model weights (`yolov8n.pt`) download automatically on first run (~6 MB).

---

## Quick Start

### If you have a webcam (for testing)

```bash
# Find your camera index
python find_camera.py

# Test the feed (replace 1 with your index)
python test_feed.py --source 1
```

### If you have a CCTV camera (RTSP)

```bash
# Test your RTSP stream first
python test_feed.py --source "rtsp://admin:password@192.168.1.10:554/stream"
```

> **How to find your RTSP URL:**
> - Log in to your DVR/NVR web interface or its mobile app
> - Look under **Network > RTSP** or **Remote Access**
> - Format is usually: `rtsp://[username]:[password]@[camera-ip]:[port]/stream`
> - Default port is `554`. Default credentials are often `admin/admin` or printed on the device.

---

## Step-by-Step Usage

### Step 1 — Test your feed
```bash
# Webcam
python test_feed.py --source 1

# CCTV
python test_feed.py --source "rtsp://admin:password@192.168.1.10/stream"
```
Press `q` to quit. Confirm you see a live image.

---

### Step 2 — Detect people
```bash
# Webcam
python person_detection.py --source 1

# CCTV
python person_detection.py --source "rtsp://admin:password@192.168.1.10/stream"
```
Green boxes appear around every detected person.

---

### Step 3 — Calibrate the camera

Stand in front of the camera so your **full body** is visible.

```bash
# Webcam
python calibrate.py --source 1 --height 170

# CCTV
python calibrate.py --source "rtsp://admin:password@192.168.1.10/stream" --height 170
```

Replace `170` with your real height in cm.

**In the window:**
1. Press `s` to start the 5-second countdown, then step back
2. After the photo is taken, click the **top of your head**
3. Click the **bottom of your feet**
4. Press `s` to save — creates `calibration.json`

> Run calibration again if you move the camera.

---

### Step 4 — Run the full system *(coming soon)*

```bash
python monitor.py --source 1
# or
python monitor.py --source "rtsp://admin:password@192.168.1.10/stream"
```

Open `http://localhost:8501` on your laptop or `http://[laptop-ip]:8501` on your phone (same Wi-Fi).

---

## Camera Tips (for CCTV)

| Requirement | Details |
|---|---|
| Resolution | 720p minimum for reliable PPE detection |
| Camera angle | 30–45° downward angle or overhead works best |
| Network | Laptop must be on the same network as the DVR/NVR |
| RTSP port | Usually `554` — must be open on the DVR firewall |

---

## Project Structure

```
cctv-ppe-detection/
├── requirements.txt        # Python dependencies
├── find_camera.py          # Scan available camera indices
├── test_feed.py            # Test webcam or RTSP stream
├── person_detection.py     # YOLOv8 person detection
├── calibrate.py            # Camera height calibration
├── monitor.py              # Full pipeline + dashboard (Step 4+)
└── calibration.json        # Generated after calibration (not in git)
```
