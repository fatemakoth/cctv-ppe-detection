import cv2
import json
import argparse
import os
import threading
import time
from collections import deque
from ultralytics import YOLO

PERSON_MODEL    = "yolov8s.pt"
PPE_MODEL_PATH  = "ppe_model/ppe_best.pt"

PERSON_CLASS    = 0
HAS_HELMET      = "Hardhat"
NO_HELMET       = "NO-Hardhat"
HAS_VEST        = "Safety Vest"
NO_VEST         = "NO-Safety Vest"

PPE_CONF          = 0.30
PERSON_CONF       = 0.50
MIN_BOX_HEIGHT_PX = 80
MIN_ASPECT_RATIO  = 0.6
SMOOTH_FRAMES     = 8   # vote over last N frames per tracked person
IOU_MATCH_THRESH  = 0.3 # min IoU to consider same person across frames


# ── threaded frame reader ──────────────────────────────────────────────────────

class RTSPReader:
    """Runs capture in a background thread, always serves the latest frame."""
    def __init__(self, source):
        self.cap = open_capture(source)
        self.ret   = False
        self.frame = None
        self._lock  = threading.Lock()
        self._stop  = False
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def _read_loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret   = ret
                self.frame = frame
            if not ret:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self._stop = True
        self.cap.release()


# ── temporal smoother ──────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)

class PersonTracker:
    """Matches detected persons across frames by IoU and smooths PPE status."""
    def __init__(self, n=SMOOTH_FRAMES):
        self.n       = n
        self.tracks  = {}   # track_id → {box, helmet_hist, vest_hist}
        self._next   = 0

    def update(self, boxes_statuses):
        """
        boxes_statuses: list of (box_xyxy, helmet_raw, vest_raw)
        Returns:        list of (box_xyxy, helmet_smooth, vest_smooth)
        """
        matched_ids = set()
        results     = []

        for box, h_raw, v_raw in boxes_statuses:
            best_id, best_iou = None, IOU_MATCH_THRESH
            for tid, t in self.tracks.items():
                s = iou(box, t["box"])
                if s > best_iou:
                    best_iou, best_id = s, tid

            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {
                    "box":        box,
                    "helmet":     deque(maxlen=self.n),
                    "vest":       deque(maxlen=self.n),
                }

            t = self.tracks[best_id]
            t["box"]    = box
            t["helmet"].append(h_raw)
            t["vest"].append(v_raw)
            matched_ids.add(best_id)

            results.append((box, smooth(t["helmet"]), smooth(t["vest"])))

        # drop stale tracks
        self.tracks = {k: v for k, v in self.tracks.items() if k in matched_ids}
        return results

def smooth(hist):
    """Majority vote over status history. Unverified '?' loses to explicit labels."""
    counts = {"OK": 0, "MISSING": 0, "?": 0}
    for s in hist:
        counts[s] += 1
    # MISSING wins if it's ever seen recently (safety-first)
    if counts["MISSING"] > 0:
        return "MISSING"
    if counts["OK"] > counts["?"]:
        return "OK"
    return "?"


# ── helpers ────────────────────────────────────────────────────────────────────

def open_capture(source):
    if not (isinstance(source, str) and source.startswith("rtsp")):
        return cv2.VideoCapture(source)

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
        cap = (cv2.VideoCapture(source, backend)
               if backend is not None else cv2.VideoCapture(source))
        if cap.isOpened():
            print(f"[INFO] RTSP opened with backend={backend}")
            return cap
        cap.release()
    return cv2.VideoCapture(source)

def check_ppe(crop, ppe_model):
    results = ppe_model(crop, verbose=False, conf=PPE_CONF)[0]
    names   = ppe_model.names
    detected = set(names[int(b.cls)] for b in results.boxes)

    has_helmet = HAS_HELMET in detected
    no_helmet  = NO_HELMET  in detected
    has_vest   = HAS_VEST   in detected
    no_vest    = NO_VEST    in detected

    helmet_status = "OK"      if (has_helmet and not no_helmet) else \
                    "MISSING" if no_helmet                       else "?"
    vest_status   = "OK"      if (has_vest   and not no_vest)   else \
                    "MISSING" if no_vest                         else "?"
    return helmet_status, vest_status

def status_color(s):
    return (0, 200, 0) if s == "OK" else (0, 0, 255) if s == "MISSING" else (0, 165, 255)

def draw_ppe_status(frame, x1, y1, x2, y2, helmet, vest, pid):
    if helmet == "MISSING" or vest == "MISSING":
        box_color = (0, 0, 255)
    elif helmet == "OK" and vest == "OK":
        box_color = (0, 200, 0)
    else:
        box_color = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    for i, (text, color) in enumerate([
        (f"P{pid}",           box_color),
        (f"Helmet: {helmet}", status_color(helmet)),
        (f"Vest  : {vest}",   status_color(vest)),
    ]):
        cv2.putText(frame, text, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if helmet == "MISSING" or vest == "MISSING":
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(source):
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        print(f"[INFO] Calibration loaded: {cal['pixels_per_cm']:.4f} px/cm")
    except FileNotFoundError:
        cal = None
        print("[WARN] No calibration.json — size filter uses pixel threshold only")

    print("[INFO] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    ppe_model    = YOLO(PPE_MODEL_PATH)
    print(f"[INFO] PPE classes: {list(ppe_model.names.values())}")
    print("[INFO] Press 'q' to quit.")

    cap = RTSPReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}  |  smoothing over {SMOOTH_FRAMES} frames")

    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        persons = person_model(frame, verbose=False, conf=PERSON_CONF)[0]

        raw = []
        for box in persons.boxes:
            if int(box.cls) != PERSON_CLASS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bh, bw  = y2 - y1, x2 - x1
            if bw == 0 or bh == 0 or bh < MIN_BOX_HEIGHT_PX or bh / bw < MIN_ASPECT_RATIO:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            h_raw, v_raw = check_ppe(crop, ppe_model)
            raw.append(((x1, y1, x2, y2), h_raw, v_raw))

        smoothed = tracker.update(raw)

        violation_count = 0
        for pid, (box, helmet, vest) in enumerate(smoothed, 1):
            x1, y1, x2, y2 = box
            draw_ppe_status(frame, x1, y1, x2, y2, helmet, vest, pid)
            if helmet == "MISSING" or vest == "MISSING":
                violation_count += 1

        if violation_count > 0:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(frame,
                        f"PPE VIOLATION — {violation_count} person(s) missing PPE",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"People: {len(smoothed)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
