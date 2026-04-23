import cv2
import argparse
import os
import threading
import time
from collections import deque
from ultralytics import YOLO

# ── config ─────────────────────────────────────────────────────────────────────

PERSON_MODEL    = "yolov8s.pt"
PPE_MODEL_PATH  = "ppe_model/ppe_best.pt"

PERSON_CONF     = 0.50
HELMET_OK_CONF  = 0.70      # must be very confident to say helmet is present
HELMET_NO_CONF  = 0.30      # lower bar to flag NO-Hardhat
VEST_OK_CONF    = 0.65
VEST_NO_CONF    = 0.30

MIN_BOX_HEIGHT  = 80
MIN_ASPECT      = 0.6
SMOOTH_FRAMES   = 8
IOU_MATCH       = 0.25

PERSON_CLASS = 0
HAS_HELMET   = "Hardhat"
NO_HELMET    = "NO-Hardhat"
HAS_VEST     = "Safety Vest"
NO_VEST      = "NO-Safety Vest"

# Body region fractions of person bounding box height
HEAD_TOP     = 0.0
HEAD_BOT     = 0.30   # top 30% = head
TORSO_TOP    = 0.15
TORSO_BOT    = 0.70   # 15-70% = torso/vest area


# ── threaded RTSP reader ───────────────────────────────────────────────────────

class FrameReader:
    def __init__(self, source):
        self.cap   = _open_cap(source)
        self.ret   = False
        self.frame = None
        self._lock = threading.Lock()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame
            if not ret:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def get(self, prop):  return self.cap.get(prop)
    def release(self):
        self._stop = True
        self.cap.release()

def _open_cap(source):
    if isinstance(source, str) and source.startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
            cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"[INFO] RTSP opened (backend={backend})")
                return cap
            cap.release()
    return cv2.VideoCapture(source)


# ── geometry ───────────────────────────────────────────────────────────────────

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def region_crop(frame, pbox, top_frac, bot_frac, frame_h):
    x1, y1, x2, y2 = pbox
    ph = y2 - y1
    ry1 = max(0, y1 + int(ph * top_frac))
    ry2 = min(frame_h, y1 + int(ph * bot_frac))
    crop = frame[ry1:ry2, x1:x2]
    return crop if crop.size > 0 else None


# ── temporal smoother ──────────────────────────────────────────────────────────

class PersonTracker:
    def __init__(self, n=SMOOTH_FRAMES):
        self.n      = n
        self.tracks = {}
        self._next  = 0

    def update(self, detections):
        matched = set()
        results = []
        for box, h, v in detections:
            best_id, best_s = None, IOU_MATCH
            for tid, t in self.tracks.items():
                s = iou(box, t["box"])
                if s > best_s:
                    best_s, best_id = s, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {"box": box,
                                        "helmet": deque(maxlen=self.n),
                                        "vest":   deque(maxlen=self.n)}
            t = self.tracks[best_id]
            t["box"] = box
            t["helmet"].append(h)
            t["vest"].append(v)
            matched.add(best_id)
            results.append((box, _vote(t["helmet"]), _vote(t["vest"])))
        self.tracks = {k: v for k, v in self.tracks.items() if k in matched}
        return results

def _vote(hist):
    c = {"OK": 0, "MISSING": 0, "?": 0}
    for s in hist: c[s] += 1
    if c["MISSING"] >= 1:  return "MISSING"
    if c["OK"] > c["?"]:  return "OK"
    return "?"


# ── region-based PPE detection ─────────────────────────────────────────────────

def check_helmet(frame, pbox, ppe_model, ppe_names, frame_h):
    """
    Crop the head region (top 30% of person box).
    Closed-world: if no Hardhat detected with sufficient confidence → MISSING.
    This avoids the model's bias of labelling every head as "Hardhat".
    """
    crop = region_crop(frame, pbox, HEAD_TOP, HEAD_BOT, frame_h)
    if crop is None:
        return "MISSING"

    results = ppe_model(crop, verbose=False, conf=HELMET_NO_CONF)[0]
    names   = ppe_names

    best_hardhat = 0.0
    best_no_hardhat = 0.0
    for box in results.boxes:
        cls  = names[int(box.cls)]
        conf = float(box.conf)
        if cls == HAS_HELMET:
            best_hardhat    = max(best_hardhat, conf)
        elif cls == NO_HELMET:
            best_no_hardhat = max(best_no_hardhat, conf)

    if best_no_hardhat >= HELMET_NO_CONF:
        return "MISSING"
    if best_hardhat >= HELMET_OK_CONF:
        return "OK"
    # Head visible but helmet not confidently detected → treat as MISSING
    return "MISSING"

def check_vest(frame, pbox, ppe_model, ppe_names, frame_h):
    """
    Crop the torso region (15-70% of person box).
    Closed-world: if no Safety Vest detected confidently → MISSING.
    """
    crop = region_crop(frame, pbox, TORSO_TOP, TORSO_BOT, frame_h)
    if crop is None:
        return "MISSING"

    results = ppe_model(crop, verbose=False, conf=VEST_NO_CONF)[0]
    names   = ppe_names

    best_vest    = 0.0
    best_no_vest = 0.0
    for box in results.boxes:
        cls  = names[int(box.cls)]
        conf = float(box.conf)
        if cls == HAS_VEST:
            best_vest    = max(best_vest, conf)
        elif cls == NO_VEST:
            best_no_vest = max(best_no_vest, conf)

    if best_no_vest >= VEST_NO_CONF:
        return "MISSING"
    if best_vest >= VEST_OK_CONF:
        return "OK"
    return "MISSING"

def detect_frame(frame, person_model, ppe_model, ppe_names, frame_w, frame_h):
    p_results = person_model(frame, verbose=False, conf=PERSON_CONF, classes=[PERSON_CLASS])[0]
    detections = []
    for box in p_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        bh, bw = y2 - y1, x2 - x1
        if bw == 0 or bh < MIN_BOX_HEIGHT or bh / max(bw, 1) < MIN_ASPECT:
            continue
        pbox  = (x1, y1, x2, y2)
        helmet = check_helmet(frame, pbox, ppe_model, ppe_names, frame_h)
        vest   = check_vest(frame, pbox, ppe_model, ppe_names, frame_h)
        detections.append((pbox, helmet, vest))
    return detections


# ── drawing ────────────────────────────────────────────────────────────────────

def _col(s):
    return (0, 200, 0) if s == "OK" else (0, 0, 255)

def draw_person(frame, box, helmet, vest, pid):
    x1, y1, x2, y2 = box
    all_ok = helmet == "OK" and vest == "OK"
    bc     = (0, 200, 0) if all_ok else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
    for i, (txt, col) in enumerate([
        (f"P{pid}",           bc),
        (f"Helmet: {helmet}", _col(helmet)),
        (f"Vest  : {vest}",   _col(vest)),
    ]):
        cv2.putText(frame, txt, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    if not all_ok:
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ── main ───────────────────────────────────────────────────────────────────────

def run(source):
    print("[INFO] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    ppe_model    = YOLO(PPE_MODEL_PATH)
    ppe_names    = ppe_model.names
    print("[INFO] Press 'q' to quit.\n")

    cap = FrameReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        raw      = detect_frame(frame, person_model, ppe_model, ppe_names, w, h)
        smoothed = tracker.update(raw)

        violations = sum(1 for _, h, v in smoothed if h != "OK" or v != "OK")

        for pid, (box, helmet, vest) in enumerate(smoothed, 1):
            draw_person(frame, box, helmet, vest, pid)

        if violations:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(frame,
                        f"PPE VIOLATION - {violations} person(s) missing PPE",
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
    parser.add_argument("--source", default="0")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
