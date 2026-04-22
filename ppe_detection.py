import cv2
import argparse
import os
import threading
import time
from collections import deque
from ultralytics import YOLO

# ── config ─────────────────────────────────────────────────────────────────────

PERSON_MODEL    = "yolov8s.pt"       # strong pretrained COCO model for person detection
PPE_MODEL_PATH  = "ppe_model/ppe_best.pt"

PERSON_CONF     = 0.50
PPE_CONF        = 0.25               # low — our model is undertrained, cast wide net
MIN_BOX_HEIGHT  = 80
MIN_ASPECT      = 0.6
SMOOTH_FRAMES   = 10
IOU_MATCH       = 0.25

PERSON_CLASS    = 0                  # COCO class index for person in yolov8s

HAS_HELMET  = "Hardhat"
NO_HELMET   = "NO-Hardhat"
HAS_VEST    = "Safety Vest"
NO_VEST     = "NO-Safety Vest"
PPE_CLASSES = {HAS_HELMET, NO_HELMET, HAS_VEST, NO_VEST}


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

def overlap_frac(ppe_box, person_box):
    """Fraction of ppe_box that lies inside person_box."""
    ix1, iy1 = max(ppe_box[0], person_box[0]), max(ppe_box[1], person_box[1])
    ix2, iy2 = min(ppe_box[2], person_box[2]), min(ppe_box[3], person_box[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area  = max(1, (ppe_box[2]-ppe_box[0]) * (ppe_box[3]-ppe_box[1]))
    return inter / area


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
    if c["MISSING"] > 0:   return "MISSING"
    if c["OK"] > c["?"]:   return "OK"
    return "?"


# ── detection ──────────────────────────────────────────────────────────────────

def detect_frame(frame, person_model, ppe_model, ppe_names, frame_w, frame_h):
    # Step 1: detect persons with strong pretrained model
    p_results = person_model(frame, verbose=False, conf=PERSON_CONF, classes=[PERSON_CLASS])[0]
    persons = []
    for box in p_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        bh, bw = y2 - y1, x2 - x1
        if bw > 0 and bh >= MIN_BOX_HEIGHT and bh / max(bw, 1) >= MIN_ASPECT:
            persons.append((x1, y1, x2, y2))

    if not persons:
        return []

    # Step 2: run PPE model on full frame to get all PPE item locations
    ppe_results = ppe_model(frame, verbose=False, conf=PPE_CONF)[0]
    ppe_hits = []
    for box in ppe_results.boxes:
        cls  = ppe_names[int(box.cls)]
        if cls not in PPE_CLASSES:
            continue
        b = tuple(map(int, box.xyxy[0]))
        ppe_hits.append((b, cls))

    # Step 3: associate PPE items to persons by overlap
    detections = []
    for pbox in persons:
        helmets = []
        vests   = []
        for ppe_box, cls in ppe_hits:
            if overlap_frac(ppe_box, pbox) < 0.35:
                continue
            if cls in (HAS_HELMET, NO_HELMET):
                helmets.append(cls)
            elif cls in (HAS_VEST, NO_VEST):
                vests.append(cls)

        helmet = _resolve(helmets, HAS_HELMET, NO_HELMET)
        vest   = _resolve(vests,   HAS_VEST,   NO_VEST)
        detections.append((pbox, helmet, vest))

    return detections

def _resolve(labels, pos, neg):
    if not labels:      return "?"
    has = any(l == pos for l in labels)
    no  = any(l == neg for l in labels)
    if has and not no:  return "OK"
    if no:              return "MISSING"
    return "?"


# ── drawing ────────────────────────────────────────────────────────────────────

def _col(s):
    return (0, 200, 0) if s == "OK" else (0, 0, 255) if s == "MISSING" else (0, 165, 255)

def draw_person(frame, box, helmet, vest, pid):
    x1, y1, x2, y2 = box
    bc = (0, 0, 255) if (helmet == "MISSING" or vest == "MISSING") else \
         (0, 200, 0) if (helmet == "OK" and vest == "OK") else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
    for i, (txt, col) in enumerate([
        (f"P{pid}",           bc),
        (f"Helmet: {helmet}", _col(helmet)),
        (f"Vest  : {vest}",   _col(vest)),
    ]):
        cv2.putText(frame, txt, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    if helmet == "MISSING" or vest == "MISSING":
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ── main ───────────────────────────────────────────────────────────────────────

def run(source):
    print("[INFO] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    ppe_model    = YOLO(PPE_MODEL_PATH)
    ppe_names    = ppe_model.names
    print(f"[INFO] PPE classes: {list(ppe_names.values())}")
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

        violations = 0
        for pid, (box, helmet, vest) in enumerate(smoothed, 1):
            draw_person(frame, box, helmet, vest, pid)
            if helmet == "MISSING" or vest == "MISSING":
                violations += 1

        if violations:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(frame,
                        f"PPE VIOLATION — {violations} person(s) missing PPE",
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
