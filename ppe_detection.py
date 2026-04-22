import cv2
import argparse
import os
import threading
import time
from collections import deque
from ultralytics import YOLO

# ── config ─────────────────────────────────────────────────────────────────────

PPE_MODEL_PATH  = "ppe_model/ppe_best.pt"
PPE_CONF        = 0.35      # confidence threshold for all PPE detections
PERSON_CONF     = 0.50
MIN_BOX_HEIGHT  = 80        # px — ignore tiny person detections
MIN_ASPECT      = 0.6       # ignore near-horizontal blobs
SMOOTH_FRAMES   = 10        # temporal vote window per tracked person
IOU_MATCH       = 0.25      # min IoU to re-identify same person next frame

# class names in our trained model
PERSON_CLS   = "Person"
HAS_HELMET   = "Hardhat"
NO_HELMET    = "NO-Hardhat"
HAS_VEST     = "Safety Vest"
NO_VEST      = "NO-Safety Vest"

PPE_CLASSES  = {HAS_HELMET, NO_HELMET, HAS_VEST, NO_VEST}


# ── threaded RTSP reader ───────────────────────────────────────────────────────

class FrameReader:
    """Background thread that continuously drains the stream buffer."""
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

    def get(self, prop):
        return self.cap.get(prop)

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


# ── geometry helpers ───────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

def overlap_fraction(ppe_box, person_box):
    """What fraction of the PPE box lies inside the person box."""
    ax1, ay1, ax2, ay2 = ppe_box
    bx1, by1, bx2, by2 = person_box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ppe_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / ppe_area


# ── temporal smoother ──────────────────────────────────────────────────────────

class PersonTracker:
    def __init__(self, n=SMOOTH_FRAMES):
        self.n      = n
        self.tracks = {}
        self._next  = 0

    def update(self, detections):
        """
        detections: list of (box, helmet_raw, vest_raw)
        returns:    list of (box, helmet_smooth, vest_smooth)
        """
        matched = set()
        results = []

        for box, h, v in detections:
            best_id, best_score = None, IOU_MATCH
            for tid, t in self.tracks.items():
                s = iou(box, t["box"])
                if s > best_score:
                    best_score, best_id = s, tid

            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {
                    "box":    box,
                    "helmet": deque(maxlen=self.n),
                    "vest":   deque(maxlen=self.n),
                }

            t = self.tracks[best_id]
            t["box"] = box
            t["helmet"].append(h)
            t["vest"].append(v)
            matched.add(best_id)
            results.append((box, _vote(t["helmet"]), _vote(t["vest"])))

        self.tracks = {k: v for k, v in self.tracks.items() if k in matched}
        return results

def _vote(hist):
    """Safety-first majority vote: MISSING wins if ever seen, then OK, then ?"""
    counts = {"OK": 0, "MISSING": 0, "?": 0}
    for s in hist:
        counts[s] += 1
    if counts["MISSING"] > 0:
        return "MISSING"
    if counts["OK"] > counts["?"]:
        return "OK"
    return "?"


# ── full-frame PPE detection ───────────────────────────────────────────────────

def detect_frame(frame, model, names):
    """
    Run PPE model on the full frame.
    Returns list of (person_box, helmet_status, vest_status).
    """
    results = model(frame, verbose=False, conf=min(PPE_CONF, PERSON_CONF))[0]

    persons  = []   # (box, conf)
    ppe_hits = []   # (box, class_name, conf)

    for box in results.boxes:
        cls  = names[int(box.cls)]
        conf = float(box.conf)
        b    = tuple(map(int, box.xyxy[0]))

        if cls == PERSON_CLS and conf >= PERSON_CONF:
            bh = b[3] - b[1]
            bw = b[2] - b[0]
            if bw > 0 and bh >= MIN_BOX_HEIGHT and bh / max(bw, 1) >= MIN_ASPECT:
                persons.append((b, conf))
        elif cls in PPE_CLASSES and conf >= PPE_CONF:
            ppe_hits.append((b, cls))

    detections = []
    for pbox, _ in persons:
        # collect PPE items whose center or majority falls inside this person box
        helmet_labels = []
        vest_labels   = []

        for ppe_box, cls in ppe_hits:
            frac = overlap_fraction(ppe_box, pbox)
            if frac < 0.4:
                continue
            if cls in (HAS_HELMET, NO_HELMET):
                helmet_labels.append(cls)
            elif cls in (HAS_VEST, NO_VEST):
                vest_labels.append(cls)

        helmet = _resolve(helmet_labels, HAS_HELMET, NO_HELMET)
        vest   = _resolve(vest_labels,   HAS_VEST,   NO_VEST)
        detections.append((pbox, helmet, vest))

    return detections

def _resolve(labels, positive, negative):
    """Given a list of detected PPE labels for one person, return OK/MISSING/?"""
    if not labels:
        return "?"
    has = any(l == positive for l in labels)
    no  = any(l == negative for l in labels)
    if has and not no:
        return "OK"
    if no:
        return "MISSING"
    return "?"


# ── drawing ────────────────────────────────────────────────────────────────────

def _color(status):
    return (0, 200, 0) if status == "OK" else (0, 0, 255) if status == "MISSING" else (0, 165, 255)

def draw_person(frame, box, helmet, vest, pid):
    x1, y1, x2, y2 = box
    if helmet == "MISSING" or vest == "MISSING":
        bc = (0, 0, 255)
    elif helmet == "OK" and vest == "OK":
        bc = (0, 200, 0)
    else:
        bc = (0, 165, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)
    for i, (txt, col) in enumerate([
        (f"P{pid}",           bc),
        (f"Helmet: {helmet}", _color(helmet)),
        (f"Vest  : {vest}",   _color(vest)),
    ]):
        cv2.putText(frame, txt, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    if helmet == "MISSING" or vest == "MISSING":
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ── main ───────────────────────────────────────────────────────────────────────

def run(source):
    print("[INFO] Loading PPE model (single full-frame model)...")
    model = YOLO(PPE_MODEL_PATH)
    names = model.names
    print(f"[INFO] Classes: {list(names.values())}")
    print("[INFO] Press 'q' to quit.\n")

    cap = FrameReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}  |  smooth window: {SMOOTH_FRAMES} frames")

    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        raw      = detect_frame(frame, model, names)
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
