"""
Unified safety monitor — PPE detection + height/elevation detection.

Models:
  yolov8n-pose.pt       person detection + ankle keypoints (height)
  ppe_model/ppe_best.pt PPE detection on head/torso crops

Requires calibration.json for height detection (run calibrate.py first).
If calibration.json is missing, height detection is skipped automatically.
"""

import cv2
import json
import argparse
import os
import threading
import time
from collections import deque
from ultralytics import YOLO

# ── config ─────────────────────────────────────────────────────────────────────

POSE_MODEL     = "yolov8n-pose.pt"      # person + keypoints in one pass
PPE_MODEL_PATH = "ppe_model/ppe_best.pt"

PERSON_CONF    = 0.50
HELMET_OK_CONF = 0.70
HELMET_NO_CONF = 0.30
VEST_OK_CONF   = 0.65
VEST_NO_CONF   = 0.30

MIN_BOX_HEIGHT = 80
MIN_ASPECT     = 0.6

SMOOTH_FRAMES  = 10
MISSING_THRESH = 4
OK_THRESH      = 6
IOU_MATCH      = 0.25

# Height thresholds
ALERT_HEIGHT_CM   = 185    # head above this (absolute, requires calibration) = ELEVATED
ANKLE_BOX_THRESH  = 0.13   # ankles more than 13% of box height above box bottom = off floor
FOOT_CONF_MIN     = 0.40   # min keypoint confidence to trust ankle
MIN_ANKLES        = 1      # need at least this many visible ankles to make a call

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# Body region fractions for PPE crop
HEAD_TOP   = 0.0
HEAD_BOT   = 0.30
TORSO_TOP  = 0.15
TORSO_BOT  = 0.70

HAS_HELMET = "Hardhat"
NO_HELMET  = "NO-Hardhat"
HAS_VEST   = "Safety Vest"
NO_VEST    = "NO-Safety Vest"


# ── calibration ────────────────────────────────────────────────────────────────

def load_calibration():
    try:
        with open("calibration.json") as f:
            data = json.load(f)
        print(f"[INFO] Calibration loaded: {data['pixels_per_cm']:.4f} px/cm  floor_y={data['floor_y']}")
        return data
    except FileNotFoundError:
        print("[WARN] calibration.json not found — height detection disabled. Run calibrate.py to enable.")
        return None


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
    ph   = y2 - y1
    ry1  = max(0, y1 + int(ph * top_frac))
    ry2  = min(frame_h, y1 + int(ph * bot_frac))
    crop = frame[ry1:ry2, x1:x2]
    return crop if crop.size > 0 else None


# ── temporal PPE smoother ──────────────────────────────────────────────────────

class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self._next  = 0

    def update(self, detections):
        """detections: list of (box, helmet_raw, vest_raw)"""
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
                self.tracks[best_id] = {"box":    box,
                                        "helmet": deque(maxlen=SMOOTH_FRAMES),
                                        "vest":   deque(maxlen=SMOOTH_FRAMES)}
            t = self.tracks[best_id]
            t["box"] = box
            t["helmet"].append(h)
            t["vest"].append(v)
            matched.add(best_id)
            results.append((box, _vote(t["helmet"]), _vote(t["vest"])))
        self.tracks = {k: v for k, v in self.tracks.items() if k in matched}
        return results

def _vote(hist):
    c = {}
    for s in hist: c[s] = c.get(s, 0) + 1
    if c.get("OK", 0) >= OK_THRESH:         return "OK"
    if c.get("MISSING", 0) >= MISSING_THRESH: return "MISSING"
    return "MISSING"   # safety-first default


# ── PPE detection ──────────────────────────────────────────────────────────────

def check_helmet(frame, pbox, ppe_model, ppe_names, frame_h):
    crop = region_crop(frame, pbox, HEAD_TOP, HEAD_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes, best_no = 0.0, 0.0
    for box in ppe_model(crop, verbose=False, conf=HELMET_NO_CONF)[0].boxes:
        cls  = ppe_names[int(box.cls)]
        conf = float(box.conf)
        if cls == HAS_HELMET: best_yes = max(best_yes, conf)
        if cls == NO_HELMET:  best_no  = max(best_no,  conf)
    if best_no  >= HELMET_NO_CONF: return "MISSING"
    if best_yes >= HELMET_OK_CONF: return "OK"
    return "MISSING"

def check_vest(frame, pbox, ppe_model, ppe_names, frame_h):
    crop = region_crop(frame, pbox, TORSO_TOP, TORSO_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes, best_no = 0.0, 0.0
    for box in ppe_model(crop, verbose=False, conf=VEST_NO_CONF)[0].boxes:
        cls  = ppe_names[int(box.cls)]
        conf = float(box.conf)
        if cls == HAS_VEST: best_yes = max(best_yes, conf)
        if cls == NO_VEST:  best_no  = max(best_no,  conf)
    if best_no  >= VEST_NO_CONF:  return "MISSING"
    if best_yes >= VEST_OK_CONF:  return "OK"
    return "MISSING"


# ── height / elevation detection ───────────────────────────────────────────────

def check_elevation(box, kps, cal):
    """
    Returns: (state, head_cm, ankle_rel)
      state: "elevated" | "bending" | "ok" | "unknown"

    Two independent checks:
      1. ELEVATED — head absolute height > ALERT_HEIGHT_CM (needs calibration).
      2. BENDING  — ankle position relative to bounding box bottom.
                    Perspective-invariant: works at any camera distance.
                    ankle_rel = (box_bottom - ankle_y) / box_height
                    ≈ 0 when standing (ankles at box bottom)
                    > ANKLE_BOX_THRESH when feet are off the floor.
    """
    x1, y1, x2, y2 = box
    box_h = y2 - y1

    # ── check 1: absolute head height (calibration required) ──────────────────
    head_cm = None
    if cal is not None:
        head_cm = (cal["floor_y"] - y1) / cal["pixels_per_cm"]
        if head_cm > ALERT_HEIGHT_CM:
            return "elevated", head_cm, None

    # ── check 2: ankle relative to box bottom (perspective-invariant) ─────────
    if kps is None or kps.data is None or len(kps.data) == 0:
        return "ok", head_cm, None   # no keypoints → can't judge feet

    kp_data   = kps.data[0]
    ankle_ys  = []
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp = kp_data[idx]
        if float(kp[2]) >= FOOT_CONF_MIN:
            ankle_ys.append(float(kp[1]))

    if len(ankle_ys) < MIN_ANKLES:
        return "ok", head_cm, None   # ankles not visible → give benefit of doubt

    avg_ankle_y = sum(ankle_ys) / len(ankle_ys)

    # How far above the box bottom are the ankles, as fraction of box height?
    # Standing normally → ≈ 0.  Feet off floor → > ANKLE_BOX_THRESH.
    ankle_rel = (y2 - avg_ankle_y) / max(box_h, 1)

    if ankle_rel > ANKLE_BOX_THRESH:
        return "bending", head_cm, ankle_rel

    return "ok", head_cm, ankle_rel


# ── drawing ────────────────────────────────────────────────────────────────────

def _ppe_col(s):
    return (0, 200, 0) if s == "OK" else (0, 0, 255)

def draw_threshold_line(frame, cal, w):
    if cal is None:
        return
    floor_y   = cal["floor_y"]
    px_per_cm = cal["pixels_per_cm"]
    alert_y   = int(floor_y - ALERT_HEIGHT_CM * px_per_cm)
    h = frame.shape[0]
    if 0 <= alert_y < h:
        cv2.line(frame, (0, alert_y), (w, alert_y), (0, 0, 255), 1)
        cv2.putText(frame, f"{ALERT_HEIGHT_CM}cm threshold",
                    (10, alert_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

def draw_ankle_dots(frame, kps, ankle_rel):
    if kps is None or kps.data is None or len(kps.data) == 0:
        return
    kp_data = kps.data[0]
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp = kp_data[idx]
        if float(kp[2]) >= FOOT_CONF_MIN:
            px, py = int(float(kp[0])), int(float(kp[1]))
            color  = (0, 0, 255) if (ankle_rel is not None and ankle_rel > ANKLE_BOX_THRESH) else (0, 255, 255)
            cv2.circle(frame, (px, py), 7, color, -1)
            if ankle_rel is not None:
                cv2.putText(frame, f"{ankle_rel:.2f}",
                            (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

def draw_person(frame, box, helmet, vest, elev_state, head_cm, ankle_rel, kps, pid, cal_enabled):
    x1, y1, x2, y2 = box
    ppe_ok    = helmet == "OK" and vest == "OK"
    elev_ok   = elev_state == "ok"

    # Box color: elevation takes priority over PPE for color
    if elev_state == "elevated":
        bc = (0, 0, 255)
    elif elev_state == "bending":
        bc = (0, 140, 255)
    elif not ppe_ok:
        bc = (0, 0, 255)
    else:
        bc = (0, 200, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)

    # Labels — built bottom-up from y1
    lines = [(f"P{pid}", bc)]
    lines.append((f"Helmet: {helmet}", _ppe_col(helmet)))
    lines.append((f"Vest  : {vest}",   _ppe_col(vest)))
    if cal_enabled and head_cm is not None:
        lines.append((f"Head: {head_cm:.0f}cm", bc))

    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

    # Bottom violation labels
    bottom_y = y2 + 22
    if elev_state == "elevated":
        cv2.putText(frame, "!! ELEVATED !!",
                    (x1, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        bottom_y += 22
    elif elev_state == "bending":
        cv2.putText(frame, "!! FEET OFF FLOOR !!",
                    (x1, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        bottom_y += 22
    if not ppe_ok:
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    draw_ankle_dots(frame, kps, ankle_rel)


# ── main loop ──────────────────────────────────────────────────────────────────

def run(source):
    cal = load_calibration()

    print("[INFO] Loading models...")
    pose_model = YOLO(POSE_MODEL)
    ppe_model  = YOLO(PPE_MODEL_PATH)
    ppe_names  = ppe_model.names
    print("[INFO] Press 'q' to quit.\n")

    cap = FrameReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}  |  height detection: {'ON' if cal else 'OFF (no calibration)'}")

    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        draw_threshold_line(frame, cal, w)

        # Single pose model pass → persons + keypoints
        pose_res = pose_model(frame, verbose=False, conf=PERSON_CONF)[0]

        raw_ppe = []   # for PPE smoother: (box, helmet_raw, vest_raw)
        per_person_elev = []   # (box, elev_state, head_cm, ankle_cm, kps)

        for i, box in enumerate(pose_res.boxes):
            if int(box.cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bh, bw  = y2 - y1, x2 - x1
            if bw == 0 or bh < MIN_BOX_HEIGHT or bh / max(bw, 1) < MIN_ASPECT:
                continue

            pbox = (x1, y1, x2, y2)
            kps  = pose_res.keypoints[i] if pose_res.keypoints is not None else None

            # PPE (raw, will be smoothed)
            helmet = check_helmet(frame, pbox, ppe_model, ppe_names, h)
            vest   = check_vest(frame, pbox, ppe_model, ppe_names, h)
            raw_ppe.append((pbox, helmet, vest))

            # Elevation (immediate, no smoothing needed — physical position)
            elev_state, head_cm, ankle_rel = check_elevation(pbox, kps, cal)
            per_person_elev.append((pbox, elev_state, head_cm, ankle_rel, kps))

        # Smooth PPE across frames
        smoothed_ppe = tracker.update(raw_ppe)

        # Draw each person
        ppe_violations  = 0
        elev_alerts     = 0
        bending_alerts  = 0

        for idx, (box, helmet, vest) in enumerate(smoothed_ppe):
            # Match elevation result by index (same order, same boxes)
            if idx < len(per_person_elev):
                _, elev_state, head_cm, ankle_rel, kps = per_person_elev[idx]
            else:
                elev_state, head_cm, ankle_rel, kps = "ok", None, None, None

            draw_person(frame, box, helmet, vest,
                        elev_state, head_cm, ankle_rel, kps,
                        idx + 1, cal is not None)

            if helmet != "OK" or vest != "OK":
                ppe_violations += 1
            if elev_state == "elevated":
                elev_alerts += 1
            elif elev_state == "bending":
                bending_alerts += 1

        # Alert banners
        banner_y = 0
        if elev_alerts > 0:
            cv2.rectangle(frame, (0, banner_y), (w, banner_y + 45), (0, 0, 160), -1)
            cv2.putText(frame, f"ELEVATION ALERT - {elev_alerts} person(s) elevated above {ALERT_HEIGHT_CM}cm",
                        (10, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            banner_y += 45
        if bending_alerts > 0:
            cv2.rectangle(frame, (0, banner_y), (w, banner_y + 45), (0, 100, 200), -1)
            cv2.putText(frame, f"FEET OFF FLOOR - {bending_alerts} person(s) elevated (bending/climbing)",
                        (10, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            banner_y += 45
        if ppe_violations > 0:
            cv2.rectangle(frame, (0, banner_y), (w, banner_y + 45), (0, 0, 180), -1)
            cv2.putText(frame, f"PPE VIOLATION - {ppe_violations} person(s) missing PPE",
                        (10, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(frame, f"People: {len(smoothed_ppe)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Safety Monitor", frame)
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
