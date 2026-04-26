"""
Unified safety monitor — PPE detection + feet-off-floor detection.

Models:
  yolov8n-pose.pt       person detection + ankle keypoints
  ppe_model/ppe_best.pt PPE detection on head/torso crops

Floor detection works in two modes:
  - Without calibration: relative ankle check only (perspective-invariant)
  - With calibration (floor_points):  relative check + floor-map check
    (catches standing on chairs/platforms too)
"""

import cv2
import json
import argparse
import os
import threading
import time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from sheets_logger import SheetsLogger

# ── Incident logging ───────────────────────────────────────────────────────────
COOLDOWN_SEC = 15   # log same violation at most once per N seconds per person

# ── config ─────────────────────────────────────────────────────────────────────

POSE_MODEL     = "yolov8n-pose.pt"
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

# Feet-off-floor thresholds
ANKLE_BOX_THRESH  = 0.07  # ankle > 7% of box height above box bottom → feet off floor
FLOOR_MAP_THRESH  = 0.14  # ankle > 14% of box height above mapped floor → feet off floor
FLOOR_DEPTH_RADIUS = 80   # max pixel distance in y to nearest calibration point — if farther, skip floor-map check
OFF_FLOOR_THRESH   = 5    # frames out of SMOOTH_FRAMES needed to declare off-floor (sustained elevation only)
FEET_MIN_DURATION  = 5.0  # seconds feet must be continuously off floor before flagging
FOOT_CONF_MIN     = 0.40
MIN_ANKLES        = 1

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

# PPE crop regions (fraction of person box height)
HEAD_TOP  = 0.0;  HEAD_BOT  = 0.30
TORSO_TOP = 0.15; TORSO_BOT = 0.70

HAS_HELMET = "Hardhat";   NO_HELMET = "NO-Hardhat"
HAS_VEST   = "Safety Vest"; NO_VEST = "NO-Safety Vest"


# ── calibration / floor map ────────────────────────────────────────────────────

def load_calibration():
    try:
        with open("calibration.json") as f:
            data = json.load(f)
        pts = data.get("floor_points", [])
        if pts:
            print(f"[INFO] Floor map loaded: {len(pts)} points")
        else:
            print("[WARN] calibration.json has no floor_points — floor-map check disabled.")
        return data
    except FileNotFoundError:
        print("[INFO] No calibration.json — using relative ankle check only.")
        return None

def floor_y_at(ankle_x, person_y2, floor_points):
    """
    Return expected floor y at the person's depth, or None if no calibration coverage.

    Uses Gaussian weighting by both y-distance (depth) and x-distance.
    Returns None when the nearest calibration point in y is farther than
    FLOOR_DEPTH_RADIUS — this prevents false positives in uncalibrated areas.
    """
    if not floor_points or len(floor_points) < 2:
        return None
    pts = np.array(floor_points, dtype=float)
    dy  = np.abs(pts[:, 1] - person_y2)

    # No calibration data near this depth → skip floor-map check entirely
    if np.min(dy) > FLOOR_DEPTH_RADIUS:
        return None

    dx      = np.abs(pts[:, 0] - ankle_x)
    weights = np.exp(-(dy / 80) ** 2) * np.exp(-(dx / 400) ** 2)
    total_w = np.sum(weights)
    if total_w < 1e-6:
        return None
    return float(np.dot(weights, pts[:, 1]) / total_w)

def draw_floor_map(frame, floor_points):
    """Draw calibration dots only — no regression line (line was misleading)."""
    for px, py in floor_points:
        cv2.circle(frame, (int(px), int(py)), 4, (0, 200, 0), -1)


# ── threaded RTSP reader ───────────────────────────────────────────────────────
# Reconnect delays (seconds) — increases with each failed attempt, caps at last value
RECONNECT_DELAYS = [2, 5, 10, 30]
RECONNECT_FAIL_THRESHOLD = 10  # consecutive bad reads before triggering reconnect

class FrameReader:
    def __init__(self, source):
        self.source = source
        self.cap    = _open_cap(source)
        self.ret    = False
        self.frame  = None
        self._lock  = threading.Lock()
        self._stop  = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        fail_streak   = 0
        retry_attempt = 0
        while not self._stop:
            ret, frame = self.cap.read()
            if ret:
                fail_streak   = 0
                retry_attempt = 0
                with self._lock:
                    self.ret, self.frame = True, frame
            else:
                fail_streak += 1
                if fail_streak >= RECONNECT_FAIL_THRESHOLD:
                    delay = RECONNECT_DELAYS[min(retry_attempt, len(RECONNECT_DELAYS) - 1)]
                    print(f"\n[WARN] Stream lost. Reconnecting in {delay}s "
                          f"(attempt {retry_attempt + 1})...")
                    time.sleep(delay)
                    self.cap.release()
                    self.cap = _open_cap(self.source)
                    fail_streak    = 0
                    retry_attempt += 1
                    if self.cap.isOpened():
                        print("[INFO] Stream reconnected.")
                        retry_attempt = 0
                else:
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
    ph  = y2 - y1
    ry1 = max(0, y1 + int(ph * top_frac))
    ry2 = min(frame_h, y1 + int(ph * bot_frac))
    crop = frame[ry1:ry2, x1:x2]
    return crop if crop.size > 0 else None


# ── temporal PPE smoother ──────────────────────────────────────────────────────

class PersonTracker:
    def __init__(self):
        self.tracks = {}
        self._next  = 0

    def update(self, detections):
        """detections: list of (box, helmet, vest, off_floor, kps)"""
        matched = set()
        results = []
        now = time.time()
        for box, h, v, off, kps in detections:
            best_id, best_s = None, IOU_MATCH
            for tid, t in self.tracks.items():
                s = iou(box, t["box"])
                if s > best_s:
                    best_s, best_id = s, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {"box":       box,
                                        "helmet":    deque(maxlen=SMOOTH_FRAMES),
                                        "vest":      deque(maxlen=SMOOTH_FRAMES),
                                        "off_floor": deque(maxlen=SMOOTH_FRAMES),
                                        "off_since": None}
            t = self.tracks[best_id]
            t["box"] = box
            t["helmet"].append(h)
            t["vest"].append(v)
            t["off_floor"].append(off)
            matched.add(best_id)

            raw_off = _vote_bool(t["off_floor"])
            if raw_off:
                if t["off_since"] is None:
                    t["off_since"] = now
                sustained = (now - t["off_since"]) >= FEET_MIN_DURATION
            else:
                t["off_since"] = None
                sustained = False

            results.append((best_id, box, _vote(t["helmet"]), _vote(t["vest"]), sustained, kps))
        self.tracks = {k: v for k, v in self.tracks.items() if k in matched}
        return results

def _vote(hist):
    c = {}
    for s in hist: c[s] = c.get(s, 0) + 1
    if c.get("OK", 0) >= OK_THRESH:            return "OK"
    if c.get("MISSING", 0) >= MISSING_THRESH:  return "MISSING"
    return "MISSING"

def _vote_bool(hist):
    """Declare off-floor when OFF_FLOOR_THRESH or more of recent frames detected it."""
    return sum(1 for x in hist if x) >= OFF_FLOOR_THRESH


# ── PPE detection ──────────────────────────────────────────────────────────────

def check_helmet(frame, pbox, ppe_model, ppe_names, frame_h):
    crop = region_crop(frame, pbox, HEAD_TOP, HEAD_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes = best_no = 0.0
    for box in ppe_model(crop, verbose=False, conf=HELMET_NO_CONF)[0].boxes:
        cls = ppe_names[int(box.cls)]; conf = float(box.conf)
        if cls == HAS_HELMET: best_yes = max(best_yes, conf)
        if cls == NO_HELMET:  best_no  = max(best_no,  conf)
    if best_no  >= HELMET_NO_CONF: return "MISSING"
    if best_yes >= HELMET_OK_CONF: return "OK"
    return "MISSING"

def check_vest(frame, pbox, ppe_model, ppe_names, frame_h):
    crop = region_crop(frame, pbox, TORSO_TOP, TORSO_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes = best_no = 0.0
    for box in ppe_model(crop, verbose=False, conf=VEST_NO_CONF)[0].boxes:
        cls = ppe_names[int(box.cls)]; conf = float(box.conf)
        if cls == HAS_VEST: best_yes = max(best_yes, conf)
        if cls == NO_VEST:  best_no  = max(best_no,  conf)
    if best_no  >= VEST_NO_CONF:  return "MISSING"
    if best_yes >= VEST_OK_CONF:  return "OK"
    return "MISSING"


# ── feet-off-floor detection ───────────────────────────────────────────────────

def check_feet(box, kps, floor_points):
    """
    Returns: (off_floor: bool, ankle_rel: float|None)

    Check 1 — Floor map (if calibrated):
      Compare ankle pixel position to the interpolated floor surface at that x.
      Normalized by box height → distance-invariant.
      Catches: standing on chair, platform, machinery.

    Check 2 — Relative to bounding box bottom (always runs):
      ankle_rel = (box_bottom - ankle_y) / box_height
      ≈ 0 when standing, > ANKLE_BOX_THRESH when feet are dangling/raised.
      Catches: sitting with feet up, climbing, hanging.
    """
    x1, y1, x2, y2 = box
    box_h = max(y2 - y1, 1)

    if kps is None or kps.data is None or len(kps.data) == 0:
        return False, None

    kp_data  = kps.data[0]
    ankle_xs, ankle_ys = [], []
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp = kp_data[idx]
        if float(kp[2]) >= FOOT_CONF_MIN:
            ankle_xs.append(float(kp[0]))
            ankle_ys.append(float(kp[1]))

    if len(ankle_ys) < MIN_ANKLES:
        return False, None

    avg_ax = sum(ankle_xs) / len(ankle_xs)
    avg_ay = sum(ankle_ys) / len(ankle_ys)

    # ── Check 1: floor map ────────────────────────────────────────────────────
    if floor_points and len(floor_points) >= 2:
        expected_floor_y = floor_y_at(avg_ax, y2, floor_points)
        if expected_floor_y is not None:
            above_floor = (expected_floor_y - avg_ay) / box_h
            if above_floor > FLOOR_MAP_THRESH:
                ankle_rel = (y2 - avg_ay) / box_h
                return True, ankle_rel

    # ── Check 2: relative to box bottom ──────────────────────────────────────
    ankle_rel = (y2 - avg_ay) / box_h
    if ankle_rel > ANKLE_BOX_THRESH:
        return True, ankle_rel

    return False, ankle_rel


# ── drawing ────────────────────────────────────────────────────────────────────

def _ppe_col(s):
    return (0, 200, 0) if s == "OK" else (0, 0, 255)

def draw_ankle_dots(frame, kps, off_floor):
    if kps is None or kps.data is None or len(kps.data) == 0:
        return
    kp_data = kps.data[0]
    color   = (0, 0, 255) if off_floor else (0, 255, 255)
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp = kp_data[idx]
        if float(kp[2]) >= FOOT_CONF_MIN:
            px, py = int(float(kp[0])), int(float(kp[1]))
            cv2.circle(frame, (px, py), 7, color, -1)

def draw_person(frame, box, helmet, vest, off_floor, kps, pid):
    x1, y1, x2, y2 = box
    ppe_ok    = helmet == "OK" and vest == "OK"
    ppe_alert = off_floor and not ppe_ok   # PPE alert only matters when elevated

    if off_floor and ppe_alert:
        bc = (0, 0, 255)         # red — elevated + missing PPE (highest risk)
    elif off_floor:
        bc = (0, 140, 255)       # orange — elevated but PPE ok
    else:
        bc = (0, 200, 0)         # green — on floor (PPE not enforced at floor level)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)

    for i, (txt, col) in enumerate([
        (f"P{pid}",           bc),
        (f"Helmet: {helmet}", _ppe_col(helmet) if off_floor else (180, 180, 180)),
        (f"Vest  : {vest}",   _ppe_col(vest)   if off_floor else (180, 180, 180)),
    ]):
        cv2.putText(frame, txt, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

    bottom_y = y2 + 22
    if off_floor:
        cv2.putText(frame, "!! FEET OFF FLOOR !!",
                    (x1, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        bottom_y += 22
    if ppe_alert:
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    draw_ankle_dots(frame, kps, off_floor)


# ── main loop ──────────────────────────────────────────────────────────────────

def run(source):
    cal          = load_calibration()
    floor_points = cal.get("floor_points", []) if cal else []

    print("[INFO] Loading models...")
    pose_model = YOLO(POSE_MODEL)
    ppe_model  = YOLO(PPE_MODEL_PATH)
    ppe_names  = ppe_model.names
    print(f"[INFO] Floor map: {'YES (' + str(len(floor_points)) + ' points)' if floor_points else 'NO — run calibrate.py to enable'}")
    print("[INFO] Press 'q' to quit.\n")

    logger  = SheetsLogger()

    cap = FrameReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    tracker     = PersonTracker()
    last_logged = {}   # "P{n}_{violation}" → epoch seconds of last log

    def should_log(pid, violation):
        key  = f"P{pid}_{violation}"
        now  = time.time()
        if now - last_logged.get(key, 0) >= COOLDOWN_SEC:
            last_logged[key] = now
            return True
        return False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Draw calibrated floor line (subtle, for reference)
        if floor_points:
            draw_floor_map(frame, floor_points)

        pose_res = pose_model(frame, verbose=False, conf=PERSON_CONF)[0]

        raw_detections = []

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

            helmet    = check_helmet(frame, pbox, ppe_model, ppe_names, h)
            vest      = check_vest(frame, pbox, ppe_model, ppe_names, h)
            off_floor, _ = check_feet(pbox, kps, floor_points)
            raw_detections.append((pbox, helmet, vest, off_floor, kps))

        smoothed     = tracker.update(raw_detections)
        ppe_violations = 0
        feet_alerts    = 0

        for tid, box, helmet, vest, off_floor, kps in smoothed:
            pid = tid
            draw_person(frame, box, helmet, vest, off_floor, kps, pid)

            if off_floor:
                feet_alerts += 1
                if helmet != "OK" or vest != "OK":
                    ppe_violations += 1   # PPE violation only counts when elevated

            # ── Incident logging ──────────────────────────────────────────────
            if off_floor and should_log(pid, "FEET_OFF_FLOOR"):
                logger.log(source, "FEET_OFF_FLOOR", pid, "feet not on floor", frame)
            if off_floor:   # PPE incidents only logged when elevated
                if helmet != "OK" and should_log(pid, "NO_HELMET"):
                    logger.log(source, "NO_HELMET", pid, f"helmet={helmet}", frame)
                if vest != "OK" and should_log(pid, "NO_VEST"):
                    logger.log(source, "NO_VEST", pid, f"vest={vest}", frame)

        # Banners
        banner_y = 0
        if feet_alerts > 0:
            cv2.rectangle(frame, (0, banner_y), (w, banner_y + 45), (0, 100, 200), -1)
            cv2.putText(frame, f"FEET OFF FLOOR - {feet_alerts} person(s) not on floor",
                        (10, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            banner_y += 45
        if ppe_violations > 0:
            cv2.rectangle(frame, (0, banner_y), (w, banner_y + 45), (0, 0, 180), -1)
            cv2.putText(frame, f"PPE VIOLATION - {ppe_violations} person(s) missing PPE",
                        (10, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(frame, f"People: {len(smoothed)}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (w - 220, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Safety Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    logger.close()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
