import cv2
import argparse
import os
import threading
import time
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from sheets_logger import SheetsLogger
from config import *

# ── threaded RTSP reader ───────────────────────────────────────────────────────

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

def suppress_merged_boxes(candidates):
    """Keep only the highest-confidence detection when two boxes overlap heavily."""
    kept = []
    for cand in sorted(candidates, key=lambda c: c[0], reverse=True):
        if all(iou(cand[2], k[2]) < MERGE_IOU_THRESH for k in kept):
            kept.append(cand)
    return kept

def region_crop(frame, pbox, top_frac, bot_frac, frame_h):
    x1, y1, x2, y2 = pbox
    ph  = y2 - y1
    ry1 = max(0, y1 + int(ph * top_frac))
    ry2 = min(frame_h, y1 + int(ph * bot_frac))
    crop = frame[ry1:ry2, x1:x2]
    return crop if crop.size > 0 else None


# ── temporal PPE smoother ──────────────────────────────────────────────────────

class PPESmoother:
    """Temporal PPE smoother keyed by ByteTrack IDs — no IoU matching needed."""
    def __init__(self):
        self.tracks = {}

    def update(self, tid, helmet, vest, off_floor):
        now = time.time()
        if tid not in self.tracks:
            self.tracks[tid] = {
                "helmet":       deque(maxlen=SMOOTH_FRAMES),
                "vest":         deque(maxlen=SMOOTH_FRAMES),
                "off_floor":    deque(maxlen=SMOOTH_FRAMES),
                "off_since":    None,
                "prev_helmet":  None,
                "prev_vest":    None,
            }
        t = self.tracks[tid]
        t["helmet"].append(helmet)
        t["vest"].append(vest)
        t["off_floor"].append(off_floor)

        raw_off = _vote_bool(t["off_floor"])
        if raw_off:
            if t["off_since"] is None:
                t["off_since"] = now
            sustained = (now - t["off_since"]) >= FEET_MIN_DURATION
        else:
            t["off_since"] = None
            sustained = False

        s_helmet = _vote(t["helmet"])
        s_vest   = _vote(t["vest"])

        # Detect removal: previously confirmed OK, now confirmed MISSING
        h_removed = t["prev_helmet"] == "OK" and s_helmet == "MISSING"
        v_removed = t["prev_vest"]   == "OK" and s_vest   == "MISSING"
        t["prev_helmet"] = s_helmet
        t["prev_vest"]   = s_vest

        return s_helmet, s_vest, sustained, h_removed, v_removed

    def cleanup(self, active_ids):
        for k in [k for k in self.tracks if k not in active_ids]:
            del self.tracks[k]

def _vote(hist):
    c = {}
    for s in hist: c[s] = c.get(s, 0) + 1
    if c.get("UNVERIFIABLE", 0) > len(hist) // 2: return "UNVERIFIABLE"
    if c.get("OK", 0) >= OK_THRESH:               return "OK"
    if c.get("MISSING", 0) >= MISSING_THRESH:      return "MISSING"
    return "MISSING"

def _vote_bool(hist):
    """Declare off-floor when OFF_FLOOR_THRESH or more of recent frames detected it."""
    return sum(1 for x in hist if x) >= OFF_FLOOR_THRESH


# ── PPE detection ──────────────────────────────────────────────────────────────

def is_occluded(pbox, kps, frame_w, frame_h):
    """
    True when the person is significantly hidden behind something or cut off by the frame:
      - bounding box clipped by frame edge (head/side cut off)
      - both shoulders invisible (low keypoint confidence)
    """
    x1, y1, x2, y2 = pbox
    if x1 <= EDGE_MARGIN or x2 >= frame_w - EDGE_MARGIN:
        return True
    if y1 <= EDGE_MARGIN:
        return True
    if kps is not None and kps.data is not None and len(kps.data) > 0:
        kp_data = kps.data[0]
        visible = sum(1 for idx in [LEFT_SHOULDER, RIGHT_SHOULDER]
                      if float(kp_data[idx][2]) >= SHOULDER_CONF_MIN)
        if visible == 0:
            return True
    return False

def is_rear_facing(kps):
    if kps is None or kps.data is None or len(kps.data) == 0:
        return False
    return float(kps.data[0][NOSE_KP][2]) < REAR_CONF_MIN

def check_helmet(frame, pbox, ppe_model, ppe_names, frame_h, rear_facing=False):
    crop = region_crop(frame, pbox, HEAD_TOP, HEAD_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes = best_no = 0.0
    for box in ppe_model(crop, verbose=False, conf=HELMET_NO_CONF)[0].boxes:
        cls = ppe_names[int(box.cls)]; conf = float(box.conf)
        if cls == HAS_HELMET: best_yes = max(best_yes, conf)
        if cls == NO_HELMET:  best_no  = max(best_no,  conf)
    if best_yes >= HELMET_OK_CONF: return "OK"
    no_thresh = HELMET_NO_CONF_REAR if rear_facing else HELMET_NO_CONF
    if best_no >= no_thresh:       return "MISSING"
    return "MISSING"

def check_vest(frame, pbox, ppe_model, ppe_names, frame_h, rear_facing=False):
    crop = region_crop(frame, pbox, TORSO_TOP, TORSO_BOT, frame_h)
    if crop is None:
        return "MISSING"
    best_yes = best_no = 0.0
    for box in ppe_model(crop, verbose=False, conf=VEST_NO_CONF)[0].boxes:
        cls = ppe_names[int(box.cls)]; conf = float(box.conf)
        if cls == HAS_VEST: best_yes = max(best_yes, conf)
        if cls == NO_VEST:  best_no  = max(best_no,  conf)
    if best_yes >= VEST_OK_CONF: return "OK"
    no_thresh = VEST_NO_CONF_REAR if rear_facing else VEST_NO_CONF
    if best_no >= no_thresh:     return "MISSING"
    return "MISSING"


# ── feet-off-floor detection ───────────────────────────────────────────────────

def check_feet(box, kps):
    """
    Returns (off_floor: bool, ankle_rel: float|None).
    Compares ankle y-position to the bounding box bottom — no calibration needed.
    ankle_rel ≈ 0 when standing, > ANKLE_BOX_THRESH when feet are raised/dangling.
    """
    x1, y1, x2, y2 = box
    box_h = max(y2 - y1, 1)

    if kps is None or kps.data is None or len(kps.data) == 0:
        return False, None

    kp_data   = kps.data[0]
    ankle_ys  = []
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp = kp_data[idx]
        if float(kp[2]) >= FOOT_CONF_MIN:
            ankle_ys.append(float(kp[1]))

    if len(ankle_ys) < MIN_ANKLES:
        return False, None

    avg_ay    = sum(ankle_ys) / len(ankle_ys)
    ankle_rel = (y2 - avg_ay) / box_h
    return ankle_rel > ANKLE_BOX_THRESH, ankle_rel


# ── drawing ────────────────────────────────────────────────────────────────────

def _ppe_col(s):
    return {
        "OK":           (0, 200, 0),
        "MISSING":      (0, 0, 255),
        "UNVERIFIABLE": (180, 180, 0),
    }.get(s, (180, 180, 0))

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
    rear             = is_rear_facing(kps)
    unverifiable     = helmet == "UNVERIFIABLE" or vest == "UNVERIFIABLE"
    ppe_ok           = helmet == "OK" and vest == "OK"
    ppe_alert        = off_floor and not ppe_ok and not unverifiable

    if unverifiable:
        bc = (180, 180, 0)       # yellow — cannot verify PPE
    elif off_floor and ppe_alert:
        bc = (0, 0, 255)         # red — elevated + missing PPE
    elif off_floor:
        bc = (0, 140, 255)       # orange — elevated but PPE ok
    else:
        bc = (0, 200, 0)         # green — on floor

    cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 2)

    pid_label = f"P{pid} [REAR]" if rear else f"P{pid}"
    ppe_col   = (180, 180, 180) if rear or not off_floor else None
    for i, (txt, col) in enumerate([
        (pid_label,           bc),
        (f"Helmet: {helmet}", ppe_col or _ppe_col(helmet)),
        (f"Vest  : {vest}",   ppe_col or _ppe_col(vest)),
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
    print("[INFO] Loading models...")
    pose_model = YOLO(POSE_MODEL)
    ppe_model  = YOLO(PPE_MODEL_PATH)
    ppe_names  = ppe_model.names
    print("[INFO] Press 'q' to quit.\n")

    logger  = SheetsLogger()

    cap = FrameReader(source)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    smoother    = PPESmoother()
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

        pose_res = pose_model.track(frame, persist=True, tracker="bytetrack_custom.yaml",
                                    verbose=False, conf=PERSON_CONF, imgsz=DETECTION_IMGSZ)[0]

        # Build candidate list, then drop heavily-overlapping (merged) boxes
        candidates = []
        for i, box in enumerate(pose_res.boxes):
            if int(box.cls) != 0:
                continue
            tid = int(box.id[0]) if box.id is not None else None
            if tid is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bh, bw  = y2 - y1, x2 - x1
            if bw == 0 or bh < MIN_BOX_HEIGHT or bh / max(bw, 1) < MIN_ASPECT:
                continue
            candidates.append((float(box.conf), tid, (x1, y1, x2, y2), i))

        candidates = suppress_merged_boxes(candidates)

        active_ids     = set()
        results        = []
        for conf, tid, pbox, i in candidates:
            kps  = pose_res.keypoints[i] if pose_res.keypoints is not None else None

            if is_occluded(pbox, kps, w, h):
                helmet = vest = "UNVERIFIABLE"
            else:
                rear   = is_rear_facing(kps)
                helmet = check_helmet(frame, pbox, ppe_model, ppe_names, h, rear)
                vest   = check_vest(frame, pbox, ppe_model, ppe_names, h, rear)
            off_floor, _ = check_feet(pbox, kps)

            s_helmet, s_vest, sustained, h_removed, v_removed = smoother.update(tid, helmet, vest, off_floor)
            active_ids.add(tid)
            results.append((tid, pbox, s_helmet, s_vest, sustained, kps, h_removed, v_removed))

        smoother.cleanup(active_ids)

        ppe_violations = 0
        feet_alerts    = 0

        for tid, box, helmet, vest, off_floor, kps, h_removed, v_removed in results:
            pid = tid
            draw_person(frame, box, helmet, vest, off_floor, kps, pid)

            if off_floor:
                feet_alerts += 1
            if helmet == "MISSING" or vest == "MISSING":
                ppe_violations += 1

            # ── Incident logging ──────────────────────────────────────────────
            if off_floor and should_log(pid, "FEET_OFF_FLOOR"):
                logger.log(source, "FEET_OFF_FLOOR", pid, "feet not on floor", frame)

            # PPE violations logged regardless of floor position
            if helmet not in ("OK", "UNVERIFIABLE") and should_log(pid, "NO_HELMET"):
                logger.log(source, "NO_HELMET", pid, "helmet missing", frame)
            if vest not in ("OK", "UNVERIFIABLE") and should_log(pid, "NO_VEST"):
                logger.log(source, "NO_VEST", pid, "vest missing", frame)

            # State transitions logged immediately — bypass cooldown
            if h_removed:
                last_logged[f"P{pid}_NO_HELMET"] = 0  # reset so next check logs it
                logger.log(source, "PPE_REMOVED", pid, "helmet removed mid-session", frame)
            if v_removed:
                last_logged[f"P{pid}_NO_VEST"] = 0
                logger.log(source, "PPE_REMOVED", pid, "vest removed mid-session", frame)

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

        cv2.putText(frame, f"People: {len(results)}", (10, h - 15),
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
