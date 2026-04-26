import cv2
import argparse
import os
import time
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from sheets_logger import SheetsLogger
from config import *

PPE_CLASSES = {HAS_HELMET, NO_HELMET, HAS_VEST, NO_VEST}


# ── Geometry ──────────────────────────────────────────────────────────────────
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def region_crop(frame, pbox, top_f, bot_f, fh):
    x1, y1, x2, y2 = pbox
    ph  = y2 - y1
    ry1 = max(0, y1 + int(ph * top_f))
    ry2 = min(fh, y1 + int(ph * bot_f))
    c   = frame[ry1:ry2, x1:x2]
    return c if c.size > 0 else None

# ── PersonTracker ─────────────────────────────────────────────────────────────
class PersonTracker:
    def __init__(self, fps=25):
        self.tracks = {}
        self._next  = 0
        self._fps   = max(fps, 1)

    def update(self, detections):
        matched = set()
        results = []
        for box, h, v, off in detections:
            best_id, best_s = None, IOU_MATCH
            for tid, t in self.tracks.items():
                s = iou(box, t["box"])
                if s > best_s:
                    best_s, best_id = s, tid
            if best_id is None:
                best_id = self._next
                self._next += 1
                self.tracks[best_id] = {
                    "box":        box,
                    "helmet":     deque(maxlen=SMOOTH_FRAMES),
                    "vest":       deque(maxlen=SMOOTH_FRAMES),
                    "off":        deque(maxlen=SMOOTH_FRAMES),
                    "off_streak": 0,
                }
            t = self.tracks[best_id]
            t["box"] = box
            t["helmet"].append(h)
            t["vest"].append(v)
            t["off"].append(off)

            raw_off = _vote_off(t["off"])
            if raw_off:
                t["off_streak"] += 1
            else:
                t["off_streak"] = 0
            sustained = (t["off_streak"] / self._fps) >= FEET_MIN_DURATION

            matched.add(best_id)
            results.append((box, _vote_ppe(t["helmet"]), _vote_ppe(t["vest"]), sustained))
        self.tracks = {k: v for k, v in self.tracks.items() if k in matched}
        return results

def _vote_ppe(hist):
    c = {}
    for s in hist: c[s] = c.get(s, 0) + 1
    if c.get("OK",      0) >= OK_THRESH:      return "OK"
    if c.get("MISSING", 0) >= MISSING_THRESH: return "MISSING"
    return "MISSING"

def _vote_off(hist):
    return sum(hist) >= 3

# ── PPE check ─────────────────────────────────────────────────────────────────
def check_helmet(frame, pbox, ppe_model, ppe_names, fh):
    crop = region_crop(frame, pbox, HEAD_TOP, HEAD_BOT, fh)
    if crop is None: return "MISSING"
    best_yes = best_no = 0.0
    for b in ppe_model(crop, verbose=False, conf=HELMET_NO_CONF)[0].boxes:
        cls = ppe_names[int(b.cls)]; conf = float(b.conf)
        if cls == HAS_HELMET: best_yes = max(best_yes, conf)
        if cls == NO_HELMET:  best_no  = max(best_no,  conf)
    if best_no  >= HELMET_NO_CONF: return "MISSING"
    if best_yes >= HELMET_OK_CONF: return "OK"
    return "MISSING"

def check_vest(frame, pbox, ppe_model, ppe_names, fh):
    crop = region_crop(frame, pbox, TORSO_TOP, TORSO_BOT, fh)
    if crop is None: return "MISSING"
    best_yes = best_no = 0.0
    for b in ppe_model(crop, verbose=False, conf=VEST_NO_CONF)[0].boxes:
        cls = ppe_names[int(b.cls)]; conf = float(b.conf)
        if cls == HAS_VEST: best_yes = max(best_yes, conf)
        if cls == NO_VEST:  best_no  = max(best_no,  conf)
    if best_no  >= VEST_NO_CONF:  return "MISSING"
    if best_yes >= VEST_OK_CONF:  return "OK"
    return "MISSING"

# ── Feet check ────────────────────────────────────────────────────────────────
def check_feet(pbox, kps):
    x1, y1, x2, y2 = pbox
    box_h = max(y2 - y1, 1)
    if kps is None or kps.data is None or len(kps.data) == 0:
        return False
    kp_data = kps.data[0]
    ankle_ys = [float(kp_data[idx][1]) for idx in [LEFT_ANKLE, RIGHT_ANKLE]
                if float(kp_data[idx][2]) >= FOOT_CONF_MIN]
    if not ankle_ys: return False
    avg_ay    = sum(ankle_ys) / len(ankle_ys)
    ankle_rel = (y2 - avg_ay) / box_h
    return ankle_rel > ANKLE_BOX_THRESH

# ── Drawing ───────────────────────────────────────────────────────────────────
def _col(s):
    return (0,200,0) if s=="OK" else (0,0,255)

def draw_person(frame, box, helmet, vest, off_floor, pid):
    x1, y1, x2, y2 = box
    ppe_ok = helmet == "OK" and vest == "OK"
    bc = (0,140,255) if off_floor else ((0,0,255) if not ppe_ok else (0,200,0))
    cv2.rectangle(frame, (x1,y1), (x2,y2), bc, 2)
    for i, (txt, col) in enumerate([
        (f"P{pid}",           bc),
        (f"Helmet:{helmet}",  _col(helmet)),
        (f"Vest:  {vest}",    _col(vest)),
    ]):
        cv2.putText(frame, txt, (x1, y1-10-i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)
    bottom = y2 + 22
    if off_floor:
        cv2.putText(frame, "!! FEET OFF FLOOR !!",
                    (x1, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)
        bottom += 22
    if not ppe_ok:
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# ── Main ──────────────────────────────────────────────────────────────────────
def run(video_path, show, skip_frames):
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Loading models...")
    pose_model = YOLO(POSE_MODEL)
    ppe_model  = YOLO(PPE_MODEL_PATH)
    ppe_names  = ppe_model.names

    logger  = SheetsLogger()
    tracker = PersonTracker(fps=fps)

    cap         = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path   = os.path.join(OUTPUT_DIR, f"{video_name}_checked.mp4")
    writer     = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / max(skip_frames, 1), (W, H))

    print(f"[INFO] Video : {video_path}")
    print(f"[INFO] Frames: {total_frames} @ {fps:.1f} FPS  ({W}x{H})")
    print(f"[INFO] Output: {out_path}")
    print(f"[INFO] Press 'q' to stop early.\n")

    # Counters for summary
    violation_frames  = 0
    total_ppe_events  = 0
    total_feet_events = 0
    last_logged       = {}

    def should_log(pid, vtype):
        key = f"P{pid}_{vtype}"
        now = time.time()
        if now - last_logged.get(key, 0) >= 10:
            last_logged[key] = now
            return True
        return False

    frame_idx = 0
    start_t   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed (still write them unmodified)
        if skip_frames > 1 and frame_idx % skip_frames != 0:
            writer.write(frame)
            continue

        # Progress
        pct = frame_idx / max(total_frames, 1) * 100
        elapsed = time.time() - start_t
        eta = (elapsed / frame_idx) * (total_frames - frame_idx) if frame_idx > 0 else 0
        print(f"\r  [{pct:5.1f}%] frame {frame_idx}/{total_frames}  "
              f"ETA {eta:.0f}s  violations={total_ppe_events+total_feet_events}",
              end="", flush=True)

        pose_res = pose_model(frame, verbose=False, conf=PERSON_CONF)[0]

        raw = []
        for i, box in enumerate(pose_res.boxes):
            if int(box.cls) != 0: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            bh,bw = y2-y1, x2-x1
            if bw==0 or bh < MIN_BOX_HEIGHT or bh/max(bw,1) < MIN_ASPECT: continue

            pbox = (x1,y1,x2,y2)
            kps  = pose_res.keypoints[i] if pose_res.keypoints is not None else None
            helmet   = check_helmet(frame, pbox, ppe_model, ppe_names, H)
            vest     = check_vest(frame, pbox, ppe_model, ppe_names, H)
            off_floor = check_feet(pbox, kps)
            raw.append((pbox, helmet, vest, off_floor))

        smoothed = tracker.update(raw)
        frame_has_violation = False

        for idx, (box, helmet, vest, off_floor) in enumerate(smoothed):
            pid = idx + 1
            draw_person(frame, box, helmet, vest, off_floor, pid)

            if helmet != "OK" or vest != "OK":
                frame_has_violation = True
                if should_log(pid, "PPE"):
                    total_ppe_events += 1
                    logger.log(video_path, "PPE_VIOLATION", pid,
                               f"helmet={helmet} vest={vest}", frame)
            if off_floor:
                frame_has_violation = True
                if should_log(pid, "FEET"):
                    total_feet_events += 1
                    logger.log(video_path, "FEET_OFF_FLOOR", pid,
                               "feet not on floor", frame)

        if frame_has_violation:
            violation_frames += 1
            cv2.rectangle(frame, (0,0), (W,45), (0,0,160), -1)
            if total_ppe_events > 0:
                cv2.putText(frame, "PPE VIOLATION DETECTED",
                            (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        # Timestamp + frame counter
        ts = f"Frame {frame_idx}/{total_frames}"
        cv2.putText(frame, ts, (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        writer.write(frame)
        if show:
            cv2.imshow("Video Check", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Stopped early by user.")
                break

    print()
    cap.release()
    writer.release()
    logger.close()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_t
    print("\n" + "="*55)
    print("  COMPLIANCE REPORT")
    print("="*55)
    print(f"  Video         : {os.path.basename(video_path)}")
    print(f"  Frames checked: {frame_idx}")
    print(f"  Duration      : {elapsed:.1f}s")
    print(f"  Violation frames  : {violation_frames} / {frame_idx}")
    print(f"  PPE incidents     : {total_ppe_events}")
    print(f"  Feet-off-floor    : {total_feet_events}")
    print(f"  Annotated video   : {out_path}")
    print(f"  Snapshots         : snapshots/")
    print("="*55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True,
                        help="Path to video file (mp4, avi, mov, etc.)")
    parser.add_argument("--show",   action="store_true",
                        help="Show annotated video while processing")
    parser.add_argument("--skip",   type=int, default=2,
                        help="Process every Nth frame (default 2, use 1 for every frame)")
    args = parser.parse_args()
    run(args.video, args.show, args.skip)
