"""
Camera calibration for height detection + floor mapping.

Modes:
  'a' — auto-detect height (YOLOv8n-pose, stand in front of camera)
  'm' — manual click height (click head + feet)
  'f' — floor map (click 5+ floor points at different depths/positions)
  'q' — quit

Usage:
  python calibrate.py --source 0 --height 170
  python calibrate.py --source "rtsp://..." --height 175
  python calibrate.py --source 0          # floor map only (--height optional)
"""

import cv2
import json
import argparse
import time
import os
import numpy as np
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
ALERT_HEIGHT_CM  = 185
AVG_FRAMES       = 60    # frames to average for stable measurement
KEYPOINT_CONF    = 0.40  # min confidence to trust a keypoint

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

manual_clicks = []
floor_clicks  = []


def on_click_height(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(manual_clicks) < 2:
        manual_clicks.append((x, y))
        label = "HEAD" if len(manual_clicks) == 1 else "FEET"
        print(f"  {label} click: ({x}, {y})")

def on_click_floor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        floor_clicks.append((x, y))
        print(f"  Floor point #{len(floor_clicks)}: ({x}, {y})")


def open_capture(source):
    if isinstance(source, str) and source.startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
            cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
            if cap.isOpened():
                print(f"[INFO] RTSP opened (backend={backend})")
                return cap
            cap.release()
    return cv2.VideoCapture(source)


def draw_preview(frame, head_y, feet_y, pixels_per_cm, floor_y, w, h):
    """Draw calibration preview: measurement line + 185cm threshold."""
    out = frame.copy()
    if head_y and feet_y:
        cy = frame.shape[1] // 2
        cv2.line(out, (cy, int(head_y)), (cy, int(feet_y)), (0, 255, 255), 2)
        cv2.circle(out, (cy, int(head_y)), 7, (0, 255, 0), -1)
        cv2.circle(out, (cy, int(feet_y)), 7, (255, 100, 0), -1)
        cv2.putText(out, "HEAD", (cy + 10, int(head_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(out, "FEET / FLOOR", (cy + 10, int(feet_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)

    if pixels_per_cm and floor_y:
        alert_y = int(floor_y - ALERT_HEIGHT_CM * pixels_per_cm)
        if 0 <= alert_y < h:
            cv2.line(out, (0, alert_y), (w, alert_y), (0, 0, 255), 2)
            cv2.putText(out, f"{ALERT_HEIGHT_CM}cm elevation threshold",
                        (10, alert_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.line(out, (0, int(floor_y)), (w, int(floor_y)), (0, 200, 0), 1)
        cv2.putText(out, "floor", (10, int(floor_y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    return out


def draw_floor_fit(frame, points, w):
    """Draw clicked floor dots and the linear regression line."""
    for px, py in points:
        cv2.circle(frame, (px, py), 7, (0, 255, 0), -1)
        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1)

    if len(points) >= 2:
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        coeffs  = np.polyfit(xs, ys, 1)
        y_left  = int(np.polyval(coeffs, 0))
        y_right = int(np.polyval(coeffs, w - 1))
        cv2.line(frame, (0, y_left), (w - 1, y_right), (0, 220, 0), 2)


def auto_measure(cap, model, height_cm, w, h):
    """
    Run pose model over AVG_FRAMES frames.
    Returns (head_y, feet_y, last_frame) averaged from detected keypoints.
    """
    print(f"[INFO] Auto-measuring over {AVG_FRAMES} frames — stand still and face the camera...")
    head_ys = []
    feet_ys = []
    collected = 0
    last_frame = None

    while collected < AVG_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame

        results = model(frame, verbose=False, conf=0.40)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue

        best_i, best_conf = None, 0.0
        for i, box in enumerate(results.boxes):
            if int(box.cls) != 0:
                continue
            c = float(box.conf)
            if c > best_conf:
                best_conf, best_i = c, i

        if best_i is None or results.keypoints is None:
            continue

        kp = results.keypoints[best_i].data[0]   # [17, 3]
        box_y1 = float(results.boxes[best_i].xyxy[0][1])

        ankle_ys = []
        for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
            if float(kp[idx][2]) >= KEYPOINT_CONF:
                ankle_ys.append(float(kp[idx][1]))

        if not ankle_ys:
            continue

        head_ys.append(box_y1)
        feet_ys.append(sum(ankle_ys) / len(ankle_ys))
        collected += 1

        bar = int(30 * collected / AVG_FRAMES)
        print(f"\r  Collecting [{('#' * bar).ljust(30)}] {collected}/{AVG_FRAMES}", end="", flush=True)

    print()

    if len(head_ys) < AVG_FRAMES // 2:
        print(f"[WARN] Only got {len(head_ys)} valid frames — auto-detect may be unreliable.")
        if not head_ys:
            return None, None, last_frame

    head_y = float(np.median(head_ys))
    feet_y = float(np.median(feet_ys))
    print(f"[INFO] Auto-detected — head_y={head_y:.1f}px  feet_y={feet_y:.1f}px  ({len(head_ys)} samples)")
    return head_y, feet_y, last_frame


def manual_measure(cap, w, h):
    """Let user click head top and feet on a live frame."""
    manual_clicks.clear()
    cv2.setMouseCallback("Calibration", on_click_height)

    print("\n[MANUAL] Click on:")
    print("  1st click — top of HEAD")
    print("  2nd click — bottom of FEET (floor level)")
    print("  'r' to reset  |  'q' to quit\n")

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        display = frame.copy()

        for i, (x, y) in enumerate(manual_clicks):
            color = (0, 255, 0) if i == 0 else (255, 100, 0)
            label = "HEAD" if i == 0 else "FEET"
            cv2.circle(display, (x, y), 8, color, -1)
            cv2.putText(display, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        if len(manual_clicks) == 2:
            cv2.line(display, manual_clicks[0], manual_clicks[1], (0, 255, 255), 2)

        cv2.putText(display, f"Clicks: {len(manual_clicks)}/2  |  r=reset  q=quit",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            manual_clicks.clear()
        elif key == ord('q'):
            return None, None, last_frame
        elif len(manual_clicks) == 2:
            head_y = float(min(manual_clicks[0][1], manual_clicks[1][1]))
            feet_y = float(max(manual_clicks[0][1], manual_clicks[1][1]))
            return head_y, feet_y, last_frame

    return None, None, last_frame


def floor_map_mode(cap, w, h):
    """
    User clicks floor points at different positions and depths.
    Shows fitted line in real-time.
    Returns list of [x, y] pairs, or None if cancelled.
    """
    floor_clicks.clear()
    cv2.setMouseCallback("Calibration", on_click_floor)

    print("\n[FLOOR MAP] Click on the floor at different positions across the frame.")
    print("  Click near/far, left/centre/right — aim for 6-10 points spread across the floor.")
    print("  'u' — undo last point")
    print("  'd' — done (need at least 4 points)")
    print("  'r' — reset all points")
    print("  'q' — quit without saving\n")

    snapshot = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        snapshot = frame.copy()
        display  = frame.copy()

        draw_floor_fit(display, floor_clicks, w)

        n = len(floor_clicks)
        status = f"Points: {n}  |  u=undo  r=reset  "
        status += "d=DONE  " if n >= 4 else f"need {4 - n} more  "
        status += "q=quit"
        cv2.putText(display, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if n >= 4:
            cv2.putText(display, "Floor line fit — looks good? Press 'd' to confirm.",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u') and floor_clicks:
            removed = floor_clicks.pop()
            print(f"  Removed point ({removed[0]}, {removed[1]}) — {len(floor_clicks)} remaining")
        elif key == ord('r'):
            floor_clicks.clear()
            print("  Reset — all points cleared.")
        elif key == ord('d'):
            if len(floor_clicks) < 4:
                print(f"  Need at least 4 points (have {len(floor_clicks)}). Keep clicking.")
            else:
                print(f"\n[INFO] Confirmed {len(floor_clicks)} floor points.")
                return list(floor_clicks), snapshot
        elif key == ord('q'):
            return None, snapshot

    return None, snapshot


def load_existing_calibration():
    """Load existing calibration.json so we can merge new data without wiping old values."""
    try:
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def run(source, height_cm):
    need_height = height_cm is not None

    print("[INFO] Loading pose model...")
    model = YOLO("yolov8n-pose.pt")

    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    cv2.namedWindow("Calibration")

    # ── Step 1: mode selection ─────────────────────────────────────────────────
    print("\n[INFO] Live preview — press:")
    if need_height:
        print("  'a' — auto-detect height (stand in front of camera)")
        print("  'm' — manual click height (click head + feet)")
    print("  'f' — floor map (click floor points for perspective correction)")
    print("  'q' — quit\n")

    mode = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        hint = "'a'=auto  'm'=manual  'f'=floor-map  'q'=quit" if need_height else "'f'=floor-map  'q'=quit"
        cv2.putText(display, hint, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a') and need_height:
            mode = "auto"
            break
        elif key == ord('m') and need_height:
            mode = "manual"
            break
        elif key == ord('f'):
            mode = "floor"
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # ── Floor map mode ─────────────────────────────────────────────────────────
    if mode == "floor":
        points, snapshot = floor_map_mode(cap, w, h)
        cap.release()

        if points is None:
            print("[INFO] Quit — nothing saved.")
            cv2.destroyAllWindows()
            return

        # Preview the fitted line on snapshot
        if snapshot is not None:
            preview = snapshot.copy()
            draw_floor_fit(preview, points, w)
            cv2.putText(preview, f"{len(points)} floor points — Press 's' to save  |  'q' to discard",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow("Calibration", preview)

        print(f"\n[RESULT] {len(points)} floor points collected.")
        print("Press 's' to save, 'q' to discard.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                data = load_existing_calibration()
                data["floor_points"] = [[int(x), int(y)] for x, y in points]
                with open(CALIBRATION_FILE, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"[SAVED] {CALIBRATION_FILE}  ({len(points)} floor points)")
                break
            elif key == ord('q'):
                print("[INFO] Discarded — nothing saved.")
                break

        cv2.destroyAllWindows()
        return

    # ── Height calibration modes ───────────────────────────────────────────────
    if mode == "auto":
        print("\n[INFO] Stand still and tall, facing the camera.")
        print("[INFO] Starting in 3 seconds...")
        start = time.time()
        while time.time() - start < 3:
            ret, frame = cap.read()
            if not ret: break
            remaining = 3 - int(time.time() - start)
            cv2.putText(frame, f"Starting in {remaining}...",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)

        head_y, feet_y, snapshot = auto_measure(cap, model, height_cm, w, h)

        if head_y is None:
            print("[WARN] Auto-detect failed — switching to manual mode.")
            mode = "manual"

    if mode == "manual":
        head_y, feet_y, snapshot = manual_measure(cap, w, h)
        if head_y is None:
            print("[INFO] Quit — nothing saved.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()

    # ── Compute + confirm ──────────────────────────────────────────────────────
    pixel_height  = feet_y - head_y
    pixels_per_cm = pixel_height / height_cm
    floor_y       = feet_y

    print(f"\n[RESULT]")
    print(f"  head_y        : {head_y:.1f} px")
    print(f"  feet_y        : {feet_y:.1f} px")
    print(f"  pixel_height  : {pixel_height:.1f} px")
    print(f"  pixels_per_cm : {pixels_per_cm:.4f}")
    print(f"  floor_y       : {floor_y:.1f} px")

    if pixels_per_cm < 1.0 or pixels_per_cm > 20.0:
        print(f"[WARN] pixels_per_cm={pixels_per_cm:.2f} looks unusual. Did you enter the right height?")

    if snapshot is not None:
        preview = draw_preview(snapshot, head_y, feet_y, pixels_per_cm, floor_y, w, h)
        cv2.putText(preview, "Press 's' to save  |  'q' to discard",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Calibration", preview)

    print("\nPress 's' to save, 'q' to discard.")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s'):
            data = load_existing_calibration()
            data.update({
                "pixels_per_cm":          round(pixels_per_cm, 4),
                "floor_y":                round(floor_y, 1),
                "reference_height_cm":    height_cm,
                "reference_pixel_height": round(pixel_height, 1),
            })
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[SAVED] {CALIBRATION_FILE}")
            break
        elif key == ord('q'):
            print("[INFO] Discarded — nothing saved.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    parser.add_argument("--height", type=float, default=None,
                        help="Your real height in cm (e.g. 170) — required for height calibration, optional for floor map")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.height)
