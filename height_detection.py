import cv2
import json
import argparse
import sys
import os
from ultralytics import YOLO

CALIBRATION_FILE  = "calibration.json"
PERSON_CLASS_ID   = 0
ALERT_HEIGHT_CM   = 185    # head above this = RED
MIN_CONFIDENCE    = 0.50
MIN_BOX_HEIGHT_PX = 80
MIN_ASPECT_RATIO  = 0.6
FOOT_CONF_MIN     = 0.40   # min keypoint confidence to trust ankle position
FOOT_TOLERANCE    = 0.18   # ankles must be within this fraction of box height from floor

LEFT_ANKLE  = 15
RIGHT_ANKLE = 16


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        print(f"[INFO] Calibration: {data['pixels_per_cm']:.4f} px/cm  floor_y={data['floor_y']}")
        return data
    except FileNotFoundError:
        print("[ERROR] calibration.json not found — run calibrate.py first.")
        sys.exit(1)


def open_capture(source):
    if not (isinstance(source, str) and source.startswith("rtsp")):
        return cv2.VideoCapture(source)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
        cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"[INFO] RTSP opened with backend={backend}")
            return cap
        cap.release()
    return cv2.VideoCapture(source)


def draw_overlay(frame, floor_y, pixels_per_cm):
    h, w = frame.shape[:2]
    alert_y = int(floor_y - ALERT_HEIGHT_CM * pixels_per_cm)
    if 0 <= alert_y < h:
        cv2.line(frame, (0, alert_y), (w, alert_y), (0, 0, 255), 2)
        cv2.putText(frame, f"{ALERT_HEIGHT_CM}cm ELEVATION THRESHOLD",
                    (10, alert_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def check_feet(kps, y1, y2, floor_y):
    """
    Use YOLOv8-pose ankle keypoints (global frame coords) to check if feet are on the floor.
    Returns: True (on floor), False (feet off floor), None (ankles not visible / inconclusive)
    """
    if kps is None or kps.data is None or len(kps.data) == 0:
        return None

    kp_data  = kps.data[0]   # shape [17, 3] — x, y, confidence
    foot_ys  = []

    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp   = kp_data[idx]
        conf = float(kp[2])
        if conf >= FOOT_CONF_MIN:
            foot_ys.append(float(kp[1]))

    if not foot_ys:
        return None   # ankles not visible — inconclusive

    avg_foot_y = sum(foot_ys) / len(foot_ys)
    box_h      = y2 - y1

    # Clamp floor reference to bounding box bottom — handles far-away people where
    # the calibrated floor_y sits below their bounding box due to perspective
    effective_floor_y = min(floor_y, y2)
    tolerance_px      = box_h * FOOT_TOLERANCE

    return avg_foot_y >= (effective_floor_y - tolerance_px)


def draw_ankle_dots(frame, kps, color):
    if kps is None or kps.data is None or len(kps.data) == 0:
        return
    kp_data = kps.data[0]
    for idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        kp   = kp_data[idx]
        conf = float(kp[2])
        if conf >= FOOT_CONF_MIN:
            px, py = int(float(kp[0])), int(float(kp[1]))
            cv2.circle(frame, (px, py), 7, (0, 255, 255), -1)
            cv2.putText(frame, f"{conf:.0%}", (px + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)


def classify(x1, y1, x2, y2, floor_y, pixels_per_cm, feet_on_floor):
    head_height_cm = (floor_y - y1) / pixels_per_cm
    body_height_cm = (y2 - y1)     / pixels_per_cm

    if head_height_cm > ALERT_HEIGHT_CM:
        state = "elevated"               # RED — head above threshold
    elif feet_on_floor is False:
        state = "bending"                # ORANGE — ankles confirmed off floor
    else:
        state = "ok"                     # GREEN — on floor or inconclusive (safe default)

    return {
        "head_height_cm": head_height_cm,
        "body_height_cm": body_height_cm,
        "state":          state,
        "feet_status":    feet_on_floor,
    }


def draw_person(frame, x1, y1, x2, y2, info, pid):
    state = info["state"]
    if state == "elevated":
        color        = (0, 0, 255)
        status       = "ELEVATED"
        bottom_label = "!! ELEVATED !!"
    elif state == "bending":
        color        = (0, 165, 255)
        status       = "FEET OFF FLOOR (BENDING)"
        bottom_label = "!! ELEVATED (BENDING) !!"
    else:
        feet_lbl = {True: "ON FLOOR", False: "OFF FLOOR", None: "FEET ?"}
        color        = (0, 255, 0)
        status       = feet_lbl.get(info["feet_status"], "ON FLOOR")
        bottom_label = None

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for i, line in enumerate([
        f"P{pid} | {status}",
        f"head:{info['head_height_cm']:.0f}cm | body:{info['body_height_cm']:.0f}cm",
    ]):
        cv2.putText(frame, line, (x1, y1 - 8 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if bottom_label:
        cv2.putText(frame, bottom_label, (x1, y2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def run(source):
    cal           = load_calibration()
    floor_y       = cal["floor_y"]
    pixels_per_cm = cal["pixels_per_cm"]

    # yolov8n-pose detects people AND their keypoints (incl. ankles) in one pass
    model = YOLO("yolov8n-pose.pt")
    print("[INFO] YOLOv8n-pose loaded. Press 'q' to quit.")

    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost stream.")
            break

        results = model(frame, verbose=False, imgsz=640)[0]
        draw_overlay(frame, floor_y, pixels_per_cm)

        alert_count   = 0
        bending_count = 0
        person_id     = 0

        for i, box in enumerate(results.boxes):
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1b, x2, y2b = map(int, box.xyxy[0])
            x1  = max(0, x1);   y1b = max(0, y1b)
            x2  = min(w, x2);   y2b = min(h, y2b)
            conf = float(box.conf[0])
            bh   = y2b - y1b
            bw   = x2 - x1

            if bw == 0 or bh == 0:
                continue
            if conf < MIN_CONFIDENCE:
                cv2.rectangle(frame, (x1, y1b), (x2, y2b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:conf {conf:.2f}", (x1, y1b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if bh < MIN_BOX_HEIGHT_PX:
                cv2.rectangle(frame, (x1, y1b), (x2, y2b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:tiny {bh}px", (x1, y1b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if bh / bw < MIN_ASPECT_RATIO:
                cv2.rectangle(frame, (x1, y1b), (x2, y2b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:shape {bh/bw:.1f}", (x1, y1b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue

            # Ankle keypoints from pose model (global frame coordinates)
            kps         = results.keypoints[i] if results.keypoints is not None else None
            feet_status = check_feet(kps, y1b, y2b, floor_y)

            person_id += 1
            info = classify(x1, y1b, x2, y2b, floor_y, pixels_per_cm, feet_status)
            draw_person(frame, x1, y1b, x2, y2b, info, person_id)
            draw_ankle_dots(frame, kps, (0, 255, 255))

            if info["state"] == "elevated":
                alert_count += 1
            elif info["state"] == "bending":
                bending_count += 1

        cv2.putText(frame, f"People: {person_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if alert_count > 0:
            cv2.putText(frame, f"ELEVATION ALERT x{alert_count}",
                        (w // 2 - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if bending_count > 0:
            cv2.putText(frame, f"ELEVATED (BENDING) x{bending_count}",
                        (w // 2 - 210, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        cv2.imshow("Height Detection", frame)
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
