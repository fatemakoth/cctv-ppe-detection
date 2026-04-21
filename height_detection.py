import cv2
import json
import argparse
import sys
import os
import mediapipe as mp
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
PERSON_CLASS_ID  = 0
ALERT_HEIGHT_CM  = 185   # head above this = RED
MIN_CONFIDENCE   = 0.50
MIN_BOX_HEIGHT_PX = 80
MIN_ASPECT_RATIO  = 0.6
FOOT_TOLERANCE    = 0.15  # ankle must be within 15% of person height from floor line

mp_pose  = mp.solutions.pose
FOOT_IDS = [27, 28, 29, 30]  # left ankle, right ankle, left heel, right heel


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


def run_pose(crop, pose_model):
    """Run MediaPipe Pose on a crop, return the result object (or None)."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return pose_model.process(rgb)


def check_feet(pose_result, crop_h, floor_y_global, y1):
    """
    Given a MediaPipe pose result, decide if feet are on the floor.
    Returns: True (on floor), False (feet off floor), None (inconclusive)
    """
    if pose_result is None or not pose_result.pose_landmarks:
        return None

    landmarks = pose_result.pose_landmarks.landmark
    foot_ys   = []
    for idx in FOOT_IDS:
        lm = landmarks[idx]
        if lm.visibility >= 0.4:
            foot_ys.append(int(lm.y * crop_h))

    if not foot_ys:
        return None

    avg_foot_y = sum(foot_ys) / len(foot_ys)

    # Clamp floor to crop bounds — far-away people have floor_y below their bounding box
    floor_in_crop = min(crop_h - 1, max(0, floor_y_global - y1))
    tolerance_px  = crop_h * FOOT_TOLERANCE

    return avg_foot_y >= (floor_in_crop - tolerance_px)


def draw_ankle_dots(frame, crop_result, x1, y1, x2, y2):
    """Draw ankle keypoints back onto the main frame."""
    if crop_result is None:
        return
    h = y2 - y1
    w = x2 - x1
    if h == 0 or w == 0:
        return
    landmarks = crop_result.pose_landmarks
    if not landmarks:
        return
    for idx in FOOT_IDS:
        lm = landmarks.landmark[idx]
        if lm.visibility >= 0.4:
            px = x1 + int(lm.x * w)
            py = y1 + int(lm.y * h)
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)


def classify(x1, y1, x2, y2, floor_y, pixels_per_cm, feet_on_floor):
    head_height_cm = (floor_y - y1) / pixels_per_cm
    body_height_cm = (y2 - y1) / pixels_per_cm

    head_elevated = head_height_cm > ALERT_HEIGHT_CM

    # RED:    head above threshold (standing elevated — most dangerous)
    # ORANGE: head below threshold but MediaPipe says feet are off ground (bending/sitting elevated)
    # GREEN:  feet on floor or inconclusive (default safe state)
    if head_elevated:
        state = "elevated"
    elif feet_on_floor is False:
        state = "bending"
    else:
        state = "ok"

    return {
        "head_height_cm": head_height_cm,
        "body_height_cm": body_height_cm,
        "state": state,
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
        color        = (0, 255, 0)
        status       = "ON FLOOR"
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
    cal          = load_calibration()
    floor_y      = cal["floor_y"]
    pixels_per_cm = cal["pixels_per_cm"]

    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8n loaded.")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    print("[INFO] MediaPipe Pose loaded. Press 'q' to quit.")

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

        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1_b, x2, y2_b = map(int, box.xyxy[0])
            x1   = max(0, x1);   y1_b = max(0, y1_b)
            x2   = min(w, x2);   y2_b = min(h, y2_b)
            conf = float(box.conf[0])
            bh   = y2_b - y1_b
            bw   = x2 - x1

            if bw == 0 or bh == 0:
                continue
            if conf < MIN_CONFIDENCE:
                cv2.rectangle(frame, (x1, y1_b), (x2, y2_b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:conf {conf:.2f}", (x1, y1_b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if bh < MIN_BOX_HEIGHT_PX:
                cv2.rectangle(frame, (x1, y1_b), (x2, y2_b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:tiny {bh}px", (x1, y1_b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if bh / bw < MIN_ASPECT_RATIO:
                cv2.rectangle(frame, (x1, y1_b), (x2, y2_b), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:shape {bh/bw:.1f}", (x1, y1_b - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue

            # Run MediaPipe once per person, reuse result for both foot check and dots
            crop        = frame[y1_b:y2_b, x1:x2]
            pose_result = run_pose(crop, pose)
            feet_status = check_feet(pose_result, y2_b - y1_b, floor_y, y1_b)
            draw_ankle_dots(frame, pose_result, x1, y1_b, x2, y2_b)

            person_id += 1
            info = classify(x1, y1_b, x2, y2_b, floor_y, pixels_per_cm, feet_status)
            draw_person(frame, x1, y1_b, x2, y2_b, info, person_id)

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
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
