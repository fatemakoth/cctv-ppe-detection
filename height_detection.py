import cv2
import json
import argparse
import sys
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
PERSON_CLASS_ID = 0
ALERT_HEIGHT_CM = 185
FLOOR_TOLERANCE_PX = 60      # feet within this many px of floor_y = on the floor
MIN_CONFIDENCE = 0.50        # ignore detections below this — filters coat/object false positives
MIN_BODY_HEIGHT_CM = 80      # ignore boxes shorter than this (hanging clothes, bags, etc.)
MIN_ASPECT_RATIO = 1.2       # height/width must exceed this — people are taller than wide


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        print(f"[INFO] Calibration loaded: {data['pixels_per_cm']:.4f} px/cm, floor_y={data['floor_y']}")
        return data
    except FileNotFoundError:
        print("[ERROR] calibration.json not found. Run calibrate.py first.")
        sys.exit(1)


def is_valid_person(x1, y1, x2, y2, conf, pixels_per_cm):
    box_h = y2 - y1
    box_w = x2 - x1
    if box_w == 0:
        return False, ""

    body_height_cm = box_h / pixels_per_cm
    aspect_ratio = box_h / box_w

    if conf < MIN_CONFIDENCE:
        return False, f"low conf ({conf:.2f})"
    if body_height_cm < MIN_BODY_HEIGHT_CM:
        return False, f"too short ({body_height_cm:.0f}cm)"
    if aspect_ratio < MIN_ASPECT_RATIO:
        return False, f"wrong shape ({aspect_ratio:.1f})"

    return True, ""


def draw_height_overlay(frame, floor_y, pixels_per_cm):
    h, w = frame.shape[:2]

    # Only show the operational alert line — remove debug reference lines
    alert_y = int(floor_y - ALERT_HEIGHT_CM * pixels_per_cm)
    if 0 <= alert_y < h:
        cv2.line(frame, (0, alert_y), (w, alert_y), (0, 0, 255), 2)
        cv2.putText(frame, f"{ALERT_HEIGHT_CM}cm ELEVATION THRESHOLD", (10, alert_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def classify_person(x1, y1, x2, y2, floor_y, pixels_per_cm):
    head_height_cm = (floor_y - y1) / pixels_per_cm
    body_height_cm = (y2 - y1) / pixels_per_cm
    feet_above_floor_px = floor_y - y2  # positive = feet above floor line

    feet_on_floor = feet_above_floor_px <= FLOOR_TOLERANCE_PX
    head_above_threshold = head_height_cm > ALERT_HEIGHT_CM

    # RED:    feet off ground + head above 185cm line → standing elevated
    # ORANGE: feet off ground + head below 185cm line → elevated but bending/crouching
    # GREEN:  feet on ground → normal
    elevated = not feet_on_floor and head_above_threshold
    elevated_bending = not feet_on_floor and not head_above_threshold

    return {
        "head_height_cm": head_height_cm,
        "body_height_cm": body_height_cm,
        "feet_on_floor": feet_on_floor,
        "head_above_threshold": head_above_threshold,
        "elevated": elevated,
        "elevated_bending": elevated_bending,
    }


def draw_person(frame, x1, y1, x2, y2, info, person_id):
    elevated = info["elevated"]

    if info["elevated"]:
        color = (0, 0, 255)        # red — feet off ground, head above threshold
    elif info["elevated_bending"]:
        color = (0, 165, 255)      # orange — feet off ground, bending below threshold
    else:
        color = (0, 255, 0)        # green — normal

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if info["feet_on_floor"]:
        feet_status = "ON FLOOR"
    elif info["elevated"]:
        feet_status = "FEET OFF FLOOR"
    else:
        feet_status = "FEET OFF FLOOR (BENDING)"

    label_lines = [
        f"P{person_id} | {feet_status}",
        f"head:{info['head_height_cm']:.0f}cm | body:{info['body_height_cm']:.0f}cm",
    ]
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line, (x1, y1 - 8 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if info["elevated"]:
        cv2.putText(frame, "!! ELEVATED !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif info["elevated_bending"]:
        cv2.putText(frame, "!! ELEVATED (BENDING) !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


def run(source):
    cal = load_calibration()
    floor_y = cal["floor_y"]
    pixels_per_cm = cal["pixels_per_cm"]

    model = YOLO("yolov8s.pt")  # small model — better pose/crouch detection than nano
    print("[INFO] YOLOv8s loaded.")
    print(f"[INFO] Filters — min confidence: {MIN_CONFIDENCE}, min body height: {MIN_BODY_HEIGHT_CM}cm, min aspect ratio: {MIN_ASPECT_RATIO}")

    if isinstance(source, str) and source.startswith("rtsp"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h} — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost stream.")
            break

        results = model(frame, verbose=False)[0]
        draw_height_overlay(frame, floor_y, pixels_per_cm)

        alert_count = 0
        bending_count = 0
        person_id = 0
        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            valid, reason = is_valid_person(x1, y1, x2, y2, conf, pixels_per_cm)
            if not valid:
                # Draw a thin grey box so you can see what was filtered and why
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(frame, f"skip:{reason}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
                continue

            person_id += 1
            info = classify_person(x1, y1, x2, y2, floor_y, pixels_per_cm)
            draw_person(frame, x1, y1, x2, y2, info, person_id)

            if info["elevated"]:
                alert_count += 1
            elif info["elevated_bending"]:
                bending_count += 1

        cv2.putText(frame, f"People: {person_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if alert_count > 0:
            cv2.putText(frame, f"ELEVATION ALERT x{alert_count}",
                        (w // 2 - 180, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        if bending_count > 0:
            cv2.putText(frame, f"ELEVATED (BENDING) x{bending_count}",
                        (w // 2 - 200, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 165, 255), 2)

        cv2.imshow("Height Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
