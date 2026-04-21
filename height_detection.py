import cv2
import json
import argparse
import sys
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
PERSON_CLASS_ID = 0
ALERT_HEIGHT_CM = 185
FLOOR_TOLERANCE_PX = 60  # feet within this many px of floor_y = on the floor


def load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        print(f"[INFO] Calibration loaded: {data['pixels_per_cm']:.4f} px/cm, floor_y={data['floor_y']}")
        return data
    except FileNotFoundError:
        print("[ERROR] calibration.json not found. Run calibrate.py first.")
        sys.exit(1)


def draw_height_overlay(frame, floor_y, pixels_per_cm):
    h, w = frame.shape[:2]

    # Floor line
    cv2.line(frame, (0, floor_y), (w, floor_y), (0, 200, 255), 1)
    cv2.putText(frame, "FLOOR", (10, floor_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # 185 cm alert threshold line
    alert_y = int(floor_y - ALERT_HEIGHT_CM * pixels_per_cm)
    if 0 <= alert_y < h:
        cv2.line(frame, (0, alert_y), (w, alert_y), (0, 0, 255), 2)
        cv2.putText(frame, f"{ALERT_HEIGHT_CM}cm ALERT LINE", (10, alert_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Reference height line (your calibrated height)
    ref_y = int(floor_y - 165.1 * pixels_per_cm)
    if 0 <= ref_y < h:
        cv2.line(frame, (0, ref_y), (w, ref_y), (200, 200, 0), 1)
        cv2.putText(frame, "165cm ref", (10, ref_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)


def classify_person(x1, y1, x2, y2, floor_y, pixels_per_cm):
    head_y = y1
    feet_y = y2

    head_height_cm = (floor_y - head_y) / pixels_per_cm
    body_height_cm = (y2 - y1) / pixels_per_cm
    feet_above_floor_px = floor_y - feet_y  # positive = feet above floor

    feet_on_floor = feet_above_floor_px <= FLOOR_TOLERANCE_PX
    elevated = head_height_cm > ALERT_HEIGHT_CM and not feet_on_floor

    return {
        "head_height_cm": head_height_cm,
        "body_height_cm": body_height_cm,
        "feet_on_floor": feet_on_floor,
        "elevated": elevated,
    }


def draw_person(frame, x1, y1, x2, y2, info, person_id):
    elevated = info["elevated"]
    color = (0, 0, 255) if elevated else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_lines = [
        f"P{person_id} | head:{info['head_height_cm']:.0f}cm",
        f"body:{info['body_height_cm']:.0f}cm | {'ON FLOOR' if info['feet_on_floor'] else 'ELEVATED'}",
    ]
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line, (x1, y1 - 8 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if elevated:
        cv2.putText(frame, "!! ELEVATED !!",
                    (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


def run(source):
    cal = load_calibration()
    floor_y = cal["floor_y"]
    pixels_per_cm = cal["pixels_per_cm"]

    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8n loaded.")

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
        person_id = 0
        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_id += 1

            info = classify_person(x1, y1, x2, y2, floor_y, pixels_per_cm)
            draw_person(frame, x1, y1, x2, y2, info, person_id)

            if info["elevated"]:
                alert_count += 1

        # Status bar
        status = f"People: {person_id}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if alert_count > 0:
            cv2.putText(frame, f"ELEVATION ALERT x{alert_count}",
                        (w // 2 - 160, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0, 0, 255), 3)

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
