import cv2
import json
import argparse
import sys
import os
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
PERSON_CLASS_ID = 0
ALERT_HEIGHT_CM = 185       # head above this = RED elevation alert
FLOOR_TOLERANCE_CM = 65     # feet this many cm above floor = elevated
                             # 65cm chosen: floor perspective error peaks ~55cm, chair starts ~87cm
MIN_CONFIDENCE = 0.50    # ignore low-confidence detections (primary filter for coats/objects)
MIN_BOX_HEIGHT_PX = 80   # minimum bounding box height in pixels — filters tiny false detections
                          # pixel-based so it works at any distance, unlike a cm threshold
MIN_ASPECT_RATIO = 0.6   # allow bent/crouching poses — only rejects nearly-horizontal blobs


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

    # Force TCP transport — more reliable than UDP on most networks
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    # Try backends in order: FFMPEG → ANY → default (no flag)
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
        cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"[INFO] RTSP opened with backend={backend}")
            return cap
        cap.release()

    print("[ERROR] All backends failed for RTSP source.")
    return cv2.VideoCapture(source)


def draw_overlay(frame, floor_y, pixels_per_cm):
    h, w = frame.shape[:2]
    alert_y = int(floor_y - ALERT_HEIGHT_CM * pixels_per_cm)
    if 0 <= alert_y < h:
        cv2.line(frame, (0, alert_y), (w, alert_y), (0, 0, 255), 2)
        cv2.putText(frame, f"{ALERT_HEIGHT_CM}cm ELEVATION THRESHOLD",
                    (10, alert_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def classify(x1, y1, x2, y2, floor_y, pixels_per_cm):
    body_height_cm = (y2 - y1) / pixels_per_cm
    head_height_cm = (floor_y - y1) / pixels_per_cm
    foot_height_cm = head_height_cm - body_height_cm  # how high feet are above calibrated floor

    feet_on_floor = foot_height_cm <= FLOOR_TOLERANCE_CM
    head_above_threshold = head_height_cm > ALERT_HEIGHT_CM

    # RED:    head above 185cm — standing elevated (feet check not needed, head says it all)
    # ORANGE: head below 185cm but feet clearly off ground — bending/sitting while elevated
    # GREEN:  feet within 65cm of floor — on the ground
    elevated = head_above_threshold
    elevated_bending = not feet_on_floor and not head_above_threshold

    return {
        "head_height_cm": head_height_cm,
        "body_height_cm": body_height_cm,
        "foot_height_cm": foot_height_cm,
        "feet_on_floor": feet_on_floor,
        "elevated": elevated,
        "elevated_bending": elevated_bending,
    }


def draw_person(frame, x1, y1, x2, y2, info, pid):
    if info["elevated"]:
        color = (0, 0, 255)
        status = "ELEVATED"
        bottom_label = "!! ELEVATED !!"
    elif info["elevated_bending"]:
        color = (0, 165, 255)
        status = "FEET OFF FLOOR (BENDING)"
        bottom_label = "!! ELEVATED (BENDING) !!"
    else:
        color = (0, 255, 0)
        status = "ON FLOOR"
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
    cal = load_calibration()
    floor_y = cal["floor_y"]
    pixels_per_cm = cal["pixels_per_cm"]

    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8n loaded. Press 'q' to quit.")

    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the latest frame, not buffered ones

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

        alert_count = 0
        bending_count = 0
        person_id = 0

        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            box_h = y2 - y1
            box_w = x2 - x1

            if box_w == 0:
                continue

            aspect = box_h / box_w

            if conf < MIN_CONFIDENCE:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:conf {conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if box_h < MIN_BOX_HEIGHT_PX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:tiny {box_h}px", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue
            if aspect < MIN_ASPECT_RATIO:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(frame, f"skip:shape {aspect:.1f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)
                continue

            person_id += 1
            info = classify(x1, y1, x2, y2, floor_y, pixels_per_cm)
            draw_person(frame, x1, y1, x2, y2, info, person_id)

            if info["elevated"]:
                alert_count += 1
            elif info["elevated_bending"]:
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
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
