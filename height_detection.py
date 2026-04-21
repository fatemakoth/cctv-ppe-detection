import cv2
import json
import argparse
from ultralytics import YOLO

PERSON_CLASS_ID = 0
ELEVATION_THRESHOLD_CM = 185
CALIBRATION_FILE = "calibration.json"

def load_calibration():
    try:
        with open(CALIBRATION_FILE) as f:
            data = json.load(f)
        print(f"[INFO] Calibration loaded: {data['pixels_per_cm']:.4f} px/cm, floor_y={data['floor_y']}")
        return data
    except FileNotFoundError:
        print("[ERROR] calibration.json not found. Run calibrate.py first.")
        return None

def estimate_height_cm(top_y, floor_y, pixels_per_cm):
    pixel_distance = floor_y - top_y
    if pixel_distance <= 0:
        return 0
    return pixel_distance / pixels_per_cm

def run(source):
    calib = load_calibration()
    if calib is None:
        return

    pixels_per_cm = calib["pixels_per_cm"]
    floor_y = calib["floor_y"]

    model = YOLO("yolov8n.pt")
    print("[INFO] YOLOv8n loaded. Press 'q' to quit.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw floor line for reference
        h, w = frame.shape[:2]
        cv2.line(frame, (0, floor_y), (w, floor_y), (255, 200, 0), 1)
        cv2.putText(frame, "floor", (10, floor_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        results = model(frame, verbose=False)[0]
        alert_count = 0

        for box in results.boxes:
            if int(box.cls) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            head_y = y1
            foot_y = y2
            head_height_cm = estimate_height_cm(head_y, floor_y, pixels_per_cm)
            foot_height_cm = estimate_height_cm(foot_y, floor_y, pixels_per_cm)

            # Elevated: head above threshold AND feet not at floor level
            feet_off_floor = foot_height_cm > 20
            elevated = head_height_cm > ELEVATION_THRESHOLD_CM and feet_off_floor

            if elevated:
                color = (0, 0, 255)
                label = f"ELEVATED {head_height_cm:.0f}cm"
                alert_count += 1
            else:
                color = (0, 255, 0)
                label = f"{head_height_cm:.0f}cm"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if alert_count > 0:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.putText(frame, f"ELEVATION ALERT — {alert_count} person(s) elevated!",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Height Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="1", help="Camera index or RTSP URL")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
