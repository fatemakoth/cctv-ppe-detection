import cv2
import json
import argparse
import mediapipe as mp
from ultralytics import YOLO

CALIBRATION_FILE = "calibration.json"
PERSON_CLASS     = 0
PERSON_CONF      = 0.45
FLOOR_TOLERANCE  = 0.10   # fraction of person height — feet within this are "on floor"

mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
ANKLE_IDS   = [27, 28]   # left ankle, right ankle
HEEL_IDS    = [29, 30]   # left heel, right heel
FOOT_IDS    = ANKLE_IDS + HEEL_IDS

def load_calibration():
    with open(CALIBRATION_FILE) as f:
        return json.load(f)

def open_capture(source):
    if not (isinstance(source, str) and source.startswith("rtsp")):
        return cv2.VideoCapture(source)
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY, None]:
        cap = cv2.VideoCapture(source, backend) if backend is not None else cv2.VideoCapture(source)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(source)

def check_feet(crop, pose, floor_y_in_crop, tolerance_px):
    """
    Run MediaPipe Pose on a person crop.
    Returns (feet_on_floor, foot_y_pixels_list, annotated_crop)
    """
    h, w = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    annotated = crop.copy()
    foot_ys = []

    if not result.pose_landmarks:
        return None, [], annotated   # pose not detected — inconclusive

    landmarks = result.pose_landmarks.landmark
    for idx in FOOT_IDS:
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            continue
        px = int(lm.x * w)
        py = int(lm.y * h)
        foot_ys.append(py)
        cv2.circle(annotated, (px, py), 5, (0, 255, 255), -1)

    if not foot_ys:
        return None, [], annotated   # keypoints not visible

    avg_foot_y = sum(foot_ys) / len(foot_ys)

    # Clamp floor_y_in_crop to within crop — for far-away people floor_y_global may be
    # below the crop bottom, which would wrongly flag everyone far back as "off floor"
    floor_y_in_crop = min(floor_y_in_crop, h - 1)

    feet_on_floor = avg_foot_y >= (floor_y_in_crop - tolerance_px)

    cv2.line(annotated, (0, int(floor_y_in_crop)),
             (w, int(floor_y_in_crop)), (255, 200, 0), 1)

    return feet_on_floor, foot_ys, annotated

def run(source):
    calib = load_calibration()
    floor_y      = calib["floor_y"]
    pixels_per_cm = calib["pixels_per_cm"]

    person_model = YOLO("yolov8n.pt")
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,        # fastest
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open source.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream {W}x{H} — press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.line(frame, (0, floor_y), (W, floor_y), (255, 200, 0), 1)

        persons = person_model(frame, verbose=False, conf=PERSON_CONF)[0]

        for i, box in enumerate(persons.boxes):
            if int(box.cls) != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            person_h = y2 - y1
            tolerance_px = person_h * FLOOR_TOLERANCE

            # floor_y relative to this crop's top
            floor_y_in_crop = floor_y - y1

            feet_on_floor, foot_ys, annotated_crop = check_feet(
                crop, pose, floor_y_in_crop, tolerance_px
            )
            frame[y1:y2, x1:x2] = annotated_crop

            if feet_on_floor is None:
                color, label = (0, 165, 255), "FEET: ?"
            elif feet_on_floor:
                color, label = (0, 200, 0),   "FEET: ON FLOOR"
            else:
                color, label = (0, 0, 255),   "FEET: OFF FLOOR"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"P{i+1}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Foot Check", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or RTSP URL")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
