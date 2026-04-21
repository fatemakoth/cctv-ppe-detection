import cv2
import json
import argparse
from ultralytics import YOLO

PERSON_MODEL   = "yolov8s.pt"      # small model — better partial-body detection than nano
PPE_MODEL_PATH = "ppe_model/ppe_best.pt"

PERSON_CLASS   = 0

# Class names in trained PPE model
HAS_HELMET     = "Hardhat"
NO_HELMET      = "NO-Hardhat"
HAS_VEST       = "Safety Vest"
NO_VEST        = "NO-Safety Vest"

PPE_CONF         = 0.30   # lower threshold — PPE on a crop is harder to detect
PERSON_CONF      = 0.50
MIN_BODY_HEIGHT_CM = 120  # skip partial/truncated detections (same logic as height_detection)
MIN_ASPECT_RATIO   = 1.2  # person boxes are taller than wide

def open_capture(source):
    if isinstance(source, str) and source.startswith("rtsp"):
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)

def check_ppe(crop, ppe_model):
    results = ppe_model(crop, verbose=False, conf=PPE_CONF)[0]
    names = ppe_model.names

    detected = set()
    for box in results.boxes:
        label = names[int(box.cls)]
        detected.add(label)

    has_helmet = HAS_HELMET in detected
    no_helmet  = NO_HELMET  in detected
    has_vest   = HAS_VEST   in detected
    no_vest    = NO_VEST    in detected

    # Explicit detection beats absence
    helmet_ok = has_helmet and not no_helmet
    vest_ok   = has_vest   and not no_vest

    # If neither detected, mark as unverified
    helmet_status = "OK"     if helmet_ok  else ("MISSING" if no_helmet  else "?")
    vest_status   = "OK"     if vest_ok    else ("MISSING" if no_vest    else "?")

    return helmet_status, vest_status

def status_color(status):
    if status == "OK":      return (0, 200, 0)
    if status == "MISSING": return (0, 0, 255)
    return (0, 165, 255)   # orange for unverified "?"

def draw_ppe_status(frame, x1, y1, x2, y2, helmet, vest, person_id):
    helmet_color = status_color(helmet)
    vest_color   = status_color(vest)

    # GREEN  = all PPE confirmed OK
    # RED    = at least one item explicitly MISSING
    # ORANGE = no item missing but some unverified (partial view, unclear)
    if helmet == "MISSING" or vest == "MISSING":
        box_color = (0, 0, 255)
    elif helmet == "OK" and vest == "OK":
        box_color = (0, 200, 0)
    else:
        box_color = (0, 165, 255)  # unverified

    overall_ok = helmet == "OK" and vest == "OK"
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    lines = [
        (f"P{person_id}", box_color),
        (f"Helmet : {helmet}", helmet_color),
        (f"Vest   : {vest}",   vest_color),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x1, y1 - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if not overall_ok:
        cv2.putText(frame, "!! PPE VIOLATION !!",
                    (x1, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def load_calibration():
    try:
        with open("calibration.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def run(source):
    cal = load_calibration()
    if cal:
        print(f"[INFO] Calibration loaded: {cal['pixels_per_cm']:.4f} px/cm")
    else:
        print("[WARN] No calibration.json — using fallback 4.0 px/cm for person size filter")

    print("[INFO] Loading models...")
    person_model = YOLO(PERSON_MODEL)
    ppe_model    = YOLO(PPE_MODEL_PATH)
    print(f"[INFO] PPE classes: {list(ppe_model.names.values())}")
    print("[INFO] Press 'q' to quit.")

    cap = open_capture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Stream: {w}x{h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = person_model(frame, verbose=False, conf=PERSON_CONF)[0]
        violation_count = 0
        person_id = 0

        for box in persons.boxes:
            if int(box.cls) != PERSON_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_h = y2 - y1
            box_w = x2 - x1
            if box_w == 0 or box_h == 0:
                continue

            # Skip partial/truncated detections and non-person shapes
            pixels_per_cm = cal.get("pixels_per_cm", 4.0) if cal else 4.0
            body_cm = box_h / pixels_per_cm
            aspect  = box_h / box_w
            if body_cm < MIN_BODY_HEIGHT_CM or aspect < MIN_ASPECT_RATIO:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            person_id += 1
            helmet, vest = check_ppe(crop, ppe_model)
            draw_ppe_status(frame, x1, y1, x2, y2, helmet, vest, person_id)

            if helmet == "MISSING" or vest == "MISSING":
                violation_count += 1

        if violation_count > 0:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(frame, f"PPE VIOLATION — {violation_count} person(s) missing PPE",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"People: {person_id}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("PPE Detection", frame)
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
