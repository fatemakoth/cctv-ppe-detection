import cv2
import json
import argparse
import time

CALIBRATION_FILE = "calibration.json"

clicks = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        print(f"  Point {len(clicks)} recorded: ({x}, {y})")

def capture_frame(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return None

    cv2.namedWindow("Calibration")

    # Warm up: show live feed until user presses 's' to start countdown
    print("[INFO] Press 's' to start the 5-second countdown, then step back and stand tall.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preview = frame.copy()
        cv2.putText(preview, "Press 's' to start 5s countdown",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Calibration", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            return None

    # Countdown: 5 → 1, then snap
    countdown_start = time.time()
    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        elapsed = time.time() - countdown_start
        if elapsed >= 5:
            break
        remaining = 5 - int(elapsed)
        preview = frame.copy()
        cv2.putText(preview, f"Stand still... {remaining}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 80, 255), 3)
        cv2.imshow("Calibration", preview)
        cv2.waitKey(1)

    cap.release()
    return frame

def run(source, height_cm):
    print("[INFO] Capturing frame from camera...")
    frame = capture_frame(source)
    if frame is None:
        return

    clone = frame.copy()
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    print("\n[INSTRUCTIONS]")
    print("  Stand in front of the camera (or place a person/object of known height).")
    print("  Click 1: top of your HEAD")
    print("  Click 2: bottom of your FEET (floor level)")
    print("  Press 'r' to reset clicks, 'q' to quit without saving.\n")

    while True:
        display = clone.copy()

        for i, (x, y) in enumerate(clicks):
            color = (0, 255, 255) if i == 0 else (0, 128, 255)
            label = "HEAD" if i == 0 else "FEET"
            cv2.circle(display, (x, y), 6, color, -1)
            cv2.putText(display, label, (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if len(clicks) == 2:
            cv2.line(display, clicks[0], clicks[1], (255, 255, 0), 2)
            pixel_height = abs(clicks[1][1] - clicks[0][1])
            pixels_per_cm = pixel_height / height_cm
            floor_y = max(clicks[0][1], clicks[1][1])

            cv2.putText(display,
                        f"Pixel height: {pixel_height}px | {pixels_per_cm:.2f} px/cm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press 's' to save, 'r' to redo",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            clicks.clear()
            print("[INFO] Clicks reset.")

        elif key == ord('s') and len(clicks) == 2:
            pixel_height = abs(clicks[1][1] - clicks[0][1])
            pixels_per_cm = pixel_height / height_cm
            floor_y = max(clicks[0][1], clicks[1][1])

            data = {
                "pixels_per_cm": round(pixels_per_cm, 4),
                "floor_y": floor_y,
                "reference_height_cm": height_cm,
                "reference_pixel_height": pixel_height
            }
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\n[SAVED] {CALIBRATION_FILE}")
            print(f"  pixels_per_cm : {pixels_per_cm:.4f}")
            print(f"  floor_y       : {floor_y} px")
            print(f"  reference     : {height_cm} cm = {pixel_height} px")
            break

        elif key == ord('q'):
            print("[INFO] Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="1", help="Camera index or RTSP URL")
    parser.add_argument("--height", type=float, required=True,
                        help="Your real height in cm (e.g. 170)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    run(source, args.height)
