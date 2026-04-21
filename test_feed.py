import cv2
import argparse
import sys

def open_capture(source):
    if isinstance(source, str) and source.startswith("rtsp"):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def test_feed(source):
    print(f"[INFO] Connecting to: {source}")
    cap = open_capture(source)

    if not cap.isOpened():
        print("[ERROR] Could not open video source. Check your RTSP URL or camera index.")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Stream opened — {width}x{height} @ {fps:.1f} FPS")
    print("[INFO] Press 'q' to quit.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame. Stream may have dropped.")
            break

        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("CCTV Feed Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Total frames read: {frame_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="0",
        help="RTSP URL (e.g. rtsp://192.168.1.10/stream) or 0 for webcam"
    )
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    test_feed(source)
