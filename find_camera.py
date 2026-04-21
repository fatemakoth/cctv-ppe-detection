import cv2

print("Scanning camera indices 0–5...")
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        has_frame = "got frame" if ret else "no frame"
        print(f"  Index {i}: {w}x{h} — {has_frame}")
        cap.release()
    else:
        print(f"  Index {i}: not available")
