from ultralytics import YOLO
import shutil
import os

print("[INFO] Training PPE detection model on downloaded dataset...")
print("[INFO] 307 training images, 15 epochs — takes ~20-40 min on CPU.\n")

model = YOLO("yolov8n.pt")

results = model.train(
    data="ppe_model/data.yaml",
    epochs=15,
    imgsz=416,
    batch=8,
    name="ppe_train",
    project="ppe_model",
    exist_ok=True,
    verbose=False,
)

best_weights = "runs/detect/ppe_model/ppe_train/weights/best.pt"
dest = "ppe_model/ppe_best.pt"

if os.path.exists(best_weights):
    shutil.copy(best_weights, dest)
    print(f"\n[DONE] Model saved to: {dest}")
    print("Run: python ppe_detection.py --source 1")
else:
    print("[ERROR] Training completed but best.pt not found.")
