import os
from ultralytics import YOLO

# Run this on your computer (not the Pi) to make yolov8n-pose.onnx.
# Needs: pip install ultralytics onnx onnxslim

HERE = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(HERE, "..", "Human-Detector", "yolov8n-pose.pt")

# 256 high x 320 wide matches the camera's 320x240 frames with almost no padding,
# so it's faster than a 320x320 square at the same detail
onnx_path = YOLO(PT_PATH).export(format="onnx", imgsz=(256, 320), opset=12, simplify=True)
os.replace(onnx_path, os.path.join(HERE, "yolov8n-pose.onnx"))
print("[INFO] Saved yolov8n-pose.onnx")
