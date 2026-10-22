import os
from ultralytics import YOLO

# Run this on your computer (not the UNO Q) to make python/yolov8n-pose.onnx.
# Needs: pip install ultralytics onnx onnxslim

HERE = os.path.dirname(os.path.abspath(__file__))
PT_PATH = os.path.join(HERE, "..", "..", "Human-Detector", "yolov8n-pose.pt")

# imgsz must match IMGSZ in python/main.py
onnx_path = YOLO(PT_PATH).export(format="onnx", imgsz=320, opset=12, simplify=True)
os.replace(onnx_path, os.path.join(HERE, "python", "yolov8n-pose.onnx"))
print("[INFO] Saved python/yolov8n-pose.onnx")
