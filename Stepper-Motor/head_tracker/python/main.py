import math
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from arduino.app_utils import App, Bridge

# Head tracker, Linux side: finds the target head and asks the sketch to turn the
# camera until that head is in the middle of the frame.
#
# Runs the pose model through ONNX Runtime instead of Ultralytics/PyTorch: much
# lighter to install on the UNO Q and faster on its CPU. Make the .onnx file on
# your computer with export_model.py and put it next to this file.

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.onnx")
IMGSZ = 320  # must match the size used in export_model.py

# COCO keypoint indices for the head: nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
KEYPOINT_CONF = 0.5
PERSON_CONF = 0.5
NMS_IOU = 0.45

# SINGLE tracks the closest person's head, MULTI tracks the average of all heads
SINGLE, MULTI = "single", "multi"
MODE = SINGLE

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 320, 240  # capture small so there's less to resize
HFOV_DEG = 70        # horizontal field of view of the camera; measure yours
STEPS_PER_REV = 400  # 200 full steps, half-stepping
DEADBAND = 0.05      # ignore errors under 5% of the frame width so it doesn't hunt
GAIN = 0.6           # turn only part of the way each time so detection lag can't overshoot
DIRECTION = 1        # set to -1 if the camera turns away from the person

def load_model():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name

def detect_heads(session, input_name, frame):
    # Returns (head x, box area) for each person whose head is visible
    h, w = frame.shape[:2]
    scale = IMGSZ / max(h, w)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)))
    # Pad to a square at the bottom/right so model coordinates are just frame coordinates * scale
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)

    # Each row: box cx, cy, w, h, person confidence, then 17 keypoints of (x, y, confidence)
    preds = session.run(None, {input_name: blob})[0][0].T
    preds = preds[preds[:, 4] > PERSON_CONF]
    if len(preds) == 0:
        return []

    boxes = np.column_stack((preds[:, 0] - preds[:, 2] / 2, preds[:, 1] - preds[:, 3] / 2,
                             preds[:, 2], preds[:, 3]))
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), preds[:, 4].tolist(), PERSON_CONF, NMS_IOU)

    heads = []
    for i in np.array(keep).flatten():
        kpts = preds[i, 5:].reshape(17, 3)[HEAD_KEYPOINTS]
        visible = kpts[kpts[:, 2] > KEYPOINT_CONF]
        if len(visible):
            heads.append((float(visible[:, 0].mean()) / scale,
                          float(preds[i, 2] * preds[i, 3]) / scale ** 2))
    return heads

def target_x(heads):
    if not heads:
        return None
    # Single mode: largest box is usually the closest person. Multi mode: average of all heads.
    if MODE == SINGLE:
        return max(heads, key=lambda h: h[1])[0]
    return sum(h[0] for h in heads) / len(heads)

def steps_to_center(x, width):
    err = x - width / 2
    if abs(err) < DEADBAND * width:
        return 0
    focal_px = (width / 2) / math.tan(math.radians(HFOV_DEG / 2))
    angle = math.degrees(math.atan(err / focal_px))
    return DIRECTION * int(round(GAIN * angle * STEPS_PER_REV / 360))

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {CAMERA_INDEX} couldn't be opened.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    # Keep only the newest frame so each detection sees where the camera points now
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

session, input_name = load_model()
cap = open_camera()
print(f"[INFO] Head tracker running in {MODE} mode.")

def loop():
    # Don't start a new move until the last one is done; frames taken mid-turn are stale
    if Bridge.call("busy"):
        time.sleep(0.01)
        return

    cap.grab()  # drop the frame buffered while the motor was turning
    ret, frame = cap.read()
    if not ret:
        return

    x = target_x(detect_heads(session, input_name, frame))
    if x is None:
        return

    steps = steps_to_center(x, frame.shape[1])
    if steps != 0:
        Bridge.call("move", steps)

App.run(user_loop=loop)
