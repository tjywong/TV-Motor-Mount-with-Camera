import math
import os
import threading
import time
import cv2
import numpy as np
import onnxruntime as ort
from stepper import Stepper

# Head tracker for a Raspberry Pi 3: finds the target head and turns the camera
# until that head is in the middle of the frame.
#
# Detection runs nonstop, even while the motor turns. Each frame is tagged with
# where the motor was when it came in, so the target is worked out as an absolute
# motor position that stays right however far the motor has moved since. The
# motor glides toward that target and is redirected smoothly as new frames arrive.
#
# Runs the pose model through ONNX Runtime instead of Ultralytics/PyTorch, which
# is much lighter on the Pi. Make the .onnx file with export_model.py.

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n-pose.onnx")

# COCO keypoint indices for the head: nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
KEYPOINT_CONF = 0.5
PERSON_CONF = 0.5
NMS_IOU = 0.45

# SINGLE tracks the closest person's head, MULTI tracks the average of all heads
SINGLE, MULTI = "single", "multi"
MODE = SINGLE

# Set True to show the camera feed (needs a desktop). Keys: m switches mode, q quits.
SHOW_WINDOW = False
# Print detections per second every this many frames (0 to turn off)
REPORT_EVERY = 30

PINS = [17, 27, 22, 23]  # BCM GPIO for IN1, IN2, IN3, IN4
MAX_SPEED = 150          # top speed in half steps per second (150 = about 135 degrees/s)
ACCEL = 300              # lower for gentler starts and stops, higher to react faster
LIMIT = 100              # half steps either side of the starting position (90 degrees)

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 320, 240  # capture small so there's less to resize
CAMERA_FPS = 15      # more frames than detection can use only wastes CPU on decoding
HFOV_DEG = 70        # horizontal field of view of the camera; measure yours
STEPS_PER_REV = 400  # 200 full steps, half-stepping
DEADBAND = 0.05      # ignore errors under 5% of the frame width so it doesn't hunt
GAIN = 0.9           # a little under 1 in case HFOV_DEG is off
DIRECTION = -1       # flip the sign if the camera turns away from the person

class Camera:
    # Reads the camera in a background thread so detection always gets the newest
    # frame, along with the motor position when that frame arrived.
    def __init__(self, stepper):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {CAMERA_INDEX} couldn't be opened.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stepper = stepper
        self._cond = threading.Condition()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._cond:
                self._latest = (frame, self.stepper.position)
                self._cond.notify()

    def read(self):
        # Wait for a frame newer than the last one returned; None if the camera stalls
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout=1.0)
            latest, self._latest = self._latest, None
        return latest

    def close(self):
        self._running = False
        self._thread.join()
        self.cap.release()

def load_model():
    opts = ort.SessionOptions()
    # Leave one core free so the motor and camera threads keep steady timing
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return session

def detect_heads(session, frame):
    # Returns (head x, head y, box area) for each person whose head is visible
    model_input = session.get_inputs()[0]
    in_h, in_w = model_input.shape[2:]
    h, w = frame.shape[:2]
    scale = min(in_w / w, in_h / h)
    if scale != 1:
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
    # Pad at the bottom/right so model coordinates are just frame coordinates * scale
    canvas = np.full((in_h, in_w, 3), 114, np.uint8)
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)

    # Each row: box cx, cy, w, h, person confidence, then 17 keypoints of (x, y, confidence)
    preds = session.run(None, {model_input.name: blob})[0][0].T
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
                          float(visible[:, 1].mean()) / scale,
                          float(preds[i, 2] * preds[i, 3]) / scale ** 2))
    return heads

def find_target(heads, mode):
    if not heads:
        return None
    # Single mode: largest box is usually the closest person. Multi mode: average of all heads.
    if mode == SINGLE:
        x, y, _ = max(heads, key=lambda h: h[2])
        return x, y
    return (sum(h[0] for h in heads) / len(heads),
            sum(h[1] for h in heads) / len(heads))

def steps_to_center(x, width):
    err = x - width / 2
    if abs(err) < DEADBAND * width:
        return 0
    focal_px = (width / 2) / math.tan(math.radians(HFOV_DEG / 2))
    angle = math.degrees(math.atan(err / focal_px))
    return DIRECTION * int(round(GAIN * angle * STEPS_PER_REV / 360))

def main():
    session = load_model()
    stepper = Stepper(PINS, max_speed=MAX_SPEED, accel=ACCEL, limit=LIMIT)
    camera = Camera(stepper)
    mode = MODE
    print(f"[INFO] Head tracker running in {mode} mode. Press Ctrl+C to stop.")

    frames = 0
    started = time.monotonic()
    try:
        while True:
            latest = camera.read()
            if latest is None:
                continue
            frame, frame_position = latest

            target = find_target(detect_heads(session, frame), mode)
            if target is not None:
                steps = steps_to_center(target[0], frame.shape[1])
                if steps != 0:
                    # Where the head was relative to where the motor pointed for this frame
                    stepper.move_to(frame_position + steps)

            frames += 1
            if REPORT_EVERY and frames % REPORT_EVERY == 0:
                now = time.monotonic()
                print(f"[INFO] {REPORT_EVERY / (now - started):.1f} detections/s")
                started = now

            if SHOW_WINDOW:
                if target is not None:
                    cv2.circle(frame, (int(target[0]), int(target[1])), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Mode: {mode}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow("Head Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    mode = MULTI if mode == SINGLE else SINGLE
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        stepper.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
