import math
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
import onnxruntime as ort
from stepper import Stepper

# Head tracker for a Raspberry Pi 3: finds the target head and turns the camera
# until that head is in the middle of the frame.
#
# Heads are found with the YuNet face detector, which is many times faster than a
# pose model on the Pi. When no face is visible (someone turned away), the YOLOv8
# pose model takes over for that frame and finds the head from the body instead.
#
# Detection runs nonstop in a background thread, even while the motor turns. Each
# frame is tagged with where the motor was when it came in, so the target is worked
# out as an absolute motor position that stays right however far the motor has moved
# since. The motor glides toward that target and is redirected smoothly as new
# frames arrive.

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(HERE, "face_detection_yunet_2023mar.onnx")
POSE_MODEL_PATH = os.path.join(HERE, "yolov8n-pose.onnx")  # made by export_model.py

FACE_CONF = 0.7
# Fall back to the pose model when no face is found. Slower, but follows people
# facing away from the camera. Set False to use faces only.
POSE_BACKUP = True

# COCO keypoint indices for the head: nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
KEYPOINT_CONF = 0.5
PERSON_CONF = 0.5
NMS_IOU = 0.45

# SINGLE tracks the closest person's head, MULTI tracks the average of all heads
SINGLE, MULTI = "single", "multi"
MODE = SINGLE

# Show the camera feed live in a window on the Pi's desktop. Keys: m switches mode, q quits.
# Skipped automatically when there's no desktop (for example, when run over SSH).
SHOW_WINDOW = True
# Serve the camera feed to a browser at http://<pi name>.local:WEB_PORT
# Frames are only encoded while a browser is watching.
WEB_VIEW = True
WEB_PORT = 8000
JPEG_QUALITY = 70
# Print detections per second every this many frames (0 to turn off)
REPORT_EVERY = 30

PINS = [17, 27, 22, 23]  # BCM GPIO for IN1, IN2, IN3, IN4
MAX_SPEED = 100          # top speed in half steps per second (150 = about 135 degrees/s)
ACCEL = 100              # lower for gentler starts and stops, higher to react faster
LIMIT = 100              # half steps either side of the starting position (90 degrees)

CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 320, 240  # raise to 640x480 if faces far away are missed (slower)
CAMERA_FPS = 15      # more frames than needed only wastes CPU on decoding
HFOV_DEG = 70        # horizontal field of view of the camera; measure yours
STEPS_PER_REV = 400  # 200 full steps, half-stepping
DEADBAND = 0.05      # ignore errors under 5% of the frame width so it doesn't hunt
GAIN = 0.9           # a little under 1 in case HFOV_DEG is off
DIRECTION = -1       # flip the sign if the camera turns away from the person

# Leave one core free so the motor and camera threads keep steady timing
THREADS = max(1, (os.cpu_count() or 4) - 1)

class Camera:
    # Reads the camera in a background thread so detection and the display always
    # get the newest frame, along with the motor position when that frame arrived.
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
        self._seq = 0
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
                self._seq += 1
                self._latest = (self._seq, frame, self.stepper.position)
                self._cond.notify_all()

    def read(self, after_seq=0):
        # Wait for a frame newer than after_seq and return (seq, frame, motor position),
        # or None if the camera stalls. Each caller tracks its own seq.
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after_seq, timeout=1.0)
            return self._latest if self._seq > after_seq else None

    def close(self):
        self._running = False
        self._thread.join()
        self.cap.release()

class State:
    # What the detection thread last found, shared with the display
    def __init__(self, mode):
        self.mode = mode
        self.heads = []      # (head x, head y, size) for every head found
        self.target = None   # (x, y) being tracked
        self.source = ""     # "face" or "pose": which model found it
        self.rate = 0.0      # detections per second
        self.running = True

def load_face_detector():
    return cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (FRAME_WIDTH, FRAME_HEIGHT),
                                     FACE_CONF, NMS_IOU, 50)

def load_pose_model():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(POSE_MODEL_PATH, opts, providers=["CPUExecutionProvider"])

def detect_faces(detector, frame):
    # Returns (face center x, face center y, face area) for each face
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    # Each row: box x, y, w, h, then 5 landmarks, then score
    return [(float(f[0] + f[2] / 2), float(f[1] + f[3] / 2), float(f[2] * f[3])) for f in faces]

def detect_pose_heads(session, frame):
    # Returns (head x, head y, body box area) for each person whose head is visible
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
    # Single mode: biggest face/body is usually the closest person. Multi mode: average of all heads.
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

def track(camera, stepper, face_detector, pose_session, state):
    # Detection loop: runs in its own thread so the display can update at camera speed
    seq = 0
    frames = 0
    started = time.monotonic()
    try:
        while state.running:
            latest = camera.read(seq)
            if latest is None:
                continue
            seq, frame, frame_position = latest

            heads, source = detect_faces(face_detector, frame), "face"
            if not heads and pose_session is not None:
                heads, source = detect_pose_heads(pose_session, frame), "pose"
            target = find_target(heads, state.mode)
            state.heads, state.target = heads, target
            state.source = source if target is not None else ""

            if target is not None:
                steps = steps_to_center(target[0], frame.shape[1])
                if steps != 0:
                    # Where the head was relative to where the motor pointed for this frame
                    stepper.move_to(frame_position + steps)

            frames += 1
            if REPORT_EVERY and frames % REPORT_EVERY == 0:
                now = time.monotonic()
                state.rate = REPORT_EVERY / (now - started)
                started = now
                print(f"[INFO] {state.rate:.1f} detections/s")
    finally:
        state.running = False

def annotate(frame, state):
    # Copy of the frame with the latest detection drawn on top
    frame = frame.copy()  # the detection thread may be using the same frame
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
    for x, y, _ in state.heads:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
    if state.target is not None:
        cv2.circle(frame, (int(state.target[0]), int(state.target[1])), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"{state.mode} | {state.source or 'none'} | {state.rate:.1f}/s",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    return frame

def toggle_mode(state):
    state.mode = MULTI if state.mode == SINGLE else SINGLE

def show(camera, state):
    # Live view at camera speed with the latest detection drawn on top
    cv2.namedWindow("Head Tracker", cv2.WINDOW_NORMAL)
    seq = 0
    while state.running:
        latest = camera.read(seq)
        if latest is None:
            continue
        seq, frame, _ = latest
        cv2.imshow("Head Tracker", annotate(frame, state))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            state.running = False
        if key == ord("m"):
            toggle_mode(state)

PAGE = b"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Head Tracker</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee; font-family: sans-serif; text-align: center; }
  img { width: 100%; max-width: 640px; border-radius: 6px; }
  button { margin-top: 12px; padding: 10px 20px; font-size: 16px; border-radius: 6px; border: 0; cursor: pointer; }
</style>
</head>
<body>
<img src="/stream" alt="Camera feed">
<br>
<button onclick="fetch('/mode', {method: 'POST'})">Switch mode</button>
</body>
</html>
"""

def start_web_view(camera, state):
    # Serves the annotated camera feed as an MJPEG stream; returns the server so it can be shut down
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = 0
                try:
                    while state.running:
                        latest = camera.read(seq)
                        if latest is None:
                            continue
                        seq, frame, _ = latest
                        _, jpg = cv2.imencode(".jpg", annotate(frame, state),
                                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser closed the page
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/mode":
                toggle_mode(state)
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass  # keep the console for the detection rate

    server = ThreadingHTTPServer(("", WEB_PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = socket.gethostname().removesuffix(".local")
    print(f"[INFO] Browser view at http://{host}.local:{WEB_PORT}")
    return server

def main():
    cv2.setNumThreads(THREADS)
    face_detector = load_face_detector()
    pose_session = load_pose_model() if POSE_BACKUP else None
    stepper = Stepper(PINS, max_speed=MAX_SPEED, accel=ACCEL, limit=LIMIT)
    camera = Camera(stepper)
    state = State(MODE)
    worker = threading.Thread(target=track, args=(camera, stepper, face_detector, pose_session, state),
                              daemon=True)
    worker.start()
    print(f"[INFO] Head tracker running in {state.mode} mode. Press Ctrl+C to stop.")
    server = start_web_view(camera, state) if WEB_VIEW else None

    # A window needs a desktop; over SSH there isn't one, so carry on without it
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if SHOW_WINDOW and not has_display:
        print("[INFO] No desktop found, so no window. Run from the Pi's desktop, "
              "or over SSH with DISPLAY=:0 in front of the command.")

    try:
        if SHOW_WINDOW and has_display:
            show(camera, state)
        else:
            while worker.is_alive():
                worker.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if server is not None:
            server.shutdown()
        worker.join()
        camera.close()
        stepper.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
