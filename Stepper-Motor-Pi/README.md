# Stepper Motor Head Tracker (Raspberry Pi 3)

Turns a camera mounted on a stepper so the person's head stays in the center of
the frame. In single mode it follows the closest person's head; in multi mode it
follows the average of everyone's heads.

The Pi does everything: it reads the camera, finds heads, and drives the stepper
through the L298N from its own GPIO pins. Heads are found with the fast YuNet
face detector. When nobody's face is visible (for example, someone has turned
away), the slower YOLOv8n-pose model takes over and finds the head from the body.

| File | What it does |
|---|---|
| `motor_test.py` | Turns the motor through direction and speed tests. Run this first |
| `head_tracker.py` | The tracker |
| `stepper.py` | Drives the motor in a background thread with smooth speed-up and slow-down (used by both scripts) |
| `face_detection_yunet_2023mar.onnx` | The YuNet face detection model, from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT license) |
| `yolov8n-pose.onnx` | The backup pose model, exported at 256×320 to match the camera's 4:3 frames |
| `export_model.py` | Remakes the `.onnx` model; run it on your computer, not the Pi |

## Parts

- Raspberry Pi 3 with its own 5 V 2.5 A power supply
- StepperOnline 17HS13 stepper motor
- L298N dual H-bridge module
- 6 V power supply for the motor
- USB webcam

## Wiring

Pin numbers below are the Pi's **physical header pins** (pin 1 is the corner pin
nearest the SD card slot), with the GPIO number in brackets. The code uses the
GPIO numbers.

### Pi to L298N

| L298N | Raspberry Pi 3 |
|---|---|
| IN1 | Pin 11 (GPIO17) |
| IN2 | Pin 13 (GPIO27) |
| IN3 | Pin 15 (GPIO22) |
| IN4 | Pin 16 (GPIO23) |
| 5V terminal | Pin 2 (5V) |
| GND terminal | Pin 6 (GND) |

### Power to L298N

| L298N | Connect to |
|---|---|
| "12V" terminal | 6 V supply + |
| GND terminal | 6 V supply −, on the same terminal as the Pi's GND wire |

**Remove the 5V-EN jumper** on the L298N before powering anything. With a 6 V
supply its onboard regulator can't make a clean 5 V, and with the jumper on it
would push power into the Pi's 5V pin. **Leave the ENA and ENB jumpers on**;
without them the motor outputs stay off.

Never connect the 6 V motor supply to any Pi pin. The Pi's GPIO pins are 3.3 V,
which the L298N reads as high, so no level shifter is needed.

### L298N to motor

| L298N | 17HS13 wire (StepperOnline colors) |
|---|---|
| OUT1 | Black (A+) |
| OUT2 | Green (A−) |
| OUT3 | Red (B+) |
| OUT4 | Blue (B−) |

Check the coil pairs with a multimeter: the two wires of one coil show resistance
between them, wires from different coils show none.

### Camera

Plug the USB webcam into any of the Pi's USB ports.

## Setup

Use **Raspberry Pi OS 64-bit**. ONNX Runtime doesn't install on the 32-bit version.

Copy this folder to the Pi (or clone the repo there), then in this folder:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`--system-site-packages` lets the virtual environment use `gpiozero`, which comes
with Raspberry Pi OS.

## Running

Test the motor first:

```bash
python3 motor_test.py
```

Put a piece of tape on the shaft. Note the fastest speed where the tape still
comes back to the same spot; `MAX_SPEED` in `head_tracker.py` must stay below
it. Press Ctrl+C to stop.

Then start the tracker:

```bash
python3 head_tracker.py
```

Every 30 frames it prints how many detections per second it's managing.

To watch the camera, open the browser address it prints (for example
`http://pi.local:8000`) on any computer on the same network. The window on the
Pi's own screen only opens when the script is started from the Pi's desktop.
Started over SSH, it opens there if you add `DISPLAY=:0` in front:

```bash
DISPLAY=:0 python3 head_tracker.py
```

Settings at the top of `head_tracker.py`:

| Setting | What it does |
|---|---|
| `MODE` | `SINGLE` or `MULTI` |
| `SHOW_WINDOW` | Shows the live camera feed in a window on the Pi's desktop, with every head found (yellow), the target (red), the center line (green), and which model found it. `m` switches mode and `q` quits. Skipped automatically when there's no desktop |
| `WEB_VIEW` | Serves the same live view to a browser at `http://<pi name>.local:8000` (the address is printed at startup), with a button to switch mode |
| `WEB_PORT`, `JPEG_QUALITY` | The browser view's port and image quality. Lower quality uses a little less CPU and network |
| `POSE_BACKUP` | `False` uses faces only: faster when nobody faces the camera, but it won't follow someone turned away |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | Raise to 640×480 if faces far from the camera are missed (slower) |
| `MAX_SPEED` | Top turning speed in half steps per second. Lower for a slower, calmer turn |
| `ACCEL` | How gently it starts and stops. Lower is smoother, higher reacts faster |
| `DIRECTION` | Flip between `1` and `-1` if the camera turns away from the person |
| `HFOV_DEG` | Your camera's horizontal field of view; a wrong value makes moves too short or too long |
| `LIMIT` | How far the camera may turn either side of where it started, so the cable doesn't wind up. 100 = 90° |
| `GAIN`, `DEADBAND` | Lower `GAIN` if it overshoots; raise `DEADBAND` if it twitches when the person is near the center |
| `REPORT_EVERY` | How often to print the detection rate; `0` turns it off |

The camera's starting position is wherever it points when the script starts, so
point it straight ahead before running.
