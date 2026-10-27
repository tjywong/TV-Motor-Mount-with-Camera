import time
from stepper import Stepper

# Standalone test for the 17HS13 stepper on an L298N, driven by a Raspberry Pi 3.
# Put a piece of tape on the shaft to see its position. Press Ctrl+C to stop.
#
# Each cycle:
#   1. Direction: quarter turn one way, pause, quarter turn back.
#   2. Speed: half a turn out and back at higher and higher top speeds.
#      If the tape doesn't come back to the same spot, the motor skipped steps
#      at that speed, so that's too fast (or the supply voltage is too low).
#   3. Release: coils switched off for 5 s. Feel the motor and driver for heat.

PINS = [17, 27, 22, 23]  # BCM GPIO for IN1, IN2, IN3, IN4
STEPS_PER_REV = 400      # 200 full steps, half-stepping
ACCEL = 1000             # half steps per second per second

# Top speeds to try, in half steps per second (200 = half a turn per second)
SPEED_TEST_SPEEDS = [100, 200, 300, 400, 600, 800]

def turn(stepper, steps, speed):
    stepper.max_speed = speed
    stepper.move(steps)
    stepper.wait()

def main():
    stepper = Stepper(PINS, accel=ACCEL)
    print("[INFO] Running motor test. Press Ctrl+C to stop.")
    try:
        while True:
            print("[INFO] Direction test")
            turn(stepper, STEPS_PER_REV // 4, 200)
            time.sleep(1)
            turn(stepper, -STEPS_PER_REV // 4, 200)
            time.sleep(2)

            for speed in SPEED_TEST_SPEEDS:
                print(f"[INFO] Speed test: {speed} steps/s")
                turn(stepper, STEPS_PER_REV // 2, speed)
                time.sleep(0.5)
                turn(stepper, -STEPS_PER_REV // 2, speed)
                time.sleep(1.5)

            print("[INFO] Coils released")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        stepper.close()

if __name__ == "__main__":
    main()
