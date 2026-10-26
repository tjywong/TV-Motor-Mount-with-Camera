import time
from stepper import Stepper

# Standalone test for the 17HS13 stepper on an L298N, driven by a Raspberry Pi 3.
# Put a piece of tape on the shaft to see its position. Press Ctrl+C to stop.
#
# Each cycle:
#   1. Direction: quarter turn one way, pause, quarter turn back.
#   2. Speed: half a turn out and back at faster and faster speeds.
#      If the tape doesn't come back to the same spot, the motor skipped steps
#      at that speed, so that's too fast (or the supply voltage is too low).
#   3. Release: coils switched off for 5 s. Feel the motor and driver for heat.

PINS = [17, 27, 22, 23]  # BCM GPIO for IN1, IN2, IN3, IN4
STEPS_PER_REV = 400      # 200 full steps, half-stepping

# Delay between half steps, slowest first (0.005 s = 200 steps/s = half a turn per second)
SPEED_TEST_INTERVALS = [0.005, 0.003, 0.002, 0.0015, 0.001, 0.00075]

def turn(stepper, steps, interval):
    stepper.step_interval = interval
    stepper.move(steps)
    stepper.wait()

def main():
    stepper = Stepper(PINS)
    print("[INFO] Running motor test. Press Ctrl+C to stop.")
    try:
        while True:
            print("[INFO] Direction test")
            turn(stepper, STEPS_PER_REV // 4, 0.005)
            time.sleep(1)
            turn(stepper, -STEPS_PER_REV // 4, 0.005)
            time.sleep(2)

            for interval in SPEED_TEST_INTERVALS:
                print(f"[INFO] Speed test: {round(1 / interval)} steps/s")
                turn(stepper, STEPS_PER_REV // 2, interval)
                time.sleep(0.5)
                turn(stepper, -STEPS_PER_REV // 2, interval)
                time.sleep(1.5)

            print("[INFO] Coils released")
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        stepper.close()

if __name__ == "__main__":
    main()
