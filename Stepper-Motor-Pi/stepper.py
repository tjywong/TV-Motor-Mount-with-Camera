import math
import threading
import time
from gpiozero import DigitalOutputDevice

# Drives the 17HS13 through the L298N by writing the coil pattern straight to IN1..IN4.
# Steps run in a background thread, so move_to() returns immediately and the caller
# can keep detecting while the motor turns. The motor speeds up and slows down
# smoothly instead of jumping, and a new target can be set at any time, even mid-move.

# Half-step sequence (IN1, IN2, IN3, IN4): IN1/IN2 drive coil A, IN3/IN4 drive coil B
HALF_STEPS = [
    (1, 0, 0, 0),  # A+
    (1, 0, 1, 0),  # A+ B+
    (0, 0, 1, 0),  #    B+
    (0, 1, 1, 0),  # A- B+
    (0, 1, 0, 0),  # A-
    (0, 1, 0, 1),  # A- B-
    (0, 0, 0, 1),  #    B-
    (1, 0, 0, 1),  # A+ B-
]

TICK = 0.0005  # seconds between speed updates; caps the top speed near 1 / TICK

class Stepper:
    def __init__(self, pins, max_speed=150, accel=300, limit=None, release_after=1.0):
        # pins: BCM GPIO numbers for IN1..IN4
        # max_speed: top speed in half steps per second
        # accel: how quickly it speeds up and slows down, in half steps per second per second
        # limit: furthest the motor may go either side of where it started, in half steps
        # release_after: switch the coils off after this many idle seconds so nothing overheats
        self.coils = [DigitalOutputDevice(p) for p in pins]
        self.max_speed = max_speed
        self.accel = accel
        self.limit = limit
        self.release_after = release_after
        self.position = 0  # half steps from the starting position
        self.target = 0
        self.speed = 0.0   # signed, half steps per second
        self._phase = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def move_to(self, target):
        # Head for an absolute position; replaces any move already in progress
        if self.limit is not None:
            target = max(-self.limit, min(self.limit, target))
        self.target = int(target)

    def move(self, steps):
        # Turn relative to the current position; positive steps turn one way, negative the other
        self.move_to(self.position + steps)

    def busy(self):
        return self.position != self.target or self.speed != 0

    def wait(self):
        while self.busy():
            time.sleep(0.005)

    def close(self):
        self._running = False
        self._thread.join()
        self._release()
        for coil in self.coils:
            coil.close()

    def _write_phase(self):
        for coil, on in zip(self.coils, HALF_STEPS[self._phase]):
            coil.value = on

    def _release(self):
        for coil in self.coils:
            coil.off()

    def _run(self):
        energized = False
        last_move = 0.0
        travelled = 0.0  # fraction of a step covered since the last step
        last = time.monotonic()
        while self._running:
            time.sleep(TICK)
            now = time.monotonic()
            dt = min(now - last, 0.01)
            last = now

            dist = self.target - self.position
            # Slow enough to stop on the spot: settle instead of creeping past the target
            if dist == 0 and abs(self.speed) <= 2 * math.sqrt(2 * self.accel):
                self.speed = 0.0
                travelled = 0.0
                if energized and now - last_move >= self.release_after:
                    self._release()
                    energized = False
                continue

            if not energized:
                # Re-apply the last coil state and let the rotor settle before stepping
                self._write_phase()
                energized = True
                time.sleep(0.01)
                last = time.monotonic()
                continue

            # Fastest speed from which it can still stop at the target, capped at max_speed
            stop_speed = math.sqrt(2 * self.accel * abs(dist))
            desired = math.copysign(min(self.max_speed, stop_speed), dist) if dist else 0.0
            change = self.accel * dt
            self.speed += max(-change, min(change, desired - self.speed))

            travelled += self.speed * dt
            if abs(travelled) >= 1:
                direction = 1 if travelled > 0 else -1
                travelled -= direction
                self._phase = (self._phase + direction) % 8
                self._write_phase()
                self.position += direction
                last_move = now
