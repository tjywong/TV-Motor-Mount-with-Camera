import threading
import time
from gpiozero import DigitalOutputDevice

# Drives the 17HS13 through the L298N by writing the coil pattern straight to IN1..IN4.
# Steps run in a background thread, so move() returns immediately and the caller can
# keep detecting while the motor turns.

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

class Stepper:
    def __init__(self, pins, step_interval=0.002, limit=None, release_after=1.0):
        # pins: BCM GPIO numbers for IN1..IN4
        # step_interval: seconds between half steps
        # limit: furthest the motor may go either side of where it started, in half steps
        # release_after: switch the coils off after this many idle seconds so nothing overheats
        self.coils = [DigitalOutputDevice(p) for p in pins]
        self.step_interval = step_interval
        self.limit = limit
        self.release_after = release_after
        self.position = 0  # half steps from the starting position
        self.target = 0
        self._phase = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def move(self, steps):
        # Turn relative to the current position; positive steps turn one way, negative the other
        target = self.position + steps
        if self.limit is not None:
            target = max(-self.limit, min(self.limit, target))
        self.target = target

    def busy(self):
        return self.position != self.target

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
        while self._running:
            target = self.target
            if self.position != target:
                if not energized:
                    # Re-apply the last coil state first so the rotor doesn't jump
                    self._write_phase()
                    energized = True
                    time.sleep(self.step_interval)
                direction = 1 if target > self.position else -1
                self._phase = (self._phase + direction) % 8
                self._write_phase()
                self.position += direction
                last_move = time.monotonic()
                time.sleep(self.step_interval)
            else:
                if energized and time.monotonic() - last_move >= self.release_after:
                    self._release()
                    energized = False
                time.sleep(0.001)
