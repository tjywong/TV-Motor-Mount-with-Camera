// Standalone test for the 17HS13 stepper on an L298N, driven by the Arduino UNO Q.
// No libraries: it writes the coil pattern directly so it also verifies the wiring.
//
// Wiring: IN1..IN4 -> D8..D11, OUT1/OUT2 -> coil A, OUT3/OUT4 -> coil B,
// L298N GND tied to UNO Q GND. Put a piece of tape on the shaft to see its position.
//
// Each cycle:
//   1. Direction: quarter turn one way, pause, quarter turn back.
//   2. Speed: half a turn out and back at faster and faster speeds.
//      If the tape doesn't come back to the same spot, the motor skipped steps
//      at that speed, so that's too fast (or the supply voltage is too low).
//   3. Release: coils switched off for 5 s. Feel the motor and driver for heat.

const int PINS[4] = {8, 9, 10, 11};  // IN1, IN2, IN3, IN4

const int STEPS_PER_REV = 400;  // 200 full steps, half-stepping

// Half-step sequence {IN1, IN2, IN3, IN4}: IN1/IN2 drive coil A, IN3/IN4 drive coil B
const byte HALF_STEPS[8][4] = {
  {1, 0, 0, 0},  // A+
  {1, 0, 1, 0},  // A+ B+
  {0, 0, 1, 0},  //    B+
  {0, 1, 1, 0},  // A- B+
  {0, 1, 0, 0},  // A-
  {0, 1, 0, 1},  // A- B-
  {0, 0, 0, 1},  //    B-
  {1, 0, 0, 1},  // A+ B-
};

// Delay between half steps, slowest first (5000 us = 200 steps/s = half a turn per second)
const unsigned int SPEED_TEST_DELAYS_US[] = {5000, 3000, 2000, 1500, 1000, 750};

int phase = 0;  // current index into HALF_STEPS

void writePhase(int p) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(PINS[i], HALF_STEPS[p][i] ? HIGH : LOW);
  }
}

// Positive steps turn one way, negative the other
void step(long steps, unsigned int delayUs) {
  int dir = steps > 0 ? 1 : -1;
  for (long i = 0; i < abs(steps); i++) {
    phase = (phase + dir + 8) % 8;
    writePhase(phase);
    delayMicroseconds(delayUs);
  }
}

// Switch both coils off so nothing heats up while idle
void release() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(PINS[i], LOW);
  }
}

void setup() {
  for (int i = 0; i < 4; i++) {
    pinMode(PINS[i], OUTPUT);
  }
  release();
  delay(2000);
}

void loop() {
  // 1. Direction
  writePhase(phase);
  delay(500);
  step(STEPS_PER_REV / 4, 5000);
  delay(1000);
  step(-STEPS_PER_REV / 4, 5000);
  delay(2000);

  // 2. Speed
  for (unsigned int delayUs : SPEED_TEST_DELAYS_US) {
    step(STEPS_PER_REV / 2, delayUs);
    delay(500);
    step(-STEPS_PER_REV / 2, delayUs);
    delay(1500);
  }

  // 3. Release
  release();
  delay(5000);
}
