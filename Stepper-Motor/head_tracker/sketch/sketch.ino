// Head tracker, microcontroller side: turns the stepper when the Python side asks.
// Python calls move(steps) to turn relative to the current position, and busy()
// to check whether the last move has finished.
//
// Wiring is the same as motor_test: IN1..IN4 -> D8..D11.

#include <Arduino_RouterBridge.h>

const int PINS[4] = {8, 9, 10, 11};  // IN1, IN2, IN3, IN4

// Delay between half steps. Use the fastest speed that came back cleanly in motor_test.
const unsigned long STEP_INTERVAL_US = 2000;

// Soft limit either side of the power-on position: 100 half steps = 90 degrees.
// Keeps the camera cable from winding up.
const long LIMIT = 100;

// Switch the coils off after this long without moving so nothing overheats
const unsigned long RELEASE_AFTER_MS = 1000;

// Half-step sequence {IN1, IN2, IN3, IN4}: IN1/IN2 drive coil A, IN3/IN4 drive coil B
const byte HALF_STEPS[8][4] = {
  {1, 0, 0, 0},
  {1, 0, 1, 0},
  {0, 0, 1, 0},
  {0, 1, 1, 0},
  {0, 1, 0, 0},
  {0, 1, 0, 1},
  {0, 0, 0, 1},
  {1, 0, 0, 1},
};

// Bridge callbacks run in a separate thread, so they only touch these two values
volatile long position = 0;  // half steps from the power-on position
volatile long target = 0;

int phase = 0;
unsigned long lastStepUs = 0;
unsigned long lastMoveMs = 0;
bool energized = false;

void writePhase(int p) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(PINS[i], HALF_STEPS[p][i] ? HIGH : LOW);
  }
}

void release() {
  for (int i = 0; i < 4; i++) {
    digitalWrite(PINS[i], LOW);
  }
}

void move(int steps) {
  target = constrain(position + steps, -LIMIT, LIMIT);
}

bool busy() {
  return position != target;
}

void setup() {
  for (int i = 0; i < 4; i++) {
    pinMode(PINS[i], OUTPUT);
  }
  release();

  Bridge.begin();
  Bridge.provide("move", move);
  Bridge.provide("busy", busy);
}

void loop() {
  long goal = target;

  if (position != goal) {
    if (!energized) {
      // Re-apply the last coil state first so the rotor doesn't jump
      writePhase(phase);
      energized = true;
      lastStepUs = micros();
    }
    if (micros() - lastStepUs >= STEP_INTERVAL_US) {
      int dir = goal > position ? 1 : -1;
      phase = (phase + dir + 8) % 8;
      writePhase(phase);
      position += dir;
      lastStepUs = micros();
      lastMoveMs = millis();
    }
  } else if (energized && millis() - lastMoveMs >= RELEASE_AFTER_MS) {
    release();
    energized = false;
  }
}
