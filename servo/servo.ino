/*
  Lane divider - 2 servos moving the barrier between lane A and lane B.

  servo1 signal -> D9     servo2 signal -> D10
  both red -> external 5-6V +     both brown -> supply GND + Uno GND

  MIRROR = true  : servos at opposite ends of the barrier (facing each other)
  MIRROR = false : both servos on the same side
*/

#include <Servo.h>

const bool MIRROR   = true;
const int  LANE_A   = 60;    // barrier position for lane A
const int  LANE_B   = 120;   // barrier position for lane B
const int  SPEED_MS = 25;    // ms per degree - higher = slower
const int  HOLD_MS  = 3000;  // pause at each lane

const int  MIN_US = 1000;    // pulse at 0 deg
const int  MAX_US = 2000;    // pulse at 180 deg
const int  TRIM2  = 0;       // fine offset for servo 2, in microseconds

Servo s1, s2;

void writeBoth(int a) {
  int us1 = map(a, 0, 180, MIN_US, MAX_US);
  int us2 = MIRROR ? map(180 - a, 0, 180, MIN_US, MAX_US) : us1;
  s1.writeMicroseconds(constrain(us1, 500, 2500));
  s2.writeMicroseconds(constrain(us2 + TRIM2, 500, 2500));
}

// Move one degree at a time so the barrier never slams.
void moveTo(int from, int to) {
  int dir = (to > from) ? 1 : -1;
  for (int a = from; a != to; a += dir) {
    writeBoth(a);
    delay(SPEED_MS);
  }
  writeBoth(to);
}

void setup() {
  s1.attach(9,  500, 2500);
  s2.attach(10, 500, 2500);
  writeBoth(LANE_A);
  delay(1000);
}

void loop() {
  moveTo(LANE_A, LANE_B);
  delay(HOLD_MS);
  moveTo(LANE_B, LANE_A);
  delay(HOLD_MS);
}
