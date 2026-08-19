/*
  Simple two-servo test - both move together, same direction.
  Pins: servo1 -> D9, servo2 -> D10
  Power: both red -> external 5-6V +, both brown -> supply GND *and* Uno GND.

  If servo 2 lags or leads servo 1, edit TRIM2 below:
  positive pushes it one way, negative the other. ~10 = about 1 degree.
*/

#include <Servo.h>

const int MIN_US = 1000;   // pulse at 0 deg
const int MAX_US = 2000;   // pulse at 180 deg
const int TRIM2  = 0;      // fine offset for servo 2 only, in microseconds
const int STEP_MS = 50;    // ms per degree -> smaller = faster

Servo s1, s2;

void writeBoth(int angle) {
  int us = map(angle, 0, 180, MIN_US, MAX_US);
  s1.writeMicroseconds(us);
  s2.writeMicroseconds(constrain(us + TRIM2, 500, 2500));
}

void setup() {
  s1.attach(9,  500, 2500);
  s2.attach(10, 500, 2500);
  writeBoth(90);          // both to centre
  delay(1000);
}

void loop() {
  for (int a = 0; a <= 180; a++) { writeBoth(a); delay(STEP_MS); }
  delay(500);
  for (int a = 180; a >= 0; a--) { writeBoth(a); delay(STEP_MS); }
  delay(500);
}
