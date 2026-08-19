/*
  Movable lane divider - two servos, one barrier, two lane positions.

  Wiring
    servo1 signal -> D9        servo2 signal -> D10
    button        -> D2 to GND (uses internal pull-up, no resistor needed)
    status LED    -> D13 (the on-board one works)
    servo power   -> external 5-6V +, grounds tied to Uno GND

  Press the button to shift the divider between lane A and lane B.

  MIRROR: set true when the two servos face each other at opposite ends of
  the barrier. Leave false when both sit on the same side.
*/

#include <Servo.h>

// ---------------------------------------------------------------- config ---
const int  PIN_S1   = 9;
const int  PIN_S2   = 10;
const int  PIN_BTN  = 2;
const int  PIN_LED  = 13;

const bool MIRROR   = true;   // servo 2 runs opposite - see note above

const int  LANE_A   = 60;     // divider parked at lane A (degrees)
const int  LANE_B   = 120;    // divider parked at lane B
const int  SPEED_MS = 25;     // ms per degree - higher = slower, gentler

const int  MIN_US = 1000;     // pulse at 0 deg   } same calibration
const int  MAX_US = 2000;     // pulse at 180 deg }  as the test sketch
const int  TRIM2  = 0;        // fine offset for servo 2, in microseconds

// ----------------------------------------------------------------- state ---
Servo s1, s2;

int  angle   = LANE_A;   // where the barrier is now
int  target  = LANE_A;   // where it is heading
bool atLaneB = false;

unsigned long lastStep   = 0;
unsigned long lastBtn    = 0;
bool          btnWasDown = false;

// -------------------------------------------------------------- movement ---
void writeBoth(int a) {
  int us1 = map(a, 0, 180, MIN_US, MAX_US);
  int us2 = MIRROR ? map(180 - a, 0, 180, MIN_US, MAX_US)
                   : us1;
  s1.writeMicroseconds(constrain(us1,         500, 2500));
  s2.writeMicroseconds(constrain(us2 + TRIM2, 500, 2500));
}

// One degree per tick - slow enough that the barrier never slams.
void stepToward() {
  if (angle == target) return;
  if (millis() - lastStep < (unsigned long)SPEED_MS) return;
  lastStep = millis();

  angle += (target > angle) ? 1 : -1;
  writeBoth(angle);
}

// ---------------------------------------------------------------- button ---
void checkButton() {
  bool down = (digitalRead(PIN_BTN) == LOW);

  if (down && !btnWasDown && millis() - lastBtn > 250) {  // debounce
    lastBtn = millis();
    atLaneB = !atLaneB;
    target  = atLaneB ? LANE_B : LANE_A;
    digitalWrite(PIN_LED, atLaneB);
  }
  btnWasDown = down;
}

// ------------------------------------------------------------------ main ---
void setup() {
  pinMode(PIN_BTN, INPUT_PULLUP);
  pinMode(PIN_LED, OUTPUT);

  s1.attach(PIN_S1, 500, 2500);
  s2.attach(PIN_S2, 500, 2500);

  writeBoth(angle);    // park at lane A
  delay(800);
}

void loop() {
  checkButton();
  stepToward();
}
