#include <FastLED.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

// ESP32 GPIO26 -> Uno D2 (RX)
// Uno D3 (TX) -> 1k/2k divider -> ESP32 GPIO25
SoftwareSerial EspLink(2, 3);

// ================= LCD =================

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ================= LED =================

#define LED_PIN 6
#define NUM_LEDS 18
#define BRIGHTNESS 100

CRGB leds[NUM_LEDS];

// ================= SERVO =================

#define SERVO_PIN 9

#define DIVIDER_LEFT   50
#define DIVIDER_CENTER 90
#define DIVIDER_RIGHT  130

#define EVENT_TIME 10000UL

int currentServoAngle = DIVIDER_CENTER;
String laneAllocation = "BALANCED_3_3";
bool laneBlinkActive = false;
bool laneBlinkOn = false;
CRGB laneBlinkColor = CRGB::Black;
unsigned long lastLaneBlink = 0;
const unsigned long LANE_BLINK_INTERVAL_MS = 80;

// ======================================================
// LCD
// ======================================================

void showLCD(const char* line1, const char* line2) {

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(line1);

  lcd.setCursor(0, 1);
  lcd.print(line2);

  Serial.println("----------------");
  Serial.println(line1);
  Serial.println(line2);
}

// ======================================================
// LED
// ======================================================

void lightsOff() {
  FastLED.clear();
  FastLED.show();
}

void setAll(CRGB color) {
  fill_solid(leds, NUM_LEDS, color);
  FastLED.show();
}

void startLaneBlink(CRGB color) {
  laneBlinkColor = color;
  laneBlinkActive = true;
  laneBlinkOn = false;
  lastLaneBlink = 0;
  lightsOff();
}

void stopLaneBlink(CRGB steadyColor) {
  laneBlinkActive = false;
  laneBlinkOn = false;
  setAll(steadyColor);
}

void updateLaneBlink() {
  if (!laneBlinkActive) return;
  unsigned long now = millis();
  if (now - lastLaneBlink < LANE_BLINK_INTERVAL_MS) return;
  lastLaneBlink = now;
  laneBlinkOn = !laneBlinkOn;
  setAll(laneBlinkOn ? laneBlinkColor : CRGB::Black);
}

// ======================================================
// SERVO WITHOUT Servo.h
// ======================================================

void servoPulse(int angle) {

  angle = constrain(angle, 0, 180);

  int pulseWidth =
    map(angle, 0, 180, 500, 2400);

  digitalWrite(SERVO_PIN, HIGH);
  delayMicroseconds(pulseWidth);

  digitalWrite(SERVO_PIN, LOW);
  delayMicroseconds(20000 - pulseWidth);
}

void holdServo(int angle, int pulses) {

  for (int i = 0; i < pulses; i++) {
    servoPulse(angle);
  }
}

void moveServoSlow(int startAngle, int endAngle) {

  if (startAngle < endAngle) {

    for (int angle = startAngle;
         angle <= endAngle;
         angle++) {

      servoPulse(angle);
      delay(40);
    }

  } else {

    for (int angle = startAngle;
         angle >= endAngle;
         angle--) {

      servoPulse(angle);
      delay(40);
    }
  }

  holdServo(endAngle, 30);
}

// ======================================================
// NORMAL
// ======================================================

void normalMode() {

  showLCD(
    "NORMAL TRAFFIC",
    "4 LANES OPEN"
  );

  setAll(CRGB::Green);

  holdServo(DIVIDER_CENTER, 30);
}

// ======================================================
// AMBULANCE
// BLUE BLUE FLASH
// ======================================================

void ambulanceMode() {

  showLCD(
    "AMBULANCE",
    "PRIORITY ACTIVE"
  );

  unsigned long startTime = millis();

  while (millis() - startTime < EVENT_TIME) {

    // BLUE FLASH 1
    setAll(CRGB::Blue);
    delay(120);

    lightsOff();
    delay(70);

    // BLUE FLASH 2
    setAll(CRGB::Blue);
    delay(120);

    lightsOff();
    delay(250);
  }

  showLCD(
    "AMBULANCE",
    "LANE READY"
  );

  setAll(CRGB::Blue);

  delay(3000);
}

// ======================================================
// POLICE
// RED RED FLASH
// ======================================================

void policeMode() {

  showLCD(
    "POLICE",
    "PRIORITY ACTIVE"
  );

  unsigned long startTime = millis();

  while (millis() - startTime < EVENT_TIME) {

    // RED FLASH 1
    setAll(CRGB::Red);
    delay(120);

    lightsOff();
    delay(70);

    // RED FLASH 2
    setAll(CRGB::Red);
    delay(120);

    lightsOff();
    delay(250);
  }

  showLCD(
    "POLICE",
    "LANE READY"
  );

  setAll(CRGB::Red);

  delay(3000);
}

// ======================================================
// ACCIDENT
// ======================================================

void accidentMode() {

  showLCD(
    "ACCIDENT",
    "ROAD BLOCKED"
  );

  unsigned long startTime = millis();

  while (millis() - startTime < EVENT_TIME) {

    setAll(CRGB::Red);
    delay(400);

    lightsOff();
    delay(250);
  }

  showLCD(
    "ROAD WARNING",
    "DRIVE SLOW"
  );

  setAll(CRGB::Red);

  delay(3000);
}

// ======================================================
// SPEED
// ======================================================

void speedMode() {

  showLCD(
    "SPEED WARNING",
    "REDUCE SPEED"
  );

  unsigned long startTime = millis();

  while (millis() - startTime < EVENT_TIME) {

    setAll(CRGB::Yellow);
    delay(450);

    lightsOff();
    delay(300);
  }

  setAll(CRGB::Yellow);

  delay(3000);
}

// ======================================================
// DIVIDER RIGHT
// Divider right = 3 lanes LEFT
// ======================================================

void dividerRightMode() {

  showLCD(
    "DIVIDER RIGHT",
    "MOVING >>>>"
  );

  setAll(CRGB::Yellow);

  delay(2500);

  moveServoSlow(
    DIVIDER_CENTER,
    DIVIDER_RIGHT
  );

  showLCD(
    "3 LANES LEFT",
    "<<<< USE LEFT"
  );

  // LED direction LEFT

  for (int repeat = 0; repeat < 4; repeat++) {

    for (int i = NUM_LEDS - 1; i >= 0; i--) {

      FastLED.clear();

      leds[i] = CRGB::Blue;

      if (i < NUM_LEDS - 1)
        leds[i + 1] = CRGB::Blue;

      if (i < NUM_LEDS - 2)
        leds[i + 2] = CRGB::White;

      FastLED.show();

      delay(120);
    }
  }

  setAll(CRGB::Blue);

  showLCD(
    "3 LANES LEFT",
    "<<<<<<"
  );

  delay(5000);
}

// ======================================================
// DIVIDER LEFT
// Divider left = 3 lanes RIGHT
// ======================================================

void dividerLeftMode() {

  showLCD(
    "DIVIDER LEFT",
    "<<<< MOVING"
  );

  setAll(CRGB::Yellow);

  delay(2500);

  moveServoSlow(
    DIVIDER_CENTER,
    DIVIDER_LEFT
  );

  showLCD(
    "3 LANES RIGHT",
    "USE RIGHT >>>>"
  );

  // LED direction RIGHT

  for (int repeat = 0; repeat < 4; repeat++) {

    for (int i = 0; i < NUM_LEDS; i++) {

      FastLED.clear();

      leds[i] = CRGB::Blue;

      if (i > 0)
        leds[i - 1] = CRGB::Blue;

      if (i > 1)
        leds[i - 2] = CRGB::White;

      FastLED.show();

      delay(120);
    }
  }

  setAll(CRGB::Blue);

  showLCD(
    "3 LANES RIGHT",
    ">>>>>>"
  );

  delay(5000);
}

// ======================================================
// CENTER
// ======================================================

void centerMode() {

  showLCD(
    "DIVIDER",
    "TO CENTER"
  );

  setAll(CRGB::Yellow);

  delay(2000);

  holdServo(DIVIDER_CENTER, 40);

  showLCD(
    "NORMAL TRAFFIC",
    "4 LANES OPEN"
  );

  setAll(CRGB::Green);

  delay(4000);
}

// ======================================================
// DASHBOARD REVERSIBLE-LANE CONTROL
// ======================================================

void sendLaneStatus(Stream& output, const char* prefix) {
  output.print(prefix);
  output.print(",LANE,");
  output.println(laneAllocation);
}

void applyLaneAllocation(int targetAngle, const String& newAllocation) {
  showLCD("LANE CHANGER", "MOVING - WAIT");
  setAll(CRGB::Yellow);

  moveServoSlow(currentServoAngle, targetAngle);
  currentServoAngle = targetAngle;
  laneAllocation = newAllocation;

  if (newAllocation == "FORWARD_4_OPPOSITE_2") {
    showLCD("FORWARD PRIORITY", "4 FWD / 2 OPP");
    startLaneBlink(CRGB::Yellow);
  } else if (newAllocation == "FORWARD_2_OPPOSITE_4") {
    showLCD("OPPOSITE PRIOR.", "2 FWD / 4 OPP");
    startLaneBlink(CRGB::Red);
  } else {
    showLCD("BALANCED TRAFFIC", "3 FWD / 3 OPP");
    stopLaneBlink(CRGB::Green);
  }
  sendLaneStatus(EspLink, "ACK");
}

void handleCommand(String command, bool fromEsp32) {
  command.trim();
  command.toUpperCase();
  if (command.length() == 0) return;

  Serial.print(fromEsp32 ? "ESP32: " : "USB: ");
  Serial.println(command);

  if (!command.startsWith("LANE_") && command != "GET_LANE_STATUS") {
    laneBlinkActive = false;
  }

  if (command == "LANE_BALANCED") {
    applyLaneAllocation(DIVIDER_CENTER, "BALANCED_3_3");
  } else if (command == "LANE_FORWARD_4") {
    applyLaneAllocation(DIVIDER_RIGHT, "FORWARD_4_OPPOSITE_2");
  } else if (command == "LANE_OPPOSITE_4") {
    applyLaneAllocation(DIVIDER_LEFT, "FORWARD_2_OPPOSITE_4");
  } else if (command == "GET_LANE_STATUS") {
    if (fromEsp32) sendLaneStatus(EspLink, "STATUS");
    else sendLaneStatus(Serial, "STATUS");
  } else if (command == "NORMAL") {
    normalMode();
    currentServoAngle = DIVIDER_CENTER;
    laneAllocation = "BALANCED_3_3";
  } else if (command == "AMBULANCE") {
    ambulanceMode();
  } else if (command == "POLICE") {
    policeMode();
  } else if (command == "ACCIDENT") {
    accidentMode();
  } else if (command == "SPEED") {
    speedMode();
  } else if (command == "RIGHT") {
    dividerRightMode();
    currentServoAngle = DIVIDER_RIGHT;
    laneAllocation = "FORWARD_4_OPPOSITE_2";
  } else if (command == "LEFT") {
    dividerLeftMode();
    currentServoAngle = DIVIDER_LEFT;
    laneAllocation = "FORWARD_2_OPPOSITE_4";
  } else if (command == "CENTER") {
    centerMode();
    currentServoAngle = DIVIDER_CENTER;
    laneAllocation = "BALANCED_3_3";
  } else {
    showLCD("INVALID COMMAND", "TRY AGAIN");
    if (fromEsp32) EspLink.println("ERROR,LANE,INVALID_COMMAND");
  }
}

// ======================================================
// SETUP
// ======================================================

void setup() {

  Serial.begin(9600);
  EspLink.begin(9600);

  // Servo
  pinMode(SERVO_PIN, OUTPUT);
  digitalWrite(SERVO_PIN, LOW);

  // LCD
  Wire.begin();

  lcd.init();
  lcd.backlight();

  // LED
  FastLED.addLeds<
    WS2811,
    LED_PIN,
    RBG
  >(leds, NUM_LEDS);

  FastLED.setBrightness(BRIGHTNESS);

  lightsOff();

  holdServo(DIVIDER_CENTER, 30);

  showLCD(
    "SMART TRAFFIC",
    "SYSTEM READY"
  );

  Serial.println();
  Serial.println("COMMANDS:");
  Serial.println("NORMAL");
  Serial.println("AMBULANCE");
  Serial.println("POLICE");
  Serial.println("ACCIDENT");
  Serial.println("SPEED");
  Serial.println("RIGHT");
  Serial.println("LEFT");
  Serial.println("CENTER");
  Serial.println("LANE_BALANCED");
  Serial.println("LANE_FORWARD_4");
  Serial.println("LANE_OPPOSITE_4");
  Serial.println("GET_LANE_STATUS");
}

// ======================================================
// COMMAND RECEIVER
// ======================================================

void loop() {
  updateLaneBlink();

  if (EspLink.available() > 0) {
    handleCommand(EspLink.readStringUntil('\n'), true);
  }

  // The original commands still work from the Uno USB Serial Monitor.
  if (Serial.available() > 0) {
    handleCommand(Serial.readStringUntil('\n'), false);
  }
}
