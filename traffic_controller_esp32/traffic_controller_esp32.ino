#include <FastLED.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ===== LANE-CHANGER UNO UART =====
#define LANE_RX_PIN 25
#define LANE_TX_PIN 26
#define LANE_BAUD 9600
HardwareSerial LaneSerial(2);
String laneAllocation = "UNKNOWN";
String laneInput = "";

// ===== SIGNAL-HEAD UNO UART =====
// The PC has one USB cable and it comes here. This is the other end of the
// chain: the seven-field signal line arrives on USB and is passed straight
// down to traffic_controller_arduino.ino, which owns the two ramp heads and
// their LCD. Only GPIO33 -> Uno D7 and a shared ground are needed; the Uno
// never answers on this link, it echoes to its own USB port instead.
//
// 9600 to match the Uno's SoftwareSerial, which is not reliable much above it.
#define SIGNAL_RX_PIN 32
#define SIGNAL_TX_PIN 33
#define SIGNAL_BAUD 9600
HardwareSerial SignalSerial(1);
unsigned long signalRelayed = 0;

// ===== LED STRIP SETTINGS =====
#define LED_PIN     5
#define NUM_LEDS    60
#define BRIGHTNESS  100
#define LED_TYPE    WS2815
#define COLOR_ORDER RBG
CRGB leds[NUM_LEDS];

// ===== LCD SETTINGS =====
LiquidCrystal_I2C lcd1(0x27, 16, 2);
LiquidCrystal_I2C lcd2(0x3F, 16, 2);
LiquidCrystal_I2C* lcd = nullptr;
bool lcdFound = false;
String ledState = "OFF";
String lcdState = "System Ready";
bool emergencyActive = false;
unsigned long heartbeat = 0;
CRGB animationColor = CRGB::Black;
bool animationActive = false;
bool blinkOn = false;
unsigned long lastAnimationUpdate = 0;
const unsigned long BLINK_INTERVAL_MS = 50;

String jsonValue(const String& json, const String& key) {
  String marker = "\"" + key + "\"";
  int keyStart = json.indexOf(marker);
  if (keyStart < 0) return "";
  int colon = json.indexOf(':', keyStart + marker.length());
  int firstQuote = json.indexOf('\"', colon + 1);
  int lastQuote = json.indexOf('\"', firstQuote + 1);
  if (colon < 0 || firstQuote < 0 || lastQuote < 0) return "";
  return json.substring(firstQuote + 1, lastQuote);
}

void setDisplay(const String& line1, const String& line2) {
  lcdState = line1 + " " + line2;
  lcdState.trim();
  if (!lcdFound) return;
  lcd->clear();
  lcd->setCursor(0, 0);
  lcd->print(line1.substring(0, 16));
  lcd->setCursor(0, 1);
  lcd->print(line2.substring(0, 16));
}

// The LCD says one thing: what the LED strip is doing. Driving it from the
// strip state in one place -- rather than from each command that happens to
// change the strip -- is what stops the two from ever disagreeing, which is
// how a display ends up announcing an ambulance that has already gone.
void showStripState() {
  if (ledState == "AMBULANCE" || ledState == "RED") {
    setDisplay("Ambulance", "passing");
  } else if (ledState == "POLICE" || ledState == "BLUE") {
    setDisplay("Police", "passing");
  } else {
    setDisplay("Safe travels", "");
  }
}

void setLed(const String& color) {
  animationActive = false;
  if (color == "RED") fill_solid(leds, NUM_LEDS, CRGB::Red);
  else if (color == "GREEN") fill_solid(leds, NUM_LEDS, CRGB::Green);
  else if (color == "YELLOW") fill_solid(leds, NUM_LEDS, CRGB::Yellow);
  else fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();
  ledState = color;
  showStripState();
}

void startLineAnimation(const String& mode, const CRGB& color) {
  ledState = mode;
  animationColor = color;
  blinkOn = false;
  animationActive = true;
  lastAnimationUpdate = 0;
  fill_solid(leds, NUM_LEDS, CRGB::Black);
  FastLED.show();
  showStripState();
}

void updateLineAnimation() {
  if (!animationActive) return;

  unsigned long now = millis();
  if (now - lastAnimationUpdate < BLINK_INTERVAL_MS) return;
  lastAnimationUpdate = now;

  blinkOn = !blinkOn;
  fill_solid(leds, NUM_LEDS, blinkOn ? animationColor : CRGB::Black);
  FastLED.show();
}

void sendAck(const String& command) {
  Serial.println("{\"type\":\"ack\",\"success\":true,\"command\":\"" + command + "\"}");
}

void sendError(const String& command, const String& message) {
  Serial.println("{\"type\":\"error\",\"success\":false,\"command\":\"" + command + "\",\"error\":\"" + message + "\"}");
}

void sendStatus() {
  heartbeat++;
  Serial.println(
    "{\"type\":\"status\",\"success\":true,\"firmware\":\"1.3-signal-relay\","
    "\"led\":\"" + ledState + "\",\"lcd\":\"" + lcdState +
    "\",\"emergency\":" + String(emergencyActive ? "true" : "false") +
    ",\"speed_bump\":\"UNAVAILABLE\",\"lane_allocation\":\"" +
    laneAllocation + "\",\"signal_relayed\":" + String(signalRelayed) +
    ",\"heartbeat\":" + String(heartbeat) + "}"
  );
}

// Whether a line off the USB port is the signal head's telemetry rather than a
// command for this board. The two are told apart by shape, which is enough
// because they have no shape in common: every command is either JSON or a
// single bare word, and the signal line is seven comma-separated fields.
bool isSignalLine(const String& line) {
  return line.length() > 0 && !line.startsWith("{") && line.indexOf(',') >= 0;
}

// Pass the line on to the signal Uno exactly as it arrived. Nothing is written
// back to the PC: the node sends these once a second and does not read a reply,
// so an acknowledgement here would only sit in the buffer waiting to be
// mistaken for the answer to the next real command.
void relaySignalLine(const String& line) {
  SignalSerial.println(line);
  signalRelayed++;
}

void readLaneMessages() {
  while (LaneSerial.available()) {
    char incoming = LaneSerial.read();
    if (incoming == '\n') {
      laneInput.trim();
      if (laneInput.startsWith("ACK,LANE,")) {
        laneAllocation = laneInput.substring(9);
      } else if (laneInput.startsWith("STATUS,LANE,")) {
        laneAllocation = laneInput.substring(12);
      }
      laneInput = "";
    } else if (incoming != '\r' && laneInput.length() < 80) {
      laneInput += incoming;
    }
  }
}

bool sendLaneCommand(const String& unoCommand, const String& expectedState) {
  while (LaneSerial.available()) LaneSerial.read();
  laneInput = "";
  LaneSerial.println(unoCommand);

  unsigned long deadline = millis() + 5000;
  while ((long)(deadline - millis()) > 0) {
    updateLineAnimation();
    while (LaneSerial.available()) {
      char incoming = LaneSerial.read();
      if (incoming == '\n') {
        laneInput.trim();
        if (laneInput == "ACK,LANE," + expectedState) {
          laneAllocation = expectedState;
          laneInput = "";
          return true;
        }
        laneInput = "";
      } else if (incoming != '\r' && laneInput.length() < 80) {
        laneInput += incoming;
      }
    }
  }
  return false;
}

void setup() {
  Serial.begin(115200);
  LaneSerial.begin(LANE_BAUD, SERIAL_8N1, LANE_RX_PIN, LANE_TX_PIN);
  SignalSerial.begin(SIGNAL_BAUD, SERIAL_8N1, SIGNAL_RX_PIN, SIGNAL_TX_PIN);

  // ---- LED strip setup ----
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
  setLed("OFF");

  // ---- LCD setup (auto-detect address) ----
  Wire.begin(22, 21); // SDA on 22, SCL on 21 -- confirmed working
  Wire.beginTransmission(0x27);
  bool found27 = (Wire.endTransmission() == 0);
  Wire.beginTransmission(0x3F);
  bool found3F = (Wire.endTransmission() == 0);

  if (found27) {
    lcd = &lcd1;
    lcdFound = true;
    Serial.println("LCD found at 0x27");
  } else if (found3F) {
    lcd = &lcd2;
    lcdFound = true;
    Serial.println("LCD found at 0x3F");
  } else {
    Serial.println("No LCD found - continuing without it");
  }

  if (lcdFound) {
    lcd->init();
    lcd->backlight();
    // The strip is off at boot, so this reads "Safe travels" -- the same one
    // of the three states it will show for the rest of the run.
    showStripState();
  }

  Serial.println("System ready. Waiting for commands...");
}

void loop() {
  updateLineAnimation();
  readLaneMessages();

  if (Serial.available()) {
    String request = Serial.readStringUntil('\n');
    request.trim();

    // Telemetry for the signal Uno, not a command for this board. Handled and
    // done with -- loop() runs again immediately, so returning here costs
    // nothing and keeps the command chain below reading as commands only.
    if (isSignalLine(request)) {
      relaySignalLine(request);
      return;
    }

    String command = jsonValue(request, "command");
    if (command.length() == 0 && !request.startsWith("{")) command = request;
    command.toUpperCase();

    if (command == "ACTIVATE_EMERGENCY" || command == "AMBULANCE_A") {
      emergencyActive = true;
      startLineAnimation("AMBULANCE", CRGB::Red);
      sendAck(command);
    }
    else if (command == "DEACTIVATE_EMERGENCY" || command == "NORMAL_A") {
      emergencyActive = false;
      setLed("OFF");
      sendAck(command);
    }
    else if (command == "SPEED_VIOLATION") {
      emergencyActive = false;
      startLineAnimation("SPEED", CRGB::Yellow);
      sendAck(command);
    }
    else if (command == "LED_COLOR") {
      String color = jsonValue(request, "color");
      color.toUpperCase();
      if (color == "RED" || color == "GREEN" || color == "YELLOW" || color == "OFF") {
        setLed(color);
        sendAck(command);
      } else {
        sendError(command, "Unsupported LED color");
      }
    }
    else if (command == "LED_MODE") {
      String mode = jsonValue(request, "mode");
      mode.toUpperCase();
      if (mode == "POLICE") {
        emergencyActive = true;
        startLineAnimation("POLICE", CRGB::Blue);
        sendAck(command);
      } else if (mode == "AMBULANCE") {
        emergencyActive = true;
        startLineAnimation("AMBULANCE", CRGB::Red);
        sendAck(command);
      } else if (mode == "SPEED") {
        emergencyActive = false;
        startLineAnimation("SPEED", CRGB::Yellow);
        sendAck(command);
      } else if (mode == "OFF") {
        emergencyActive = false;
        setLed("OFF");
        sendAck(command);
      } else {
        sendError(command, "Unsupported LED mode");
      }
    }
    else if (command == "LCD_MESSAGE") {
      String message = jsonValue(request, "message");
      setDisplay(message.substring(0, 16), message.substring(16, 32));
      sendAck(command);
    }
    else if (command == "GET_STATUS") {
      sendStatus();
    }
    else if (command == "SET_LANE_ALLOCATION") {
      String allocation = jsonValue(request, "allocation");
      allocation.toUpperCase();
      String unoCommand = "";
      String expectedState = "";
      if (allocation == "BALANCED") {
        unoCommand = "LANE_BALANCED";
        expectedState = "BALANCED_3_3";
      } else if (allocation == "FORWARD_4") {
        unoCommand = "LANE_FORWARD_4";
        expectedState = "FORWARD_4_OPPOSITE_2";
      } else if (allocation == "OPPOSITE_4") {
        unoCommand = "LANE_OPPOSITE_4";
        expectedState = "FORWARD_2_OPPOSITE_4";
      } else {
        sendError(command, "Invalid lane allocation");
      }

      if (unoCommand.length() > 0) {
        if (sendLaneCommand(unoCommand, expectedState)) {
          sendAck(command);
        } else {
          sendError(command, "Lane changer Uno did not acknowledge");
        }
      }
    }
    else if (command == "RAISE_SPEED_BUMP" || command == "LOWER_SPEED_BUMP") {
      sendError(command, "No speed-bump controller is connected");
    }
    else {
      sendError(command, "Unknown command");
    }
  }
}
