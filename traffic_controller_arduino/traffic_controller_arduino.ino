/*
 * traffic_controller.ino
 * ======================
 * Arduino Uno — Smart Traffic System Hardware Controller
 *
 * Receives one line per second from broadcast_server.py over USB Serial and
 * drives a demand-responsive traffic signal:
 *
 *   main road HEAVY        ->  RED     (close the merge, do not add to the jam)
 *   main road MODERATE/FREE->  GREEN   (there is room, let the side road in)
 *   every change between them          ->  YELLOW for 5 s
 *
 * The PC decides the phase; this sketch only displays it. That is deliberate —
 * the timing rules live in exactly one place (traffic_light.py) instead of
 * being duplicated in two languages that could drift apart.
 *
 * An approaching emergency vehicle does NOT take over the signal, because
 * emergency traffic proceeds through any indication. It lights a dedicated
 * LED strip instead, telling drivers to leave space, while the signal carries
 * on serving normal traffic.
 *
 * Two ways in, one format:
 *   The line may arrive on USB (the Uno plugged straight into the PC) or on
 *   D7 from the ESP32, which is the hub the PC normally talks to: one USB
 *   cable goes to the ESP32 and it passes this line down untouched. Both are
 *   read every loop, so either works with no change here and neither has to be
 *   chosen at boot. Keeping USB live means the Serial Monitor and sketch
 *   uploads still work with the ESP32 wire attached.
 *
 * Serial protocol (from PC or ESP32, 9600 baud, newline-terminated, 7 fields):
 *   PHASE,LEVEL,CI,VEHICLES,EMG,LINE1,LINE2
 *
 *   PHASE     GREEN | YELLOW | RED   — the lamp to light (authoritative)
 *   LEVEL     FREE | MODERATE | HEAVY — measured congestion, for the LCD
 *   CI        congestion index, 0.000–1.000
 *   VEHICLES  vehicles currently in frame
 *   EMG       1 while the emergency strip should be lit, else 0
 *   LINE1/2   16-character LCD rows, already padded by the PC
 *
 *   e.g.  GREEN,HEAVY,0.740,11,0,LIGHT: GREEN,HEAVY CI:0.74
 *   e.g.  YELLOW,FREE,0.120,0,0,YELLOW 3s>RED,FREE CI:0.12
 *   e.g.  RED,FREE,0.050,1,1,LIGHT: RED,!! AMBULANCE !!
 *
 * Two signal heads — one per merging side road:
 *   The layout is two main carriageways running opposite ways (north→south and
 *   south→north), each fed by a side road that merges into it. The signals are
 *   on the SIDE ROADS, not on the carriageways: they hold merging traffic back
 *   while the main road is heavy and let it in once there is room. The main
 *   roads themselves run free and are never signalled.
 *
 *   Both heads show the same phase, because congestion is measured as one
 *   reading for the whole scene. They are free to be green together or red
 *   together — nothing here crosses, so there is no conflicting movement to
 *   protect against. Wiring the second head is optional; leave those three
 *   pins empty and the sketch behaves exactly as it did with one.
 *
 * Wiring:
 *   LCD SDA  → A4
 *   LCD SCL  → A5
 *   LCD VCC  → 5V,  LCD GND → GND
 *
 *   RAMP 1 head (N→S side road)        RAMP 2 head (S→N side road)
 *   Green LED  → 330Ω → pin 9  → GND   Green LED  → 330Ω → pin 2 → GND
 *   Yellow LED → 330Ω → pin 10 → GND   Yellow LED → 330Ω → pin 3 → GND
 *   Red LED    → 330Ω → pin 11 → GND   Red LED    → 330Ω → pin 4 → GND
 *
 *   Blue LED        → 330Ω → pin 8  → GND   (emergency indicator)
 *   Emergency strip → pin 6 → strip driver   (dedicated output, see below)
 *
 *   Every LED's short leg (cathode) goes to GND through its resistor; the long
 *   leg goes to the pin. Pins 0 and 1 are the USB serial link — using them for
 *   LEDs breaks the connection to the PC.
 *
 *   ESP32 link (optional, and how the PC reaches this board by default):
 *   ESP32 GPIO33 → pin 7  (this board's RX)
 *   ESP32 GND    → GND    (required — without a shared ground there is no
 *                          reference for the signal and nothing arrives)
 *   Pin 12 is this board's TX on that link. It is reserved but need not be
 *   wired: the ESP32 does not read it, and the echo below goes to USB.
 *   The ESP32's 3.3 V output is above the Uno's 3.0 V threshold for a HIGH, so
 *   this direction needs no level shifting. The other direction would: 5 V into
 *   an ESP32 pin needs a 1k/2k divider, which is why only one wire is used.
 *
 * Emergency LED strip (pin 6):
 *   Pin 6 is a PWM-capable output reserved for the "leave space for the
 *   emergency vehicle" strip. A short strip may be driven through a resistor
 *   like any LED; anything longer than a few LEDs draws more than the 40 mA an
 *   Arduino pin can source, so drive it through a transistor/MOSFET or a relay
 *   with its own supply. The pin is switched HIGH/LOW here, so either works.
 *   For an addressable (WS2812/NeoPixel) strip, keep pin 6 as the data line and
 *   replace setEmergencyStrip() with your library's calls.
 *
 * Library needed (install via Arduino IDE Library Manager):
 *   "LiquidCrystal I2C" by Frank de Brabander
 */

#include <SoftwareSerial.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ── LCD (I2C address 0x27 is most common; try 0x3F if blank) ─────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── Link to the ESP32 hub (RX on 7, TX on 12) ─────────────────────────────
// A software port rather than the hardware one on pins 0/1, so the USB link to
// the PC stays free: uploading a sketch or opening the Serial Monitor would
// otherwise mean unplugging the ESP32's wire every time.
SoftwareSerial EspLink(7, 12);

// ── LED pins ──────────────────────────────────────────────────────────────
// RAMP 1 head — the side road merging into the north→south carriageway
const int PIN_GREEN     = 9;
const int PIN_YELLOW    = 10;
const int PIN_RED       = 11;
// RAMP 2 head — the side road merging into the south→north carriageway.
// Optional; leave these unwired to run a single signal.
const int PIN_GREEN_X   = 2;
const int PIN_YELLOW_X  = 3;
const int PIN_RED_X     = 4;

const int PIN_BLUE      = 8;   // Emergency indicator LED
const int PIN_EMG_STRIP = 6;   // Dedicated emergency "leave space" strip

// ── Serial buffer ─────────────────────────────────────────────────────────
// Reserved up front so appending characters does not repeatedly reallocate,
// which fragments the Uno's 2 KB of SRAM over a long session.
String inputBuffer = "";
const unsigned int MAX_LINE = 96;

// ── State ─────────────────────────────────────────────────────────────────
String lastPhase = "";
bool   lastEmergency = false;
unsigned long lastPacketMs = 0;
const unsigned long TIMEOUT_MS = 5000;  // Show "NO SIGNAL" after 5s of silence

// Whether any byte at all has arrived on the port. This separates the two ways
// the link can be wrong: nothing arriving means the PC is not sending to this
// board, while bytes arriving that never form a packet means it is sending at
// the wrong speed. Both used to look like "Waiting..." forever.
bool anyBytesSeen = false;
const unsigned long SILENCE_MS = 8000;  // Long enough for the PC to boot and open


// ══════════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  EspLink.begin(9600);
  inputBuffer.reserve(MAX_LINE);

  // LCD init
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Traffic System");
  lcd.setCursor(0, 1);
  lcd.print("Waiting...");

  // LED pins
  pinMode(PIN_GREEN,     OUTPUT);
  pinMode(PIN_YELLOW,    OUTPUT);
  pinMode(PIN_RED,       OUTPUT);
  pinMode(PIN_GREEN_X,   OUTPUT);
  pinMode(PIN_YELLOW_X,  OUTPUT);
  pinMode(PIN_RED_X,     OUTPUT);
  pinMode(PIN_BLUE,      OUTPUT);
  pinMode(PIN_EMG_STRIP, OUTPUT);
  allLedsOff();

  // Startup self-test: cycle every output so wiring can be verified at a glance
  startupBlink();
}


// ══════════════════════════════════════════════════════════════════════════
void loop() {
  // ── Read serial bytes into buffer ──────────────────────────────────────
  // Both ports feed the one buffer. They carry the same format from the same
  // program — only the route differs — and only one of them is ever wired at a
  // time in practice, so there is nothing to keep apart.
  readPort(Serial);
  readPort(EspLink);

  // The emergency strip is latched solid in processCommand() rather than
  // flashed here: a steady red reads as "emergency vehicle present" at a
  // glance, and holding a level costs the loop nothing, so every millisecond
  // stays available for reading serial.

  // ── Never heard from the PC at all ─────────────────────────────────────
  // "Waiting..." on its own is a mystery: it looks identical whether the node
  // was never started, is driving a different board, or is talking at a speed
  // this sketch cannot read. Once it is clear nothing is coming, say which.
  if (lastPacketMs == 0 && millis() > SILENCE_MS) {
    if (anyBytesSeen && lastPhase != "BADDATA") {
      lastPhase = "BADDATA";
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Serial garbled  ");
      lcd.setCursor(0, 1);
      lcd.print("Need baud 9600  ");
    } else if (!anyBytesSeen && lastPhase != "NODATA") {
      lastPhase = "NODATA";
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("No data from PC ");
      lcd.setCursor(0, 1);
      // Neither route is delivering. The usual cause is the relay wire: the
      // node talks to the ESP32 and the ESP32's GPIO33 has to reach pin 7 here,
      // with the two grounds tied together.
      lcd.print("Check D7 + GND  ");
    }
  }

  // ── Timeout: no packet for TIMEOUT_MS → show warning ───────────────────
  if (lastPacketMs > 0 && (millis() - lastPacketMs) > TIMEOUT_MS) {
    if (lastPhase != "TIMEOUT") {
      lastPhase = "TIMEOUT";
      lastEmergency = false;
      allLedsOff();
      // Hold BOTH ramps red on a lost connection — fail safe, not fail open. A
      // green left standing here would be inviting traffic to merge onto a
      // carriageway nothing is measuring any more.
      digitalWrite(PIN_RED, HIGH);
      digitalWrite(PIN_RED_X, HIGH);
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("NO SIGNAL       ");
      lcd.setCursor(0, 1);
      lcd.print("Check PC cable  ");
    }
  }
}


// ══════════════════════════════════════════════════════════════════════════
// Drain one port into the line buffer, acting on each complete line
// ══════════════════════════════════════════════════════════════════════════
void readPort(Stream& port) {
  while (port.available()) {
    char ch = (char)port.read();
    anyBytesSeen = true;
    if (ch == '\n') {
      processCommand(inputBuffer);
      inputBuffer = "";
    } else if (ch != '\r') {
      // Drop anything absurdly long rather than growing the buffer without
      // bound if a newline is ever lost to line noise.
      if (inputBuffer.length() < MAX_LINE) {
        inputBuffer += ch;
      }
    }
  }
}


// ══════════════════════════════════════════════════════════════════════════
// Parse "PHASE,LEVEL,CI,VEHICLES,EMG,LINE1,LINE2" and drive the hardware
// ══════════════════════════════════════════════════════════════════════════
void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  // Split on commas — exactly 7 fields
  String parts[7];
  int partIdx = 0;
  int start   = 0;
  for (int i = 0; i <= cmd.length() && partIdx < 7; i++) {
    if (i == cmd.length() || cmd[i] == ',') {
      parts[partIdx++] = cmd.substring(start, i);
      start = i + 1;
    }
  }
  if (partIdx < 7) return;   // malformed, ignore and wait for the next line

  // Counted only once the line has parsed. A wrong baud rate delivers a steady
  // stream of rubbish that sometimes contains a newline, and treating that as
  // contact would report a healthy link that has never carried one usable
  // packet — and hide the "Serial garbled" hint that names the real fault.
  lastPacketMs = millis();

  String phase    = parts[0];
  String level    = parts[1];
  String ci       = parts[2];
  String vehicles = parts[3];
  bool   emergency = (parts[4] == "1");
  String line1    = parts[5];
  String line2    = parts[6];

  // Pad/trim to exactly 16 chars for clean LCD display
  while (line1.length() < 16) line1 += ' ';
  while (line2.length() < 16) line2 += ' ';
  line1 = line1.substring(0, 16);
  line2 = line2.substring(0, 16);

  // Only touch the lamps when the phase actually changes (prevents flicker)
  if (phase != lastPhase) {
    lastPhase = phase;
    setPhaseLEDs(phase);
  }

  // Emergency strip: solid on for as long as an emergency vehicle is present,
  // off the moment it is not.
  if (emergency != lastEmergency) {
    lastEmergency = emergency;
    setEmergencyStrip(emergency);
    digitalWrite(PIN_BLUE, emergency ? HIGH : LOW);
  }

  // LCD always updates (the countdown and CI change every second)
  lcd.setCursor(0, 0);
  lcd.print(line1);
  lcd.setCursor(0, 1);
  lcd.print(line2);

  // Echo back to PC for debugging (visible in Serial Monitor)
  Serial.print("[OK] ");
  Serial.print(phase);
  Serial.print(" ");
  Serial.print(level);
  Serial.print(" CI=");
  Serial.print(ci);
  Serial.print(" V=");
  Serial.print(vehicles);
  Serial.println(emergency ? " EMG" : "");
}


// ══════════════════════════════════════════════════════════════════════════
// Signal lamps — exactly one is lit at a time, like a real signal head
// ══════════════════════════════════════════════════════════════════════════
void setPhaseLEDs(String phase) {
  digitalWrite(PIN_GREEN,  LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_RED,    LOW);

  if (phase == "GREEN") {
    digitalWrite(PIN_GREEN, HIGH);
  } else if (phase == "YELLOW") {
    digitalWrite(PIN_YELLOW, HIGH);
  } else if (phase == "RED") {
    digitalWrite(PIN_RED, HIGH);
  }
  // An unrecognised phase leaves every lamp dark, which is visibly wrong
  // rather than quietly showing the wrong colour.

  setRamp2LEDs(phase);
}


// ══════════════════════════════════════════════════════════════════════════
// Second ramp head — shows the same phase as the first
// ══════════════════════════════════════════════════════════════════════════
// Both side roads are metered from one congestion reading for the whole scene,
// so they open and close together. Nothing crosses here — two ramps merging
// into two separate carriageways — so both being green at once is correct, and
// the yellow between phases is simply shown on both.
void setRamp2LEDs(String phase) {
  digitalWrite(PIN_GREEN_X,  LOW);
  digitalWrite(PIN_YELLOW_X, LOW);
  digitalWrite(PIN_RED_X,    LOW);

  if (phase == "GREEN") {
    digitalWrite(PIN_GREEN_X, HIGH);
  } else if (phase == "YELLOW") {
    digitalWrite(PIN_YELLOW_X, HIGH);
  } else if (phase == "RED") {
    digitalWrite(PIN_RED_X, HIGH);
  }
}


// Dedicated emergency strip output. Swap the body for your LED library if you
// fit an addressable strip; the rest of the sketch does not care.
void setEmergencyStrip(bool on) {
  digitalWrite(PIN_EMG_STRIP, on ? HIGH : LOW);
}


void allLedsOff() {
  digitalWrite(PIN_GREEN,     LOW);
  digitalWrite(PIN_YELLOW,    LOW);
  digitalWrite(PIN_RED,       LOW);
  digitalWrite(PIN_GREEN_X,   LOW);
  digitalWrite(PIN_YELLOW_X,  LOW);
  digitalWrite(PIN_RED_X,     LOW);
  digitalWrite(PIN_BLUE,      LOW);
  digitalWrite(PIN_EMG_STRIP, LOW);
}


// Startup self-test: cycle each output so a miswired LED is obvious at boot.
// The order matches the wiring comment at the top — main green/yellow/red,
// then cross green/yellow/red, then the two emergency outputs — so an LED that
// lights out of turn names the pin that is wrong.
void startupBlink() {
  int pins[] = {PIN_GREEN, PIN_YELLOW, PIN_RED,
                PIN_GREEN_X, PIN_YELLOW_X, PIN_RED_X,
                PIN_BLUE, PIN_EMG_STRIP};
  for (unsigned int i = 0; i < sizeof(pins) / sizeof(pins[0]); i++) {
    digitalWrite(pins[i], HIGH);
    delay(300);
    digitalWrite(pins[i], LOW);
  }
}
