"""Checks for the broadcast server's hardware decisions and HTTP contract.

Two things are verified here that nothing else covers:

1. **The hardware mapping** — that a signal phase becomes the right lamp, servo
   angle, LCD text and emergency-strip state, and that the serial line sent to
   the Arduino matches the format the sketch parses. A mismatch here is silent:
   the Python side keeps running happily while the hardware shows the wrong
   thing.
2. **The HTTP endpoints** the dashboard depends on, driven through Flask's test
   client against a real logger writing to a temp directory.

    python test_server.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import broadcast_server as bs
import traffic_vision as tv
from traffic_light import (
    DEFAULT_GREEN_LEVELS,
    MERGE_GREEN_LEVELS,
    LightState,
    Phase,
    TrafficLightConfig,
    TrafficLightController,
)

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    """Print a section banner."""
    print(f"\n[{title}]")


def state_for(phase: Phase, remaining: float = 0.0,
              target: Phase | None = None) -> LightState:
    """Build a light state directly, without running the clock."""
    return LightState(
        phase=phase,
        target=target or phase,
        remaining=remaining,
        transitioning=phase is Phase.YELLOW,
        elapsed=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════
section("A] Hardware mapping - lamps, servo and LCD")

green = bs.decide_hardware_commands(state_for(Phase.GREEN), "HEAVY", 0.74, False)
check("heavy traffic shows GREEN", green["phase"] == "GREEN", green["phase"])
check("only the green lamp is lit",
      green["green_led"] and not green["red_led"] and not green["yellow_led"])
check("the gate opens on green", green["servo_angle"] == bs.SERVO_OPEN,
      f"{green['servo_angle']}deg")
check("the LCD names the phase", "GREEN" in green["lcd_line1"], repr(green["lcd_line1"]))
check("the LCD reports the congestion reading",
      "HEAVY" in green["lcd_line2"] and "0.74" in green["lcd_line2"],
      repr(green["lcd_line2"]))

red = bs.decide_hardware_commands(state_for(Phase.RED), "FREE", 0.05, False)
check("an empty road shows RED", red["phase"] == "RED", red["phase"])
check("only the red lamp is lit",
      red["red_led"] and not red["green_led"] and not red["yellow_led"])
check("the gate closes on red", red["servo_angle"] == bs.SERVO_CLOSED,
      f"{red['servo_angle']}deg")

yellow = bs.decide_hardware_commands(
    state_for(Phase.YELLOW, remaining=3.0, target=Phase.GREEN), "MODERATE", 0.45, False)
check("the transition shows YELLOW", yellow["phase"] == "YELLOW")
check("only the yellow lamp is lit",
      yellow["yellow_led"] and not yellow["red_led"] and not yellow["green_led"])
check("the gate is part-open during the transition",
      yellow["servo_angle"] == bs.SERVO_PARTIAL, f"{yellow['servo_angle']}deg")
check("the LCD counts the transition down",
      "3s" in yellow["lcd_line1"] and "GREEN" in yellow["lcd_line1"],
      repr(yellow["lcd_line1"]))
check("the remaining time is published", yellow["phase_remaining"] == 3.0)
check("the destination phase is published", yellow["phase_target"] == "GREEN")

for command in (green, red, yellow):
    check(f"{command['phase']}: LCD line 1 is 16 columns",
          len(command["lcd_line1"]) == 16, str(len(command["lcd_line1"])))
    check(f"{command['phase']}: LCD line 2 is 16 columns",
          len(command["lcd_line2"]) == 16, str(len(command["lcd_line2"])))


# ══════════════════════════════════════════════════════════════════════════
section("B] An emergency does not seize the signal")

emergency = bs.decide_hardware_commands(state_for(Phase.RED), "FREE", 0.05,
                                        True, "ambulance")
check("the phase is unchanged by an emergency", emergency["phase"] == "RED",
      "emergency traffic proceeds through any indication")
check("the red lamp stays lit", emergency["red_led"])
check("the dedicated emergency strip is driven", emergency["emergency_strip"] is True)
check("the vehicle type reaches the LCD", "AMBULANCE" in emergency["lcd_line2"],
      repr(emergency["lcd_line2"]))
check("the signal line is untouched by the emergency",
      "RED" in emergency["lcd_line1"], repr(emergency["lcd_line1"]))

police = bs.decide_hardware_commands(state_for(Phase.GREEN), "HEAVY", 0.8,
                                     True, "police")
check("a police car is named too", "POLICE" in police["lcd_line2"],
      repr(police["lcd_line2"]))
check("green stays green during an emergency", police["phase"] == "GREEN")

unknown = bs.decide_hardware_commands(state_for(Phase.GREEN), "HEAVY", 0.8, True,
                                      "not-a-type")
check("an unidentified emergency still raises the strip",
      unknown["emergency_strip"] and "EMERGENCY" in unknown["lcd_line2"],
      repr(unknown["lcd_line2"]))

quiet = bs.decide_hardware_commands(state_for(Phase.GREEN), "HEAVY", 0.8, False)
check("no emergency leaves the strip dark", quiet["emergency_strip"] is False)


# ══════════════════════════════════════════════════════════════════════════
section("C] Serial wire format matches the Arduino sketch")

sent: list[bytes] = []


class FakeSerial:
    """Captures what would have gone down the wire."""

    is_open = True

    def write(self, payload: bytes) -> None:
        """Record one written line."""
        sent.append(payload)


# The seven-field format belongs to the Uno sketch; the ESP32 is driven with
# JSON instead, so the board has to be pinned for this section to mean anything.
original = bs.arduino
original_board = bs.board_kind
bs.arduino = FakeSerial()
bs.board_kind = "uno"
try:
    bs.send_to_arduino(yellow, 0.456, 7)
    bs.send_to_arduino(emergency, 0.05, 1)
finally:
    bs.arduino = original
    bs.board_kind = original_board

line = sent[0].decode("ascii")
fields = line.rstrip("\n").split(",")
check("a line is sent per call", len(sent) == 2, f"{len(sent)} lines")
check("the line is newline terminated", line.endswith("\n"))
check("the sketch's 7 fields are present", len(fields) == 7, f"{len(fields)}: {fields}")
check("field 1 is the phase", fields[0] == "YELLOW", fields[0])
check("field 2 is the congestion level", fields[1] == "MODERATE", fields[1])
check("field 3 is the congestion index", fields[2] == "0.456", fields[2])
check("field 4 is the vehicle count", fields[3] == "7", fields[3])
check("field 5 is the emergency flag", fields[4] == "0", fields[4])
check("fields 6 and 7 are the LCD rows",
      "YELLOW" in fields[5] and "MODERATE" in fields[6], f"{fields[5]!r} {fields[6]!r}")

emergency_fields = sent[1].decode("ascii").rstrip("\n").split(",")
check("the emergency flag is set when a vehicle is present",
      emergency_fields[4] == "1", emergency_fields[4])
check("the phase is still sent during an emergency",
      emergency_fields[0] == "RED", emergency_fields[0])
check("the line is plain ASCII the Arduino can parse",
      all(ord(char) < 128 for char in line), "no multi-byte characters")

check("a missing Arduino is a no-op, not a crash",
      bs.send_to_arduino(green, 0.1, 1) is None)


# ══════════════════════════════════════════════════════════════════════════
section("C2] The ESP32 is driven with its own JSON protocol")


class FakeEsp32:
    """A stand-in for the ESP32 sketch: JSON in, one acknowledgement out."""

    is_open = True

    def __init__(self, reply: str = '{"type":"ack","success":true}') -> None:
        """Record written requests and hand back a canned reply line."""
        self.written: list[str] = []
        self._reply = reply
        self._pending: list[str] = []

    def reset_input_buffer(self) -> None:
        """Drop replies nobody is waiting for, as pyserial does."""
        self._pending.clear()

    def flush(self) -> None:
        """No buffering to flush."""

    def write(self, payload: bytes) -> None:
        """Record one request and queue the sketch's answer to it."""
        self.written.append(payload.decode("utf-8"))
        self._pending.append(self._reply)

    def readline(self) -> bytes:
        """Return the next queued reply line, or nothing."""
        if not self._pending:
            return b""
        return (self._pending.pop(0) + "\n").encode("utf-8")


board = FakeEsp32()
original = bs.arduino
original_board = bs.board_kind
bs.arduino = board
bs.board_kind = "esp32"
try:
    acked = bs.send_json_to_board("LED_MODE", {"mode": "AMBULANCE"})
    request = json.loads(board.written[0])

    check("the request is JSON the sketch can parse", isinstance(request, dict))
    check("the command survives the trip", request["command"] == "LED_MODE",
          str(request.get("command")))
    check("the payload the sketch reads is carried through",
          request["payload"]["mode"] == "AMBULANCE", str(request.get("payload")))
    check("the line is newline terminated", board.written[0].endswith("\n"))
    check("an acknowledged command reports as delivered", acked is True, str(acked))

    # With no Uno of its own on the node, the ESP32 is the way to it: the same
    # seven-field line is written to the hub, which passes it down UART1
    # untouched. The format must survive that trip unchanged, because the Uno
    # sketch is the thing parsing it at the far end.
    before = len(board.written)
    bs.send_to_arduino(yellow, 0.456, 7)
    check("the signal line is relayed through the ESP32",
          len(board.written) == before + 1, f"{len(board.written) - before} lines")
    relayed = board.written[-1]
    check("and goes down the wire in the Uno's own format, not wrapped in JSON",
          relayed.startswith("YELLOW,MODERATE,0.456,7,0,") and relayed.endswith("\n"),
          repr(relayed))
    check("the relayed line still has the sketch's 7 fields",
          len(relayed.rstrip("\n").split(",")) == 7, repr(relayed))

    refused = FakeEsp32('{"type":"error","success":false,"error":"Unknown command"}')
    bs.arduino = refused
    check("a refusal is reported, not swallowed",
          bs.send_json_to_board("RAISE_SPEED_BUMP", {}) is False)
finally:
    bs.arduino = original
    bs.board_kind = original_board

bs.board_kind = "esp32"
try:
    check("a missing board is a no-op, not a crash",
          bs.send_json_to_board("GET_STATUS", {}) is False)
finally:
    bs.board_kind = original_board


# ══════════════════════════════════════════════════════════════════════════
section("D] The signal drives from the congestion level end to end")

controller = TrafficLightController(TrafficLightConfig(yellow_seconds=5.0))
commands = []
now = 0.0
for step in range(200):
    now = step * 0.5
    level = "HEAVY" if 20 <= now < 60 else "FREE"
    light_state = controller.update(level, now)
    commands.append((now, bs.decide_hardware_commands(
        light_state, level, 0.5, False)))

phases = [(t, c["phase"]) for t, c in commands]
changes = [phases[i] for i in range(len(phases))
           if i == 0 or phases[i][1] != phases[i - 1][1]]
sequence = [phase for _, phase in changes]
check("the full sequence is red -> yellow -> green -> yellow -> red",
      sequence == ["RED", "YELLOW", "GREEN", "YELLOW", "RED"], f"{sequence}")

yellow_spans = [changes[i + 1][0] - changes[i][0]
                for i in range(len(changes) - 1) if changes[i][1] == "YELLOW"]
check("each yellow lasted 5 seconds",
      all(abs(span - 5.0) < 0.51 for span in yellow_spans), f"{yellow_spans}")
check("exactly one lamp is lit at every instant",
      all(sum([c["red_led"], c["yellow_led"], c["green_led"]]) == 1
          for _, c in commands), "no ambiguous indications")
check("the servo always agrees with the lamp",
      all(c["servo_angle"] == bs.SERVO_FOR_PHASE[Phase(c["phase"])]
          for _, c in commands))


# ══════════════════════════════════════════════════════════════════════════
section("D2] A merging side road is metered the other way round")

# The signal sits on the ramp, not on the carriageway it feeds, so it closes
# while the main road is heavy and opens once there is room -- the opposite of
# the junction signal in section D above.
meter = TrafficLightController(TrafficLightConfig(green_levels=MERGE_GREEN_LEVELS))
now = 10.0
meter.update("HEAVY", now)
check("HEAVY closes the merge", meter.phase is Phase.RED, str(meter.phase))

now += 10.0
opening = meter.update("MODERATE", now)
check("MODERATE begins opening it", opening.phase is Phase.YELLOW, str(opening.phase))
check("with green as the target", opening.target is Phase.GREEN, str(opening.target))

now += 5.1
meter.update("MODERATE", now)
check("MODERATE opens the merge", meter.phase is Phase.GREEN, str(meter.phase))

now += 10.0
meter.update("FREE", now)
check("FREE keeps it open", meter.phase is Phase.GREEN, str(meter.phase))

now += 10.0
meter.update("HEAVY", now)
now += 5.1
meter.update("HEAVY", now)
check("and HEAVY closes it again", meter.phase is Phase.RED, str(meter.phase))

check("a junction signal is unaffected",
      TrafficLightConfig().green_levels == DEFAULT_GREEN_LEVELS,
      str(TrafficLightConfig().green_levels))


# ══════════════════════════════════════════════════════════════════════════
section("D3] Both boards are driven from one detection")


class FakeUnoLink:
    """Records the seven-field lines written to it."""

    is_open = True

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def write(self, payload: bytes) -> None:
        """Record one written line."""
        self.sent.append(payload)


class FakeEsp32Link(FakeUnoLink):
    """Answers every JSON command with an ack, as the sketch does."""

    def reset_input_buffer(self) -> None:
        """No buffered replies to discard."""

    def flush(self) -> None:
        """Nothing is buffered on the way out."""

    def write(self, payload: bytes) -> None:
        """Record the command and queue the acknowledgement it earns."""
        self.sent.append(payload)
        self.acks.append(b'{"type":"ack","success":true}\n')

    def readline(self) -> bytes:
        """Return the next queued acknowledgement."""
        return self.acks.pop(0) if self.acks else b""

    acks: list[bytes] = []


uno_link, esp_link = FakeUnoLink(), FakeEsp32Link()
esp_link.acks = []
original_links = dict(bs.board_links)
original_arduino, original_kind = bs.arduino, bs.board_kind
bs.board_links.update({"uno": uno_link, "esp32": esp_link})
bs.arduino = None
try:
    check("both boards count as connected", bs.any_board_connected())
    check("each kind resolves to its own port",
          bs.board_link("uno") is uno_link and bs.board_link("esp32") is esp_link)

    busy = {"phase": "RED", "congestion_level": "HEAVY", "emergency_strip": True,
            "lcd_line1": "LIGHT: RED", "lcd_line2": "HEAVY CI:0.81"}
    bs.send_to_arduino(busy, 0.81, 12)
    check("the seven-field line goes to the Uno only",
          len(uno_link.sent) == 1 and not esp_link.sent, str(uno_link.sent[:1]))

    # The ESP32 has no camera: until it is told, a detected ambulance lights the
    # Uno's strip and leaves the ESP32 dark. This is that gap closed.
    bs._esp32_state.update({"emergency": None})
    bs.push_state_to_esp32(busy, True)
    sent = [json.loads(payload.decode())["command"] for payload in esp_link.sent]
    check("a detected emergency reaches the ESP32 by itself",
          "ACTIVATE_EMERGENCY" in sent, str(sent))
    check("and the dashboard's view of the board follows it",
          bs.manual_hardware["led_mode"] == "AMBULANCE",
          bs.manual_hardware["led_mode"])

    settled = len(esp_link.sent)
    bs.push_state_to_esp32(busy, True)
    bs.push_state_to_esp32(busy, True)
    check("an unchanged state is not re-sent every second",
          len(esp_link.sent) == settled, f"{len(esp_link.sent) - settled} extra")

    esp_link.sent.clear()
    clear = {"phase": "GREEN", "congestion_level": "FREE", "emergency_strip": False,
             "lcd_line1": "LIGHT: GREEN", "lcd_line2": "FREE CI:0.05"}
    bs.push_state_to_esp32(clear, False)
    sent = [json.loads(payload.decode()) for payload in esp_link.sent]
    commands = [entry["command"] for entry in sent]
    check("clearing the emergency reaches it too",
          "DEACTIVATE_EMERGENCY" in commands, str(commands))

    # The ESP32's LCD reports what its LED strip is doing and nothing else, so
    # the congestion reading is never pushed at it. That text belongs on the
    # Uno's LCD, which already receives it in the seven-field line above.
    check("the congestion reading is not pushed to its LCD",
          "LCD_MESSAGE" not in commands, str(commands))

    esp_link.sent.clear()
    busier = dict(clear, congestion_level="MODERATE", lcd_line2="MODERATE CI:0.44")
    bs.push_state_to_esp32(busier, False)
    check("nor is a change of congestion level", not esp_link.sent, str(esp_link.sent))

    # A direct link is preferred over the relay when both are open: the line
    # then reaches the Uno without a second board in the path, and the hub is
    # not asked to spend its UART on something it does not need to carry.
    esp_link.sent.clear()
    uno_link.sent.clear()
    bs.send_to_arduino(busy, 0.81, 12)
    check("with both attached the direct link wins",
          len(uno_link.sent) == 1 and not esp_link.sent,
          f"uno={len(uno_link.sent)} esp32={len(esp_link.sent)}")

    # One USB cable to the hub is the normal wiring, and the signal heads have
    # to keep working on it. This used to go nowhere -- the heads stayed dark
    # and the fault looked like dead hardware.
    bs.board_links.clear()
    bs.board_links["esp32"] = esp_link
    esp_link.sent.clear()
    uno_link.sent.clear()
    bs.send_to_arduino(busy, 0.81, 12)
    check("with only the hub attached the line is relayed through it",
          not uno_link.sent and len(esp_link.sent) == 1,
          f"uno={len(uno_link.sent)} esp32={len(esp_link.sent)}")
    check("the relayed line is the Uno's format verbatim",
          esp_link.sent[0].decode().startswith("RED,HEAVY,0.810,12,1,"),
          esp_link.sent[0].decode())
finally:
    bs.board_links.clear()
    bs.board_links.update(original_links)
    bs.arduino, bs.board_kind = original_arduino, original_kind


# ══════════════════════════════════════════════════════════════════════════
section("E] HTTP endpoints the dashboard depends on")

workdir = Path(tempfile.mkdtemp())


def sample_row(index: int, level: str = "FREE", phase: str = "RED") -> dict:
    """One telemetry row in the logged schema."""
    return {
        "Date": "2026-01-01", "Timestamp": f"12:00:{index % 60:02d}",
        "Congestion Index": 0.2, "Level": level, "Light Phase": phase,
        "Total Vehicles": 3, "Cars": 3, "Motorcycles": 0, "Buses": 0, "Trucks": 0,
        "Avg Speed (km/h)": 30.0, "Max Speed (km/h)": 40.0, "Moving": 3, "Stopped": 0,
        "Weighted Density (PCE)": 3.0, "Servo Angle (deg)": 10, "Emergency": "no",
        "Emergency Type": "none", "Emergency Vehicles": 0, "Unique Vehicles": 3,
        "FPS": 25.0, "Speed Source": "auto",
    }


logger = tv.TelemetryLogger(workdir / "traffic_history.csv",
                            workdir / "traffic_history.xlsx")
light = TrafficLightController()
bs._register_runtime(logger, light)
bs._paths["csv"] = str(workdir / "traffic_history.csv")
bs._paths["excel"] = str(workdir / "traffic_history.xlsx")
bs._paths["master_csv"] = str(logger.master_csv_path)
bs._paths["master_excel"] = str(logger.master_excel_path)

for index in range(10):
    logger.log(sample_row(index))
logger.save_now()

client = bs.flask_app.test_client()

try:
    response = client.get("/record/status")
    body = response.get_json()
    check("/record/status responds", response.status_code == 200)
    check("status reports the open session", body["rows"] == 10, f"{body}")
    check("status starts un-armed", body["recording"] is False)

    response = client.post("/record/start", json={"name": "http test"})
    check("/record/start responds", response.status_code == 200)
    check("starting arms the recording", response.get_json()["recording"] is True)
    check("the recording takes the given name",
          response.get_json()["name"] == "http test")

    for index in range(4):
        logger.log(sample_row(index, "HEAVY", "GREEN"))

    check("rows land in the new recording",
          client.get("/record/status").get_json()["rows"] == 4)

    stopped = client.post("/record/stop").get_json()
    check("/record/stop returns what was captured",
          stopped["rows"] == 4 and stopped["name"] == "http test", f"{stopped}")
    check("stopping disarms", client.get("/record/status").get_json()["recording"] is False)

    saved = client.post("/record/save").get_json()
    check("/record/save writes both workbooks", "session" in saved and "master" in saved,
          f"{saved}")

    sessions = client.get("/sessions").get_json()
    check("/sessions lists every recorded run", len(sessions) >= 2, f"{len(sessions)}")
    check("the named recording is listed",
          any(entry["Session Name"] == "http test" for entry in sessions),
          f"{[e['Session Name'] for e in sessions]}")
    check("session summaries carry their statistics",
          all("Peak Vehicles" in entry and "Mean Congestion" in entry
              for entry in sessions))

    records = client.get("/records").get_json()
    check("/records returns every row across sessions", len(records) >= 14,
          f"{len(records)}")
    check("records carry a session stamp", all(row.get("Session") for row in records))
    check("records use the logged schema",
          set(tv.LOG_HEADERS).issubset(set(records[0])), f"{sorted(records[0])[:5]}")

    one = [entry for entry in sessions if entry["Session Name"] == "http test"][0]
    scoped = client.get(f"/records?session={one['Session']}").get_json()
    check("/records filters to one session", len(scoped) == 4, f"{len(scoped)}")
    check("the filter really restricts the session",
          {row["Session"] for row in scoped} == {one["Session"]})

    limited = client.get("/records?limit=3").get_json()
    check("/records honours the limit", len(limited) == 3, f"{len(limited)}")
    check("a bad limit falls back instead of failing",
          client.get("/records?limit=abc").status_code == 200)

    check("/export/all serves the master workbook",
          client.get("/export/all").status_code == 200)
    check("/export/all.csv serves the master CSV",
          client.get("/export/all.csv").status_code == 200)
    check("/export still serves this session only",
          client.get("/export").status_code == 200)

    # ── Signal endpoint ─────────────────────────────────────────────────────
    light_body = client.get("/light").get_json()
    check("/light reports the phase", light_body["phase"] in {"RED", "YELLOW", "GREEN"},
          light_body["phase"])
    check("/light publishes its timing config",
          light_body["config"]["yellow_seconds"] == 5.0, f"{light_body['config']}")
    check("/light starts in automatic mode", light_body["manual"] is None)

    forced = client.post("/light", json={"phase": "GREEN"}).get_json()
    check("/light accepts a forced phase", forced["manual"] == "GREEN", f"{forced}")
    check("the controller received the override", light.forced is Phase.GREEN)

    released = client.post("/light", json={"phase": "AUTO"}).get_json()
    check("/light releases the override", released["manual"] is None)
    check("the controller is back on automatic", light.forced is None)

    bad = client.post("/light", json={"phase": "PURPLE"})
    check("an unknown phase is rejected with 400", bad.status_code == 400,
          f"{bad.status_code}")

    health = client.get("/health").get_json()
    check("/health reports the light phase", "light_phase" in health, f"{sorted(health)}")
    check("/health reports the recording state", health["recording"] is False)

    index_page = client.get("/").get_data(as_text=True)
    for endpoint in ("/export/all", "/sessions", "/records", "/record/start", "/light"):
        check(f"the landing page documents {endpoint}", endpoint in index_page)

finally:
    logger.close()
    bs._register_runtime(None, None)
    shutil.rmtree(workdir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
section("E2] Hardware commands forwarded on the dashboard's behalf")

# The dashboard has no serial port: this node owns the Arduino, so manual
# hardware controls arrive over HTTP and are translated into the words the
# lane-changer/LED board reads. A wrong word here is silent -- the dashboard
# reports success while the barrier never moves.

reply = client.post("/command", json={"command": "SET_LANE_ALLOCATION",
                                      "payload": {"allocation": "FORWARD_4"}}).get_json()
check("a lane allocation is acknowledged", reply["success"], reply.get("ack"))
check("it becomes the word the board reads", reply["sent"] == "LANE_FORWARD_4",
      reply["sent"])
check("the resulting allocation is reported back",
      reply["lane_allocation"] == "FORWARD_4_OPPOSITE_2", reply["lane_allocation"])

for allocation, word in (("BALANCED", "LANE_BALANCED"),
                         ("OPPOSITE_4", "LANE_OPPOSITE_4")):
    sent = client.post("/command", json={"command": "SET_LANE_ALLOCATION",
                                         "payload": {"allocation": allocation}}).get_json()
    check(f"{allocation} maps to {word}", sent["sent"] == word, sent["sent"])


def move_divider(direction):
    """Step the divider one lane and return the node's reply."""
    return client.post("/command", json={"command": "MOVE_LANE_DIVIDER",
                                         "payload": {"direction": direction}}).get_json()


# The dashboard drives the barrier with two arrows, so the node -- which is the
# only side that knows where the divider currently is -- resolves the step. A
# wrong neighbour here moves the barrier the wrong way across live traffic.
# The loop above leaves it at the left-most position.
step = move_divider("RIGHT")
check("a divider step right lands on the middle position",
      step["lane_allocation"] == "BALANCED_3_3", step["lane_allocation"])
check("the step is sent as the board's own lane word",
      step["sent"] == "LANE_BALANCED", step["sent"])
step = move_divider("RIGHT")
check("a second step right reaches the far lane",
      step["lane_allocation"] == "FORWARD_4_OPPOSITE_2", step["lane_allocation"])

# At the outermost lane the arrow must not claim a move it did not make, and
# must not spend the link re-commanding the position already held.
step = move_divider("RIGHT")
check("stepping past the far lane reports the limit",
      step["at_limit"] is True and step["sent"] is None, step["sent"])
check("the position at the limit is unchanged",
      step["lane_allocation"] == "FORWARD_4_OPPOSITE_2", step["lane_allocation"])

step = move_divider("LEFT")
check("a divider step left comes back through the middle",
      step["lane_allocation"] == "BALANCED_3_3" and step["at_limit"] is False,
      step["lane_allocation"])
check("an unknown divider direction is refused",
      client.post("/command", json={"command": "MOVE_LANE_DIVIDER",
                                    "payload": {"direction": "UP"}}).status_code == 400)

# Left of centre is where the other tests below expect to find it.
move_divider("LEFT")

# The LED strip is asked for a meaning, not a colour: the board picks the
# flash pattern that goes with each one.
for mode, word in (("POLICE", "POLICE"), ("AMBULANCE", "AMBULANCE"),
                   ("SPEED", "SPEED"), ("OFF", "NORMAL")):
    sent = client.post("/command", json={"command": "LED_MODE",
                                         "payload": {"mode": mode}}).get_json()
    check(f"LED mode {mode} maps to {word}", sent["sent"] == word, sent["sent"])

bump = client.post("/command", json={"command": "RAISE_SPEED_BUMP"}).get_json()
check("the speed bump remembers it is raised", bump["speed_bump"] == "Raised")
lcd = client.post("/command", json={"command": "LCD_MESSAGE",
                                    "payload": {"message": "TEST"}}).get_json()
check("an LCD message is forwarded verbatim", lcd["sent"] == "LCD:TEST", lcd["sent"])

check("an unknown command is refused, not forwarded",
      client.post("/command", json={"command": "MAKE_TEA"}).status_code == 400)
check("an unknown allocation is refused",
      client.post("/command", json={"command": "SET_LANE_ALLOCATION",
                                    "payload": {"allocation": "SIDEWAYS"}}).status_code == 400)
check("an empty command is refused",
      client.post("/command", json={}).status_code == 400)

# With no board attached the command still updates what the dashboard shows,
# but says plainly that nothing reached the wire.
check("a command with no board attached reports it was not delivered",
      bump["delivered"] is False and bump["arduino_connected"] is False)

# A command line carries no commas, and the traffic-controller sketch drops any
# line without seven fields -- that is what lets one port carry both.
check("command lines cannot be mistaken for a telemetry packet",
      "," not in reply["sent"])

snapshot = bs._manual_hardware_snapshot()
check("telemetry echoes the commanded lane allocation",
      snapshot["lane_allocation"] == "FORWARD_2_OPPOSITE_4", snapshot["lane_allocation"])
check("telemetry reports whether a board is attached",
      snapshot["arduino_connected"] is False)


# ══════════════════════════════════════════════════════════════════════════
section("F] Endpoints degrade safely before the pipeline is running")

bs._register_runtime(None, None)
check("/sessions returns an empty list, not an error",
      client.get("/sessions").get_json() == [])
check("/records returns an empty list, not an error",
      client.get("/records").get_json() == [])
check("/record/status reports idle", client.get("/record/status").get_json()["rows"] == 0)
check("/record/start reports unavailable", client.post("/record/start").status_code == 503)
check("/light reports unavailable", client.get("/light").status_code == 503)


# ══════════════════════════════════════════════════════════════════════════
section("G] Configuration defaults")

args = bs.parse_args([])
check("yellow defaults to 5 seconds", args.yellow_seconds == 5.0,
      f"{args.yellow_seconds}")
check("a master CSV is configured by default",
      args.master_csv == "traffic_all_sessions.csv", args.master_csv)
check("a master workbook is configured by default",
      args.master_excel == "traffic_all_sessions.xlsx", args.master_excel)
check("the session file stays the familiar name",
      args.csv == "traffic_history.csv", args.csv)
check("the yellow interval is overridable",
      bs.parse_args(["--yellow-seconds", "3"]).yellow_seconds == 3.0)
check("MODERATE traffic is served green by default",
      "MODERATE" in TrafficLightConfig().green_levels,
      str(TrafficLightConfig().green_levels))
check("a merging side road is metered by default",
      bs.parse_args([]).signal_mode == "merge", bs.parse_args([]).signal_mode)

# The serial speed follows the sketch named by --board. Left to one fixed
# number it silently disagreed with whichever board did not use it, and a
# mismatched speed is indistinguishable from an unplugged cable.
check("a uno is read at the 9600 its sketch opens with",
      bs.resolve_baud(None, "uno") == 9600, str(bs.resolve_baud(None, "uno")))
check("an esp32 is read at 115200",
      bs.resolve_baud(None, "esp32") == 115200, str(bs.resolve_baud(None, "esp32")))
check("and an explicit --arduino-baud still wins",
      bs.resolve_baud(57600, "uno") == 57600, str(bs.resolve_baud(57600, "uno")))
check("the baud is not pinned to one board's speed",
      bs.parse_args([]).arduino_baud is None, str(bs.parse_args([]).arduino_baud))


# ══════════════════════════════════════════════════════════════════════════
section("H] The launcher hands the CV node its options")

# start.py keeps a few options for itself and passes everything else through.
# If one of the node's options were ever shadowed by one of the launcher's, it
# would be swallowed in silence — the system would start and simply watch the
# wrong thing.
import start  # noqa: E402 - imported here so the section reads in order

own, passthrough = start.parse_args([
    "--dashboard-port", "9000", "--source", "road.mp4", "--no-arduino",
    "--imgsz", "1280", "--calibration", "junction_a.json",
])
check("the launcher keeps its own options", own.dashboard_port == 9000)
check("--source reaches the CV node", "--source" in passthrough and "road.mp4" in passthrough,
      str(passthrough))
check("so does every other node option",
      {"--no-arduino", "--imgsz", "1280", "--calibration", "junction_a.json"}
      .issubset(set(passthrough)), str(passthrough))
check("and nothing of the launcher's is mixed in with them",
      "--dashboard-port" not in passthrough and "9000" not in passthrough,
      str(passthrough))

defaults, _rest = start.parse_args([])
check("the dashboard is told the node is local by default",
      defaults.node == "127.0.0.1", defaults.node)
check("the launcher and the node agree on the stream port",
      defaults.stream_port == bs.parse_args([]).stream_port,
      f"{defaults.stream_port} vs {bs.parse_args([]).stream_port}")
check("the upload limit is raised above Streamlit's 200 MB default",
      defaults.max_upload_mb > 200, str(defaults.max_upload_mb))
check("either half can be started alone",
      not defaults.no_server and not defaults.no_dashboard)

_, leaked = start.parse_args(["--dashboard-port", "9000", "--node", "10.0.0.2",
                              "--no-browser"])
check("the launcher's own options are not passed on to the node", not leaked,
      str(leaked))

# --max-upload-mb is the one name both programs want, and they mean the same
# thing by it, so the launcher raises both limits from the single number rather
# than silently keeping it to itself.
command = start.node_command(start.parse_args(["--max-upload-mb", "500"])[0], [])
check("an upload limit given to the launcher also reaches the node",
      "--max-upload-mb" in command and "500" in command, str(command[-6:]))
check("the node's stream port is set from the launcher's",
      "--stream-port" in command, str(command[-6:]))


# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f" RESULT: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(1 if failed else 0)
