"""
broadcast_server.py  —  Smart Traffic CV Node
=============================================
Runs on the laptop/PC with the camera. All of the detection, speed and logging
logic lives in ``traffic_vision.py``; this file is the orchestration layer.

Responsibilities:
  1. Run YOLO11-m detection + BoT-SORT tracking (see traffic_vision.py)
  2. Measure every vehicle's speed in **km/h** and compute a congestion index
  2b. Detect working siren beacons (siren_vision.py) and name the vehicle
      type from its livery (emergency_vision.py)
  3. Decide hardware commands (servo angle, LED states, LCD messages)
  4. Send a compact command string over USB Serial to the Arduino
  5. Log one telemetry row per second to CSV (instantly) and to a styled
     traffic_history.xlsx (periodically + on exit)
  6. Push the same JSON over TCP to the Raspberry Pi controller
  7. Serve the live annotated feed + telemetry over HTTP so the Streamlit
     dashboard on the same WiFi can display it

Quick start
-----------
    python broadcast_server.py                       # webcam 0, yolo11m
    python broadcast_server.py --source road.mp4     # a recorded video
    python broadcast_server.py --source rtsp://...   # a roadside camera
    python broadcast_server.py --imgsz 1280          # more accuracy, less FPS
    python broadcast_server.py --model yolo12n.pt    # fall back to the nano model
    python broadcast_server.py --no-display          # headless / server mode
    python broadcast_server.py --no-emergency-ai     # siren detection only

Where the frames come from
--------------------------
The source is not fixed at start-up. A video can be uploaded, or a camera URL
pasted, from the dashboard at any time and the pipeline carries straight on —
same detector, same km/h, same signal, same logs — because everything about
*where frames come from* is confined to ``video_source.py``. A recorded video
runs on its own media clock rather than the wall clock, so the speeds it
reports are the speeds it was filmed at even when the machine cannot decode it
at full rate. See ``/source``, ``/sources``, ``/source/upload`` and
``/source/control`` below.

Speeds are reported in km/h. Accuracy depends on calibration — run
``python calibrate_speed.py`` once per camera position for ±5% accuracy.
Without it the system self-calibrates from vehicle sizes (±25%) so it still
reports real km/h out of the box.

Emergency mode is raised by a **working siren light** — a beacon that actually
switches on and off — and its colour names the vehicle: blue is police, red is
an ambulance (a fire engine on a large vehicle), alternating red and blue is
police. See ``siren_vision.py``.

Livery recognition (``emergency_vision.py``) runs alongside but does *not*
trigger: it names the type when a red beacon could be either an ambulance or a
fire engine, and it marks vehicles that look like emergency vehicles but are
running no beacon with a "?" and no priority. An ambulance parked outside a
hospital is still an ambulance and does not need the junction held for it.
Pass ``--livery-triggers-emergency`` to let markings trigger as well, which
suits a camera watching a hospital approach rather than a public road.

Serial protocol to Arduino (one line per second, newline-terminated):
  LEVEL,CI,VEHICLES,LINE1,LINE2
  e.g.  HEAVY,0.74,11,TRAFFIC: HEAVY  ,CI: 0.74 STOP
  e.g.  EMERGENCY,1.00,3,!! AMBULANCE !!,CLEAR RIGHT LANE

Keyboard controls (press while the video window is focused):
  q  — quit                     e  — toggle Emergency mode
  g  — force GREEN  (Free)      y  — force YELLOW (Moderate)
  r  — force RED    (Heavy)     a  — back to automatic mode
  x  — write the Excel workbook right now
  c  — reload speed_calibration.json without restarting
  p  — save a snapshot PNG of the current annotated frame

Video-file playback (ignored for live sources):
  space — pause / resume          [ ] — jump back / forward 5 seconds
  l     — toggle looping          0   — restart from the beginning
"""

from __future__ import annotations

import argparse
import ctypes
import json
import signal
import socket
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file

from emergency_vision import LCD_NAMES
from traffic_light import (
    DEFAULT_GREEN_LEVELS,
    MERGE_GREEN_LEVELS,
    LightState,
    Phase,
    TrafficLightConfig,
    TrafficLightController,
    lcd_lines,
)
from traffic_vision import (
    TelemetryLogger,
    TrafficVisionPipeline,
    VisionConfig,
    annotate,
    build_log_row,
    fit_to_display,
)
from video_source import (
    SourceError,
    VideoSource,
    available_cameras,
    delete_upload,
    list_cameras,
    list_uploads,
    probe,
    store_upload,
)

# ── Arduino Serial ────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed. Run: pip install pyserial")

ROOT_DIR = Path(__file__).resolve().parent

# ══════════════════════════════════════════════
# CONFIGURATION — defaults, all overridable on the command line
# ══════════════════════════════════════════════

# One USB cable leaves this PC, and it goes to the ESP32
# (``traffic_controller_esp32/``) on COM6 at 115200. The ESP32 is the hub: it
# drives the LED strip and its own LCD directly, and reaches both Unos over its
# own UARTs -- the lane changer on UART2, the signal-head controller
# (``traffic_controller_arduino/``) on UART1. Its protocol is newline-delimited
# JSON, the same messages the dashboard POSTs to ``/command``.
#
# The seven-field signal line still belongs to the Uno sketch; with no Uno on a
# port of its own it is written to the ESP32, which recognises it and passes it
# down UART1 untouched (see ``send_to_arduino``). Plugging the Uno straight into
# the PC as well still works -- ``--board uno``, or ``--second-port`` for both --
# and then the line goes to it directly rather than through the hub.
#
# Naming the wrong board here is silent: the port opens and the writes succeed,
# but nothing the sketch understands is ever sent, so it sits on its start-up
# message looking like dead hardware.
ARDUINO_PORT = "COM6"
ARDUINO_BAUDRATE = 115200   # fallback only; resolve_baud() prefers the board's own
ARDUINO_BOARD = "esp32"

#: How long to wait for the board's acknowledgement of a dashboard command.
#: Shorter than the dashboard's own 4 s HTTP timeout, so a board that goes
#: quiet is reported as such rather than showing up as an unreachable node.
BOARD_REPLY_TIMEOUT = 3.0

SERVER_HOST = "0.0.0.0"      # TCP telemetry socket (Raspberry Pi)
SERVER_PORT = 5000

STREAM_HOST = "0.0.0.0"      # HTTP video + telemetry (dashboard)
STREAM_PORT = 8502
JPEG_QUALITY = 80

#: Annotated preview frames per second. Detection is unaffected by this — it
#: runs on every frame the source delivers — but drawing the overlays and
#: encoding a JPEG is pure presentation cost, so it is capped at the rate a
#: person can actually perceive rather than repeated as fast as the GPU allows.
STREAM_FPS = 15.0

CSV_FILE = "traffic_history.csv"            # current session only
EXCEL_FILE = "traffic_history.xlsx"         # current session only
MASTER_CSV_FILE = "traffic_all_sessions.csv"    # every session ever recorded
MASTER_EXCEL_FILE = "traffic_all_sessions.xlsx"

# Traffic signal timing. Which levels call for green depends on --signal-mode:
# a merging side road is held RED while the main carriageway is HEAVY, while a
# junction approach shows GREEN whenever there is traffic to serve. Either way
# the timings below govern the change between them.
YELLOW_SECONDS = 5.0
MIN_PHASE_SECONDS = 3.0

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Servo positions for the lane divider / gate.
SERVO_OPEN = 90
SERVO_PARTIAL = 45
SERVO_CLOSED = 10
SERVO_EMERGENCY = 180

HISTORY_MAXLEN = 3600        # ~1 hour of 1 Hz telemetry kept in memory

#: Largest video the dashboard may upload. Werkzeug spools the body to a temp
#: file rather than holding it in memory, so this bounds disk, not RAM.
MAX_UPLOAD_MB = 4096


# ══════════════════════════════════════════════
# COMMAND LINE
# ══════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for this run.

    ``argv`` defaults to the real command line; passing a list makes the
    defaults testable without spawning a process.
    """
    parser = argparse.ArgumentParser(
        description="Smart Traffic CV broadcast node (YOLO11-m, km/h speeds).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="2",
                        help="camera name (e.g. droidcam), camera index, video file, or "
                             "stream URL. Can be changed live from the dashboard.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe camera indices, print which ones deliver a picture, and exit")
    # Defaults come from VisionConfig rather than being repeated here: this one
    # said yolo12m.pt long after the config had moved to yolo11m.pt, so every
    # run silently used the slower model regardless of what the config declared.
    parser.add_argument("--model", default=VisionConfig.model_path,
                        help="YOLO weights to load")
    parser.add_argument("--imgsz", type=int, default=VisionConfig.imgsz,
                        help="inference resolution; 1280 finds ~25%% more vehicles "
                             "than 960 at night, 960 is faster")
    parser.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu | 0")
    parser.add_argument("--no-half", action="store_true", help="disable FP16 inference")
    parser.add_argument("--augment", action="store_true",
                        help="test-time augmentation: more accurate, ~3x slower")
    parser.add_argument("--calibration", default="speed_calibration.json",
                        help="speed calibration file produced by calibrate_speed.py")
    parser.add_argument("--no-emergency-ai", action="store_true",
                        help="disable CLIP ambulance/police/fire recognition "
                             "(siren-light detection continues unaffected)")
    parser.add_argument("--emergency-backend", default="auto",
                        choices=["auto", "clip", "trained"],
                        help="how to recognise livery. 'auto' (default) uses "
                             "emergency_cls.pt if present and falls back to "
                             "CLIP; 'trained' recognises 71%% of emergency "
                             "vehicles against CLIP's 45%% and runs 7x faster, "
                             "but only knows what it was shown; 'clip' needs no "
                             "training and knows any country's vehicles. Build "
                             "a trained model with train_emergency_classifier.py")
    parser.add_argument("--emergency-model", default="emergency_cls.pt",
                        help="weights to load when --emergency-backend trained")
    parser.add_argument("--clip-model", default="ViT-B/16",
                        help="CLIP model for emergency recognition: "
                             "ViT-B/32 (fastest) | ViT-B/16 | ViT-L/14 (slowest)")
    parser.add_argument("--emergency-interval", type=int, default=5,
                        help="frames between emergency re-checks of each vehicle")
    parser.add_argument("--no-siren-detection", action="store_true",
                        help="disable flashing-beacon detection; with it off and "
                             "--livery-triggers-emergency unset, nothing can raise "
                             "emergency mode automatically")
    parser.add_argument("--livery-triggers-emergency", action="store_true",
                        help="let a recognised ambulance/police car open the lane "
                             "even with its lights off. Off by default: a marked "
                             "vehicle not on a call does not need priority, and "
                             "livery alone is what made 29%% of night traffic "
                             "read as police")
    parser.add_argument("--siren-blue", default="police",
                        help="vehicle type a flashing blue beacon means here")
    parser.add_argument("--siren-red", default="ambulance",
                        help="vehicle type a flashing red beacon means here")
    parser.add_argument("--siren-both", default="police",
                        help="vehicle type alternating red+blue means here")
    parser.add_argument("--free-flow-kmh", type=float, default=50.0,
                        help="speed at which traffic counts as completely free-flowing")
    parser.add_argument("--capacity", type=float, default=20.0,
                        help="weighted vehicle count that counts as a full road")
    parser.add_argument("--arduino-port", default=ARDUINO_PORT,
                        help="serial port the controller board is on")
    parser.add_argument("--arduino-baud", type=int, default=None,
                        help="serial speed. Defaults to whatever the sketch named by "
                             "--board uses -- 9600 for a uno, 115200 for an esp32 -- "
                             "so it only needs setting if you changed Serial.begin()")
    parser.add_argument("--board", default=ARDUINO_BOARD, choices=["esp32", "uno"],
                        help="which sketch is on the other end of --arduino-port. "
                             "'esp32' speaks JSON and is what the dashboard's hardware "
                             "controls drive; 'uno' is the bare-word and seven-field "
                             "format the signal heads use")
    parser.add_argument("--second-port", default=None,
                        help="serial port of the OTHER board, so both can be driven at "
                             "once: the ESP32 for the LED strip, LCD and lane changer, "
                             "the Uno for the signal heads. Whichever kind --board is "
                             "not, this port is assumed to be")
    parser.add_argument("--second-baud", type=int, default=None,
                        help="serial speed of --second-port (default: 9600 for a uno, "
                             "115200 for an esp32)")
    parser.add_argument("--signal-mode", default="merge", choices=["merge", "junction"],
                        help="what the signal governs. 'merge' meters a side road "
                             "joining a main carriageway: RED while the main road is "
                             "HEAVY, GREEN once it is MODERATE or FREE. 'junction' is "
                             "the older behaviour, GREEN whenever traffic is present")
    parser.add_argument("--arduino", action="store_true",
                        help="kept so existing commands and scripts keep working; the "
                             "serial link is already on unless --no-arduino is given")
    parser.add_argument("--no-arduino", action="store_true",
                        help="do not open the serial port. Use this when no board is "
                             "attached, or when another program already holds the port "
                             "-- on Windows only one process can have it open.")
    parser.add_argument("--yellow-seconds", type=float, default=YELLOW_SECONDS,
                        help="how long the light holds YELLOW between green and red")
    parser.add_argument("--min-phase-seconds", type=float, default=MIN_PHASE_SECONDS,
                        help="minimum time GREEN or RED is held before changing again")
    parser.add_argument("--master-csv", default=MASTER_CSV_FILE,
                        help="all-sessions CSV that every run appends to")
    parser.add_argument("--master-excel", default=MASTER_EXCEL_FILE,
                        help="all-sessions Excel workbook")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT, help="HTTP stream port")
    parser.add_argument("--stream-fps", type=float, default=STREAM_FPS,
                        help="annotated preview frames per second; detection still runs "
                             "on every frame. 0 draws every frame (slowest)")
    parser.add_argument("--tcp-port", type=int, default=SERVER_PORT, help="TCP telemetry port")
    parser.add_argument("--csv", default=CSV_FILE, help="telemetry CSV path")
    parser.add_argument("--excel", default=EXCEL_FILE, help="telemetry Excel path")
    parser.add_argument("--no-display", action="store_true", help="run without a preview window")
    parser.add_argument("--camera-width", type=int, default=1920, help="requested capture width")
    parser.add_argument("--camera-height", type=int, default=1080, help="requested capture height")
    parser.add_argument("--loop-video", action="store_true", help="restart video files at the end")
    parser.add_argument("--playback-rate", type=float, default=1.0,
                        help="video file speed multiplier; 0 processes every frame as "
                             "fast as possible. Reported km/h are unaffected either way.")
    parser.add_argument("--no-drop-frames", action="store_true",
                        help="decode every frame of a video file instead of skipping "
                             "to stay in real time (slower than life on a busy machine)")
    parser.add_argument("--no-reconnect", action="store_true",
                        help="do not try to reconnect a camera or stream that drops out")
    parser.add_argument("--exit-on-end", action="store_true",
                        help="stop when a video file finishes instead of holding on "
                             "the last frame and waiting for another source")
    parser.add_argument("--max-upload-mb", type=int, default=MAX_UPLOAD_MB,
                        help="largest video the dashboard may upload")
    return parser.parse_args(argv)


# ══════════════════════════════════════════════
# ARDUINO SERIAL
# ══════════════════════════════════════════════

arduino: "serial.Serial | None" = None

#: Which sketch is on the other end of the wire. Set once by :func:`init_arduino`
#: and read by everything that puts bytes on the port, because the two boards do
#: not understand the same messages.
board_kind: str = ARDUINO_BOARD

#: Every open board link, keyed by sketch kind. Both boards can be attached at
#: once -- the Uno drives the signal heads, the ESP32 the LED strip, LCD and
#: lane changer -- so a sender names the board it is addressing instead of
#: assuming the node owns exactly one port.
board_links: dict[str, "serial.Serial"] = {}

#: Serial speed each sketch's ``Serial.begin()`` uses.
BOARD_BAUD = {"uno": 9600, "esp32": 115200}

#: Held for the length of one exchange with a board. Two threads reach the same
#: port now that the signal line is relayed through the ESP32 -- the capture
#: loop writes telemetry every second while a dashboard command arrives on a
#: Flask thread -- and a command's reply is read line by line, so an unguarded
#: telemetry write could land in the middle of one and be mistaken for it.
board_write_lock = threading.RLock()


def resolve_baud(explicit: int | None, board: str) -> int:
    """Pick the serial speed for a board, preferring an explicit setting.

    Defaulting from the board kind is what stops the commonest silent failure
    on this link: a speed that does not match the sketch delivers bytes which
    decode into nothing, so the board sits on its start-up message looking
    exactly as if the cable were dead.
    """
    return explicit if explicit else BOARD_BAUD.get(board, ARDUINO_BAUDRATE)


def board_link(kind: str) -> "serial.Serial | None":
    """Return the open serial port for one board kind, or ``None``.

    A node configured the old way -- one ``--arduino-port`` and one ``--board``
    -- keeps working: when the registry holds nothing for ``kind`` the legacy
    single-board globals are consulted, so existing setups need no new flags.
    """
    link = board_links.get(kind)
    if link is None and board_kind == kind:
        link = arduino
    if link is None:
        return None
    try:
        return link if link.is_open else None
    except Exception:  # noqa: BLE001  a closed or unplugged port
        return None


def any_board_connected() -> bool:
    """Whether at least one controller board is reachable."""
    return any(board_link(kind) is not None for kind in ("uno", "esp32"))


def init_arduino(port: str, baud: int = ARDUINO_BAUDRATE, enabled: bool = True,
                 board: str = ARDUINO_BOARD) -> None:
    """Open the controller's serial port, degrading gracefully when unavailable."""
    global arduino, board_kind
    board_kind = board
    if not enabled or not SERIAL_AVAILABLE:
        return
    try:
        arduino = serial.Serial(port, baud, timeout=1)
        board_links[board] = arduino
        time.sleep(2)  # the board reboots when the port opens
        print(f"[ARD] Connected on {port} @ {baud} baud ({board})")
    except Exception as exc:  # noqa: BLE001
        print(f"[ARD] Could not open {port}: {exc}")
        print(f"[ARD] Hardware output disabled. Check --arduino-port (now {port}), "
              "and that nothing else -- the Arduino IDE's Serial Monitor above all "
              "-- is holding the port.")
        arduino = None


def init_second_board(port: str | None, baud: int | None, primary: str,
                      enabled: bool = True) -> None:
    """Open the other board's port so both can be driven from one detection.

    The second board is whichever kind ``--board`` is not, because there is one
    sketch of each: the Uno owns the signal heads and the ESP32 owns the strip,
    LCD and lane changer. A failure here is reported and then ignored -- one
    board being absent must not stop the other from working.
    """
    if not port or not enabled or not SERIAL_AVAILABLE:
        return
    kind = "esp32" if primary == "uno" else "uno"
    speed = resolve_baud(baud, kind)
    try:
        link = serial.Serial(port, speed, timeout=1)
        time.sleep(2)  # the board reboots when the port opens
        board_links[kind] = link
        print(f"[ARD] Connected on {port} @ {speed} baud ({kind}, second board)")
    except Exception as exc:  # noqa: BLE001
        print(f"[ARD] Could not open second board on {port}: {exc}")
        print(f"[ARD] The {kind} will not be driven. The {primary} is unaffected.")


def send_line_to_arduino(line: str) -> bool:
    """Send one bare command line to the Arduino, outside the telemetry format.

    The traffic-controller sketch ignores any line that does not carry seven
    comma-separated fields, so these commands pass it by untouched and are
    picked up only by the lane-changer/LED board, which reads whole words.
    That is what lets one serial link carry both without a second port.
    """
    link = board_link("uno")
    if link is None:
        return False
    try:
        with board_write_lock:
            link.write((line.strip() + "\n").encode("ascii", errors="ignore"))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[ARD] Command error: {exc}")
        return False


def send_json_to_board(command: str, payload: dict) -> bool:
    """Send one JSON command to the ESP32 and wait for its acknowledgement.

    This is the sketch's own protocol -- ``{"command": ..., "payload": {...}}``
    in, ``{"type": "ack" | "error" | "status", ...}`` back -- and it is exactly
    what the dashboard POSTs to ``/command``, so the request is forwarded as it
    arrived rather than translated into the Uno's bare command words.

    Waiting for the reply is what makes the dashboard's confirmation mean
    something: the board answers every command it recognises, so a write that
    goes into the void is reported as undelivered instead of as success.
    """
    link = board_link("esp32")
    if link is None:
        return False
    request = json.dumps({"command": command, "payload": payload or {}})
    # The lock spans the reply as well as the request: the relayed signal line
    # shares this port, and a line arriving between the two would be read here
    # as the board's answer.
    with board_write_lock:
        try:
            link.reset_input_buffer()   # discard replies nobody is waiting for
            link.write((request + "\n").encode("utf-8", errors="ignore"))
            link.flush()
        except Exception as exc:  # noqa: BLE001
            print(f"[ARD] Command error: {exc}")
            return False

        deadline = time.monotonic() + BOARD_REPLY_TIMEOUT
        while time.monotonic() < deadline:
            try:
                line = link.readline().decode("utf-8", errors="replace").strip()
            except Exception as exc:  # noqa: BLE001
                print(f"[ARD] Reply error: {exc}")
                return False
            if not line:
                continue
            try:
                reply = json.loads(line)
            except json.JSONDecodeError:
                continue    # the sketch also prints plain-text notes at boot
            if reply.get("type") == "error":
                print(f"[ARD] {command} refused: {reply.get('error', 'no reason given')}")
                return False
            if reply.get("type") in {"ack", "status"}:
                return True
    print(f"[ARD] {command} was not acknowledged within {BOARD_REPLY_TIMEOUT:.0f}s")
    return False


def send_to_arduino(hardware: dict, congestion_index: float, vehicles: int) -> None:
    """Send one command line to the Arduino, ignoring transient write errors.

    Wire format (see ``traffic_controller_arduino.ino``)::

        PHASE,LEVEL,CI,VEHICLES,EMG,LINE1,LINE2

    ``PHASE`` is the lamp to light (GREEN/YELLOW/RED) and is authoritative --
    the Arduino no longer derives the lamp from the congestion level, so the
    signal logic lives in exactly one place. ``EMG`` is 1 while the dedicated
    emergency LED strip should be lit.

    Only the Uno sketch acts on this. It is addressed by name when the Uno has
    a port of its own; otherwise it goes to the ESP32, whose sketch recognises
    a comma-separated line as signal telemetry rather than a command and passes
    it down UART1 to the Uno untouched. That is what lets one USB cable to the
    hub drive every board, and it is why the line is never answered: relaying it
    is silent, so nothing arrives on the port to be mistaken for a command's
    acknowledgement.

    A second's telemetry is dropped rather than queued when a dashboard command
    is mid-exchange on the same port. The next line follows a second later and
    carries the same state, so waiting out a reply here would stall the capture
    loop to deliver something already superseded.
    """
    link = board_link("uno") or board_link("esp32")
    if link is None:
        return
    message = (
        f"{hardware['phase']},"
        f"{hardware['congestion_level']},"
        f"{congestion_index:.3f},"
        f"{vehicles},"
        f"{1 if hardware['emergency_strip'] else 0},"
        f"{hardware['lcd_line1'].strip()},"
        f"{hardware['lcd_line2'].strip()}\n"
    )
    if not board_write_lock.acquire(blocking=False):
        return
    try:
        link.write(message.encode("ascii", errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        print(f"[ARD] Send error: {exc}")
    finally:
        board_write_lock.release()


# ══════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════

#: Gate/lane-divider angle for each signal phase.
SERVO_FOR_PHASE = {
    Phase.GREEN: SERVO_OPEN,
    Phase.YELLOW: SERVO_PARTIAL,
    Phase.RED: SERVO_CLOSED,
}


def decide_hardware_commands(state: LightState, level: str, congestion_index: float,
                             emergency: bool, emergency_type: str = "unknown") -> dict:
    """Map a light state and traffic reading onto every hardware output.

    The signal is demand-responsive, and ``--signal-mode`` decides what that
    means: a merging side road is held RED while the carriageway it feeds is
    HEAVY, while a junction approach shows GREEN whenever there is traffic to
    serve. That decision is made by
    :class:`~traffic_light.TrafficLightController`; this function only renders
    it onto lamps, the servo and the LCD, so it is the same either way.

    An emergency vehicle does **not** seize the signal -- emergency traffic
    proceeds through any indication -- so it drives a dedicated LED strip
    output instead, leaving the phase alone. When the vehicle has been
    identified its type is named on the LCD (``AMBULANCE``, ``POLICE``,
    ``FIRE TRUCK``) rather than a generic warning.
    """
    banner = None
    if emergency:
        name = LCD_NAMES.get(emergency_type, "EMERGENCY")
        banner = f"!! {name} !!"

    line1, line2 = lcd_lines(state, level, congestion_index, emergency_text=banner)

    return {
        "phase": state.phase.value,
        "phase_target": state.target.value,
        "phase_remaining": round(state.remaining, 1),
        "transitioning": state.transitioning,
        "congestion_level": level,
        "servo_angle": SERVO_FOR_PHASE[state.phase],
        "red_led": state.phase is Phase.RED,
        "yellow_led": state.phase is Phase.YELLOW,
        "green_led": state.phase is Phase.GREEN,
        # Dedicated output: a strip that tells drivers to leave space for an
        # approaching emergency vehicle. Independent of the signal phase.
        "emergency_strip": emergency,
        "blue_led": emergency,
        "lcd_line1": line1,
        "lcd_line2": line2,
    }


#: Keyboard keys that force a phase, and the phase each one requests.
MANUAL_PHASES: dict[str, Phase] = {
    "g": Phase.GREEN,
    "y": Phase.YELLOW,
    "r": Phase.RED,
}


# ══════════════════════════════════════════════
# SHARED STATE FOR THE NETWORK SERVERS
# ══════════════════════════════════════════════

stream_lock = threading.Lock()
latest_jpeg: bytes | None = None
latest_telemetry_snapshot: dict = {}
history_buffer: deque = deque(maxlen=HISTORY_MAXLEN)
connected_clients: list[socket.socket] = []

flask_app = Flask(__name__)
_paths = {
    "csv": CSV_FILE,
    "excel": EXCEL_FILE,
    "master_csv": MASTER_CSV_FILE,
    "master_excel": MASTER_EXCEL_FILE,
}

# The live logger, signal controller and video source, published for the HTTP
# handlers once the capture loop has built them. Kept in a dict rather than as
# module globals so the handlers always see the current objects.
_runtime: dict[str, Any] = {"logger": None, "light": None, "source": None}

# Hardware the dashboard drives directly rather than the congestion reading:
# the reversible divider and the LED guide. Held here because the board itself
# cannot be queried mid-stream, so the node remembers what it last commanded
# and republishes it in telemetry.
hardware_lock = threading.Lock()
manual_hardware: dict[str, Any] = {
    "lane_allocation": "BALANCED_3_3",
    "led_mode": "OFF",
    "speed_bump": "Lowered",
    "lcd_message": "",
}

# What the ESP32 was last told, so the once-a-second loop can send it something
# only when it actually changed. The board answers every command and the reply
# is waited for, so re-sending an unchanged state every second would spend the
# link -- and the loop's time budget -- saying nothing new.
_esp32_state: dict[str, Any] = {"emergency": None}


def push_state_to_esp32(hardware: dict, emergency: bool) -> None:
    """Mirror the detected road state onto the ESP32, automatically.

    The ESP32 has no camera. Until now it only ever moved when somebody pressed
    a button on the dashboard, so a detected ambulance lit the Uno's strip and
    left the ESP32's dark. This closes that gap: the same detection that drives
    the signal heads drives the strip, and because it also updates
    ``manual_hardware`` the dashboard shows the board's real state rather than
    the last thing a human clicked.

    Emergency is the only thing sent, and only when it changes. The congestion
    reading is deliberately not pushed here: the ESP32's LCD reports what its
    LED strip is doing and nothing else, while the level, the index and the
    signal countdown belong on the Uno's LCD, which already carries them in the
    seven-field line.
    """
    if board_link("esp32") is None:
        return
    if emergency == _esp32_state["emergency"]:
        return

    # ACTIVATE_EMERGENCY turns the strip red, DEACTIVATE_EMERGENCY clears it,
    # and the sketch's LCD follows the strip on its own. What those commands do
    # on the board is its business -- this only decides when they are sent,
    # which is the one thing the board cannot work out for itself.
    command = "ACTIVATE_EMERGENCY" if emergency else "DEACTIVATE_EMERGENCY"
    if send_json_to_board(command, {}):
        _esp32_state["emergency"] = emergency
        with hardware_lock:
            manual_hardware["led_mode"] = "AMBULANCE" if emergency else "OFF"


#: Dashboard command → the word the lane-changer/LED board understands.
LANE_ALLOCATION_LINES = {
    "BALANCED": ("LANE_BALANCED", "BALANCED_3_3"),
    "FORWARD_4": ("LANE_FORWARD_4", "FORWARD_4_OPPOSITE_2"),
    "OPPOSITE_4": ("LANE_OPPOSITE_4", "FORWARD_2_OPPOSITE_4"),
}

#: The divider's three positions in physical order, left to right. The
#: dashboard drives the barrier with two arrows that step along this list, so a
#: press means "one lane that way" rather than "jump to the far end" -- which is
#: how a divider is actually thought about, and keeps the operator from swinging
#: it across two lanes of traffic with one click.
LANE_DIVIDER_ORDER = ("OPPOSITE_4", "BALANCED", "FORWARD_4")

#: Where each reported state sits in that order, so the next position can be
#: worked out from the one the node last commanded.
LANE_DIVIDER_INDEX = {
    LANE_ALLOCATION_LINES[key][1]: index
    for index, key in enumerate(LANE_DIVIDER_ORDER)
}

LED_MODE_LINES = {
    "POLICE": "POLICE",
    "AMBULANCE": "AMBULANCE",
    "SPEED": "SPEED",
    "OFF": "NORMAL",
}


def _manual_hardware_snapshot() -> dict[str, Any]:
    """Return the dashboard-driven hardware state, plus whether a board is on it."""
    with hardware_lock:
        snapshot = dict(manual_hardware)
    snapshot["arduino_connected"] = any_board_connected()
    return snapshot


def _register_runtime(logger_obj, light_obj, source_obj=None) -> None:
    """Expose the logger, light controller and video source to the HTTP layer."""
    _runtime["logger"] = logger_obj
    _runtime["light"] = light_obj
    _runtime["source"] = source_obj


@flask_app.route("/video_feed")
def video_feed() -> Response:
    """Serve the annotated camera feed as an MJPEG stream."""
    def generate():
        """Yield JPEG frames as they are produced by the capture loop."""
        while True:
            with stream_lock:
                frame_bytes = latest_jpeg
            if frame_bytes is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.033)  # cap at roughly 30 fps
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/telemetry")
def telemetry_feed():
    """Return the most recent telemetry snapshot as JSON.

    The source block is read live rather than taken from the snapshot: a
    paused video produces no new frames, so a paused flag carried on the
    snapshot could never arrive.
    """
    with stream_lock:
        payload = dict(latest_telemetry_snapshot)
    source = _runtime["source"]
    if source is not None:
        payload["source"] = source.status()

    # The hardware block is normally built by the capture loop, but the board
    # is attached whether or not frames are flowing. Filling it in here means a
    # node waiting for a camera still reports its wiring honestly, instead of
    # looking to the dashboard like a node with nothing plugged in.
    #
    # The commanded values overwrite rather than fill gaps: the loop's copy is
    # only as fresh as the last frame, so a command issued between frames would
    # otherwise read back as the old position for up to a frame's worth of time.
    hardware = dict(payload.get("hardware") or {})
    hardware.update(_manual_hardware_snapshot())
    payload["hardware"] = hardware
    return jsonify(payload)


@flask_app.route("/history")
def history_feed():
    """Return the rolling buffer of recorded telemetry rows as JSON."""
    with stream_lock:
        return jsonify(list(history_buffer))


@flask_app.route("/export")
def export_excel():
    """Build the session workbook from the current rows, then serve it.

    Built here rather than kept fresh by a background timer: a rebuild is a
    full regeneration, and paying for one every 20s in the process running the
    detector cost far more than the occasional download it was anticipating.
    """
    logger_obj = _runtime["logger"]
    if logger_obj is not None:
        logger_obj.write_excel()
    path = Path(_paths["excel"]).resolve()
    if not path.exists():
        return jsonify({"error": "no workbook yet"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@flask_app.route("/export.csv")
def export_csv():
    """Serve the raw telemetry CSV for download."""
    path = Path(_paths["csv"]).resolve()
    if not path.exists():
        return jsonify({"error": "no csv yet"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@flask_app.route("/export/all")
def export_master_excel():
    """Build the all-sessions workbook from the master CSV, then serve it.

    The archive only grows, so this is the expensive one; it is built on the
    request that actually wants it rather than every minute in the background.
    """
    logger_obj = _runtime["logger"]
    if logger_obj is not None:
        logger_obj.write_master_excel()
    path = Path(_paths["master_excel"]).resolve()
    if not path.exists():
        return jsonify({"error": "no master workbook yet"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@flask_app.route("/export/all.csv")
def export_master_csv():
    """Serve the all-sessions CSV."""
    path = Path(_paths["master_csv"]).resolve()
    if not path.exists():
        return jsonify({"error": "no master csv yet"}), 404
    return send_file(path, as_attachment=True, download_name=path.name)


@flask_app.route("/sessions")
def sessions_index():
    """List every recorded session with its summary statistics."""
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify([])
    return jsonify(logger_obj.session_index())


@flask_app.route("/records")
def records_feed():
    """Return telemetry rows across *all* sessions, not just the live hour.

    ``/history`` only holds the rolling in-memory buffer; this reads the master
    record from disk, which is what lets the dashboard chart past sessions.
    Pass ``?limit=N`` to cap the number of rows returned, and ``?session=ID``
    to restrict it to one session.
    """
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify([])
    try:
        limit = int(request.args.get("limit", 20000))
    except ValueError:
        limit = 20000
    rows = logger_obj.read_master_rows(limit)
    wanted = request.args.get("session")
    if wanted:
        rows = [row for row in rows if str(row.get("Session")) == wanted]
    return jsonify(rows)


@flask_app.route("/record/status")
def record_status():
    """Report whether a named recording is currently running."""
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify({"recording": False, "rows": 0, "name": "", "id": ""})
    return jsonify(logger_obj.session_info())


@flask_app.route("/record/start", methods=["POST", "GET"])
def record_start():
    """Close the current session and begin a new, named recording."""
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify({"error": "logger not ready"}), 503
    payload = request.get_json(silent=True) or {}
    name = payload.get("name") or request.args.get("name") or ""
    return jsonify(logger_obj.start_session(name))


@flask_app.route("/record/stop", methods=["POST", "GET"])
def record_stop():
    """Finalise the named recording and return what it captured."""
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify({"error": "logger not ready"}), 503
    return jsonify(logger_obj.stop_session())


@flask_app.route("/record/save", methods=["POST", "GET"])
def record_save():
    """Force both workbooks to be written right now."""
    logger_obj = _runtime["logger"]
    if logger_obj is None:
        return jsonify({"error": "logger not ready"}), 503
    return jsonify(logger_obj.save_now())


@flask_app.route("/light", methods=["GET", "POST"])
def light_control():
    """Read the signal state, or force a phase.

    ``POST {"phase": "GREEN"}`` forces a phase; ``{"phase": null}`` or
    ``{"phase": "AUTO"}`` hands control back to the congestion reading. A
    forced phase is still reached through the normal yellow interval.
    """
    controller = _runtime["light"]
    if controller is None:
        return jsonify({"error": "light not ready"}), 503

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        wanted = payload.get("phase") or request.args.get("phase")
        if wanted in (None, "", "AUTO", "auto"):
            controller.request_phase(None)
        else:
            try:
                controller.request_phase(Phase(str(wanted).upper()))
            except ValueError:
                return jsonify({"error": f"unknown phase {wanted!r}"}), 400

    with stream_lock:
        snapshot = dict(latest_telemetry_snapshot)
    return jsonify({
        "phase": snapshot.get("light_phase", controller.phase.value),
        "target": snapshot.get("light_target", controller.phase.value),
        "remaining": snapshot.get("light_remaining", 0.0),
        "transitioning": snapshot.get("light_transitioning", False),
        "manual": controller.forced.value if controller.forced else None,
        "config": controller.config.to_dict(),
    })


# ── Hardware commands ────────────────────────────────────────────────────────
# The dashboard has no serial port of its own: this node owns the Arduino, so
# manual hardware controls arrive here and are forwarded down the same link.
# That keeps one owner of the port and lets the dashboard sit on another
# machine, which is the whole point of splitting the two.


@flask_app.route("/command", methods=["POST"])
def hardware_command():
    """Forward one dashboard hardware command to the Arduino.

    ``POST {"command": "SET_LANE_ALLOCATION", "payload": {"allocation": "FORWARD_4"}}``
    ``POST {"command": "MOVE_LANE_DIVIDER", "payload": {"direction": "RIGHT"}}``

    Returns the acknowledgement the dashboard expects, including the resulting
    lane allocation so the page can show confirmed state rather than a guess.
    """
    body = request.get_json(silent=True) or {}
    command = str(body.get("command", "")).strip().upper()
    payload = body.get("payload") or {}
    if not command:
        return jsonify({"success": False, "error": "no command given"}), 400

    connected = any_board_connected()
    line: str | None = None
    at_limit = False

    # What actually goes on the wire, which is not always what arrived: the
    # divider's relative move is resolved to a position here, so both boards
    # keep the one absolute lane command they already understand.
    board_command: str | None = command
    board_payload: dict = payload

    with hardware_lock:
        if command == "SET_LANE_ALLOCATION":
            allocation = str(payload.get("allocation", "")).strip().upper()
            if allocation not in LANE_ALLOCATION_LINES:
                return jsonify({
                    "success": False,
                    "error": f"unknown allocation {allocation!r}",
                }), 400
            line, state = LANE_ALLOCATION_LINES[allocation]
            manual_hardware["lane_allocation"] = state

        elif command == "MOVE_LANE_DIVIDER":
            direction = str(payload.get("direction", "")).strip().upper()
            if direction not in ("LEFT", "RIGHT"):
                return jsonify({
                    "success": False,
                    "error": f"unknown direction {direction!r}",
                }), 400
            # An unrecognised held state means the node has not commanded the
            # divider yet, or was restarted; the centre is where it is parked at
            # power-on, so stepping from there is the honest guess.
            index = LANE_DIVIDER_INDEX.get(
                str(manual_hardware["lane_allocation"]).upper(),
                LANE_DIVIDER_ORDER.index("BALANCED"),
            )
            step = 1 if direction == "RIGHT" else -1
            target = index + step
            if not 0 <= target < len(LANE_DIVIDER_ORDER):
                # Already against the far lane. Re-commanding the same position
                # would replay the servo's move-and-blink routine and spend the
                # link to change nothing, so nothing is sent and the reply says
                # why rather than claiming a move.
                at_limit = True
                board_command = None
            else:
                allocation = LANE_DIVIDER_ORDER[target]
                line, state = LANE_ALLOCATION_LINES[allocation]
                manual_hardware["lane_allocation"] = state
                board_command = "SET_LANE_ALLOCATION"
                board_payload = {"allocation": allocation}

        elif command == "LED_MODE":
            mode = str(payload.get("mode", "OFF")).strip().upper()
            if mode not in LED_MODE_LINES:
                return jsonify({
                    "success": False,
                    "error": f"unknown LED mode {mode!r}",
                }), 400
            line = LED_MODE_LINES[mode]
            manual_hardware["led_mode"] = mode

        elif command == "LED_COLOR":
            colour = str(payload.get("color", "")).strip().upper()
            line = colour or None
            manual_hardware["led_mode"] = colour

        elif command in ("RAISE_SPEED_BUMP", "LOWER_SPEED_BUMP"):
            raised = command == "RAISE_SPEED_BUMP"
            line = "BUMP_UP" if raised else "BUMP_DOWN"
            manual_hardware["speed_bump"] = "Raised" if raised else "Lowered"

        elif command == "LCD_MESSAGE":
            message = str(payload.get("message", "")).strip()
            line = f"LCD:{message}" if message else None
            manual_hardware["lcd_message"] = message

        elif command == "ACTIVATE_EMERGENCY":
            line = "AMBULANCE"
            manual_hardware["led_mode"] = "AMBULANCE"

        elif command == "DEACTIVATE_EMERGENCY":
            line = "NORMAL"
            manual_hardware["led_mode"] = "OFF"

        else:
            return jsonify({
                "success": False,
                "error": f"unknown command {command!r}",
            }), 400

        state_now = dict(manual_hardware)

    # The command is validated and recorded above either way; only the wire
    # format differs. The ESP32 is given the request as it arrived, because that
    # is already its protocol; the Uno gets the bare word translated for it.
    # These are all strip, LCD, divider and bump commands, which belong to the
    # ESP32, so it is preferred whenever it is attached. With only a Uno on the
    # node the bare word goes to it instead, exactly as before.
    if board_command is None:
        delivered = False
    elif board_link("esp32") is not None:
        delivered = send_json_to_board(board_command, board_payload)
    else:
        delivered = send_line_to_arduino(line) if line else False

    # A command with no board attached still updates what the dashboard shows;
    # saying so plainly beats a silent success or a hard failure during a demo.
    return jsonify({
        "success": True,
        "command": command,
        "payload": payload,
        "ack": f"ACK:{command}",
        "sent": line,
        "delivered": delivered,
        "at_limit": at_limit,
        "arduino_connected": connected,
        **state_now,
    })


# ── Video source ─────────────────────────────────────────────────────────────
# The camera is not the only thing this node can watch. A recorded video or a
# roadside camera's URL goes through exactly the same pipeline, and can replace
# the live camera without restarting anything.


def _payload() -> dict:
    """Merge JSON body, form fields and query string into one dict.

    Uploads arrive as multipart form data and the keyboard-free curl case as a
    query string, so a handler that only read JSON would work from the
    dashboard and nowhere else.
    """
    body = request.get_json(silent=True) or {}
    merged = {**request.args.to_dict(), **request.form.to_dict(), **body}
    return merged


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
    """Coerce a JSON/query value to a bool, or ``default`` when absent."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _require_source():
    """Return the live video source, or ``None`` before the loop has one."""
    return _runtime["source"]


@flask_app.route("/source", methods=["GET", "POST"])
def source_control():
    """Report the current video source, or switch to another one.

    ``POST {"source": "..."}`` accepts anything the ``--source`` flag does: a
    camera index or name, a path to a video file, or a stream URL. The new
    source is opened before the old one is dropped, so a URL that turns out to
    be unreachable leaves the running feed untouched and reports why.
    """
    source = _require_source()
    if source is None:
        return jsonify({"error": "no capture loop running"}), 503

    if request.method == "POST":
        payload = _payload()
        wanted = payload.get("source") or payload.get("spec")
        if not wanted:
            return jsonify({"error": "give a 'source': a camera, a file or a URL"}), 400
        try:
            return jsonify(source.switch(str(wanted), loop=_as_bool(payload.get("loop"))))
        except SourceError as exc:
            return jsonify({"error": str(exc)}), 400

    return jsonify(source.status())


@flask_app.route("/sources")
def sources_index():
    """List everything this node could watch: cameras, uploads, and the current one."""
    source = _require_source()
    return jsonify({
        "current": source.status() if source is not None else {},
        "cameras": available_cameras(),
        "uploads": list_uploads(),
    })


@flask_app.route("/source/upload", methods=["POST"])
def source_upload():
    """Accept a video file and, unless told otherwise, start playing it.

    The file is decoded once before the pipeline is pointed at it, so a
    corrupt or unsupported upload fails here with a readable message instead
    of becoming a black feed.

    Each distinct video is kept once. Sending one that is already stored --
    the usual case when the same clip is analysed again -- answers with the
    existing file and ``"reused": true`` instead of writing a second copy.
    """
    source = _require_source()
    if source is None:
        return jsonify({"error": "no capture loop running"}), 503

    upload = request.files.get("file") or request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify({"error": "no file in the request (field name: 'file')"}), 400

    target, reused = store_upload(upload.save, upload.filename)

    try:
        info = probe(str(target))
    except SourceError as exc:
        # Only bin the file when this upload is what created it. A reused copy
        # was already here and may be somebody else's source.
        if not reused:
            target.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400

    payload = _payload()
    if not _as_bool(payload.get("play"), True):
        return jsonify({"stored": target.name, "reused": reused,
                        "info": info, "playing": False})

    try:
        status = source.switch(str(target), loop=_as_bool(payload.get("loop")))
    except SourceError as exc:
        return jsonify({"error": str(exc), "stored": target.name}), 400
    return jsonify({"stored": target.name, "reused": reused,
                    "info": info, "playing": True, "source": status})


@flask_app.route("/source/upload/<path:name>", methods=["DELETE", "POST"])
def source_upload_delete(name: str):
    """Delete one uploaded video.

    Only the filename is honoured, so this cannot reach outside the upload
    directory however the path is dressed up.
    """
    wanted = Path(name.replace("\\", "/")).name
    source = _require_source()
    if (source is not None and source.spec is not None and source.is_file
            and Path(source.spec.target).name == wanted):
        return jsonify({"error": f"{wanted} is playing right now — "
                                 "switch to another source first"}), 409
    if not delete_upload(wanted):
        return jsonify({"error": f"no uploaded video named {wanted!r}"}), 404
    return jsonify({"deleted": wanted, "uploads": list_uploads()})


@flask_app.route("/source/control", methods=["POST"])
def source_playback():
    """Drive playback of a video file: pause, seek, loop, or change speed.

    Actions: ``pause`` · ``resume`` · ``toggle`` · ``restart`` · ``seek``
    (with ``fraction``, ``seconds`` or ``delta``) · ``loop`` (``value``) ·
    ``rate`` (``value``) · ``drop`` (``value``).
    """
    source = _require_source()
    if source is None:
        return jsonify({"error": "no capture loop running"}), 503

    payload = _payload()
    action = str(payload.get("action", "")).strip().lower()

    try:
        if action == "pause":
            return jsonify(source.pause(True))
        if action == "resume":
            return jsonify(source.pause(False))
        if action == "toggle":
            return jsonify(source.toggle_pause())
        if action == "restart":
            return jsonify(source.restart())
        if action == "seek":
            return jsonify(source.seek(
                fraction=_as_float(payload.get("fraction")),
                seconds=_as_float(payload.get("seconds")),
                delta_seconds=_as_float(payload.get("delta")),
            ))
        if action == "loop":
            return jsonify(source.set_loop(bool(_as_bool(payload.get("value"), True))))
        if action == "rate":
            return jsonify(source.set_rate(_as_float(payload.get("value")) or 0.0))
        if action == "drop":
            return jsonify(source.set_drop_frames(bool(_as_bool(payload.get("value"), True))))
    except SourceError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"error": f"unknown action {action!r}"}), 400


def _as_float(value: Any) -> float | None:
    """Coerce a JSON/query value to a float, or ``None`` when absent/unparseable."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@flask_app.errorhandler(413)
def upload_too_large(_error):
    """Answer an oversized upload with JSON, like every other failure here."""
    limit = flask_app.config.get("MAX_CONTENT_LENGTH") or 0
    return jsonify({"error": f"file is larger than the {limit // (1024 * 1024)} MB "
                             "upload limit (raise it with --max-upload-mb)"}), 413


@flask_app.route("/health")
def health():
    """Report node liveness and basic pipeline stats for monitoring."""
    with stream_lock:
        snapshot = dict(latest_telemetry_snapshot)
    logger_obj = _runtime["logger"]
    source = _runtime["source"]
    return jsonify({
        "status": "ok" if snapshot else "starting",
        "fps": snapshot.get("fps", 0),
        "speed_source": snapshot.get("speed_source"),
        "samples_logged": snapshot.get("samples_logged", 0),
        "arduino": any_board_connected(),
        "light_phase": snapshot.get("light_phase"),
        "recording": bool(logger_obj.recording) if logger_obj else False,
        "session": snapshot.get("session", {}),
        "source": source.status() if source is not None else {},
    })


@flask_app.route("/")
def stream_index() -> str:
    """Simple landing page listing the available endpoints."""
    return ("Smart Traffic stream is running.<br>"
            "Video: /video_feed | Telemetry: /telemetry | History: /history<br>"
            "This session: /export | /export.csv<br>"
            "All sessions: /export/all | /export/all.csv | /sessions | /records<br>"
            "Recording: /record/status | /record/start | /record/stop | /record/save<br>"
            "Source: /source | /sources | /source/upload | /source/control<br>"
            "Signal: /light | Health: /health")


def start_tcp_server(port: int) -> None:
    """Accept Raspberry Pi clients that want the telemetry JSON stream."""
    def serve() -> None:
        """Background accept loop for the TCP telemetry socket."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((SERVER_HOST, port))
        except OSError as exc:
            print(f"[NET] TCP port {port} unavailable: {exc}")
            return
        server.listen(5)
        print(f"[NET] TCP server listening on port {port}")
        while True:
            client, address = server.accept()
            connected_clients.append(client)
            print(f"[NET] RPi connected: {address}")

    threading.Thread(target=serve, daemon=True).start()


def start_http_server(port: int) -> None:
    """Serve the video/telemetry HTTP endpoints on a background thread."""
    def serve() -> None:
        """Run Flask without the reloader, since this is not the main thread."""
        try:
            flask_app.run(host=STREAM_HOST, port=port, threaded=True, use_reloader=False)
        except OSError as exc:
            print(f"[HTTP] Could not bind port {port}: {exc}")

    threading.Thread(target=serve, daemon=True).start()


def broadcast_tcp(telemetry: dict) -> None:
    """Push one telemetry frame to every connected TCP client."""
    if not connected_clients:
        return
    wire = (json.dumps(telemetry) + "\n").encode("utf-8")
    dead = []
    for client in connected_clients:
        try:
            client.sendall(wire)
        except Exception:  # noqa: BLE001
            dead.append(client)
    for client in dead:
        connected_clients.remove(client)
        print("[NET] Client disconnected.")


# ══════════════════════════════════════════════
# CAPTURE
# ══════════════════════════════════════════════
# Everything about *where frames come from* — cameras, files, stream URLs, the
# media clock that keeps a recorded video's km/h honest — lives in
# ``video_source.py``, so that this file only ever asks for the next frame.


def build_config(args: argparse.Namespace) -> VisionConfig:
    """Translate command-line options into a :class:`VisionConfig`."""
    return VisionConfig(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=not args.no_half,
        augment=args.augment,
        calibration_path=args.calibration,
        free_flow_speed_kmh=args.free_flow_kmh,
        max_capacity_pce=args.capacity,
        emergency_appearance=not args.no_emergency_ai,
        emergency_backend=args.emergency_backend,
        emergency_model_path=args.emergency_model,
        emergency_clip_model=args.clip_model,
        emergency_check_interval=args.emergency_interval,
        emergency_require_siren=not args.livery_triggers_emergency,
        siren_detection=not args.no_siren_detection,
        siren_blue_type=args.siren_blue,
        siren_red_type=args.siren_red,
        siren_both_type=args.siren_both,
    )


def open_initial_source(args: argparse.Namespace) -> VideoSource:
    """Open the source named on the command line, or start with none.

    A missing camera used to end the process. It no longer needs to: the node
    can come up empty and be pointed at an uploaded video or a camera URL from
    the dashboard, which is a great deal more useful than a server that
    refused to start because a webcam was unplugged.
    """
    options = {
        "loop": args.loop_video,
        "playback_rate": args.playback_rate,
        "drop_frames": not args.no_drop_frames,
        "reconnect": not args.no_reconnect,
    }
    try:
        return VideoSource(args.source, args.camera_width, args.camera_height, **options)
    except SourceError as exc:
        print(f"[CV] {exc}")
        print("[CV] Starting with no source. Upload a video or set a camera URL "
              "on the dashboard's Live Camera page (or POST to /source).")
        return VideoSource(None, args.camera_width, args.camera_height, **options)


def install_interrupt_handlers() -> None:
    """Make Ctrl-Break stop the node as tidily as Ctrl-C does.

    Windows terminates a Python process outright on Ctrl-Break — no exception,
    no ``finally`` — which would abandon the workbook mid-session, leave the
    vehicles still in frame unlogged and hold the serial port open. Anything
    that stops this node from outside its console (a launcher, a service
    wrapper, a scheduled task) has only Ctrl-Break to send it, so it has to
    mean the same thing as the key a person presses.
    """
    def interrupt(_signum, _frame) -> None:
        """Turn the signal into the exception the capture loop already handles."""
        raise KeyboardInterrupt

    for name in ("SIGBREAK", "SIGTERM"):
        handled = getattr(signal, name, None)
        if handled is not None:
            try:
                signal.signal(handled, interrupt)
            except (OSError, ValueError):   # not supported on this platform
                pass


#: Whether ``timeBeginPeriod`` is currently raised, so it is lowered once.
_TIMER_RAISED = False


def raise_timer_resolution() -> bool:
    """Ask Windows for a 1 ms scheduler tick, and report whether it agreed.

    ``cv2.waitKey(1)`` is documented as a 1 ms wait, but it sleeps on the
    system timer, whose default period is 15.625 ms — so it returns a whole
    tick later. That wait sits in the middle of the capture loop and is paid
    on every frame the window is polled: measured here, 15.58 ms with the
    default timer against 1.96 ms with the tick raised, or about 13.6 ms per
    frame handed back to detection.

    This is the same call media players make for the same reason. It costs a
    little idle power, so it is released again on shutdown.
    """
    global _TIMER_RAISED
    if _TIMER_RAISED or not sys.platform.startswith("win"):
        return False
    try:
        # 0 is TIMERR_NOERROR; anything else means the period was refused.
        if ctypes.windll.winmm.timeBeginPeriod(1) != 0:
            return False
    except (AttributeError, OSError):   # not Windows, or winmm unavailable
        return False
    _TIMER_RAISED = True
    return True


def restore_timer_resolution() -> None:
    """Hand back the 1 ms tick raised by ``raise_timer_resolution``."""
    global _TIMER_RAISED
    if not _TIMER_RAISED:
        return
    try:
        ctypes.windll.winmm.timeEndPeriod(1)
    except (AttributeError, OSError):
        pass
    _TIMER_RAISED = False


def waiting_frame(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    """Render the card shown in place of video when there is no source.

    OpenCV's built-in fonts are ASCII only, so this text has to be.
    """
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    cv2.putText(canvas, "SMART TRAFFIC - NO VIDEO SOURCE", (60, height // 2 - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 200, 255), 2)
    cv2.putText(canvas, message, (60, height // 2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
    cv2.putText(canvas, "Upload a video or set a camera URL on the Live Camera page.",
                (60, height // 2 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
    return canvas


def publish_frame(frame: np.ndarray, telemetry: dict | None = None) -> None:
    """Encode a frame for the MJPEG stream, optionally with fresh telemetry."""
    global latest_jpeg, latest_telemetry_snapshot

    # INTER_LINEAR rather than the INTER_AREA default: this frame is only ever
    # looked at by a person, and on a 1920x1080 source the area filter cost
    # 6.9 ms of a 47 ms budget — more than the JPEG encode it feeds, and more
    # than the overlays. The difference is invisible at a 0.67x downscale.
    ok, buffer = cv2.imencode(
        ".jpg", fit_to_display(frame, 1280, 720, interpolation=cv2.INTER_LINEAR),
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return
    with stream_lock:
        latest_jpeg = buffer.tobytes()
        if telemetry is not None:
            latest_telemetry_snapshot = telemetry


def publish_telemetry(telemetry: dict) -> None:
    """Refresh the telemetry snapshot without redrawing or re-encoding a frame.

    The numbers are cheap and the picture is not, so on frames where the stream
    is not due the dashboard still gets a fully current reading.
    """
    global latest_telemetry_snapshot
    with stream_lock:
        latest_telemetry_snapshot = telemetry


def show_and_poll(frame: np.ndarray | None, show_window: bool,
                  redraw: bool = True, poll: bool = True) -> int:
    """Draw the preview window when enabled and return the key pressed.

    ``redraw`` says whether ``frame`` differs from what is already on screen.
    The overlay is only rebuilt at ``--stream-fps``, so a machine detecting
    faster than that used to resize and blit an identical picture on most
    iterations.

    ``poll`` gates ``cv2.waitKey``, which is the expensive half. It is
    documented as a 1 ms wait, but it sleeps on the Windows timer and so
    returns a whole system tick later -- measured at 15.5 ms on this machine,
    paid serially in the middle of the capture loop on *every* frame. Polling
    only on the frames that redraw puts key handling on the same ~15 Hz
    cadence, far quicker than anyone can press a key, and hands the rest of
    that time back to detection.

    Both are left on whenever no frame is arriving: a paused or finished video
    still has to answer the keyboard, and the wait is also what stops the loop
    spinning against a source that has nothing to give.

    ``imshow`` only paints when a ``waitKey`` follows it, so a redraw always
    implies a poll -- never pass ``redraw=True, poll=False``.
    """
    if not show_window:
        return 255
    if frame is not None and redraw:
        cv2.imshow("Smart Traffic Broadcast Server",
                   fit_to_display(frame, DISPLAY_WIDTH, DISPLAY_HEIGHT,
                                  interpolation=cv2.INTER_LINEAR))
    if not poll:
        return 255
    return cv2.waitKey(1) & 0xFF


#: Playback keys, and what each one does to the source. All are no-ops on a
#: live camera, which reports as much rather than failing silently.
PLAYBACK_KEYS: dict[int, str] = {
    32: "pause/resume", ord("l"): "loop", ord("["): "back 5s",
    ord("]"): "forward 5s", ord("0"): "restart",
}


def handle_playback_key(key: int, source: VideoSource) -> bool:
    """Apply a video-playback key. Returns True when the key was one of these."""
    if key not in PLAYBACK_KEYS:
        return False

    try:
        if key == 32:
            source.toggle_pause()
        elif key == ord("l"):
            source.set_loop(not source.status()["loop"])
        elif key == ord("["):
            source.seek(delta_seconds=-5.0)
        elif key == ord("]"):
            source.seek(delta_seconds=5.0)
        else:
            source.restart()
    except SourceError as exc:
        print(f"[CV] {exc}")
        return True

    print(f"[CV] {PLAYBACK_KEYS[key]}: {source.hud_text()}")
    return True


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

def main() -> int:
    """Run the capture → detection → control → logging loop."""
    args = parse_args()

    if args.list_cameras:
        # Before anything heavy: no model load, no serial port, no servers.
        return list_cameras(args.camera_width, args.camera_height)

    _paths["csv"], _paths["excel"] = args.csv, args.excel
    flask_app.config["MAX_CONTENT_LENGTH"] = args.max_upload_mb * 1024 * 1024
    install_interrupt_handlers()
    fine_timer = raise_timer_resolution()

    print("=" * 66)
    print(" SMART TRAFFIC BROADCAST SERVER".center(66))
    print("=" * 66)
    if fine_timer:
        print("[CV] Scheduler tick raised to 1 ms — the preview's key poll costs "
              "~2 ms a frame instead of ~15 ms.")

    init_arduino(args.arduino_port, resolve_baud(args.arduino_baud, args.board),
                 enabled=not args.no_arduino, board=args.board)
    init_second_board(args.second_port, args.second_baud, args.board,
                      enabled=not args.no_arduino)
    if not args.no_arduino and board_link("uno") is None:
        if board_link("esp32") is not None:
            print("[ARD] No Uno on a port of its own, so the signal line is being "
                  "relayed through the ESP32's UART1. If the heads stay dark, that "
                  "wire (ESP32 GPIO33 -> Uno D7) is where to look.")
        else:
            print("[ARD] No board on the link, so the signal heads are not being "
                  "driven and their LCD will sit on its start-up message. Check "
                  "--arduino-port and --board.")
    start_tcp_server(args.tcp_port)
    start_http_server(args.stream_port)
    print(f"[HTTP] Video     http://<your-ip>:{args.stream_port}/video_feed")
    print(f"[HTTP] Telemetry http://<your-ip>:{args.stream_port}/telemetry")

    config = build_config(args)
    pipeline = TrafficVisionPipeline(config)
    logger = TelemetryLogger(args.csv, args.excel,
                             master_csv=args.master_csv, master_excel=args.master_excel)
    # A merging side road is metered the other way round from a junction: the
    # signal is on the ramp, so it closes while the carriageway it feeds is
    # heavy and opens once there is room.
    green_levels = (MERGE_GREEN_LEVELS if args.signal_mode == "merge"
                    else DEFAULT_GREEN_LEVELS)
    light = TrafficLightController(TrafficLightConfig(
        yellow_seconds=args.yellow_seconds,
        min_phase_seconds=args.min_phase_seconds,
        green_levels=green_levels,
    ))
    source = open_initial_source(args)
    _paths["master_csv"] = str(Path(logger.master_csv_path).resolve())
    _paths["master_excel"] = str(Path(logger.master_excel_path).resolve())
    _register_runtime(logger, light, source)
    if args.signal_mode == "merge":
        print(f"[SIG] Merge metering: RED while the main road is HEAVY, GREEN once it"
              f" is MODERATE or FREE, {args.yellow_seconds:.0f}s YELLOW between")
    else:
        print(f"[SIG] Junction signal: GREEN while traffic is present, RED on an empty"
              f" road, {args.yellow_seconds:.0f}s YELLOW between")
    print(f"[HTTP] Source    http://<your-ip>:{args.stream_port}/sources"
          "   (upload a video or point at a camera URL from the dashboard)")

    show_window = not args.no_display
    if show_window:
        cv2.namedWindow("Smart Traffic Broadcast Server", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Traffic Broadcast Server", DISPLAY_WIDTH, DISPLAY_HEIGHT)
        print("[CV] Keys: q=quit e=emergency g/y/r=manual a=auto x=excel c=recalibrate p=snapshot")
        print("[CV] Video files: space=pause [/]=seek 5s l=loop 0=restart")

    emergency_mode = False       # manual emergency toggle
    auto_mode = True             # False while a manual g/y/r override is held
    last_log_time = 0.0
    telemetry: dict = {}
    annotated = None             # last frame drawn, redisplayed while idle
    #: Whether the preview holds something not yet on screen. Starts True so
    #: the placeholder is drawn once; set again each time the overlay is
    #: redrawn, so the window is only repainted when the picture has changed.
    preview_dirty = True
    idle_reported = False
    last_publish = 0.0
    #: Seconds between annotated preview frames; 0 redraws on every frame.
    stream_interval = 1.0 / args.stream_fps if args.stream_fps > 0 else 0.0

    # Publish a placeholder straight away so the dashboard shows "no source"
    # rather than "stream offline" while a source is being chosen.
    placeholder = waiting_frame("Waiting for a video source.")
    publish_frame(placeholder)

    try:
        while True:
            # A jump in the picture — a new source, a looped file wrapping
            # round, someone scrubbing the timeline — means every track in
            # flight belongs to a scene that no longer exists.
            if source.pop_discontinuity():
                jumped_to = source.media_time
                logger.log_vehicles(pipeline.reset(jumped_to))
                light.rebase(jumped_to)
                last_log_time = 0.0

            read = source.read()
            frame = read.frame
            # No frame means paused, reconnecting, or played out. The node stays
            # up either way — the whole point is that another source can be
            # loaded without restarting — so the loop carries on to the display
            # and the keyboard rather than skipping them.
            idle = frame is None

            if idle:
                if read.ended and not idle_reported and source.spec is not None:
                    idle_reported = True
                    if args.exit_on_end:
                        print(f"[CV] {source.spec.label} finished.")
                        break
                    print(f"[CV] {source.spec.label} finished — holding on the "
                          "last frame. Load another source from the dashboard, "
                          "or press q to stop.")
            else:
                idle_reported = False

                # The scene's clock, which for a video file is the video's
                # own — that is what keeps its km/h independent of how fast
                # this machine happens to be.
                now = read.time
                result = pipeline.process(frame, now, wall=time.time())

                # ── Decide hardware state ────────────────────────────────────────
                # Two different questions, and they get two different answers.
                # ``auto_priority`` is what the junction acts on and needs a
                # beacon actually flashing; ``auto_emergency`` is what gets
                # reported and includes a vehicle recognised by its livery
                # alone — an ambulance stopped at a crash scene in daylight,
                # whose beacon blooms white and so cannot be read as a colour.
                auto_priority = len(result.priority_ids) > 0
                auto_emergency = len(result.emergency_ids) > 0
                effective_emergency = emergency_mode or auto_priority
                reported_emergency = emergency_mode or auto_emergency

                # The signal follows demand, in whichever direction --signal-mode
                # calls for, with five seconds of yellow in between. An emergency
                # does not seize it — emergency traffic proceeds through any
                # indication — it lights the dedicated strip instead.
                light_state = light.update(result.congestion_level, now)
                hardware = decide_hardware_commands(
                    light_state, result.congestion_level, result.congestion_index,
                    effective_emergency,
                    result.primary_emergency_type if auto_priority else "unknown",
                )

                emergency_source = ("auto" if auto_emergency
                                    else "manual" if emergency_mode else "none")

                telemetry = {
                    "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                    "timestamp_ms": int(now * 1000),
                    "congestion_index": result.congestion_index,
                    "congestion_level": hardware["congestion_level"],
                    # The signal itself, so the dashboard can mirror the real lamp.
                    "light_phase": hardware["phase"],
                    "light_target": hardware["phase_target"],
                    "light_remaining": hardware["phase_remaining"],
                    "light_transitioning": hardware["transitioning"],
                    "light_manual": light.forced.value if light.forced else None,
                    "total_vehicles": result.total_vehicles,
                    "vehicle_counts": result.vehicle_counts,
                    "avg_speed_kmh": result.avg_speed_kmh,
                    "max_speed_kmh": result.max_speed_kmh,
                    "moving_vehicles": result.moving_vehicles,
                    "stopped_vehicles": result.stopped_vehicles,
                    "unique_vehicles": result.unique_vehicles_session,
                    # Detected, then excluded by the region of interest. A class
                    # that rides in a lane the region misses would otherwise look
                    # like a class the detector cannot find.
                    "outside_roi": result.outside_roi,
                    "outside_roi_counts": result.outside_roi_counts,
                    "weighted_density": result.weighted_density,
                    # Recognised, and therefore worth showing the operator.
                    "emergency_vehicle_detected": reported_emergency,
                    "emergency_vehicle_count": len(result.emergency_ids),
                    # Running a beacon, and therefore actually being given the
                    # lane. Always a subset of the above.
                    "emergency_priority_active": effective_emergency,
                    "emergency_priority_count": len(result.priority_ids),
                    "emergency_source": emergency_source,
                    "emergency_types": result.emergency_types,
                    "emergency_primary_type": (result.primary_emergency_type
                                               if auto_emergency else "unknown"),
                    "emergency_recognition": pipeline.appearance.status,
                    "siren_detection": pipeline.siren.status,
                    "siren_colours": sorted({det.siren_colour
                                             for det in result.detections
                                             if det.is_emergency
                                             and det.siren_colour != "none"}),
                    # Marked vehicles that are not running a beacon: shown so
                    # the operator can see them, deliberately given no priority.
                    "marked_no_siren": len(result.suspected_ids),
                    # A beacon flashes 1-4 times a second; sampling slower than
                    # that can miss one, and saying so is more honest than
                    # reporting a clear road.
                    "siren_sample_hz": result.siren_sample_hz,
                    "siren_undersampled": result.siren_undersampled,
                    # Vehicles in shot too briefly for the flash test to have
                    # judged them at all — "not enough footage to say" rather
                    # than "nothing there".
                    "siren_pending": result.siren_pending,
                    "siren_tracked": result.siren_tracked,
                    # False when the calibrated region did not fit this picture
                    # and the whole frame is being monitored instead.
                    "roi_active": result.roi_active,
                    "fps": round(result.fps, 1),
                    "inference_ms": result.inference_ms,
                    "emergency_ms": result.emergency_ms,
                    "speed_source": result.speed_source,
                    "speed_accuracy": pipeline.ground.accuracy_note,
                    "samples_logged": logger.row_count,
                    "detections": [
                        {
                            "id": det.track_id,
                            "class": det.label,
                            "confidence": round(det.confidence, 3),
                            "speed_kmh": det.speed_kmh,
                            "bbox": list(det.bbox),
                            "emergency": det.is_emergency,
                            "emergency_confidence": det.emergency_confidence,
                            "emergency_type": det.emergency_type,
                            "emergency_evidence": det.emergency_evidence,
                            "emergency_suspected": det.emergency_suspected,
                            "siren_colour": det.siren_colour,
                            "siren_rate_hz": det.siren_rate_hz,
                        }
                        for det in result.detections
                    ],
                    "hardware": {
                        "servo_angle": hardware["servo_angle"],
                        "red_led": hardware["red_led"],
                        "yellow_led": hardware["yellow_led"],
                        "green_led": hardware["green_led"],
                        "blue_led": hardware["blue_led"],
                        "emergency_strip": hardware["emergency_strip"],
                        "lcd_line1": hardware["lcd_line1"],
                        "lcd_line2": hardware["lcd_line2"],
                        # Set from the dashboard, not from the road reading, so
                        # it is echoed back rather than recomputed.
                        **_manual_hardware_snapshot(),
                    },
                    "session": logger.session_info(),
                }

                # ── Once-per-second outputs ──────────────────────────────────────
                if now - last_log_time >= 1.0:
                    logger.log(build_log_row(
                        result, hardware["servo_angle"], effective_emergency,
                        level_override=hardware["congestion_level"],
                        light_phase=hardware["phase"],
                    ))
                    logger.log_vehicles(pipeline.drain_completed())
                    send_to_arduino(hardware, result.congestion_index, result.total_vehicles)
                    push_state_to_esp32(hardware, effective_emergency)
                    broadcast_tcp(telemetry)
                    with stream_lock:
                        history_buffer.append({
                            key: value for key, value in telemetry.items()
                            if key != "detections"  # keep the history payload small
                        })
                    last_log_time = now

                # ── Overlay + publish ────────────────────────────────────────────
                if effective_emergency:
                    kind = (result.primary_emergency_type.upper() if auto_emergency
                            else "MANUAL")
                    mode_label = f"EMERGENCY ({kind})"
                elif auto_mode:
                    mode_label = "AUTO"
                else:
                    mode_label = f"MANUAL: {hardware['congestion_level']}"
                # Drawing the overlays, downscaling and JPEG-encoding cost about
                # 12 ms per frame together — a quarter of the budget spent on a
                # picture for a human, competing with the detection it depicts.
                # Nobody can see more than ~15 fps of traffic footage, so the
                # preview is capped there while detection keeps every frame.
                # The telemetry numbers are cheap and stay fully live.
                if stream_interval <= 0.0 or now - last_publish >= stream_interval:
                    playback = source.status()
                    hint = ("q:quit e:emergency g/y/r:manual a:auto x:excel "
                            "c:recalibrate p:snapshot")
                    if source.is_file:
                        hint += " | space:pause [/]:seek l:loop"
                    annotated = annotate(
                        frame, result,
                        mode_label=mode_label,
                        servo_angle=hardware["servo_angle"],
                        emergency=effective_emergency,
                        roi=pipeline.ground.roi_polygon(frame.shape),
                        hint=hint,
                        source_label=source.hud_text(),
                        progress=playback["progress"] if source.is_file else None,
                    )
                    publish_frame(annotated, telemetry)
                    last_publish = now
                    preview_dirty = True
                else:
                    publish_telemetry(telemetry)

            # Display and keyboard happen whether or not a frame arrived, so a
            # paused or finished video is still fully controllable. While
            # frames *are* flowing they ride the preview's cadence rather than
            # the detector's -- see show_and_poll for why that is worth doing.
            serve_window = preview_dirty or idle
            key = show_and_poll(annotated if annotated is not None else placeholder,
                                show_window, redraw=preview_dirty, poll=serve_window)
            preview_dirty = False

            # ── Keyboard ─────────────────────────────────────────────────────
            if key == ord("q"):
                break

            if key == ord("e"):
                emergency_mode = not emergency_mode
                if emergency_mode:
                    auto_mode = False
                print(f"[SIM] Emergency {'ON' if emergency_mode else 'OFF'}")

            elif key in (ord("g"), ord("y"), ord("r")):
                # Forced phases still run through the yellow interval, so a
                # manual override can never flip green to red instantly.
                phase = MANUAL_PHASES[chr(key)]
                light.request_phase(phase)
                auto_mode = False
                print(f"[MAN] Forcing {phase.value} (yellow interval still applies)")

            elif key == ord("a"):
                emergency_mode, auto_mode = False, True
                light.request_phase(None)
                print("[AUTO] Returned to automatic detection mode")

            elif key == ord("x"):
                print("[LOG] Writing Excel workbook...")
                print(f"[LOG] {'Written' if logger.write_excel() else 'Failed'}: {args.excel}")

            elif key == ord("c"):
                ground = pipeline.reload_calibration(args.calibration)
                print(f"[CAL] Reloaded: {ground.accuracy_note}")

            elif key == ord("p"):
                if annotated is None:
                    print("[CV] Nothing to snapshot yet.")
                else:
                    name = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(name, annotated)
                    print(f"[CV] Saved {name}")

            else:
                handle_playback_key(key, source)

            if idle:
                time.sleep(0.03)   # nothing to process; do not spin the CPU

            # Pacing a video file to its real frame rate happens inside
            # VideoSource, alongside the media clock it has to agree with.

    except KeyboardInterrupt:
        print("\n[CV] Interrupted.")
    finally:
        # finalize() also retires vehicles still in frame, so the last few
        # never go missing from the per-vehicle speed log.
        logger.log_vehicles(pipeline.finalize(source.media_time))
        logger.close()
        for kind in ("uno", "esp32"):
            link = board_link(kind)
            if link is not None:
                link.close()
                print(f"[ARD] Serial port closed ({kind}).")
        source.release()
        cv2.destroyAllWindows()
        restore_timer_resolution()
        print("[CV] Broadcast server stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
