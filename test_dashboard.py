"""
test_dashboard.py — verification suite for the Streamlit dashboard
==================================================================

Runs every page headlessly through Streamlit's AppTest harness and checks that
they render without exceptions, in both modes:

* **offline** — the broadcast server is not running, so the dashboard must fall
  back to simulated data instead of erroring;
* **live** — telemetry/history HTTP calls are stubbed with realistic km/h
  payloads from ``broadcast_server.py``, so the pages must show real values.

Run with::

    python test_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

APP_DIR = Path(__file__).resolve().parent / "SmartTrafficSystem"
sys.path.insert(0, str(APP_DIR))

from config.project import TEAM, roster_is_complete  # noqa: E402

PAGES = [
    "app.py",
    "pages/1_Operations_Dashboard.py",
    "pages/2_Live_Camera.py",
    "pages/3_Traffic_Analytics.py",
    "pages/4_Emergency_Control.py",
    "pages/5_Hardware_Monitor.py",
    "pages/6_System_Logs.py",
    "pages/7_Settings.py",
    "pages/8_About.py",
]

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record the outcome of one assertion."""
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append((name, detail))
        print(f"  [FAIL] {name}  {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# FAKE LIVE TELEMETRY — exactly the shape broadcast_server.py publishes
# ═══════════════════════════════════════════════════════════════════════════

TELEMETRY = {
    "timestamp": "12:00:00",
    "timestamp_ms": 1_700_000_000_000,
    "congestion_index": 0.42,
    # An ambulance is in frame, so the broadcast server reports EMERGENCY and
    # drives the hardware accordingly — the payload is kept self-consistent.
    "congestion_level": "EMERGENCY",
    "total_vehicles": 7,
    "vehicle_counts": {"car": 5, "motorcycle": 1, "bus": 0, "truck": 1},
    "avg_speed_kmh": 34.6,
    "max_speed_kmh": 51.2,
    "moving_vehicles": 6,
    "stopped_vehicles": 1,
    "unique_vehicles": 118,
    "weighted_density": 8.5,
    "emergency_vehicle_detected": True,
    "emergency_vehicle_count": 1,
    "emergency_source": "auto",
    "emergency_types": ["ambulance"],
    "emergency_primary_type": "ambulance",
    "emergency_recognition": "CLIP ViT-B/16 on cuda (ambulance/police/fire)",
    "siren_detection": "siren-light detection on (blue->police, red->ambulance)",
    "siren_colours": ["red"],
    "marked_no_siren": 1,
    "siren_sample_hz": 27.4,
    "siren_undersampled": False,
    "fps": 27.4,
    "inference_ms": 35.7,
    "emergency_ms": 4.2,
    "speed_source": "homography",
    "speed_accuracy": "homography calibration (±5%)",
    "samples_logged": 240,
    # Served live by /telemetry so a paused video still reports itself.
    "source": None,   # filled in below, once SOURCE is defined
    "detections": [
        {"id": 12, "class": "car", "confidence": 0.91, "speed_kmh": 38.2,
         "bbox": [100, 200, 220, 300], "emergency": False, "emergency_confidence": 0.0,
         "emergency_type": "ordinary", "emergency_evidence": "none",
         "emergency_suspected": False, "siren_colour": "none", "siren_rate_hz": 0.0},
        {"id": 13, "class": "truck", "confidence": 0.84, "speed_kmh": 44.0,
         "bbox": [400, 210, 620, 360], "emergency": True, "emergency_confidence": 0.97,
         "emergency_type": "ambulance", "emergency_evidence": "siren-red+livery",
         "emergency_suspected": False, "siren_colour": "red", "siren_rate_hz": 2.1},
        # Recognised livery, beacon off: shown, but given no priority.
        {"id": 14, "class": "car", "confidence": 0.79, "speed_kmh": 31.0,
         "bbox": [700, 220, 820, 320], "emergency": False, "emergency_confidence": 0.81,
         "emergency_type": "police", "emergency_evidence": "livery-no-siren",
         "emergency_suspected": True, "siren_colour": "none", "siren_rate_hz": 0.0},
    ],
    "hardware": {"servo_angle": 90, "red_led": False, "yellow_led": False,
                 "green_led": True, "blue_led": True, "emergency_strip": True,
                 "lcd_line1": "LIGHT: GREEN    ",
                 "lcd_line2": "!! AMBULANCE !! ",
                 # Set from the dashboard and echoed back by the node, so it is
                 # independent of the phase servo above.
                 "lane_allocation": "FORWARD_4_OPPOSITE_2",
                 "led_mode": "AMBULANCE",
                 "speed_bump": "Lowered",
                 "arduino_connected": True},
    # The signal is demand-responsive: heavy traffic is being served, so GREEN.
    "light_phase": "GREEN",
    "light_target": "GREEN",
    "light_remaining": 0.0,
    "light_transitioning": False,
    "light_manual": None,
    "session": {"id": "20260101_120000", "name": "auto", "recording": False,
                "rows": 240, "elapsed_s": 240.0},
}

RECORD_STATUS = {"id": "20260101_120000", "name": "auto", "recording": False,
                 "rows": 240, "started_ms": 1767268800000, "elapsed_s": 240.0}

# What the node is watching. A video file rather than the camera, because that
# is the case with something to render: a position, a length and transport
# controls that a live camera has no use for.
SOURCE = {
    "kind": "file", "label": "bridge_cam.mp4", "spec": "bridge_cam.mp4",
    "target": "C:/GDP/uploads/bridge_cam.mp4", "is_file": True, "live": False,
    "width": 1920, "height": 1080, "native_fps": 30.0, "frame_count": 5400,
    "duration_s": 180.0, "duration_text": "3:00", "position_s": 45.0,
    "position_text": "0:45", "progress": 0.25, "paused": False, "finished": False,
    "loop": False, "playback_rate": 1.0, "drop_frames": True,
    "frames_read": 1350, "frames_skipped": 12, "reconnects": 0, "switches": 1,
    "media_time": 1767268800.0, "media_ahead_s": 0.0,
}

CATALOGUE = {
    "current": SOURCE,
    "cameras": [{"index": 0, "name": "HP Wide Vision HD Camera", "spec": "0"},
                {"index": 1, "name": "DroidCam Source 3", "spec": "1"}],
    "uploads": [{"name": "bridge_cam.mp4", "spec": "C:/GDP/uploads/bridge_cam.mp4",
                 "size_mb": 84.2, "uploaded_at": "2026-01-01 11:58",
                 "modified": 1767268680.0}],
}

TELEMETRY["source"] = SOURCE

SESSIONS = [
    {"Session": "20251231_090000", "Session Name": "yesterday", "Date": "2025-12-31",
     "Start": "09:00:00", "End": "09:05:00", "Duration (s)": 300, "Samples": 300,
     "Peak Vehicles": 9, "Mean Speed (km/h)": 31.0, "Peak Speed (km/h)": 52.0,
     "Mean Congestion": 0.42, "Peak Congestion": 0.81, "FREE": 100, "MODERATE": 120,
     "HEAVY": 80, "Emergency Samples": 3},
    {"Session": "20260101_120000", "Session Name": "auto", "Date": "2026-01-01",
     "Start": "12:00:00", "End": "12:04:00", "Duration (s)": 240, "Samples": 240,
     "Peak Vehicles": 12, "Mean Speed (km/h)": 34.6, "Peak Speed (km/h)": 51.0,
     "Mean Congestion": 0.55, "Peak Congestion": 0.90, "FREE": 40, "MODERATE": 100,
     "HEAVY": 100, "Emergency Samples": 5},
]


def _records(count: int = 120) -> list[dict]:
    """Build a plausible all-sessions archive in the raw log-row schema."""
    rows = []
    for index in range(count):
        busy = index >= count // 2
        rows.append({
            "Session": "20260101_120000" if busy else "20251231_090000",
            "Session Name": "auto" if busy else "yesterday",
            "Date": "2026-01-01" if busy else "2025-12-31",
            "Timestamp": f"{17 if busy else 9:02d}:00:{index % 60:02d}",
            "Congestion Index": 0.8 if busy else 0.15,
            "Level": "HEAVY" if busy else "FREE",
            "Light Phase": "GREEN" if busy else "RED",
            "Total Vehicles": 12 if busy else 2,
            "Cars": 8 if busy else 2, "Motorcycles": 1 if busy else 0,
            "Buses": 2 if busy else 0, "Trucks": 1 if busy else 0,
            "Avg Speed (km/h)": 21.0 if busy else 48.0,
            "Max Speed (km/h)": 34.0 if busy else 57.0,
            "Moving": 6 if busy else 2, "Stopped": 6 if busy else 0,
            "Weighted Density (PCE)": 18.0 if busy else 2.0,
            "Servo Angle (deg)": 90 if busy else 10,
            "Emergency": "YES" if busy and index % 20 == 0 else "no",
            "Emergency Type": "ambulance" if busy and index % 20 == 0 else "none",
            "Emergency Vehicles": 1 if busy and index % 20 == 0 else 0,
            "Unique Vehicles": 30 if busy else 4,
            "FPS": 23.0, "Speed Source": "homography",
        })
    return rows


def _history(count: int = 90) -> list[dict]:
    """Build a plausible telemetry history spanning a few minutes."""
    rows = []
    for index in range(count):
        row = dict(TELEMETRY)
        row.pop("detections", None)
        row["timestamp_ms"] = TELEMETRY["timestamp_ms"] + index * 1000
        row["avg_speed_kmh"] = 20.0 + (index % 30)
        row["total_vehicles"] = 3 + (index % 9)
        row["congestion_index"] = 0.2 + (index % 50) / 100.0
        row["congestion_level"] = ["FREE", "MODERATE", "HEAVY"][index % 3]
        rows.append(row)
    return rows


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload: object, content: bytes = b"") -> None:
        """Store the JSON payload and raw content this response should return."""
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        """No-op: the fake response is always a success."""

    def json(self) -> object:
        """Return the stored payload."""
        return self._payload


def fake_get(url: str, *args, **kwargs) -> FakeResponse:
    """Route stubbed HTTP calls to the right fake payload."""
    if url.endswith("/telemetry"):
        return FakeResponse(TELEMETRY)
    if url.endswith("/history"):
        return FakeResponse(_history())
    if url.endswith("/record/status"):
        return FakeResponse(RECORD_STATUS)
    if url.endswith("/sessions"):
        return FakeResponse(SESSIONS)
    if url.endswith("/records"):
        # The dashboard filters by session via a query parameter; mirror that
        # so a scoped request returns only the matching rows.
        wanted = (kwargs.get("params") or {}).get("session")
        rows = _records()
        if wanted:
            rows = [row for row in rows if row["Session"] == wanted]
        return FakeResponse(rows)
    if url.endswith("/light"):
        return FakeResponse({"phase": "GREEN", "target": "GREEN", "remaining": 0.0,
                             "transitioning": False, "manual": None,
                             "config": {"yellow_seconds": 5.0}})
    if url.endswith("/export/all"):
        return FakeResponse({}, content=b"PK\x03\x04fake-master-xlsx")
    if url.endswith("/export/all.csv"):
        return FakeResponse({}, content=b"Session,Date\n")
    if url.endswith("/export"):
        return FakeResponse({}, content=b"PK\x03\x04fake-xlsx")
    if url.endswith("/sources"):
        return FakeResponse(CATALOGUE)
    if url.endswith("/source"):
        return FakeResponse(SOURCE)
    raise AssertionError(f"unexpected URL requested: {url}")


def fake_post(url: str, *args, **kwargs) -> FakeResponse:
    """Route stubbed recording/signal commands."""
    if url.endswith("/record/start"):
        return FakeResponse({**RECORD_STATUS, "recording": True, "name": "test run",
                             "rows": 0})
    if url.endswith("/record/stop"):
        return FakeResponse({**RECORD_STATUS, "recording": False, "rows": 240})
    if url.endswith("/record/save"):
        return FakeResponse({"session": True, "master": True})
    if url.endswith("/light"):
        return FakeResponse({"phase": "GREEN", "target": "GREEN", "remaining": 0.0,
                             "transitioning": False, "manual": None, "config": {}})
    if url.endswith("/source/control"):
        action = (kwargs.get("json") or {}).get("action")
        return FakeResponse({**SOURCE, "paused": action == "pause"})
    if url.endswith("/source/upload"):
        return FakeResponse({"stored": "uploaded.mp4", "playing": True, "source": SOURCE})
    if url.endswith("/source"):
        return FakeResponse(SOURCE)
    if url.endswith("/command"):
        # The node owns the divider's position, so a step comes back as the
        # place it landed -- named the way the controller names it, not the way
        # the page shows it.
        body = kwargs.get("json") or {}
        direction = (body.get("payload") or {}).get("direction")
        return FakeResponse({
            "success": True, "command": body.get("command"), "delivered": True,
            "at_limit": False,
            "lane_allocation": ("FORWARD_4_OPPOSITE_2" if direction == "RIGHT"
                                else "BALANCED_3_3"),
        })
    raise AssertionError(f"unexpected URL posted: {url}")


def offline_get(url: str, *args, **kwargs):
    """Simulate the broadcast server being unreachable."""
    import requests

    raise requests.exceptions.ConnectionError("stream offline")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE RUNS
# ═══════════════════════════════════════════════════════════════════════════


def run_pages(label: str, patched_get) -> dict[str, object]:
    """Run every page with the given HTTP stub and report exceptions.

    Sub-pages are reached with ``switch_page`` from the main app rather than
    loaded directly, because ``st.page_link`` needs the multipage registry that
    only exists once the entrypoint script has run.
    """
    from streamlit.testing.v1 import AppTest

    results: dict[str, object] = {}
    for page in PAGES:
        post_stub = fake_post if patched_get is fake_get else patched_get
        with mock.patch("requests.get", side_effect=patched_get),                 mock.patch("requests.post", side_effect=post_stub):
            app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=90)
            # Every page's auto-refresh loop calls st.rerun() forever, which
            # the harness would sit through until it timed out.
            for flag in ("auto_refresh", "camera_auto_refresh", "emergency_auto_refresh",
                         "hardware_auto_refresh", "log_auto_refresh"):
                app.session_state[flag] = False
            try:
                if page != "app.py":
                    app.switch_page(page)
                app.run()
                errors = [str(e.value) for e in app.exception]
                check(f"{label}: {page} renders", not errors, "; ".join(errors)[:250])
                results[page] = app
            except Exception as exc:  # noqa: BLE001
                check(f"{label}: {page} renders", False, f"{type(exc).__name__}: {exc}")
                results[page] = None
    return results


def widget_labels(app: object, kind: str) -> list[str]:
    """Return every label of one widget type, e.g. ``button`` or ``toggle``."""
    try:
        return [str(widget.label) for widget in getattr(app, kind)]
    except Exception:  # noqa: BLE001 - the page may have none of that kind
        return []


def page_text(app: object) -> str:
    """Concatenate every readable element on a rendered page."""
    parts: list[str] = []
    for collection in ("markdown", "caption", "info", "success", "warning", "error",
                       "text", "header", "subheader", "title"):
        try:
            parts += [str(element.value) for element in getattr(app, collection)]
        except Exception:  # noqa: BLE001 - not every element type exists on every page
            continue
    try:
        for metric in app.get("metric"):
            parts += [str(metric.label), str(metric.value), str(metric.delta)]
    except Exception:  # noqa: BLE001
        pass
    return " ".join(parts)


def test_offline_mode() -> None:
    """Every page must render with the broadcast server unreachable."""
    print("\n[1] Dashboard with the broadcast server OFFLINE")
    apps = run_pages("offline", offline_get)

    dashboard = apps.get("pages/1_Operations_Dashboard.py")
    if dashboard is not None:
        check("offline dashboard still shows a speed in km/h",
              "km/h" in page_text(dashboard))

    # The front page carries no live figures, so the one thing that must
    # survive the node being down is the project's own identity.
    home = apps.get("app.py")
    if home is not None:
        text = page_text(home)
        check("offline front page still names the project",
              "Adaptive Signal Optimization" in text, text[:150])
        check("offline front page still lists the group",
              all(member.name in text for member in TEAM if member.filled),
              text[:250])

    camera = apps.get("pages/2_Live_Camera.py")
    if camera is not None:
        text = page_text(camera)
        check("offline camera page warns the stream is down",
              "offline" in text.lower(), text[:150])


def test_live_mode() -> None:
    """Every page must render live telemetry and display km/h speeds."""
    print("\n[2] Dashboard with the broadcast server LIVE")
    apps = run_pages("live", fake_get)

    dashboard = apps.get("pages/1_Operations_Dashboard.py")
    if dashboard is not None:
        text = page_text(dashboard)
        check("live speed reaches the dashboard", "34.6 km/h" in text,
              "expected the telemetry's avg_speed_kmh")
        check("no pixel-per-second label anywhere", "px/s" not in text)
        check("live vehicle count reaches the dashboard", "7" in text)

    home = apps.get("app.py")
    if home is not None:
        text = page_text(home)
        check("the front page reports the node is connected",
              "Live system connected" in text, text[:200])
        # A roster half-filled with placeholders in front of an examiner is
        # the failure this page exists to prevent, so it has to be said out
        # loud while any slot is empty -- and stop being said once none is.
        check("an unfilled roster is called out, a complete one is not",
              ("still to fill in" in text) != roster_is_complete(),
              f"complete={roster_is_complete()}")

    camera = apps.get("pages/2_Live_Camera.py")
    if camera is not None:
        text = page_text(camera)
        check("camera page reports km/h", "34.6 km/h" in text, text[-200:])
        check("camera page shows peak speed", "51" in text)
        check("camera page shows the calibration mode", "homography" in text)
        check("camera page has no px/s", "px/s" not in text)
        check("camera page reports pipeline FPS", "27.4" in text)
        check("camera page names the emergency vehicle type",
              "Ambulance" in text, text[-250:])
        check("camera page reports the recogniser", "CLIP ViT-B/16" in text)
        # The source is no longer fixed at start-up, so the page has to say
        # what is being watched and how far through it is.
        check("camera page names the video source", "bridge_cam.mp4" in text,
              text[-250:])
        check("camera page shows where the video has got to",
              "0:45" in text and "3:00" in text, text[-250:])
        # Button and toggle labels are widgets, not readable text, so these
        # are read off the widget list rather than the rendered page.
        buttons = widget_labels(camera, "button")
        check("camera page offers transport controls for a video",
              any("Pause" in label for label in buttons)
              and any("Restart" in label for label in buttons), str(buttons))
        check("camera page offers a way to change source",
              any("Change source" in label
                  for label in widget_labels(camera, "toggle")),
              str(widget_labels(camera, "toggle")))

    emergency = apps.get("pages/4_Emergency_Control.py")
    if emergency is not None:
        text = page_text(emergency)
        check("emergency page identifies the vehicle type", "Ambulance" in text,
              text[-250:])
        check("emergency page shows live detection badge", "Live Detection" in text)
        check("emergency page shows which cue fired",
              "siren-red+livery" in text, text[-250:])
        check("emergency page reports the beacon colour and rate",
              "red" in text and "2.1 Hz" in text, text[-250:])
        check("emergency page separates marked vehicles running no siren",
              "no priority given" in text, text[-250:])
        check("emergency page mirrors real hardware state",
              "CLEAR RIGHT LANE" in text or "Blue" in text, text[-250:])

    analytics = apps.get("pages/3_Traffic_Analytics.py")
    if analytics is not None:
        text = page_text(analytics)
        check("analytics reports live session data",
              "live broadcast" in text.lower(), text[:200])
        check("analytics shows an average speed KPI", "km/h" in text)


def test_service_layer() -> None:
    """The status and analytics services must map telemetry fields correctly."""
    print("\n[3] Service layer field mapping")
    from backend.services.analytics_service import AnalyticsService
    from backend.services.emergency_control_service import EmergencyControlService
    from backend.services.system_status_service import SystemStatusService

    with mock.patch("requests.get", side_effect=fake_get),             mock.patch("requests.post", side_effect=fake_post):
        status = SystemStatusService()
        metrics = status.get_dashboard_metrics()
        check("live flag set", status.is_live_stream_connected)
        check("speed mapped from avg_speed_kmh",
              abs(metrics.traffic.average_speed_kmh - 34.6) < 0.01,
              f"{metrics.traffic.average_speed_kmh}")
        check("fps mapped from telemetry", abs(metrics.traffic.fps - 27.4) < 0.01,
              f"{metrics.traffic.fps}")
        check("vehicle count mapped", metrics.traffic.vehicle_count == 7)
        check("density mapped from congestion level",
              metrics.traffic.traffic_density == "Critical",
              metrics.traffic.traffic_density)
        check("raw telemetry exposed to pages",
              status.last_telemetry.get("speed_source") == "homography")
        check("emergency status raised", metrics.emergency_status.value == "Active")

        emergency_service = EmergencyControlService()
        state = emergency_service.get_emergency_state()
        check("emergency service uses live telemetry", emergency_service.is_live)
        check("emergency vehicle type identified",
              state.detection.vehicle_type == "Ambulance", state.detection.vehicle_type)
        check("emergency confidence from the flagged vehicle",
              abs(state.detection.confidence - 0.97) < 0.01, f"{state.detection.confidence}")
        check("hardware signal mirrors the blue emergency LED",
              state.hardware.current_signal == "Blue", state.hardware.current_signal)
        check("LCD text mirrored from the CV node",
              "AMBULANCE" in state.hardware.lcd_status, state.hardware.lcd_status)

        # The divider is shown from whichever end the button commanded. Driving
        # real hardware, that is the node -- and it must not be derived from the
        # phase servo, which swings on every light change.
        from config.settings import Settings
        live_settings = Settings()
        live_settings.simulation_mode = False
        live_state = EmergencyControlService(settings=live_settings).get_emergency_state()
        check("lane allocation read from the node, not the phase servo",
              live_state.hardware.lane_allocation == "Forward: 4 / Opposite: 2",
              live_state.hardware.lane_allocation)

        # In simulation the button never reaches the node, so the node's copy
        # must not overwrite what the operator just set -- otherwise the card
        # snaps back the moment the page redraws.
        # Simulation is asked for explicitly: the shipped default is live mode,
        # where these buttons really do reach the board over the node.
        from backend.simulation.emergency_simulator import EmergencySimulator
        sim_settings = Settings()
        sim_settings.simulation_mode = True
        sim_service = EmergencyControlService(settings=sim_settings)
        sim_service.set_lane_allocation("OPPOSITE_4")
        check("a simulated allocation survives the next redraw",
              sim_service.get_emergency_state().hardware.lane_allocation
              == "Forward: 2 / Opposite: 4",
              sim_service.get_emergency_state().hardware.lane_allocation)
        check("a simulated allocation is held across service instances",
              EmergencyControlService(settings=sim_settings)
              .get_emergency_state().hardware.lane_allocation
              == "Forward: 2 / Opposite: 4")
        check("an unknown simulated allocation is refused",
              EmergencySimulator().set_lane_allocation("SIDEWAYS") is False)
        EmergencyControlService(settings=sim_settings).set_lane_allocation("BALANCED")

        # The page drives the divider with two arrows, one lane per press, so
        # what matters is which neighbour a step lands on -- and that an arrow
        # at the outermost lane says so instead of reporting a move that never
        # happened, which would leave the card and the barrier disagreeing.
        step = sim_service.move_lane_divider("LEFT")
        check("a step left gives the opposite direction the extra lane",
              step["lane_allocation"] == "Forward: 2 / Opposite: 4", step["lane_allocation"])
        check("a step is shown on the card it moved",
              sim_service.get_emergency_state().hardware.lane_allocation
              == "Forward: 2 / Opposite: 4")
        check("a step past the outermost lane is refused, not silently applied",
              sim_service.move_lane_divider("LEFT")["at_limit"] is True)
        sim_service.move_lane_divider("RIGHT")
        check("stepping back right crosses the even split to the far lane",
              sim_service.move_lane_divider("RIGHT")["lane_allocation"]
              == "Forward: 4 / Opposite: 2")
        check("a step past the far lane is refused too",
              sim_service.move_lane_divider("RIGHT")["at_limit"] is True)
        check("an unknown direction never moves the divider",
              sim_service.move_lane_divider("SIDEWAYS")["success"] is False)
        EmergencyControlService(settings=sim_settings).set_lane_allocation("BALANCED")

        # Driving hardware the step is resolved by the node, which answers in
        # the controller's vocabulary; the page shows it to an operator.
        live_service = EmergencyControlService(settings=live_settings)
        check("a live divider step is reported in the operator's words",
              live_service.move_lane_divider("right")["lane_allocation"]
              == "Forward: 4 / Opposite: 2",
              live_service.move_lane_divider("right")["lane_allocation"])
        check("an unknown live direction never reaches the node",
              live_service.move_lane_divider("UP")["success"] is False)

        analytics = AnalyticsService()
        data = analytics.get_analytics(hours=24, interval_minutes=15)
        check("analytics uses live history", analytics.is_live_data)
        check("speed timeline built from live km/h", data.speed_timeline is not None
              and "avg_speed_kmh" in data.speed_timeline.columns)
        check("speed timeline has values",
              data.speed_timeline["avg_speed_kmh"].max() > 0,
              f"max {data.speed_timeline['avg_speed_kmh'].max()}")
        check("csv export includes the speed section",
              "speed_timeline_kmh" in data.to_export_csv())

    with mock.patch("requests.get", side_effect=offline_get),             mock.patch("requests.post", side_effect=offline_get):
        status2 = SystemStatusService()
        metrics2 = status2.get_dashboard_metrics()
        check("offline falls back to simulation", not status2.is_live_stream_connected)
        check("offline still returns a metrics object", metrics2.traffic.vehicle_count >= 0)

        analytics2 = AnalyticsService()
        data2 = analytics2.get_analytics(hours=6, interval_minutes=15)
        check("offline analytics simulated", not analytics2.is_live_data)
        check("simulated data still has a speed timeline",
              data2.speed_timeline is not None and not data2.speed_timeline.empty)

        emergency2 = EmergencyControlService()
        state2 = emergency2.get_emergency_state()
        check("offline emergency service falls back", not emergency2.is_live)
        check("offline emergency service still returns state",
              state2.detection is not None and state2.hardware is not None)


def test_settings_roundtrip() -> None:
    """Stream host/port must persist through the settings file."""
    print("\n[4] Settings persistence")
    from config.settings import SETTINGS_PATH, Settings

    backup = SETTINGS_PATH.read_text() if SETTINGS_PATH.exists() else None
    try:
        settings = Settings()
        settings.stream_host = "127.0.0.1"
        settings.stream_port = 9001
        settings.save()
        reloaded = Settings.load()
        check("stream host persisted", reloaded.stream_host == "127.0.0.1")
        check("stream port persisted", reloaded.stream_port == 9001)

        # A settings file from an older build must not break startup.
        data = json.loads(SETTINGS_PATH.read_text())
        data["a_removed_option"] = True
        SETTINGS_PATH.write_text(json.dumps(data))
        check("unknown settings keys ignored", Settings.load().stream_port == 9001)
    finally:
        if backup is not None:
            SETTINGS_PATH.write_text(backup)
        elif SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()


def main() -> int:
    """Run the dashboard verification suite."""
    print("=" * 70)
    print(" SMART TRAFFIC DASHBOARD - VERIFICATION SUITE".center(70))
    print("=" * 70)

    for test in (test_service_layer, test_settings_roundtrip,
                 test_offline_mode, test_live_mode):
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            import traceback

            FAILED.append((test.__name__, str(exc)))
            print(f"  [ERROR] {test.__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f" RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name, detail in FAILED:
        print(f"   FAILED: {name}  {detail}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
