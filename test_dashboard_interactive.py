"""
test_dashboard_interactive.py — widget-level verification of the dashboard
=========================================================================

``test_dashboard.py`` checks that every page *renders*. This suite drives the
widgets and checks that they actually *do* something: log filters narrow the
records, analytics re-bins when the interval changes, every hardware command
runs, the confidence slider really drops detections, and a saved setting
round-trips to disk.

Run with::

    python test_dashboard_interactive.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "SmartTrafficSystem"
sys.path.insert(0, str(APP_DIR))

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

st.page_link = lambda *a, **k: None  # AppTest cannot resolve sibling pages

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{f'  ({detail})' if detail else ''}")


def load(page: str, **state) -> AppTest:
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=90)
    for key, value in {
        "auto_refresh": False,
        "camera_auto_refresh": False,
        "emergency_auto_refresh": False,
        "hardware_auto_refresh": False,
        "log_auto_refresh": False,
        **state,
    }.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    return at


# ---------------------------------------------------------------------------
print("\n[A] System Logs — records, filters, search, export")
# ---------------------------------------------------------------------------
at = load("pages/6_System_Logs.py")

rows_all = at.dataframe[0].value
check("log records render", len(rows_all) > 0, f"{len(rows_all)} rows")
check(
    "log table has the expected columns",
    list(rows_all.columns) == ["Timestamp", "Level", "Source", "Module", "Message"],
    str(list(rows_all.columns)),
)

# Level filter
at.selectbox[0].select("ERROR").run()
errors = at.dataframe[0].value
check(
    "level filter returns only ERROR rows",
    len(errors) > 0 and set(errors["Level"]) == {"ERROR"},
    f"{len(errors)} rows",
)

# Source filter (reset level first)
at.selectbox[0].select("All").run()
at.selectbox[1].select("Camera").run()
by_source = at.dataframe[0].value
check(
    "source filter returns only Camera rows",
    len(by_source) > 0 and set(by_source["Source"]) == {"Camera"},
    f"{len(by_source)} rows",
)

# Search
at.selectbox[1].select("All").run()
at.text_input[0].input("WiFi").run()
searched = at.dataframe[0].value
check(
    "search matches message/source/module",
    len(searched) > 0
    and all(
        "wifi" in " ".join(str(v) for v in row).lower() for _, row in searched.iterrows()
    ),
    f"{len(searched)} rows",
)

# Export payload — AppTest exposes only the click state, so rebuild the CSV
# from the exact frame the page hands to st.download_button.
at.text_input[0].input("").run()
check("log export button offered", len(at.download_button) == 1)
csv_text = at.dataframe[0].value.to_csv(index=False)
check(
    "log CSV export carries the records",
    "Timestamp,Level,Source,Module,Message" in csv_text
    and len(csv_text.strip().splitlines()) > 1,
    f"{len(csv_text.strip().splitlines())} lines",
)

# Statistics add up
stats = [m for m in at.markdown if "kpi-value" in m.value]
check("log statistic cards rendered", len(stats) >= 5, f"{len(stats)} cards")

# Clear
at.button[0].click().run()
check("clear logs empties the table", len(at.dataframe) == 0 or at.dataframe[0].value.empty)

# ---------------------------------------------------------------------------
print("\n[B] Traffic Analytics — datasets, controls, exports")
# ---------------------------------------------------------------------------
at = load("pages/3_Traffic_Analytics.py")

# Two tab bars: six archive-insight tabs plus the seven live-session tabs.
tab_labels = [t.label for t in at.tabs]
check("live session tabs are all present", len(at.tabs) == 13, f"{len(at.tabs)} tabs")
for label in ["📉 Timeline", "⚡ Speed", "🚦 Density", "🥧 Vehicle Types",
              "🔥 Congestion", "⏱️ Waiting Time", "🗺️ Heatmap"]:
    check(f"live tab '{label}' present", label in tab_labels)
for label in ["🕒 Demand by Hour", "📉 Speed vs Congestion", "🚦 Signal Behaviour",
              "📊 Distribution", "🗂️ Session Comparison", "🌊 Session Shape"]:
    check(f"archive tab '{label}' present", label in tab_labels)

from backend.services.analytics_service import AnalyticsService  # noqa: E402

csv_text = AnalyticsService().export_csv(hours=24, interval_minutes=15)
for section in [
    "# vehicle_timeline",
    "# speed_timeline_kmh",
    "# density_timeline",
    "# vehicle_types",
    "# congestion_trend",
    "# waiting_time",
    "# heatmap",
]:
    check(f"analytics CSV has {section}", section in csv_text)

raw = at.dataframe[0].value
check("raw-data viewer shows a dataset", len(raw) > 0, f"{len(raw)} rows")

# Widgets are addressed by key, not position, so adding a control to the page
# cannot silently make these assertions test the wrong thing.
before = len(at.dataframe[-1].value)
at.selectbox(key="interval").select(5).run()
after = len(at.dataframe[-1].value)
check("finer interval yields more data points", after > before, f"{before} -> {after}")

at.selectbox(key="time_range").select(48).run()
check("48h window regenerates without error", not at.exception)
check("regenerate button works",
      not at.button(key="regenerate").click().run().exception)

# ── Scope selector: the archive vs the live session ─────────────────────────
scope = at.selectbox(key="scope")
check("data scope selector offers session and archive views",
      {"This session", "All sessions"}.issubset(set(scope.options)),
      f"{scope.options[:4]}")
check("switching to the full archive renders cleanly",
      not at.selectbox(key="scope").select("All sessions").run().exception)

at = load("pages/3_Traffic_Analytics.py")
check("comparison metric selector works",
      not at.selectbox(key="compare_metric").select("Peak Vehicles").run().exception)

# ---------------------------------------------------------------------------
print("\n[B2] Session recording controls")
# ---------------------------------------------------------------------------
at = load("pages/3_Traffic_Analytics.py")
sidebar_buttons = [b.label for b in at.sidebar.button]
check("a record button is offered", any("Record" in label for label in sidebar_buttons),
      f"{sidebar_buttons}")
check("a save-now button is offered", any("Save" in label for label in sidebar_buttons),
      f"{sidebar_buttons}")

at.text_input(key="recording_name").set_value("unit test run").run()
check("a recording can be named",
      at.session_state["recording_name"] == "unit test run")

started = at.button(key="start_recording").click().run()
check("start recording runs without error", not started.exception)

at = load("pages/3_Traffic_Analytics.py")
check("save-now runs without error",
      not at.button(key="save_now").click().run().exception)

# Stop the recording started above before going on. While one is running every
# page offers "Stop" instead of "Start" — correctly — so leaving it going makes
# the dashboard check below hunt for a button that is rightly absent, and the
# suite dies on a KeyError. This only bites when the CV node is actually up:
# with nothing listening the start click silently does nothing, so the button
# to press here is whichever one the page is showing. Probed rather than
# assumed, because both states are legitimate and the suite has to run in each.
at = load("pages/3_Traffic_Analytics.py")
stop_button = [b for b in at.button if getattr(b, "key", None) == "stop_recording"]
if stop_button:
    stop_button[0].click().run()

# The operations dashboard carries a compact version of the same control.
dash = load("pages/1_Operations_Dashboard.py")
dash_buttons = [b.label for b in dash.sidebar.button]
check("the main dashboard also offers recording",
      any("Record" in label for label in dash_buttons), f"{dash_buttons}")
check("the dashboard record button works",
      not dash.button(key="dash_start_recording").click().run().exception)

at = load("pages/3_Traffic_Analytics.py")
export_labels = [b.label for b in at.button] + [d.label for d in at.download_button]
check("a this-session export is offered",
      any("This Session" in label for label in export_labels), f"{export_labels}")
check("an all-sessions export is offered",
      any("All Sessions" in label for label in export_labels), f"{export_labels}")

# ---------------------------------------------------------------------------
print("\n[C] Emergency Control — every hardware command")
# ---------------------------------------------------------------------------
at = load("pages/4_Emergency_Control.py")
labels = [b.label for b in at.button]
print(f"    buttons: {labels}")

for index, button in enumerate(at.button):
    label = button.label
    result = at.button[index].click().run()
    check(f"command '{label}' runs cleanly", not result.exception)

check("no error box after commands", len(at.error) == 0, f"{[e.value for e in at.error]}")

# LCD message round trip
at = load("pages/4_Emergency_Control.py")
at.text_input[0].input("TEST MESSAGE").run()
send = [i for i, b in enumerate(at.button) if b.label == "Send to LCD"][0]
at.button[send].click().run()
check("LCD message send succeeds", not at.exception and len(at.error) == 0)

# ---------------------------------------------------------------------------
print("\n[D] Live Camera — overlay toggles and detection table")
# ---------------------------------------------------------------------------
at = load("pages/2_Live_Camera.py")
check("camera falls back to the simulated feed", len(at.warning) == 1)
check("detection table rendered", len(at.dataframe) == 1)

detections = at.dataframe[0].value
check(
    "detection table columns",
    {"ID", "Class", "Lane", "Confidence", "Emergency"}.issubset(set(detections.columns)),
    str(list(detections.columns)),
)

for index, box in enumerate(at.checkbox):
    label = box.label
    at.checkbox[index].uncheck().run()
    check(f"overlay toggle '{label}' works", not at.exception)

# The simulator emits confidences in 0.70-0.99, so a high threshold must
# actually drop rows. This caught the slider writing to a Settings instance
# the CameraService did not hold.
at = load("pages/2_Live_Camera.py")
base = len(at.dataframe[0].value) if at.dataframe else 0
at.sidebar.slider[0].set_value(0.99).run()
high = len(at.dataframe[0].value) if at.dataframe else 0
check("confidence slider actually filters detections", high < base, f"{base} -> {high}")

at.sidebar.slider[0].set_value(0.50).run()
low = len(at.dataframe[0].value) if at.dataframe else 0
check("lowering the threshold restores detections", low > high, f"{high} -> {low}")

# ---------------------------------------------------------------------------
print("\n[D2] Live Camera — choosing and driving the video source")
# ---------------------------------------------------------------------------
# The source picker is the one part of the dashboard that changes what the CV
# node is doing rather than just describing it, so every control here is driven
# and the resulting request inspected. A picker that renders beautifully and
# posts nothing would otherwise look perfectly healthy.
import types  # noqa: E402

import requests  # noqa: E402

import backend.services.video_source_service as vss  # noqa: E402

PLAYING = {
    "kind": "file", "label": "bridge.mp4", "spec": "bridge.mp4",
    "target": "C:/GDP/uploads/bridge.mp4", "is_file": True, "live": False,
    "width": 1920, "height": 1080, "native_fps": 30.0, "frame_count": 3600,
    "duration_s": 120.0, "duration_text": "2:00", "position_s": 30.0,
    "position_text": "0:30", "progress": 0.25, "paused": False, "finished": False,
    "loop": False, "playback_rate": 1.0, "drop_frames": True, "frames_read": 900,
    "frames_skipped": 0, "reconnects": 0, "switches": 1, "media_time": 0.0,
    "media_ahead_s": 0.0,
}
CATALOGUE = {
    "current": PLAYING,
    "cameras": [{"index": 0, "name": "HP Wide Vision HD Camera", "spec": "0"},
                {"index": 2, "name": "DroidCam Source 3", "spec": "2"}],
    "uploads": [{"name": "bridge.mp4", "spec": "C:/GDP/uploads/bridge.mp4",
                 "size_mb": 12.0, "uploaded_at": "2026-01-01 10:00",
                 "modified": 0.0}],
}
REQUESTS: list[tuple[str, dict]] = []


class _Reply:
    """The two methods VideoSourceService uses off a response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Never fails: these stubs only model the successful path."""

    def json(self) -> dict:
        """Return the stubbed payload."""
        return self._payload


def _stub_get(url, *_args, **_kwargs):
    REQUESTS.append((f"GET {url}", {}))
    return _Reply(CATALOGUE if url.endswith("/sources") else PLAYING)


def _stub_post(url, *_args, **kwargs):
    REQUESTS.append((f"POST {url}", kwargs.get("json") or {}))
    return _Reply(PLAYING)


# Rebinding the name inside the service leaves the real `requests` module
# untouched for every other suite sharing this process.
vss.requests = types.SimpleNamespace(get=_stub_get, post=_stub_post,
                                     delete=_stub_get, exceptions=requests.exceptions)


def by_label(widgets, text: str):
    """Return the first widget whose label contains ``text``."""
    for widget in widgets:
        if text in str(widget.label):
            return widget
    raise AssertionError(f"no widget labelled {text!r} among "
                         f"{[str(w.label) for w in widgets]}")


at = load("pages/2_Live_Camera.py", show_source_panel=True, camera_auto_refresh=True)
check("the source picker opens", len(at.tabs) >= 4, f"{len(at.tabs)} tabs")
check("it names what is playing",
      any("bridge.mp4" in m.value for m in at.markdown),
      str([m.value for m in at.markdown][:3]))
check("live refresh is suspended while the picker is open",
      not at.exception, "the page must return rather than sleep-and-rerun forever")

REQUESTS.clear()
by_label(at.text_input, "Address").input("rtsp://10.0.0.5:554/stream1").run()
by_label(at.button, "Connect").click().run()
posts = [(url, body) for url, body in REQUESTS if url.startswith("POST")]
check("connecting to a URL posts it to the node",
      any(body.get("source") == "rtsp://10.0.0.5:554/stream1" for _u, body in posts),
      str(posts))
check("and posts it to /source", any(url.endswith("/source") for url, _b in posts),
      str([u for u, _ in posts]))

at = load("pages/2_Live_Camera.py", show_source_panel=True)
REQUESTS.clear()
by_label(at.selectbox, "Camera").select("2").run()
by_label(at.button, "Use this camera").click().run()
check("picking a connected camera posts its index",
      any(body.get("source") == "2" for url, body in REQUESTS if url.startswith("POST")),
      str(REQUESTS))

at = load("pages/2_Live_Camera.py", show_source_panel=True)
check("a playing video offers a position control",
      any("Jump" in str(s.label) for s in at.slider), str([str(s.label) for s in at.slider]))
REQUESTS.clear()
by_label(at.slider, "Jump to").set_value(75.0).run()
by_label(at.button, "Jump").click().run()
seeks = [body for url, body in REQUESTS
         if url.endswith("/source/control") and body.get("action") == "seek"]
check("scrubbing posts a seek", bool(seeks), str(REQUESTS))
check("the seek carries the position asked for",
      seeks and abs(float(seeks[0].get("fraction", 0)) - 0.75) < 0.01, str(seeks))

at = load("pages/2_Live_Camera.py", show_source_panel=True)
REQUESTS.clear()
by_label(at.select_slider, "Playback speed").set_value(0.0).run()
rates = [body for url, body in REQUESTS
         if url.endswith("/source/control") and body.get("action") == "rate"]
check("changing playback speed posts a rate", bool(rates), str(REQUESTS))
check("'as fast as possible' is sent as rate 0",
      rates and float(rates[0].get("value", -1)) == 0.0, str(rates))

at = load("pages/2_Live_Camera.py", show_source_panel=True)
REQUESTS.clear()
by_label(at.button, "🗑").click().run()
check("deleting an upload calls the node",
      any("source/upload" in url for url, _b in REQUESTS), str(REQUESTS))

# Closing the picker must let the live refresh resume, or the page would be
# frozen from then on.
at = load("pages/2_Live_Camera.py", show_source_panel=False)
check("the picker collapses again", len(at.tabs) == 0, f"{len(at.tabs)} tabs")
check("with the picker closed the page still renders", not at.exception)

# ---------------------------------------------------------------------------
print("\n[E] Settings — save round trip")
# ---------------------------------------------------------------------------
from config.settings import SETTINGS_PATH, Settings  # noqa: E402

backup = None
if SETTINGS_PATH.exists():
    backup = SETTINGS_PATH.read_bytes()

try:
    at = load("pages/7_Settings.py")
    at.text_input[0].input("Renamed System").run()
    save = [i for i, b in enumerate(at.button) if "Save" in b.label][0]
    at.button[save].click().run()

    check("settings file written", SETTINGS_PATH.exists())
    reloaded = Settings.load()
    check(
        "saved title persisted",
        reloaded.app_title == "Renamed System",
        reloaded.app_title,
    )
    check(
        "serial port persisted as a real port, not None",
        isinstance(reloaded.serial_port, str) and reloaded.serial_port != "",
        repr(reloaded.serial_port),
    )
    check(
        "camera index persisted as an int",
        isinstance(reloaded.camera_index, int),
        repr(reloaded.camera_index),
    )
finally:
    if backup is None:
        SETTINGS_PATH.unlink(missing_ok=True)
    else:
        SETTINGS_PATH.write_bytes(backup)
    print("    (settings.json restored)")

# ---------------------------------------------------------------------------
print("\n[F] Dashboard — KPI values")
# ---------------------------------------------------------------------------
at = load("pages/1_Operations_Dashboard.py")
kpis = [m.value for m in at.markdown if "kpi-value" in m.value]
check("dashboard renders KPI cards", len(kpis) >= 13, f"{len(kpis)} cards")
check("time/date/updated metrics present", len(at.metric) == 3, f"{len(at.metric)} metrics")

empty = [k for k in kpis if re.search(r'class="kpi-value[^"]*">\s*</div>', k)]
check("no KPI card renders an empty value", not empty, f"{len(empty)} empty")

# The icon slot must hold an emoji, not a wordy label (the swapped-argument bug).
icons = re.findall(r'class="kpi-icon">([^<]*)</div>', "\n".join(kpis))
check(
    "KPI icon slot holds icons, not labels",
    all(len(i.strip()) <= 4 for i in icons),
    f"longest={max((i.strip() for i in icons), key=len, default='')!r}",
)

# ---------------------------------------------------------------------------
print("\n[F2] Front page — the project and the group")
# ---------------------------------------------------------------------------
from config import project  # noqa: E402  (needs APP_DIR on sys.path, set above)

at = load("app.py")
# The theme injects its stylesheet as a markdown element, and it names every
# class the page can use -- so counting classes across all markdown counts the
# CSS as well as the page. Only the rendered blocks are the page.
front = "\n".join(m.value for m in at.markdown if "<style>" not in m.value)

check("the front page carries the project title",
      project.PROJECT_TITLE in front, front[:120])
check("it states the aim", project.PROJECT_AIM[:40] in front)
check("it states the group objective", project.GROUP_OBJECTIVE[:40] in front)

# One card per member, filled or not: a missing card would silently drop
# somebody from the group rather than showing an obvious gap.
team_card_count = front.count('class="team-card')
check("every member has a card",
      team_card_count == len(project.TEAM),
      f"{team_card_count} cards for {len(project.TEAM)} members")
check("every real name is shown",
      all(m.name in front for m in project.TEAM if m.filled))
check("unfilled slots are marked, not hidden",
      front.count("is-unfilled") == project.missing_names(),
      f"{front.count('is-unfilled')} marked, {project.missing_names()} missing")

check("every headline result is shown",
      all(r.figure in front for r in project.HEADLINE_RESULTS))
# The qualifier is what keeps the figure honest -- "0 false alarms" without
# "night footage" is a different and much larger claim.
check("each result keeps its qualifier",
      all(r.note in front for r in project.HEADLINE_RESULTS))

# It is a title page: the live figures belong on the dashboard behind it, and
# a stray KPI card here would mean the two had started to overlap again.
check("no live KPI cards on the title page", "kpi-value" not in front)
check("no auto-refresh timer on the title page", not at.toggle)

# ---------------------------------------------------------------------------
print("\n[G] KPI card slots on the pages that used positional arguments")
# ---------------------------------------------------------------------------
for page in [
    "pages/4_Emergency_Control.py",
    "pages/5_Hardware_Monitor.py",
    "pages/6_System_Logs.py",
]:
    at = load(page)
    cards = [m.value for m in at.markdown if "kpi-card" in m.value]
    icons = re.findall(r'class="kpi-icon">([^<]*)</div>', "\n".join(cards))
    labels = re.findall(r'class="kpi-label">([^<]*)</div>', "\n".join(cards))
    check(f"{page}: cards rendered", len(cards) > 0, f"{len(cards)} cards")
    check(
        f"{page}: icon slot holds an icon",
        icons and all(len(i.strip()) <= 4 for i in icons),
        f"longest={max((i.strip() for i in icons), key=len, default='')!r}",
    )
    check(
        f"{page}: label slot holds a word",
        labels and any(len(l.strip()) > 4 for l in labels),
        f"sample={labels[:3]}",
    )

print("\n" + "=" * 70)
print(f" RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"   FAILED: {name}")
print("=" * 70)
sys.exit(1 if FAIL else 0)