"""Service providing aggregated dashboard metrics from the live broadcast stream."""

from __future__ import annotations

import os
import time

import requests

from backend.models.system_state import ConnectionStatus, DashboardMetrics, EmergencyStatus
from backend.simulation.simulator import TrafficSimulator
from config.settings import Settings, get_settings

# Where broadcast_server.py is running. Resolution order:
#   1. the TRAFFIC_STREAM_HOST / TRAFFIC_STREAM_PORT environment variables
#   2. stream_host / stream_port in data/settings.json (the Settings page)
#   3. the built-in default
# Use 127.0.0.1 when the CV node and this dashboard run on the same machine.
_saved = get_settings()
LIVE_STREAM_HOST: str = os.environ.get("TRAFFIC_STREAM_HOST", _saved.stream_host)
LIVE_STREAM_PORT: int = int(os.environ.get("TRAFFIC_STREAM_PORT", _saved.stream_port))
BASE_URL: str = f"http://{LIVE_STREAM_HOST}:{LIVE_STREAM_PORT}"
TELEMETRY_URL: str = f"{BASE_URL}/telemetry"
VIDEO_FEED_URL: str = f"{BASE_URL}/video_feed"
HISTORY_URL: str = f"{BASE_URL}/history"
EXPORT_URL: str = f"{BASE_URL}/export"
EXPORT_CSV_URL: str = f"{BASE_URL}/export.csv"
# All-sessions record: every run ever logged, not just the live rolling hour.
EXPORT_ALL_URL: str = f"{BASE_URL}/export/all"
EXPORT_ALL_CSV_URL: str = f"{BASE_URL}/export/all.csv"
RECORDS_URL: str = f"{BASE_URL}/records"
SESSIONS_URL: str = f"{BASE_URL}/sessions"
RECORD_STATUS_URL: str = f"{BASE_URL}/record/status"
RECORD_START_URL: str = f"{BASE_URL}/record/start"
RECORD_STOP_URL: str = f"{BASE_URL}/record/stop"
RECORD_SAVE_URL: str = f"{BASE_URL}/record/save"
LIGHT_URL: str = f"{BASE_URL}/light"
# Where the CV node is looking: a camera, an uploaded video, or a stream URL.
SOURCE_URL: str = f"{BASE_URL}/source"
SOURCES_URL: str = f"{BASE_URL}/sources"
SOURCE_UPLOAD_URL: str = f"{BASE_URL}/source/upload"
SOURCE_CONTROL_URL: str = f"{BASE_URL}/source/control"
REQUEST_TIMEOUT_SECONDS: float = 1.5
# How old a telemetry payload may be before it stops counting as live. The node
# keeps serving its last snapshot after a video ends or the camera drops, so a
# reachable /telemetry is not proof of a running pipeline -- without this the
# dashboard displays the final frame's numbers indefinitely and looks healthy.
TELEMETRY_STALE_AFTER_SECONDS: float = 6.0
# Opening a camera or reaching out to a remote stream can take a few seconds
# before it either delivers a picture or admits it cannot.
SOURCE_TIMEOUT_SECONDS: float = 45.0
# An upload is a whole video file crossing the network; it gets as long as it
# needs, since the alternative is a half-written file and a confusing error.
UPLOAD_TIMEOUT_SECONDS: float = 1800.0
# Reading the whole master record touches disk on the CV node, so it gets a
# longer budget than the 1.5s used for the once-per-second telemetry poll.
ARCHIVE_TIMEOUT_SECONDS: float = 15.0

# Maps the broadcast server's congestion levels onto the dashboard's density scale.
CONGESTION_TO_DENSITY: dict[str, str] = {
    "FREE": "Low",
    "MODERATE": "Moderate",
    "HEAVY": "High",
    "EMERGENCY": "Critical",
}


class SystemStatusService:
    """Aggregates dashboard metrics, preferring the live broadcast stream when reachable."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service with optional settings override."""
        self._settings = settings or get_settings()
        self._simulator = TrafficSimulator(self._settings)
        self._last_live_ok = False
        self._last_telemetry: dict = {}
        self._last_stale = False
        #: Last telemetry clock seen, and the monotonic time we first saw it, so
        #: a snapshot that stops advancing can be told from a live one.
        self._last_stamp: float | None = None
        self._last_stamp_at = 0.0

    @property
    def is_live_stream_connected(self) -> bool:
        """Return whether the most recent telemetry fetch succeeded."""
        return self._last_live_ok

    @property
    def is_telemetry_stale(self) -> bool:
        """Return whether the node answered but with an out-of-date snapshot.

        Distinct from ``is_live_stream_connected``: a node whose video has ended
        is still reachable and still serving its final snapshot. Pages should
        say the feed has stopped rather than presenting frozen numbers as live.
        """
        return self._last_stale

    def _is_stale(self, telemetry: dict) -> bool:
        """Return whether the node's snapshot has stopped advancing.

        A frozen feed is detected by its clock standing still, not by comparing
        the node's clock to ours. Comparing against wall time would assume the
        two machines agree -- they need not, the node may be on another host --
        and would condemn any payload whose timestamp is merely unfamiliar.
        Watching for the stamp to stop changing needs no such assumption.

        The first sighting of any stamp is always fresh: there is no history to
        judge it against yet, and refusing data on first contact would break
        every caller that fetches once.
        """
        stamp = telemetry.get("timestamp_ms")
        if stamp is None:
            return False
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            return False

        now = time.monotonic()
        if stamp != self._last_stamp:
            self._last_stamp = stamp
            self._last_stamp_at = now
            return False
        # Same snapshot as last time: stale once it has stood still too long.
        return (now - self._last_stamp_at) > TELEMETRY_STALE_AFTER_SECONDS

    @property
    def last_telemetry(self) -> dict:
        """Return the raw telemetry payload from the most recent fetch.

        Pages use this for fields that have no slot in ``DashboardMetrics``,
        such as per-vehicle detections and the speed-calibration mode.
        """
        return self._last_telemetry

    def _fetch_live_telemetry(self) -> dict | None:
        """Fetch telemetry JSON from the live broadcast server, or None if unreachable."""
        try:
            response = requests.get(TELEMETRY_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError):
            return None

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Return current dashboard metrics, overlaying live telemetry when available."""
        metrics = self._simulator.get_dashboard_metrics()
        telemetry = self._fetch_live_telemetry()

        if telemetry is None:
            self._last_live_ok = False
            self._last_stale = False
            self._last_telemetry = {}
            metrics.system_health.camera_status = ConnectionStatus.OFFLINE
            metrics.system_health.yolo_status = ConnectionStatus.OFFLINE
            return metrics

        # A reachable node serving a frozen snapshot is not a live feed. Keep the
        # payload -- pages still want the last known detections -- but report the
        # camera as offline so the numbers are not presented as current.
        if self._is_stale(telemetry):
            self._last_live_ok = False
            self._last_stale = True
            self._last_telemetry = telemetry
            metrics.system_health.camera_status = ConnectionStatus.OFFLINE
            metrics.system_health.yolo_status = ConnectionStatus.OFFLINE
            return metrics

        self._last_live_ok = True
        self._last_stale = False
        self._last_telemetry = telemetry

        traffic = metrics.traffic
        traffic.vehicle_count = int(telemetry.get("total_vehicles", traffic.vehicle_count))
        # Real km/h from the calibrated speed estimator in traffic_vision.py.
        traffic.average_speed_kmh = float(
            telemetry.get("avg_speed_kmh", traffic.average_speed_kmh)
        )
        traffic.fps = float(telemetry.get("fps", traffic.fps))
        congestion_level = str(telemetry.get("congestion_level", "")).upper()
        traffic.traffic_density = CONGESTION_TO_DENSITY.get(congestion_level, traffic.traffic_density)

        # The real signal phase, so the dashboard mirrors the physical lamp
        # rather than guessing it from the congestion level.
        traffic.light_phase = str(telemetry.get("light_phase", "") or "")
        traffic.light_target = str(telemetry.get("light_target", "") or "")
        traffic.light_remaining = float(telemetry.get("light_remaining", 0.0) or 0.0)
        traffic.light_transitioning = bool(telemetry.get("light_transitioning"))

        emergency_detected = bool(telemetry.get("emergency_vehicle_detected"))
        traffic.emergency_vehicle_count = int(
            telemetry.get("emergency_vehicle_count", 1 if emergency_detected else 0)
        )
        metrics.emergency_status = EmergencyStatus.ACTIVE if emergency_detected else EmergencyStatus.NONE

        metrics.system_health.camera_status = ConnectionStatus.ONLINE
        metrics.system_health.yolo_status = ConnectionStatus.ONLINE

        # The node reports whether it actually has the board open, so outside
        # simulation the controller lights are reported rather than assumed.
        if not self._settings.simulation_mode:
            hardware = telemetry.get("hardware") or {}
            board = (
                ConnectionStatus.ONLINE
                if hardware.get("arduino_connected", bool(hardware))
                else ConnectionStatus.OFFLINE
            )
            metrics.system_health.esp32_status = board
            metrics.system_health.serial_status = board

        return metrics
