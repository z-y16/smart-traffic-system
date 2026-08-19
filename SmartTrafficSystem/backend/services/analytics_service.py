"""Traffic analytics data service."""

from __future__ import annotations

import pandas as pd
import requests

from backend.models.analytics import TrafficAnalyticsData
from backend.services.system_status_service import (
    ARCHIVE_TIMEOUT_SECONDS,
    EXPORT_URL,
    HISTORY_URL,
    REQUEST_TIMEOUT_SECONDS,
)
from backend.simulation.analytics_simulator import AnalyticsSimulator
from config.settings import Settings, get_settings

# Maps the broadcast server's congestion levels onto the dashboard's density scale.
CONGESTION_LEVEL_TO_DENSITY: dict[str, str] = {
    "FREE": "Low",
    "MODERATE": "Moderate",
    "HEAVY": "High",
    "EMERGENCY": "Critical",
}
DENSITY_SCORE_MAP: dict[str, int] = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
SCORE_DENSITY_MAP: dict[int, str] = {score: density for density, score in DENSITY_SCORE_MAP.items()}


class AnalyticsService:
    """Provides analytics datasets for the Traffic Analytics page."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize analytics service."""
        self._settings = settings or get_settings()
        self._simulator = AnalyticsSimulator(self._settings)
        self._cache: TrafficAnalyticsData | None = None
        self._cache_key: tuple[int, int] | None = None
        self._last_live_ok = False

    @property
    def is_live_data(self) -> bool:
        """Return whether the cached analytics came from the live broadcast history."""
        return self._last_live_ok

    def get_analytics(self, hours: int = 24, interval_minutes: int = 15) -> TrafficAnalyticsData:
        """Return analytics data, regenerating when parameters change."""
        key = (hours, interval_minutes)
        if self._cache is None or self._cache_key != key:
            self._cache = self._generate(hours=hours, interval_minutes=interval_minutes)
            self._cache_key = key
        return self._cache

    def refresh(self, hours: int = 24, interval_minutes: int = 15) -> TrafficAnalyticsData:
        """Force regeneration of analytics data."""
        self._cache = self._generate(hours=hours, interval_minutes=interval_minutes)
        self._cache_key = (hours, interval_minutes)
        return self._cache

    def export_csv(self, hours: int = 24, interval_minutes: int = 15) -> str:
        """Return combined analytics CSV string."""
        data = self.get_analytics(hours=hours, interval_minutes=interval_minutes)
        return data.to_export_csv()

    def fetch_export_bytes(self) -> bytes | None:
        """Download the live broadcast server's Excel traffic log, or None if unreachable.

        Gets the archive budget rather than a few seconds: the node builds the
        workbook when this asks for it, so the reply waits on a long session
        being serialised. Timing out here returns None, which reads on screen
        as a node that is simply down.
        """
        try:
            response = requests.get(EXPORT_URL, timeout=ARCHIVE_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException:
            return None

    def _generate(self, hours: int, interval_minutes: int) -> TrafficAnalyticsData:
        """Build analytics from live broadcast history, falling back to simulation."""
        history = self._fetch_live_history()
        if history:
            self._last_live_ok = True
            return self._build_from_history(history, interval_minutes)
        self._last_live_ok = False
        return self._simulator.generate(hours=hours, interval_minutes=interval_minutes)

    def _fetch_live_history(self) -> list[dict] | None:
        """Fetch recorded telemetry rows from the live broadcast server."""
        try:
            response = requests.get(HISTORY_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            rows = response.json()
        except (requests.exceptions.RequestException, ValueError):
            return None
        return rows or None

    def _build_from_history(self, history: list[dict], interval_minutes: int) -> TrafficAnalyticsData:
        """Convert raw broadcast telemetry rows into dashboard analytics datasets."""
        frame = pd.DataFrame(history)
        frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms")
        frame = frame.set_index("timestamp").sort_index()
        rule = f"{interval_minutes}min"

        vehicle_timeline = (
            frame["total_vehicles"]
            .resample(rule)
            .mean()
            .round()
            .fillna(0)
            .astype(int)
            .rename("vehicle_count")
            .reset_index()
        )

        density_score = (
            frame["congestion_level"].map(CONGESTION_LEVEL_TO_DENSITY).map(DENSITY_SCORE_MAP)
        )
        density_score_resampled = (
            density_score.resample(rule).mean().round().clip(1, 4).fillna(1).astype(int)
        )
        density_timeline = pd.DataFrame(
            {
                "timestamp": density_score_resampled.index,
                "density_score": density_score_resampled.values,
            }
        )
        density_timeline["density"] = density_timeline["density_score"].map(SCORE_DENSITY_MAP)
        density_timeline = density_timeline[["timestamp", "density", "density_score"]]

        vehicle_counts_df = pd.json_normalize(frame["vehicle_counts"].tolist()).fillna(0)
        vehicle_totals = vehicle_counts_df.sum().sort_values(ascending=False)
        if vehicle_totals.empty or vehicle_totals.sum() == 0:
            vehicle_types = pd.DataFrame({"vehicle_type": ["No data"], "count": [1]})
        else:
            vehicle_types = pd.DataFrame(
                {
                    "vehicle_type": [str(name).capitalize() for name in vehicle_totals.index],
                    "count": vehicle_totals.values.astype(int),
                }
            )

        congestion_trend = (
            (frame["congestion_index"] * 100.0)
            .resample(rule)
            .mean()
            .round(1)
            .fillna(0)
            .rename("congestion_index")
            .reset_index()
        )

        # Real measured speeds in km/h. Samples with no vehicles in frame log
        # 0 km/h, which would drag the average down, so they are excluded.
        speed_column = frame.get("avg_speed_kmh")
        if speed_column is None:
            speed_timeline = None
        else:
            measured = speed_column.where(speed_column > 0)
            speed_timeline = (
                measured.resample(rule)
                .mean()
                .round(1)
                .fillna(0.0)
                .rename("avg_speed_kmh")
                .reset_index()
            )

        # No direct wait-time sensor on the live camera — estimated from congestion.
        estimated_wait = (congestion_trend["congestion_index"] * 0.5 + 3.0).clip(0, 180).round(1)
        waiting_time = pd.DataFrame(
            {
                "timestamp": congestion_trend["timestamp"],
                "avg_waiting_seconds": estimated_wait,
            }
        )

        heatmap = self._simulator.build_placeholder_heatmap()

        return TrafficAnalyticsData(
            vehicle_timeline=vehicle_timeline,
            density_timeline=density_timeline,
            vehicle_types=vehicle_types,
            congestion_trend=congestion_trend,
            waiting_time=waiting_time,
            heatmap=heatmap,
            speed_timeline=speed_timeline,
        )
