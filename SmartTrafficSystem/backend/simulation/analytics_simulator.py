"""Simulated historical traffic analytics data generation."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backend.models.analytics import TrafficAnalyticsData
from config.settings import Settings, get_settings


class AnalyticsSimulator:
    """Generates realistic dummy analytics datasets for demonstration."""

    DENSITY_MAP: dict[str, int] = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
    VEHICLE_TYPES: tuple[str, ...] = ("Car", "Truck", "Bus", "Motorcycle", "Ambulance")
    LANES: tuple[str, ...] = ("Lane A", "Lane B", "Lane C", "Lane D")
    HOURS: tuple[str, ...] = tuple(f"{h:02d}:00" for h in range(24))

    def __init__(self, settings: Settings | None = None, seed: int = 42) -> None:
        """Initialize simulator with reproducible random seed."""
        self._settings = settings or get_settings()
        self._rng = random.Random(seed)
        np.random.seed(seed)

    def generate(self, hours: int = 24, interval_minutes: int = 15) -> TrafficAnalyticsData:
        """Generate a full analytics dataset for the requested time window."""
        end_time = datetime.now().replace(second=0, microsecond=0)
        start_time = end_time - timedelta(hours=hours)
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{interval_minutes}min")

        vehicle_timeline = self._build_vehicle_timeline(timestamps)
        density_timeline = self._build_density_timeline(timestamps, vehicle_timeline)
        vehicle_types = self._build_vehicle_types()
        congestion_trend = self._build_congestion_trend(timestamps, vehicle_timeline)
        waiting_time = self._build_waiting_time(timestamps, congestion_trend)
        heatmap = self._build_heatmap()
        speed_timeline = self._build_speed_timeline(timestamps, congestion_trend)

        return TrafficAnalyticsData(
            vehicle_timeline=vehicle_timeline,
            density_timeline=density_timeline,
            vehicle_types=vehicle_types,
            congestion_trend=congestion_trend,
            waiting_time=waiting_time,
            heatmap=heatmap,
            speed_timeline=speed_timeline,
        )

    def _build_speed_timeline(
        self, timestamps: pd.DatetimeIndex, congestion_trend: pd.DataFrame
    ) -> pd.DataFrame:
        """Build average speed in km/h, falling as congestion rises."""
        speeds: list[float] = []
        for congestion in congestion_trend["congestion_index"]:
            # 55 km/h on an empty road down to ~8 km/h in a jam.
            value = 55.0 - (congestion / 100.0) * 47.0 + self._rng.uniform(-3, 3)
            speeds.append(round(max(3.0, min(90.0, value)), 1))
        return pd.DataFrame({"timestamp": timestamps, "avg_speed_kmh": speeds})

    def _rush_hour_multiplier(self, timestamp: pd.Timestamp) -> float:
        """Return traffic multiplier based on time of day."""
        hour = timestamp.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 1.6
        if 11 <= hour <= 14:
            return 1.2
        if 22 <= hour or hour <= 5:
            return 0.45
        return 1.0

    def _build_vehicle_timeline(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Build vehicle count over time with rush-hour patterns."""
        counts: list[int] = []
        base = 18.0
        for index, ts in enumerate(timestamps):
            wave = 6.0 * np.sin(index / 8.0)
            noise = self._rng.uniform(-4, 4)
            value = base * self._rush_hour_multiplier(ts) + wave + noise
            counts.append(max(0, int(round(value))))
        return pd.DataFrame({"timestamp": timestamps, "vehicle_count": counts})

    def _build_density_timeline(
        self, timestamps: pd.DatetimeIndex, vehicle_timeline: pd.DataFrame
    ) -> pd.DataFrame:
        """Map vehicle counts to density levels over time."""
        densities: list[str] = []
        scores: list[int] = []
        for count in vehicle_timeline["vehicle_count"]:
            if count < 12:
                level = "Low"
            elif count < 22:
                level = "Moderate"
            elif count < 32:
                level = "High"
            else:
                level = "Critical"
            densities.append(level)
            scores.append(self.DENSITY_MAP[level])
        return pd.DataFrame(
            {"timestamp": timestamps, "density": densities, "density_score": scores}
        )

    def _build_vehicle_types(self) -> pd.DataFrame:
        """Build vehicle type distribution for pie chart."""
        weights = [0.58, 0.18, 0.10, 0.12, 0.02]
        total = self._rng.randint(850, 1200)
        counts = [max(1, int(total * w + self._rng.uniform(-5, 5))) for w in weights]
        return pd.DataFrame({"vehicle_type": list(self.VEHICLE_TYPES), "count": counts})

    def _build_congestion_trend(
        self, timestamps: pd.DatetimeIndex, vehicle_timeline: pd.DataFrame
    ) -> pd.DataFrame:
        """Build congestion index trend correlated with vehicle volume."""
        indices: list[float] = []
        for count, ts in zip(vehicle_timeline["vehicle_count"], timestamps):
            rush = 15.0 if self._rush_hour_multiplier(ts) > 1.3 else 0.0
            index = min(100.0, max(5.0, count * 2.8 + rush + self._rng.uniform(-3, 3)))
            indices.append(round(index, 1))
        return pd.DataFrame({"timestamp": timestamps, "congestion_index": indices})

    def _build_waiting_time(
        self, timestamps: pd.DatetimeIndex, congestion_trend: pd.DataFrame
    ) -> pd.DataFrame:
        """Build average waiting time correlated with congestion."""
        waits: list[float] = []
        for congestion in congestion_trend["congestion_index"]:
            base_wait = congestion * 0.35 + self._rng.uniform(2, 8)
            waits.append(round(min(180.0, max(5.0, base_wait)), 1))
        return pd.DataFrame({"timestamp": timestamps, "avg_waiting_seconds": waits})

    def build_placeholder_heatmap(self) -> pd.DataFrame:
        """Public accessor for the lane×hour placeholder heatmap.

        The live camera has no per-lane sensors, so this placeholder is
        reused even when the rest of the analytics dataset is real.
        """
        return self._build_heatmap()

    def _build_heatmap(self) -> pd.DataFrame:
        """Build lane-by-hour traffic intensity matrix."""
        data: dict[str, list[int]] = {"hour": list(self.HOURS)}
        for lane in self.LANES:
            lane_values: list[int] = []
            for hour in range(24):
                base = 20
                if lane in ("Lane A", "Lane B"):
                    base += 8
                if 7 <= hour <= 9 or 17 <= hour <= 19:
                    base += 25
                elif 22 <= hour or hour <= 5:
                    base -= 12
                lane_values.append(max(1, base + self._rng.randint(-6, 6)))
            data[lane] = lane_values
        return pd.DataFrame(data)
