"""Analytics domain models."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrafficAnalyticsData:
    """Container for all traffic analytics datasets."""

    vehicle_timeline: pd.DataFrame
    density_timeline: pd.DataFrame
    vehicle_types: pd.DataFrame
    congestion_trend: pd.DataFrame
    waiting_time: pd.DataFrame
    heatmap: pd.DataFrame
    # Average vehicle speed in km/h over time. Real measured data when the
    # broadcast stream is live, simulated otherwise.
    speed_timeline: pd.DataFrame | None = None

    def to_export_csv(self) -> str:
        """Combine all datasets into a single CSV string for download."""
        sections: list[str] = []

        datasets = [
            ("vehicle_timeline", self.vehicle_timeline),
            ("density_timeline", self.density_timeline),
            ("vehicle_types", self.vehicle_types),
            ("congestion_trend", self.congestion_trend),
            ("waiting_time", self.waiting_time),
            ("heatmap", self.heatmap),
        ]
        if self.speed_timeline is not None:
            datasets.insert(1, ("speed_timeline_kmh", self.speed_timeline))

        for name, frame in datasets:
            sections.append(f"# {name}")
            sections.append(frame.to_csv(index=False))
            sections.append("")

        return "\n".join(sections)
