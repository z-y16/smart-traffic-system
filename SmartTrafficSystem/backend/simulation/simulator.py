"""Simulated data generation for demo and development mode."""

from __future__ import annotations

import math
import random
import time
from datetime import datetime

import psutil

from backend.models.system_state import (
    ConnectionStatus,
    DashboardMetrics,
    EmergencyStatus,
    SystemHealth,
    TrafficMetrics,
)
from config.settings import Settings, get_settings


class TrafficSimulator:
    """Generates realistic traffic and system telemetry in simulation mode."""

    DENSITY_LEVELS: tuple[str, ...] = ("Low", "Moderate", "High", "Critical")
    LANES: tuple[str, ...] = ("Lane A", "Lane B", "Lane C", "Lane D")

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize simulator with optional settings override."""
        self._settings = settings or get_settings()
        self._seed_time = time.time()
        # Prime psutil's CPU counters. The non-blocking form measures against
        # the previous call, so without this the first reading is always 0.0.
        try:
            psutil.cpu_percent(interval=None)
        except Exception:  # noqa: BLE001
            pass

    def _wave(self, period: float, amplitude: float, offset: float) -> float:
        """Return a smooth oscillating value based on elapsed time."""
        elapsed = time.time() - self._seed_time
        return offset + amplitude * (0.5 + 0.5 * math.sin(elapsed / period))

    def get_emergency_status(self) -> EmergencyStatus:
        """Return simulated emergency status with occasional detection events."""
        elapsed = time.time() - self._seed_time
        cycle = int(elapsed // 120) % 4
        if cycle == 2:
            return EmergencyStatus.DETECTED
        if cycle == 3:
            return EmergencyStatus.ACTIVE
        return EmergencyStatus.NONE

    def get_system_health(self) -> SystemHealth:
        """Return simulated subsystem connectivity when in simulation mode."""
        if self._settings.simulation_mode:
            return SystemHealth(
                camera_status=ConnectionStatus.SIMULATED,
                yolo_status=ConnectionStatus.SIMULATED,
                esp32_status=ConnectionStatus.SIMULATED,
                serial_status=ConnectionStatus.SIMULATED,
                database_status=ConnectionStatus.ONLINE,
                system_status=ConnectionStatus.ONLINE,
            )
        return SystemHealth(
            camera_status=ConnectionStatus.OFFLINE,
            yolo_status=ConnectionStatus.OFFLINE,
            esp32_status=ConnectionStatus.OFFLINE,
            serial_status=ConnectionStatus.OFFLINE,
            database_status=ConnectionStatus.DEGRADED,
            system_status=ConnectionStatus.DEGRADED,
        )

    def get_traffic_metrics(self) -> TrafficMetrics:
        """Return simulated live traffic measurements."""
        vehicle_count = int(self._wave(period=45.0, amplitude=18.0, offset=24.0))
        vehicle_count = max(0, vehicle_count + random.randint(-3, 3))

        if vehicle_count < 12:
            density = "Low"
        elif vehicle_count < 28:
            density = "Moderate"
        elif vehicle_count < 40:
            density = "High"
        else:
            density = "Critical"

        lane_index = int(time.time() // 30) % len(self.LANES)
        average_speed = round(self._wave(period=60.0, amplitude=15.0, offset=42.0), 1)
        fps = round(self._wave(period=20.0, amplitude=4.0, offset=26.0), 1)

        return TrafficMetrics(
            vehicle_count=vehicle_count,
            traffic_density=density,
            current_lane=self.LANES[lane_index],
            average_speed_kmh=max(5.0, average_speed),
            fps=max(15.0, fps),
        )

    def get_resource_usage(self) -> tuple[float, float]:
        """Return CPU and RAM usage percentages.

        ``interval=None`` measures against the previous call instead of
        sleeping. The blocking ``interval=0.1`` this replaces stalled the
        script for 100ms on every reading, and the Live Camera page takes two
        a second — a fifth of that page's time spent asleep inside the call
        reporting how busy the machine was.
        """
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except Exception:
            cpu = round(self._wave(period=30.0, amplitude=12.0, offset=35.0), 1)
            ram = round(self._wave(period=40.0, amplitude=10.0, offset=52.0), 1)
        return cpu, ram

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Build a complete dashboard snapshot."""
        cpu, ram = self.get_resource_usage()
        return DashboardMetrics(
            timestamp=datetime.now(),
            system_health=self.get_system_health(),
            traffic=self.get_traffic_metrics(),
            emergency_status=self.get_emergency_status(),
            cpu_usage_percent=cpu,
            ram_usage_percent=ram,
            simulation_mode=self._settings.simulation_mode,
        )
