"""Typed domain models for dashboard and system telemetry."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


class ConnectionStatus(str, Enum):
    """Represents connectivity state for hardware and services."""

    ONLINE = "Online"
    OFFLINE = "Offline"
    DEGRADED = "Degraded"
    SIMULATED = "Simulated"


class EmergencyStatus(str, Enum):
    """Emergency vehicle priority state."""

    NONE = "None"
    DETECTED = "Detected"
    ACTIVE = "Active"
    CLEARED = "Cleared"


@dataclass
class SystemHealth:
    """Health indicators for integrated subsystems."""

    camera_status: ConnectionStatus
    yolo_status: ConnectionStatus
    esp32_status: ConnectionStatus
    serial_status: ConnectionStatus
    database_status: ConnectionStatus
    system_status: ConnectionStatus


@dataclass
class TrafficMetrics:
    """Live traffic measurements from detection pipeline."""

    vehicle_count: int
    traffic_density: str
    current_lane: str
    average_speed_kmh: float
    fps: float
    emergency_vehicle_count: int = 0

    # Live traffic signal state. The light is demand-responsive: GREEN while
    # there is traffic to serve, RED on an empty road, YELLOW in between.
    light_phase: str = ""
    light_target: str = ""
    light_remaining: float = 0.0
    light_transitioning: bool = False


@dataclass
class DashboardMetrics:
    """Aggregated metrics displayed on the main dashboard."""

    timestamp: datetime
    system_health: SystemHealth
    traffic: TrafficMetrics
    emergency_status: EmergencyStatus
    cpu_usage_percent: float
    ram_usage_percent: float
    simulation_mode: bool


@dataclass
class EmergencyDetection:
    """Emergency vehicle detection details."""

    detected: bool
    confidence: float
    detected_lane: str
    estimated_arrival_seconds: int
    vehicle_type: str


@dataclass
class HardwareStatus:
    """Current status of traffic control hardware."""

    current_signal: str
    speed_bump_status: str
    lane_allocation: str
    led_status: str
    lcd_status: str


@dataclass
class EmergencyControlState:
    """Complete state for emergency control module."""

    detection: EmergencyDetection
    hardware: HardwareStatus
    emergency_active: bool
    priority_lane: str | None


@dataclass
class HardwareMonitorState:
    """Complete hardware monitoring state for all connected devices."""

    esp32_connected: bool
    arduino_connected: bool
    servo_status: str
    led_strip_status: str
    lcd_status: str
    wifi_status: str
    voltage: float
    current: float
    heartbeat: int
    firmware_version: str


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Single log entry with metadata."""

    timestamp: datetime
    level: LogLevel
    message: str
    source: str
    module: str
