"""Simulated hardware monitoring data for demo and development mode."""

from __future__ import annotations

import random
import time

from backend.models.system_state import HardwareMonitorState
from config.settings import Settings, get_settings


class HardwareSimulator:
    """Generates realistic hardware monitoring telemetry in simulation mode."""

    SERVO_STATES: tuple[str, ...] = ("Idle", "Active", "Moving", "Error")
    # The strip is a driver-facing guide, so it shows the meanings the board
    # actually animates rather than raw colours.
    LED_STRIP_STATES: tuple[str, ...] = ("Off", "Police", "Ambulance", "Speed")
    LCD_STATES: tuple[str, ...] = ("Idle", "Displaying", "Error")
    WIFI_STATES: tuple[str, ...] = ("Connected", "Disconnected", "Weak Signal")
    FIRMWARE_VERSION = "v2.4.1"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize simulator with optional settings override."""
        self._settings = settings or get_settings()
        self._seed_time = time.time()
        self._heartbeat_counter = 0

    def _wave(self, period: float, amplitude: float, offset: float) -> float:
        """Return a smooth oscillating value based on elapsed time."""
        elapsed = time.time() - self._seed_time
        import math
        return offset + amplitude * (0.5 + 0.5 * math.sin(elapsed / period))

    def get_hardware_monitor_state(self) -> HardwareMonitorState:
        """Return complete hardware monitoring state."""
        if self._settings.simulation_mode:
            esp32_connected = True
            arduino_connected = True
        else:
            esp32_connected = False
            arduino_connected = False

        servo_status = random.choice(self.SERVO_STATES)
        led_strip_status = random.choice(self.LED_STRIP_STATES)
        lcd_status = random.choice(self.LCD_STATES)
        wifi_status = random.choice(self.WIFI_STATES)

        voltage = round(self._wave(period=60.0, amplitude=0.5, offset=5.0), 2)
        current = round(self._wave(period=45.0, amplitude=0.3, offset=1.2), 2)

        self._heartbeat_counter = (self._heartbeat_counter + 1) % 65536
        heartbeat = self._heartbeat_counter

        return HardwareMonitorState(
            esp32_connected=esp32_connected,
            arduino_connected=arduino_connected,
            servo_status=servo_status,
            led_strip_status=led_strip_status,
            lcd_status=lcd_status,
            wifi_status=wifi_status,
            voltage=voltage,
            current=current,
            heartbeat=heartbeat,
            firmware_version=self.FIRMWARE_VERSION,
        )
