"""Service for hardware device monitoring and telemetry."""

from __future__ import annotations

from backend.communication.node_link import get_node_link
from backend.models.system_state import HardwareMonitorState
from backend.services.system_status_service import BASE_URL
from backend.simulation.hardware_simulator import HardwareSimulator
from config.settings import Settings, get_settings

#: What each LED guide mode is called on the monitor page.
LED_MODE_LABELS: dict[str, str] = {
    "POLICE": "Police",
    "AMBULANCE": "Ambulance",
    "SPEED": "Speed",
    "OFF": "Off",
}


class HardwareMonitorService:
    """Manages hardware device monitoring and status tracking."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service with optional settings."""
        self._settings = settings or get_settings()
        self._simulator = HardwareSimulator(self._settings)

    def get_hardware_state(self) -> HardwareMonitorState:
        """Return current hardware monitoring state.

        Outside simulation mode the board's real state is read from the CV
        node, which owns the serial link. Voltage and current stay simulated:
        this rig has no power sensor to read them from.
        """
        state = self._simulator.get_hardware_monitor_state()
        if self._settings.simulation_mode:
            return state

        link = get_node_link(BASE_URL)
        status = link.get_status()
        node_up = link.is_connected() and bool(status)

        # Reaching the node is not the same as the node having the board open:
        # reporting the controller as connected because its *host* answered
        # would hide an unplugged Arduino behind a healthy-looking page.
        board_up = node_up and bool(status.get("arduino_connected"))
        state.esp32_connected = board_up
        state.arduino_connected = board_up
        state.wifi_status = "Not Used (USB)"

        if node_up:
            mode = str(status.get("led_mode", "OFF")).upper()
            state.led_strip_status = LED_MODE_LABELS.get(mode, mode.title())
            lcd = " ".join(
                part for part in (
                    str(status.get("lcd_line1", "")).strip(),
                    str(status.get("lcd_line2", "")).strip(),
                ) if part
            )
            state.lcd_status = "Displaying" if lcd else "Idle"
            # The signal servo is the one output the node reports an angle for.
            state.servo_status = f"{int(status.get('servo_angle', 90))}°"
        return state

    def is_esp32_connected(self) -> bool:
        """Check if ESP32 is connected."""
        state = self.get_hardware_state()
        return state.esp32_connected

    def is_arduino_connected(self) -> bool:
        """Check if Arduino is connected."""
        state = self.get_hardware_state()
        return state.arduino_connected

    def get_power_metrics(self) -> tuple[float, float]:
        """Return voltage and current readings."""
        state = self.get_hardware_state()
        return state.voltage, state.current

    def get_heartbeat(self) -> int:
        """Return current heartbeat counter."""
        state = self.get_hardware_state()
        return state.heartbeat
