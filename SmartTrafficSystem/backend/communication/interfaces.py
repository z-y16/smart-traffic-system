"""Communication interfaces for hardware integration."""

from abc import ABC, abstractmethod
from typing import Any


class CommunicationInterface(ABC):
    """Abstract base for serial, UART, and MQTT transports."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the target device or broker."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the active connection."""

    @abstractmethod
    def send_command(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a command and return the acknowledgement response."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the interface is currently connected."""


class SimulatedCommunication(CommunicationInterface):
    """Simulated transport that returns realistic acknowledgements."""

    def __init__(self) -> None:
        """Initialize simulated connection state."""
        self._connected = False

    def connect(self) -> bool:
        """Simulate a successful connection."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Simulate disconnect."""
        self._connected = False

    def send_command(self, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return a simulated acknowledgement for the given command."""
        emergency_commands = {
            "ACTIVATE_EMERGENCY",
            "DEACTIVATE_EMERGENCY",
            "SET_LANE_ALLOCATION",
            "MOVE_LANE_DIVIDER",
            "RAISE_SPEED_BUMP",
            "LOWER_SPEED_BUMP",
            "LED_COLOR",
            "LED_MODE",
            "LCD_MESSAGE",
        }
        return {
            "success": True,
            "command": command,
            "payload": payload or {},
            "ack": f"ACK:{command}",
            "simulated": True,
            "emergency_related": command in emergency_commands,
        }

    def is_connected(self) -> bool:
        """Return simulated connection state."""
        return self._connected
