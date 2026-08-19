"""USB serial transport for the ESP32 traffic-light controller."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

try:
    import serial
    from serial import SerialException
except ImportError:  # Keeps simulation mode usable before dependencies are installed.
    serial = None  # type: ignore[assignment]

    class SerialException(Exception):
        """Fallback exception used when pyserial is unavailable."""

from backend.communication.interfaces import CommunicationInterface


class SerialCommunication(CommunicationInterface):
    """Exchange newline-delimited JSON messages with an ESP32 over USB."""

    def __init__(self, port: str, baud_rate: int, timeout: float = 6.0) -> None:
        """Store connection settings without opening the serial port."""
        self._port = port
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()
        self._last_status: dict[str, Any] = {}
        self._last_error = ""

    @property
    def last_error(self) -> str:
        """Return the latest connection or protocol error."""
        return self._last_error

    @property
    def last_status(self) -> dict[str, Any]:
        """Return the most recently confirmed hardware status."""
        return dict(self._last_status)

    def connect(self) -> bool:
        """Open the configured COM port and allow the ESP32 reset to finish."""
        if self.is_connected():
            return True
        if serial is None:
            self._last_error = "pyserial is not installed; run pip install -r requirements.txt"
            return False
        try:
            self._serial = serial.Serial(
                self._port,
                self._baud_rate,
                timeout=0.1,
                write_timeout=self._timeout,
            )
            time.sleep(2.0)
            self._serial.reset_input_buffer()
            self._last_error = ""
            return True
        except (SerialException, OSError) as exc:
            self._last_error = str(exc)
            self._serial = None
            return False

    def disconnect(self) -> None:
        """Close the active serial port."""
        if self._serial is not None:
            try:
                self._serial.close()
            except (SerialException, OSError):
                pass
        self._serial = None

    def is_connected(self) -> bool:
        """Return whether the serial port is currently open."""
        return self._serial is not None and bool(self._serial.is_open)

    def _read_response(self) -> dict[str, Any]:
        """Read JSON lines until a protocol response or timeout is received."""
        if self._serial is None:
            return {"success": False, "error": "Serial port is not open"}
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            try:
                line = self._serial.readline().decode("utf-8", errors="replace").strip()
            except (SerialException, OSError) as exc:
                self._last_error = str(exc)
                self.disconnect()
                return {"success": False, "error": self._last_error}
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("type") == "status":
                self._last_status = message
            if message.get("type") in {"ack", "error", "status"}:
                message.setdefault("success", message.get("type") != "error")
                return message
        self._last_error = "Controller response timed out"
        return {"success": False, "error": self._last_error}

    def send_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one JSON command and wait for its hardware acknowledgement."""
        with self._lock:
            if not self.connect():
                return {"success": False, "error": self._last_error}
            request = {"command": command, "payload": payload or {}}
            try:
                assert self._serial is not None
                self._serial.write((json.dumps(request) + "\n").encode("utf-8"))
                self._serial.flush()
            except (SerialException, OSError) as exc:
                self._last_error = str(exc)
                self.disconnect()
                return {"success": False, "error": self._last_error}
            return self._read_response()

    def get_status(self) -> dict[str, Any]:
        """Request and return the ESP32's current physical/controller state."""
        response = self.send_command("GET_STATUS")
        return response if response.get("success") else self.last_status


_instances: dict[tuple[str, int], SerialCommunication] = {}


def get_serial_communication(port: str, baud_rate: int) -> SerialCommunication:
    """Return the shared transport for a COM-port/baud-rate pair."""
    key = (port, baud_rate)
    if key not in _instances:
        _instances[key] = SerialCommunication(port, baud_rate)
    return _instances[key]
