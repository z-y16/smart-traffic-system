"""Hardware transport that routes commands through the CV node.

The dashboard does not open a serial port. ``broadcast_server.py`` already owns
the Arduino -- on Windows a second process cannot open the same COM port at all
-- so manual hardware controls are POSTed to the node over the same HTTP link
that already carries telemetry, and the node forwards them down the wire.

That also means the dashboard keeps working when it is running on a different
machine from the camera, which is the arrangement ``start.py --node`` exists to
support.
"""

from __future__ import annotations

from typing import Any

import requests

from backend.communication.interfaces import CommunicationInterface


#: A hardware command should either land quickly or be reported as not landing;
#: the servo sweep itself is acknowledged by the node, not waited on here.
COMMAND_TIMEOUT_SECONDS: float = 4.0


class NodeCommandLink(CommunicationInterface):
    """Send hardware commands to the CV node's ``/command`` endpoint."""

    def __init__(self, base_url: str, timeout: float = COMMAND_TIMEOUT_SECONDS) -> None:
        """Store the node address without contacting it."""
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._connected = False
        self._last_error = ""
        self._last_status: dict[str, Any] = {}

    @property
    def last_error(self) -> str:
        """Return the latest transport error, or an empty string."""
        return self._last_error

    @property
    def last_status(self) -> dict[str, Any]:
        """Return the most recent hardware state the node reported."""
        return dict(self._last_status)

    def connect(self) -> bool:
        """Check the node is reachable, without holding anything open."""
        try:
            response = requests.get(f"{self._base_url}/health", timeout=self._timeout)
            self._connected = response.status_code == 200
            self._last_error = "" if self._connected else f"node returned {response.status_code}"
        except requests.exceptions.RequestException as exc:
            self._connected = False
            self._last_error = str(exc)
        return self._connected

    def disconnect(self) -> None:
        """Mark the link unused; HTTP holds no resource to release."""
        self._connected = False

    def is_connected(self) -> bool:
        """Return whether the node answered the last time it was asked."""
        return self._connected

    def send_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST one command to the node and return its acknowledgement."""
        try:
            response = requests.post(
                f"{self._base_url}/command",
                json={"command": command, "payload": payload or {}},
                timeout=self._timeout,
            )
        except requests.exceptions.RequestException as exc:
            self._connected = False
            self._last_error = str(exc)
            return {
                "success": False,
                "command": command,
                "error": f"CV node unreachable: {exc}",
            }

        try:
            body = response.json()
        except ValueError:
            self._last_error = f"node returned non-JSON ({response.status_code})"
            return {"success": False, "command": command, "error": self._last_error}

        self._connected = True
        if body.get("success"):
            self._last_error = ""
            self._last_status = body
        else:
            self._last_error = str(body.get("error", "command refused"))
        return body

    def get_status(self) -> dict[str, Any]:
        """Return the node's current hardware state, read from telemetry."""
        try:
            response = requests.get(f"{self._base_url}/telemetry", timeout=self._timeout)
            response.raise_for_status()
            telemetry = response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._connected = False
            self._last_error = str(exc)
            return self.last_status

        self._connected = True
        self._last_error = ""
        status = dict(telemetry.get("hardware") or {})
        if status:
            self._last_status = status
        return status


_instances: dict[str, NodeCommandLink] = {}


def get_node_link(base_url: str) -> NodeCommandLink:
    """Return the shared command link for a node address."""
    if base_url not in _instances:
        _instances[base_url] = NodeCommandLink(base_url)
    return _instances[base_url]
