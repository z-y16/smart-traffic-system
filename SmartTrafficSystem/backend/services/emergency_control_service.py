"""Service for emergency vehicle detection and control operations."""

from __future__ import annotations

from typing import Any

import requests

from backend.communication.interfaces import CommunicationInterface, SimulatedCommunication
from backend.communication.node_link import get_node_link
from backend.models.system_state import EmergencyControlState
from backend.services.system_status_service import (
    BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    TELEMETRY_URL,
)
from backend.simulation.emergency_simulator import EmergencySimulator
from config.settings import Settings, get_settings

# How the broadcast server's recognised vehicle types are shown to operators.
VEHICLE_TYPE_LABELS: dict[str, str] = {
    "ambulance": "Ambulance",
    "police": "Police Vehicle",
    "fire truck": "Fire Truck",
    "unknown": "Emergency Vehicle (type unconfirmed)",
}

#: How the controller names each divider position, and how an operator reads it.
LANE_ALLOCATION_LABELS: dict[str, str] = {
    "BALANCED_3_3": "Balanced: 3 / 3",
    "FORWARD_4_OPPOSITE_2": "Forward: 4 / Opposite: 2",
    "FORWARD_2_OPPOSITE_4": "Forward: 2 / Opposite: 4",
    "UNKNOWN": "Unknown — check lane changer",
}


class EmergencyControlService:
    """Manages emergency vehicle detection and hardware control."""

    def __init__(
        self,
        settings: Settings | None = None,
        communication: CommunicationInterface | None = None,
    ) -> None:
        """Initialize service with optional settings and communication interface."""
        self._settings = settings or get_settings()
        self._communication = communication
        self._simulator = EmergencySimulator(self._settings)
        self._last_live_ok = False
        self._last_telemetry: dict = {}

    def _get_communication(self) -> CommunicationInterface:
        """Return the injected transport, or the live link to the CV node.

        In simulation mode nothing should reach hardware, so the simulated
        transport stands in; otherwise commands go to the node that owns the
        serial port.
        """
        if self._communication is not None:
            return self._communication
        if self._settings.simulation_mode:
            return SimulatedCommunication()
        return get_node_link(BASE_URL)

    def _send_command(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one hardware command through the active transport."""
        return self._get_communication().send_command(command, payload)

    @property
    def is_live(self) -> bool:
        """Return whether the last state came from the live broadcast stream."""
        return self._last_live_ok

    @property
    def last_telemetry(self) -> dict:
        """Return the raw telemetry payload behind the most recent state."""
        return self._last_telemetry

    def _fetch_live_telemetry(self) -> dict | None:
        """Fetch telemetry from the broadcast server, or None if unreachable."""
        try:
            response = requests.get(TELEMETRY_URL, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError):
            return None

    def get_emergency_state(self) -> EmergencyControlState:
        """Return current emergency control state, preferring live telemetry.

        When the broadcast server is reachable, the detection block reflects
        what the camera actually recognised (ambulance / police / fire truck)
        and the hardware block mirrors the servo, LED and LCD state the CV node
        is really driving. Lane assignment stays simulated — a single camera
        has no per-lane sensor.
        """
        state = self._simulator.get_emergency_control_state()
        telemetry = self._fetch_live_telemetry()

        if telemetry is None:
            self._last_live_ok = False
            self._last_telemetry = {}
            return state

        self._last_live_ok = True
        self._last_telemetry = telemetry

        detected = bool(telemetry.get("emergency_vehicle_detected"))
        detection = state.detection
        detection.detected = detected
        if detected:
            vehicle_type = str(telemetry.get("emergency_primary_type", "unknown"))
            detection.vehicle_type = VEHICLE_TYPE_LABELS.get(
                vehicle_type, vehicle_type.title()
            )
            confidences = [
                float(d.get("emergency_confidence", 0.0))
                for d in telemetry.get("detections", [])
                if d.get("emergency")
            ]
            detection.confidence = max(confidences) if confidences else 1.0
        else:
            detection.confidence = 0.0

        hardware_state = telemetry.get("hardware") or {}
        if hardware_state:
            hardware = state.hardware
            # The emergency strip is reported ahead of the signal phase,
            # because on this page the question being asked is "is a priority
            # vehicle coming?", not "what colour is the light?". The phase
            # itself is shown on the Hardware Monitor page.
            strip = bool(hardware_state.get("emergency_strip",
                                            hardware_state.get("blue_led")))
            phase = str(telemetry.get("light_phase", "") or "")
            hardware.current_signal = (
                "Blue" if strip else phase.capitalize() if phase else "Off"
            )
            hardware.led_status = hardware.current_signal
            # Both rows matter: line 1 carries the signal, line 2 the emergency
            # banner, so reading only one would drop whichever is interesting.
            lcd = " ".join(
                part for part in (
                    str(hardware_state.get("lcd_line1", "")).strip(),
                    str(hardware_state.get("lcd_line2", "")).strip(),
                ) if part
            )
            hardware.lcd_status = lcd or "Idle"

            # The divider is not derived from the signal servo: that servo is
            # the phase gate and swings with every light change, whereas the
            # divider only moves when an operator moves it.
            #
            # It is read back from whichever end the button actually commanded.
            # In simulation the button never reaches the node, so overlaying the
            # node's copy here would show the barrier snapping back to its old
            # position the instant the page redrew.
            if not self._settings.simulation_mode:
                raw_allocation = str(
                    hardware_state.get("lane_allocation", "UNKNOWN")
                ).upper()
                hardware.lane_allocation = LANE_ALLOCATION_LABELS.get(
                    raw_allocation, raw_allocation.replace("_", " ").title()
                )

        state.emergency_active = detected
        return state

    def activate_emergency_mode(self, lane: str) -> dict[str, Any]:
        """Activate emergency mode for specified lane."""
        if self._settings.simulation_mode:
            success = self._simulator.activate_emergency(lane)
            return {
                "success": success,
                "command": "ACTIVATE_EMERGENCY",
                "lane": lane,
                "simulated": True,
            }

        return self._send_command("ACTIVATE_EMERGENCY", {"lane": lane})

    def deactivate_emergency_mode(self) -> dict[str, Any]:
        """Deactivate emergency mode."""
        if self._settings.simulation_mode:
            success = self._simulator.deactivate_emergency()
            return {
                "success": success,
                "command": "DEACTIVATE_EMERGENCY",
                "simulated": True,
            }

        return self._send_command("DEACTIVATE_EMERGENCY")

    def set_lane_allocation(self, allocation: str) -> dict[str, Any]:
        """Move the reversible divider to the requested lane allocation.

        ``allocation`` is one of ``BALANCED``, ``FORWARD_4`` or ``OPPOSITE_4``:
        six highway lanes split three/three, or four one way and two the other.
        """
        normalized = allocation.strip().upper()
        if self._settings.simulation_mode:
            success = self._simulator.set_lane_allocation(normalized)
            return {
                "success": success,
                "command": "SET_LANE_ALLOCATION",
                "allocation": normalized,
                "simulated": True,
            }

        return self._send_command(
            "SET_LANE_ALLOCATION", {"allocation": normalized}
        )

    def move_lane_divider(self, direction: str) -> dict[str, Any]:
        """Step the divider one lane ``LEFT`` or ``RIGHT``.

        The barrier has three positions — four lanes one way, an even three /
        three, four the other — and this walks between neighbours rather than
        naming one. Which lanes that leaves is worked out where the current
        position is actually held: the CV node when driving hardware, the
        simulator otherwise. Pressing an arrow at the outermost lane reports
        ``at_limit`` instead of moving, so the page can say so rather than
        claiming a move that never happened.
        """
        normalized = direction.strip().upper()
        if normalized not in ("LEFT", "RIGHT"):
            return {
                "success": False,
                "error": f"unknown direction {direction!r}",
            }

        if self._settings.simulation_mode:
            result = self._simulator.move_lane_divider(normalized)
            return {
                "command": "MOVE_LANE_DIVIDER",
                "direction": normalized,
                "simulated": True,
                **result,
            }

        result = self._send_command("MOVE_LANE_DIVIDER", {"direction": normalized})
        # The node answers in the controller's own vocabulary; the page shows
        # the position to an operator, so it is translated here rather than
        # leaving every caller to know both names for the same barrier.
        raw_allocation = str(result.get("lane_allocation", "")).upper()
        if raw_allocation:
            result["lane_allocation"] = LANE_ALLOCATION_LABELS.get(
                raw_allocation, raw_allocation.replace("_", " ").title()
            )
        return result

    def raise_speed_bump(self) -> dict[str, Any]:
        """Raise speed bump to slow traffic."""
        if self._settings.simulation_mode:
            success = self._simulator.raise_speed_bump()
            return {
                "success": success,
                "command": "RAISE_SPEED_BUMP",
                "simulated": True,
            }

        return self._send_command("RAISE_SPEED_BUMP")

    def lower_speed_bump(self) -> dict[str, Any]:
        """Lower speed bump for emergency vehicle passage."""
        if self._settings.simulation_mode:
            success = self._simulator.lower_speed_bump()
            return {
                "success": success,
                "command": "LOWER_SPEED_BUMP",
                "simulated": True,
            }

        return self._send_command("LOWER_SPEED_BUMP")

    def set_led_color(self, color: str) -> dict[str, Any]:
        """Set LED strip color."""
        if self._settings.simulation_mode:
            success = self._simulator.set_led_color(color)
            return {
                "success": success,
                "command": "LED_COLOR",
                "color": color,
                "simulated": True,
            }

        return self._send_command("LED_COLOR", {"color": color})

    def set_led_mode(self, mode: str) -> dict[str, Any]:
        """Start the labelled LED guide animation for a vehicle type.

        The strip is a driver-facing guide rather than a lamp, so it is asked
        for a meaning (``POLICE``, ``AMBULANCE``, ``SPEED``, ``OFF``) and the
        board chooses the colour and flash pattern that goes with it.
        """
        normalized = mode.strip().upper()
        if self._settings.simulation_mode:
            simulation_colors = {
                "POLICE": "Blue",
                "AMBULANCE": "Red",
                "SPEED": "Yellow",
                "OFF": "Off",
            }
            success = self._simulator.set_led_color(
                simulation_colors.get(normalized, "Off")
            )
            return {
                "success": success,
                "command": "LED_MODE",
                "mode": normalized,
                "simulated": True,
            }

        return self._send_command("LED_MODE", {"mode": normalized})

    def send_lcd_message(self, message: str) -> dict[str, Any]:
        """Send message to LCD display."""
        if self._settings.simulation_mode:
            success = self._simulator.send_lcd_message(message)
            return {
                "success": success,
                "command": "LCD_MESSAGE",
                "message": message,
                "simulated": True,
            }

        return self._send_command("LCD_MESSAGE", {"message": message})
