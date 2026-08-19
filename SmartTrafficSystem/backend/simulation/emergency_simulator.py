"""Simulated emergency control data for demo and development mode."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from backend.models.system_state import (
    EmergencyControlState,
    EmergencyDetection,
    HardwareStatus,
)
from config.settings import Settings, get_settings


# Streamlit rebuilds the services on every rerun, so anything an operator sets
# has to outlive the simulator instance or the card would snap back to its
# default the moment the page redraws -- a button that reports success and then
# visibly does nothing. On real hardware the CV node holds this; in simulation
# it is held here for the same reason.
_simulated_lane_allocation: str = "Balanced: 3 / 3"


class EmergencySimulator:
    """Generates realistic emergency control telemetry in simulation mode."""

    VEHICLE_TYPES: tuple[str, ...] = ("Ambulance", "Fire Truck", "Police Car")
    LANES: tuple[str, ...] = ("Lane A", "Lane B", "Lane C", "Lane D")
    SIGNALS: tuple[str, ...] = ("Red", "Green", "Yellow")
    SPEED_BUMP_STATES: tuple[str, ...] = ("Lowered", "Raised")
    LANE_ALLOCATION_STATES: tuple[str, ...] = (
        "Balanced: 3 / 3",
        "Forward: 4 / Opposite: 2",
        "Forward: 2 / Opposite: 4",
    )
    #: The same three states in physical order, left to right, so the
    #: dashboard's arrows can step the divider one lane at a time. Moving the
    #: barrier left hands the extra lane to the opposite direction.
    LANE_DIVIDER_POSITIONS: tuple[str, ...] = (
        "Forward: 2 / Opposite: 4",
        "Balanced: 3 / 3",
        "Forward: 4 / Opposite: 2",
    )
    LED_STATES: tuple[str, ...] = ("Off", "Red", "Green", "Yellow")
    LCD_STATES: tuple[str, ...] = ("Idle", "Displaying Message")

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize simulator with optional settings override."""
        self._settings = settings or get_settings()
        self._seed_time = time.time()
        self._emergency_active = False
        self._priority_lane = None

    def _get_emergency_detection(self) -> EmergencyDetection:
        """Return simulated emergency detection with occasional events."""
        elapsed = time.time() - self._seed_time
        cycle = int(elapsed // 120) % 4

        if cycle == 2:
            detected = True
            confidence = round(random.uniform(0.85, 0.98), 3)
            detected_lane = random.choice(self.LANES)
            estimated_arrival = random.randint(15, 45)
            vehicle_type = random.choice(self.VEHICLE_TYPES)
        elif cycle == 3:
            detected = True
            confidence = round(random.uniform(0.90, 0.99), 3)
            detected_lane = self._priority_lane or random.choice(self.LANES)
            estimated_arrival = random.randint(5, 20)
            vehicle_type = random.choice(self.VEHICLE_TYPES)
        else:
            detected = False
            confidence = 0.0
            detected_lane = "N/A"
            estimated_arrival = 0
            vehicle_type = "N/A"

        return EmergencyDetection(
            detected=detected,
            confidence=confidence,
            detected_lane=detected_lane,
            estimated_arrival_seconds=estimated_arrival,
            vehicle_type=vehicle_type,
        )

    def _get_hardware_status(self) -> HardwareStatus:
        """Return simulated hardware status based on emergency state."""
        if self._emergency_active:
            current_signal = "Green"
            speed_bump_status = "Lowered"
            led_status = "Green"
            lcd_status = "Displaying Message"
        else:
            current_signal = random.choice(self.SIGNALS)
            speed_bump_status = random.choice(self.SPEED_BUMP_STATES)
            led_status = random.choice(self.LED_STATES)
            lcd_status = random.choice(self.LCD_STATES)

        # The divider only moves when it is told to, so it is held state
        # rather than something to re-roll on every read.
        return HardwareStatus(
            current_signal=current_signal,
            speed_bump_status=speed_bump_status,
            lane_allocation=_simulated_lane_allocation,
            led_status=led_status,
            lcd_status=lcd_status,
        )

    def get_emergency_control_state(self) -> EmergencyControlState:
        """Return complete emergency control state."""
        detection = self._get_emergency_detection()

        if detection.detected and not self._emergency_active:
            self._emergency_active = True
            self._priority_lane = detection.detected_lane
        elif not detection.detected and self._emergency_active:
            elapsed = time.time() - self._seed_time
            cycle = int(elapsed // 120) % 4
            if cycle == 0:
                self._emergency_active = False
                self._priority_lane = None

        hardware = self._get_hardware_status()

        return EmergencyControlState(
            detection=detection,
            hardware=hardware,
            emergency_active=self._emergency_active,
            priority_lane=self._priority_lane,
        )

    def activate_emergency(self, lane: str) -> bool:
        """Activate emergency mode for specified lane."""
        self._emergency_active = True
        self._priority_lane = lane
        return True

    def deactivate_emergency(self) -> bool:
        """Deactivate emergency mode."""
        self._emergency_active = False
        self._priority_lane = None
        return True

    def set_lane_allocation(self, allocation: str) -> bool:
        """Set the simulated reversible-lane allocation."""
        global _simulated_lane_allocation
        labels = {
            "BALANCED": "Balanced: 3 / 3",
            "FORWARD_4": "Forward: 4 / Opposite: 2",
            "OPPOSITE_4": "Forward: 2 / Opposite: 4",
        }
        if allocation not in labels:
            return False
        _simulated_lane_allocation = labels[allocation]
        return True

    def move_lane_divider(self, direction: str) -> dict[str, object]:
        """Step the simulated divider one lane left or right.

        Unlike the other simulated controls this reports more than success:
        an arrow pressed at the outermost lane is not a failure and not a move
        either, so the caller is told which of the two happened along with the
        position now held.
        """
        global _simulated_lane_allocation
        if direction not in ("LEFT", "RIGHT"):
            return {
                "success": False,
                "error": f"unknown direction {direction!r}",
                "lane_allocation": _simulated_lane_allocation,
            }

        positions = self.LANE_DIVIDER_POSITIONS
        try:
            index = positions.index(_simulated_lane_allocation)
        except ValueError:
            index = positions.index("Balanced: 3 / 3")

        target = index + (1 if direction == "RIGHT" else -1)
        at_limit = not 0 <= target < len(positions)
        if not at_limit:
            _simulated_lane_allocation = positions[target]

        return {
            "success": True,
            "at_limit": at_limit,
            "lane_allocation": _simulated_lane_allocation,
        }

    def raise_speed_bump(self) -> bool:
        """Raise speed bump."""
        return True

    def lower_speed_bump(self) -> bool:
        """Lower speed bump."""
        return True

    def set_led_color(self, color: str) -> bool:
        """Set LED color."""
        return True

    def send_lcd_message(self, message: str) -> bool:
        """Send message to LCD."""
        return True
