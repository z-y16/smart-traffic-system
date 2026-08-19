"""Simulated vehicle detections for the live camera feed."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from backend.models.detection import BoundingBox, VehicleClass, VehicleDetection
from config.settings import Settings, get_settings


@dataclass
class _SimVehicle:
    """Internal state for a simulated tracked vehicle."""

    tracking_id: int
    vehicle_class: VehicleClass
    lane_index: int
    x_offset: float
    speed: float
    width: int
    height: int
    confidence: float
    is_emergency: bool = False


class DetectionSimulator:
    """Generates animated fake detections across multiple lanes."""

    FRAME_WIDTH: int = 960
    FRAME_HEIGHT: int = 540
    LANE_Y_POSITIONS: tuple[int, ...] = (95, 195, 295, 395)
    LANE_NAMES: tuple[str, ...] = ("Lane A", "Lane B", "Lane C", "Lane D")

    VEHICLE_PROFILES: tuple[tuple[VehicleClass, int, int, float], ...] = (
        (VehicleClass.CAR, 52, 28, 0.91),
        (VehicleClass.CAR, 48, 26, 0.88),
        (VehicleClass.TRUCK, 68, 34, 0.85),
        (VehicleClass.BUS, 72, 36, 0.87),
        (VehicleClass.MOTORCYCLE, 36, 22, 0.79),
        (VehicleClass.CAR, 50, 27, 0.92),
        (VehicleClass.TRUCK, 65, 32, 0.83),
    )

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize tracked vehicles for simulation."""
        self._settings = settings or get_settings()
        self._seed_time = time.time()
        self._vehicles = self._build_initial_vehicles()

    def _build_initial_vehicles(self) -> list[_SimVehicle]:
        """Create the initial set of simulated vehicles."""
        vehicles: list[_SimVehicle] = []
        for index, (vclass, width, height, confidence) in enumerate(self.VEHICLE_PROFILES):
            lane = index % len(self.LANE_Y_POSITIONS)
            vehicles.append(
                _SimVehicle(
                    tracking_id=100 + index,
                    vehicle_class=vclass,
                    lane_index=lane,
                    x_offset=float(index * 110 + 40),
                    speed=1.8 + (index % 3) * 0.6,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
            )
        return vehicles

    def _emergency_cycle_active(self) -> bool:
        """Return whether the emergency ambulance should be visible."""
        elapsed = time.time() - self._seed_time
        return int(elapsed // 90) % 2 == 1

    def _get_emergency_vehicle(self) -> _SimVehicle | None:
        """Return a simulated ambulance when the emergency cycle is active."""
        if not self._emergency_cycle_active():
            return None
        elapsed = time.time() - self._seed_time
        x_pos = (elapsed * 120.0) % (self.FRAME_WIDTH + 100) - 50
        return _SimVehicle(
            tracking_id=999,
            vehicle_class=VehicleClass.AMBULANCE,
            lane_index=1,
            x_offset=x_pos,
            speed=3.5,
            width=58,
            height=30,
            confidence=0.96,
            is_emergency=True,
        )

    def get_detections(self) -> list[VehicleDetection]:
        """Return current frame detections with updated positions."""
        elapsed = time.time() - self._seed_time
        detections: list[VehicleDetection] = []

        for vehicle in self._vehicles:
            x_pos = (vehicle.x_offset + elapsed * vehicle.speed * 40.0) % (self.FRAME_WIDTH + 80)
            x_pos -= 40
            detections.append(self._to_detection(vehicle, x_pos))

        emergency = self._get_emergency_vehicle()
        if emergency is not None:
            detections.append(self._to_detection(emergency, emergency.x_offset))

        return detections

    def _to_detection(self, vehicle: _SimVehicle, x_pos: float) -> VehicleDetection:
        """Convert internal vehicle state to a public detection model."""
        lane_y = self.LANE_Y_POSITIONS[vehicle.lane_index]
        x1 = int(max(0, min(self.FRAME_WIDTH - vehicle.width, x_pos)))
        y1 = lane_y
        bbox = BoundingBox(x1=x1, y1=y1, x2=x1 + vehicle.width, y2=y1 + vehicle.height)

        confidence = vehicle.confidence + 0.02 * math.sin(time.time() / 3.0 + vehicle.tracking_id)
        confidence = round(min(0.99, max(0.70, confidence)), 2)

        return VehicleDetection(
            tracking_id=vehicle.tracking_id,
            vehicle_class=vehicle.vehicle_class,
            confidence=confidence,
            bbox=bbox,
            lane=self.LANE_NAMES[vehicle.lane_index],
            is_emergency=vehicle.is_emergency,
            speed_kmh=round(vehicle.speed * 18.0, 1),
        )

    @property
    def lane_names(self) -> list[str]:
        """Return configured lane labels."""
        return list(self.LANE_NAMES)
