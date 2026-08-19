"""Detection and camera frame domain models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class VehicleClass(str, Enum):
    """Supported vehicle classification labels."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    AMBULANCE = "ambulance"


@dataclass
class BoundingBox:
    """Pixel coordinates for a detection bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return bounding box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return bounding box height in pixels."""
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class VehicleDetection:
    """Single detected vehicle from the vision pipeline."""

    tracking_id: int
    vehicle_class: VehicleClass
    confidence: float
    bbox: BoundingBox
    lane: str
    is_emergency: bool = False
    speed_kmh: float = 0.0


@dataclass
class CameraFrame:
    """Rendered camera frame with associated detections."""

    timestamp: datetime
    image_bytes: bytes
    detections: list[VehicleDetection] = field(default_factory=list)
    fps: float = 0.0
    frame_width: int = 960
    frame_height: int = 540
    source: str = "simulated"
