"""Simulated detection provider implementing Member 1 interface."""

from backend.interfaces.detection_provider import DetectionProvider
from backend.models.detection import VehicleDetection
from backend.simulation.detection_simulator import DetectionSimulator
from config.settings import Settings, get_settings


class SimulatedDetectionProvider(DetectionProvider):
    """Exposes simulated detections through the Member 1 integration interface."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize with optional settings."""
        self._settings = settings or get_settings()
        self._simulator = DetectionSimulator(self._settings)

    def get_detections(self) -> list[VehicleDetection]:
        """Return detections above the configured confidence threshold."""
        threshold = self._settings.detection_threshold
        return [d for d in self._simulator.get_detections() if d.confidence >= threshold]

    def get_vehicle_count(self) -> int:
        """Return the current number of detected vehicles."""
        return len(self.get_detections())

    def get_vehicle_classes(self) -> list[str]:
        """Return class labels for all current detections."""
        return [d.vehicle_class.value for d in self.get_detections()]

    def get_tracking_ids(self) -> list[int]:
        """Return tracking IDs for all current detections."""
        return [d.tracking_id for d in self.get_detections()]

    def get_traffic_density(self) -> str:
        """Return current traffic density level."""
        count = self.get_vehicle_count()
        if count < 4:
            return "Low"
        if count < 7:
            return "Moderate"
        if count < 9:
            return "High"
        return "Critical"

    def get_emergency_detected(self) -> bool:
        """Return whether an emergency vehicle is detected."""
        return any(d.is_emergency for d in self.get_detections())

    def get_yolo_confidence(self) -> float:
        """Return the highest YOLO confidence score in the current frame."""
        detections = self.get_detections()
        if not detections:
            return 0.0
        return max(d.confidence for d in detections)
