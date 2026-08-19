"""Detection provider interface for Member 1 integration."""

from abc import ABC, abstractmethod

from backend.models.detection import VehicleDetection


class DetectionProvider(ABC):
    """Interface for receiving detections from the YOLO pipeline."""

    @abstractmethod
    def get_vehicle_count(self) -> int:
        """Return the current number of detected vehicles."""

    @abstractmethod
    def get_vehicle_classes(self) -> list[str]:
        """Return class labels for all current detections."""

    @abstractmethod
    def get_tracking_ids(self) -> list[int]:
        """Return tracking IDs for all current detections."""

    @abstractmethod
    def get_traffic_density(self) -> str:
        """Return current traffic density level."""

    @abstractmethod
    def get_emergency_detected(self) -> bool:
        """Return whether an emergency vehicle is detected."""

    @abstractmethod
    def get_yolo_confidence(self) -> float:
        """Return the highest YOLO confidence score in the current frame."""

    @abstractmethod
    def get_detections(self) -> list[VehicleDetection]:
        """Return full detection objects for the current frame."""
