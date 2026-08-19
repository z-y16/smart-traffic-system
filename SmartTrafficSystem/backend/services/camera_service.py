"""Camera frame rendering and detection overlay service."""

from __future__ import annotations

import io
import time
from abc import ABC, abstractmethod
from datetime import datetime

from PIL import Image, ImageDraw

from backend.models.detection import CameraFrame, VehicleDetection
from backend.simulation.detection_simulator import DetectionSimulator
from config.settings import Settings, get_settings


class CameraProvider(ABC):
    """Abstract camera source — replace with OpenCV for live capture."""

    @abstractmethod
    def capture_frame(self) -> Image.Image:
        """Return a raw camera frame without overlays."""


class SimulatedCameraProvider(CameraProvider):
    """Generates a synthetic top-down road scene as placeholder video."""

    FRAME_WIDTH: int = 960
    FRAME_HEIGHT: int = 540
    LANE_Y: tuple[int, ...] = (80, 180, 280, 380)
    LANE_HEIGHT: int = 70

    def __init__(self) -> None:
        """Initialize the simulated camera provider."""
        self._start = time.time()

    def capture_frame(self) -> Image.Image:
        """Render a static road background with subtle animation."""
        image = Image.new("RGB", (self.FRAME_WIDTH, self.FRAME_HEIGHT), "#1a1a2e")
        draw = ImageDraw.Draw(image)

        draw.rectangle([0, 0, self.FRAME_WIDTH, 60], fill="#0f3460")
        draw.text((20, 18), "SMART TRAFFIC CAM — SIMULATED FEED", fill="#94a3b8")

        draw.rectangle([0, 60, self.FRAME_WIDTH, self.FRAME_HEIGHT], fill="#2d3436")

        for index in range(5):
            y = 60 + index * ((self.FRAME_HEIGHT - 60) // 4)
            draw.line([(0, y), (self.FRAME_WIDTH, y)], fill="#636e72", width=2)

        elapsed = time.time() - self._start
        dash_offset = int(elapsed * 60) % 40
        for lane_y in self.LANE_Y:
            center_y = lane_y + self.LANE_HEIGHT // 2
            for x in range(-dash_offset, self.FRAME_WIDTH, 40):
                draw.line([(x, center_y), (x + 20, center_y)], fill="#f1c40f", width=2)

        lane_labels = ["LANE A", "LANE B", "LANE C", "LANE D"]
        for label, lane_y in zip(lane_labels, self.LANE_Y):
            draw.rectangle([8, lane_y + 8, 78, lane_y + 28], fill="#0f172a", outline="#334155")
            draw.text((14, lane_y + 10), label, fill="#38bdf8")

        draw.rectangle([0, self.FRAME_HEIGHT - 36, self.FRAME_WIDTH, self.FRAME_HEIGHT], fill="#0f172a")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, self.FRAME_HEIGHT - 26), f"REC  {timestamp}", fill="#ef4444")
        draw.text((self.FRAME_WIDTH - 180, self.FRAME_HEIGHT - 26), "SIMULATION MODE", fill="#64748b")

        return image


class CameraService:
    """Renders camera frames with detection overlays."""

    NORMAL_BOX_COLOR: str = "#22c55e"
    EMERGENCY_BOX_COLOR: str = "#ef4444"
    ID_BG_COLOR: str = "#0f172a"

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize camera service with simulated provider."""
        self._settings = settings or get_settings()
        self._provider = SimulatedCameraProvider()
        self._detector = DetectionSimulator(self._settings)
        self._last_frame_time = time.time()
        self._fps: float = 24.0

    def _update_fps(self) -> None:
        """Calculate rolling FPS from frame intervals."""
        now = time.time()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self._fps = round(0.85 * self._fps + 0.15 * instant_fps, 1)

    def get_detections(self) -> list[VehicleDetection]:
        """Return current detections from the vision pipeline."""
        threshold = self._settings.detection_threshold
        return [d for d in self._detector.get_detections() if d.confidence >= threshold]

    def render_frame(
        self,
        show_boxes: bool = True,
        show_ids: bool = True,
        show_lanes: bool = True,
        show_labels: bool = True,
    ) -> CameraFrame:
        """Capture and render a frame with optional overlays."""
        self._update_fps()
        base = self._provider.capture_frame()
        detections = self.get_detections()
        frame = base.copy()
        draw = ImageDraw.Draw(frame)

        if show_lanes:
            self._draw_lane_guides(draw)

        for detection in detections:
            self._draw_vehicle_silhouette(draw, detection)
            if show_boxes:
                self._draw_bounding_box(draw, detection, show_ids, show_labels)

        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=85)
        return CameraFrame(
            timestamp=datetime.now(),
            image_bytes=buffer.getvalue(),
            detections=detections,
            fps=self._fps,
            frame_width=SimulatedCameraProvider.FRAME_WIDTH,
            frame_height=SimulatedCameraProvider.FRAME_HEIGHT,
            source="simulated" if self._settings.simulation_mode else "camera",
        )

    def _draw_lane_guides(self, draw: ImageDraw.ImageDraw) -> None:
        """Highlight lane boundaries on the frame."""
        width = SimulatedCameraProvider.FRAME_WIDTH
        for lane_y in SimulatedCameraProvider.LANE_Y:
            y2 = lane_y + SimulatedCameraProvider.LANE_HEIGHT
            draw.rectangle([0, lane_y, width, y2], outline="#334155", width=1)

    def _draw_vehicle_silhouette(self, draw: ImageDraw.ImageDraw, detection: VehicleDetection) -> None:
        """Draw a filled vehicle shape beneath the bounding box."""
        bbox = detection.bbox
        color = "#7f1d1d" if detection.is_emergency else "#475569"
        draw.rounded_rectangle(
            [bbox.x1 + 2, bbox.y1 + 2, bbox.x2 - 2, bbox.y2 - 2],
            radius=4,
            fill=color,
        )

    def _draw_bounding_box(
        self,
        draw: ImageDraw.ImageDraw,
        detection: VehicleDetection,
        show_ids: bool,
        show_labels: bool,
    ) -> None:
        """Draw bounding box, tracking ID, and class label."""
        bbox = detection.bbox
        color = self.EMERGENCY_BOX_COLOR if detection.is_emergency else self.NORMAL_BOX_COLOR
        width = 3 if detection.is_emergency else 2

        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color, width=width)

        if detection.is_emergency:
            draw.rectangle([bbox.x1 - 1, bbox.y1 - 1, bbox.x2 + 1, bbox.y2 + 1], outline="#fca5a5", width=1)

        label_parts: list[str] = []
        if show_ids:
            label_parts.append(f"ID:{detection.tracking_id}")
        if show_labels:
            label_parts.append(detection.vehicle_class.value.upper())
            label_parts.append(f"{detection.confidence:.0%}")

        if label_parts:
            label = " | ".join(label_parts)
            text_y = max(62, bbox.y1 - 18)
            text_width = len(label) * 7 + 8
            draw.rectangle([bbox.x1, text_y, bbox.x1 + text_width, text_y + 16], fill=self.ID_BG_COLOR)
            draw.text((bbox.x1 + 4, text_y + 1), label, fill=color)
