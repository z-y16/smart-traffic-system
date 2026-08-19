"""
traffic_vision.py — Smart Traffic System vision & telemetry engine
==================================================================

Everything that makes the detection pipeline *accurate* lives in this one
module, so `broadcast_server.py` stays a thin orchestration layer.

What this module upgrades over the original inline pipeline
-----------------------------------------------------------
1.  **YOLOv12-m instead of YOLOv12-n** — roughly +12 mAP on COCO. Small and
    distant vehicles that the nano model silently dropped are now detected.
2.  **FP16 (half precision) on CUDA** — nearly doubles throughput, which is
    what buys the headroom to run the bigger model at a bigger input size.
3.  **Higher inference resolution (`imgsz`, default 960)** — the single most
    effective knob for detecting far-away/small vehicles.
4.  **Class-filtered inference** — only vehicle classes are ever decoded, so
    people/traffic lights can never become false "vehicles", and NMS competes
    only among vehicles.
5.  **Class-agnostic NMS** — removes the classic duplicate box where the same
    van is emitted as both `car` and `truck`.
6.  **BoT-SORT tracker with global motion compensation** (tuned config shipped
    in `trackers/traffic_botsort.yaml`) — far fewer ID switches than the stock
    ByteTrack config, which directly improves speed accuracy.
7.  **Track confirmation** — a track must be seen `min_track_frames` times
    before it counts as a vehicle, which removes single-frame false positives.
8.  **Majority-vote class stabilisation** — a vehicle stops flickering between
    `car` and `truck`; its reported class is the mode of its history.
9.  **Real speed in km/h**, not pixels/second (see below).
10. **Hysteresis everywhere** — congestion level and emergency flags need
    sustained evidence to flip, so the LEDs/servo stop chattering.
11. **Crash-safe telemetry logging** — CSV appended every second (never
    corrupts), styled Excel written atomically on a timer and at shutdown.

Speed in km/h
-------------
Pixels-per-second is meaningless without knowing how many metres a pixel is
worth, and that number changes across the image because of perspective. Two
calibration modes are supported:

* ``homography`` (accurate, ~±5%): you mark 4 points on the road surface and
  give their real-world dimensions once, with ``calibrate_speed.py``. Every
  detection's ground-contact point is projected onto the road plane, so
  displacement is measured in real metres regardless of where in the frame it
  happens.
* ``auto`` (zero setup, ~±25%): each vehicle's own bounding-box height is
  compared against the known typical height of its class (a car is ~1.5 m
  tall), which yields a metres-per-pixel scale *at that vehicle's exact
  position* — so perspective is handled implicitly because every vehicle
  carries its own local scale.

Both feed the same estimator, which measures displacement over a short time
window (not frame-to-frame, which is dominated by box jitter), rejects
physically impossible values, and smooths with an EMA.
"""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from emergency_vision import (
    LCD_NAMES,
    EmergencyAppearanceClassifier,
    EmergencyConfig,
    EmergencyVerdict,
    fuse as fuse_emergency,
)
from siren_vision import SirenConfig, SirenDetector

# ---------------------------------------------------------------------------
# Optional imports — kept optional so the module can be unit-tested without a
# GPU/torch install, and so the dashboard can import the constants safely.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly at runtime
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]
    ULTRALYTICS_AVAILABLE = False

try:  # pragma: no cover
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
# VEHICLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# COCO class ids for the vehicle classes we care about.
COCO_VEHICLE_IDS: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck

# Passenger-car-equivalent weights: a bus occupies the road like 3 cars.
CLASS_WEIGHTS: dict[str, float] = {
    "car": 1.0,
    "motorcycle": 0.5,
    "bus": 3.0,
    "truck": 3.0,
}

# Typical real-world dimensions in metres, used by the auto-calibrator.
# Height is used as the primary cue because, unlike width, it barely changes
# with the vehicle's heading relative to the camera (a car seen from the side
# is ~4.5 m "wide" in the image but still ~1.5 m tall).
VEHICLE_HEIGHT_M: dict[str, float] = {
    "car": 1.50,
    "motorcycle": 1.60,
    "bus": 3.20,
    "truck": 3.50,
}
VEHICLE_WIDTH_M: dict[str, float] = {
    "car": 1.80,
    "motorcycle": 0.80,
    "bus": 2.55,
    "truck": 2.55,
}

CONGESTION_LEVELS: tuple[str, ...] = ("FREE", "MODERATE", "HEAVY", "EMERGENCY")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VisionConfig:
    """Tunable parameters for the whole detection/telemetry pipeline."""

    # ── Model ───────────────────────────────────────────────────────────────
    # imgsz 1280 rather than 960 because at night the vehicles that matter are
    # small and dark. Measured over 30 frames of the night footage: 960 finds
    # 15.6 vehicles per frame, 1280 finds 19.6 — and a *larger model* at 960
    # does not close that gap, because the limit is how many pixels a distant
    # car occupies, not how much capacity is spent on them. yolo12x at 1280
    # actually found fewer (17.9) while running 2.5x slower, so resolution is
    # bought here and model size is not.
    #
    # yolo11m rather than yolo12m: on the same night footage it is 33% faster
    # (24.6 ms vs 37.1 ms per frame) *and* finds more vehicles (20.4 vs 19.9).
    # YOLO12's attention blocks (A2C2f) cost real time on an Ada laptop GPU and
    # buy nothing at this resolution, so the family — not just the size — was
    # the wrong choice. Full pipeline: 11.9 -> 15.1 FPS from this alone.
    model_path: str = "yolo11m.pt"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.50
    max_det: int = 100
    device: str = "auto"          # "auto" | "cuda" | "cpu" | "0"
    half: bool = True             # FP16 — ignored on CPU
    agnostic_nms: bool = True
    augment: bool = False         # test-time augmentation: +accuracy, ~3x slower
    tracker: str = "trackers/traffic_botsort.yaml"

    # ── Track quality gates ─────────────────────────────────────────────────
    min_track_frames: int = 3     # frames a track must survive before counting
    class_vote_window: int = 30   # frames of class history used for the vote
    track_ttl_seconds: float = 3.0  # drop bookkeeping for tracks unseen this long

    # ── Speed ───────────────────────────────────────────────────────────────
    calibration_path: str = "speed_calibration.json"
    speed_window_seconds: float = 0.80   # measurement window per vehicle
    speed_min_window_seconds: float = 0.25
    speed_min_samples: int = 3
    speed_ema_alpha: float = 0.35
    speed_max_plausible_kmh: float = 200.0
    auto_scale_correction: float = 1.0   # multiply auto-calibrated scale if biased
    stationary_kmh: float = 3.0          # below this a vehicle counts as stopped

    # ── Congestion ──────────────────────────────────────────────────────────
    free_flow_speed_kmh: float = 50.0
    max_capacity_pce: float = 20.0       # weighted vehicles = "full" road
    speed_weight: float = 0.60
    density_weight: float = 0.40
    speed_relevance_vehicles: int = 3    # slow traffic only means congestion above this
    threshold_low: float = 0.30
    threshold_medium: float = 0.60
    level_hysteresis: float = 0.05       # CI must overshoot by this to change level
    level_min_dwell_seconds: float = 1.5

    # ── Emergency vehicle recognition ───────────────────────────────────────
    # Two cues, with a deliberate hierarchy: the *siren light* decides whether
    # a vehicle is on a call, and CLIP's reading of the livery only names the
    # type when the beacon colour cannot (red is both ambulance and fire).
    # Set ``emergency_require_siren=False`` to let a clearly marked vehicle
    # trigger with its lights off — appropriate for a hospital approach, and
    # the setting to avoid on a public road, where it is what made the system
    # call 29% of ordinary night traffic a police car.
    emergency_appearance: bool = True    # recognise ambulance/police/fire by sight
    # clip | trained | auto. CLIP by default: livery now reports an emergency
    # rather than only annotating one, and on footage it was not trained on the
    # fine-tuned model called six ordinary night cars "fire truck" at 0.94-1.00
    # confidence while missing a real ambulance. See EmergencyConfig.backend.
    emergency_backend: str = "clip"
    emergency_model_path: str = "emergency_cls.pt"   # for backend="trained"
    emergency_clip_model: str = "ViT-B/16"
    emergency_check_interval: int = 5    # frames between re-checks of a track
    emergency_require_siren: bool = True

    # ── Siren beacon detection (see siren_vision.py) ────────────────────────
    siren_detection: bool = True
    siren_min_box_px: int = 26           # ignore vehicles smaller than this
    siren_window_seconds: float = 3.4    # must hold 3 flashes of a slow beacon
    siren_min_flashes: int = 3
    siren_hold_seconds: float = 3.0      # stay flagged this long between flashes
    siren_blue_type: str = "police"      # colour → vehicle type, by country
    siren_red_type: str = "ambulance"
    siren_both_type: str = "police"

    def resolved_device(self) -> str:
        """Return the concrete torch device string to run inference on."""
        if self.device != "auto":
            return self.device
        if TORCH_AVAILABLE and torch.cuda.is_available():  # pragma: no cover
            return "cuda"
        return "cpu"

    def use_half(self) -> bool:
        """Return whether FP16 inference should actually be enabled."""
        return bool(self.half) and self.resolved_device() != "cpu"

    @classmethod
    def load(cls, path: str | Path) -> "VisionConfig":
        """Load config from JSON, ignoring unknown keys, or return defaults."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Write the config to JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.__dict__, handle, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VehicleDetection:
    """One confirmed, tracked vehicle in the current frame."""

    track_id: int
    label: str                      # majority-vote stabilised class
    raw_label: str                  # what YOLO said this frame
    confidence: float
    bbox: tuple[int, int, int, int]
    speed_kmh: float | None
    is_emergency: bool
    emergency_confidence: float
    age_frames: int
    world_xy: tuple[float, float] | None = None
    emergency_type: str = "ordinary"   # ambulance | police | fire truck | unknown
    emergency_evidence: str = "none"   # siren-blue | siren-red+livery | none
    #: Recognised livery but no working beacon — shown, but given no priority.
    emergency_suspected: bool = False
    siren_colour: str = "none"         # blue | red | red+blue | none
    siren_rate_hz: float = 0.0         # flashes per second actually measured
    #: Whether this vehicle should actually be given priority by the hardware.
    #: An emergency vehicle recognised only by its livery is reported but not
    #: prioritised, so the two flags differ and both are needed.
    emergency_priority: bool = False
    #: Reported despite falling outside the monitored region — only emergency
    #: vehicles ever are, and saying so keeps them out of the traffic counts.
    outside_region: bool = False

    @property
    def speed_text(self) -> str:
        """Human-readable speed for on-screen labels."""
        return "--" if self.speed_kmh is None else f"{self.speed_kmh:.0f}km/h"

    @property
    def emergency_name(self) -> str:
        """Uppercase vehicle type for on-screen labels, e.g. ``AMBULANCE``."""
        if self.emergency_priority:
            return LCD_NAMES.get(self.emergency_type, "EMERGENCY")
        if self.is_emergency or self.emergency_suspected:
            return f"{LCD_NAMES.get(self.emergency_type, 'EMERGENCY')}?"
        return ""


@dataclass
class FrameResult:
    """Everything derived from a single processed frame."""

    frame_index: int
    timestamp: float
    detections: list[VehicleDetection]
    vehicle_counts: dict[str, int]
    total_vehicles: int
    avg_speed_kmh: float
    max_speed_kmh: float
    moving_vehicles: int
    stopped_vehicles: int
    weighted_density: float
    congestion_index: float
    congestion_level: str
    emergency_ids: set[int]
    fps: float
    speed_source: str
    unique_vehicles_session: int
    inference_ms: float = 0.0
    # Which kinds of emergency vehicle are in frame right now, most confident
    # first, e.g. ``["ambulance"]``.
    emergency_types: list[str] = field(default_factory=list)
    emergency_ms: float = 0.0
    #: Vehicles that look like emergency vehicles but are not running a beacon.
    suspected_ids: set[int] = field(default_factory=set)
    #: Rate at which vehicles are being sampled, and whether that is too slow
    #: for a beacon's flashes to survive. A siren missed because the machine
    #: was overloaded should be visible, not silent.
    siren_sample_hz: float = 0.0
    siren_undersampled: bool = False
    #: Vehicles the detector found but the region of interest excluded, by
    #: class. Reported rather than dropped in silence: a region drawn around
    #: one carriageway discards everything in the next lane over, and a class
    #: that keeps to that lane — motorcycles, typically — then looks to the
    #: operator like a class the detector cannot see.
    outside_roi_counts: dict[str, int] = field(default_factory=dict)
    #: Emergency vehicles whose beacon is actually running, and which therefore
    #: earn priority. Always a subset of ``emergency_ids``: a marked ambulance
    #: with its lights off appears in that set and not in this one.
    priority_ids: set[int] = field(default_factory=set)
    #: Vehicles in shot that have not yet been watched long enough for the
    #: flash test to have any opinion about them. A clip shorter than
    #: ``SirenConfig.min_active_seconds`` produces nothing but these, and
    #: reporting them is the difference between "no emergency vehicle" and
    #: "not enough footage to say".
    siren_pending: int = 0
    #: Vehicles the siren detector is holding evidence for. When
    #: ``siren_pending`` equals this, nothing in shot has been watched long
    #: enough to have been cleared, and "no emergency vehicle" would be a claim
    #: about footage nobody could have judged.
    siren_tracked: int = 0
    #: Whether the calibrated region of interest was applied to this frame. It
    #: is ignored when its aspect ratio does not match the picture, because a
    #: polygon stretched across a different shape excludes an arbitrary part of
    #: the road. See :meth:`GroundPlane.roi_applies_to`.
    roi_active: bool = True

    @property
    def outside_roi(self) -> int:
        """How many detected vehicles fell outside the monitored region."""
        return sum(self.outside_roi_counts.values())

    @property
    def primary_emergency_type(self) -> str:
        """The emergency vehicle type to display and send to the hardware."""
        return self.emergency_types[0] if self.emergency_types else "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION  (pixels → metres)
# ═══════════════════════════════════════════════════════════════════════════


class GroundPlane:
    """Converts image coordinates into real-world metres on the road plane.

    Three modes, in descending order of accuracy:

    * ``homography`` — a 4-point perspective transform produced by
      ``calibrate_speed.py``. Exact for anything moving on the road surface.
    * ``scale`` — a single global metres-per-pixel value. Fine for a
      near-overhead camera where perspective is negligible.
    * ``auto`` — no calibration file: scale is inferred per-vehicle from its
      bounding-box height versus the known typical height of its class.
    """

    def __init__(self, calibration: dict[str, Any] | None = None,
                 auto_scale_correction: float = 1.0) -> None:
        """Build a ground plane from a calibration dict (or None for auto)."""
        self._auto_correction = float(auto_scale_correction)
        self._matrix: np.ndarray | None = None
        self._calib_size: tuple[int, int] | None = None
        self._meters_per_pixel: float | None = None
        self._roi: np.ndarray | None = None
        self.mode = "auto"
        self.notes = ""
        #: World-Y of the counting line; None until a homography is loaded.
        self._count_line_y: float | None = None

        if not calibration:
            return

        self.notes = str(calibration.get("notes", ""))
        size = calibration.get("frame_size")
        if size and len(size) == 2:
            self._calib_size = (int(size[0]), int(size[1]))

        roi = calibration.get("roi")
        if roi and len(roi) >= 3:
            self._roi = np.array(roi, dtype=np.float32)

        mode = str(calibration.get("mode", "")).lower()
        if mode == "homography":
            image_points = np.array(calibration["image_points"], dtype=np.float32)
            world_points = np.array(calibration["world_points"], dtype=np.float32)
            if image_points.shape != (4, 2) or world_points.shape != (4, 2):
                raise ValueError("homography calibration needs exactly 4 image and 4 world points")
            self._matrix = cv2.getPerspectiveTransform(image_points, world_points)
            self.mode = "homography"
            # World Y runs along the road, so the halfway point across the
            # calibrated patch is a counting line without asking the user to
            # draw a second thing. Vehicles are counted as they cross it.
            self._count_line_y = float(
                (world_points[:, 1].min() + world_points[:, 1].max()) / 2.0)
        elif mode == "scale":
            self._meters_per_pixel = float(calibration["meters_per_pixel"])
            self.mode = "scale"

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, auto_scale_correction: float = 1.0) -> "GroundPlane":
        """Load calibration from JSON; fall back to auto mode when missing."""
        path = Path(path)
        if not path.exists():
            return cls(None, auto_scale_correction)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return cls(data, auto_scale_correction)
        except Exception as exc:  # noqa: BLE001 - never let bad calibration crash the pipeline
            print(f"[CAL] Ignoring unusable calibration '{path}': {exc}")
            return cls(None, auto_scale_correction)

    @property
    def is_calibrated(self) -> bool:
        """Return whether a real metric calibration is in use."""
        return self.mode in ("homography", "scale")

    @property
    def counting_line_y(self) -> float | None:
        """World-Y of the counting line, or None when uncalibrated.

        Counting a vehicle as it crosses a line is immune to the tracker losing
        and re-acquiring it: a car that changes id three times still crosses
        the line once. Counting distinct ids instead — which is what
        ``len(_seen_ids)`` does — reports that same car three times.
        """
        return self._count_line_y

    @property
    def accuracy_note(self) -> str:
        """Short human description of the expected speed accuracy."""
        return {
            "homography": "homography calibration (±5%)",
            "scale": "fixed scale calibration (±10%)",
            "auto": "auto size-based calibration (±25%) — run calibrate_speed.py for accuracy",
        }[self.mode]

    # -- geometry ------------------------------------------------------------

    def _rescale(self, point: Sequence[float], frame_shape: tuple[int, int]) -> tuple[float, float]:
        """Map a point from the live frame onto the frame size used at calibration."""
        if self._calib_size is None:
            return float(point[0]), float(point[1])
        height, width = frame_shape[:2]
        if width == self._calib_size[0] and height == self._calib_size[1]:
            return float(point[0]), float(point[1])
        return (
            float(point[0]) * self._calib_size[0] / max(width, 1),
            float(point[1]) * self._calib_size[1] / max(height, 1),
        )

    def image_to_world(self, point: Sequence[float],
                       frame_shape: tuple[int, int]) -> tuple[float, float] | None:
        """Project an image point onto the road plane, in metres.

        Returns ``None`` in auto mode, where no global mapping exists.
        """
        x, y = self._rescale(point, frame_shape)
        if self.mode == "homography" and self._matrix is not None:
            src = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self._matrix)
            return float(dst[0][0][0]), float(dst[0][0][1])
        if self.mode == "scale" and self._meters_per_pixel is not None:
            return x * self._meters_per_pixel, y * self._meters_per_pixel
        return None

    def local_scale(self, bbox: Sequence[float], label: str) -> float | None:
        """Estimate metres-per-pixel at a detection, from its apparent size.

        Only used in auto mode. The bounding-box *height* is compared with the
        known typical height of the vehicle class, which gives the scale at
        that vehicle's exact depth in the scene.
        """
        real_height = VEHICLE_HEIGHT_M.get(label)
        if real_height is None:
            return None
        box_height = float(bbox[3]) - float(bbox[1])
        if box_height < 8.0:  # too small to measure reliably
            return None
        return (real_height / box_height) * self._auto_correction

    @property
    def has_roi(self) -> bool:
        """Whether a region of interest was loaded at all."""
        return self._roi is not None

    @property
    def calibration_size(self) -> tuple[int, int]:
        """Frame size the calibration was drawn on, or ``(0, 0)`` if unknown."""
        return self._calib_size or (0, 0)

    def roi_applies_to(self, frame_shape: tuple[int, int]) -> bool:
        """Whether the stored region of interest means anything for this frame.

        A region of interest is a polygon drawn around a particular stretch of
        road as one particular camera saw it. Rescaling it to a frame of the
        same shape is sound; stretching it onto a frame of a *different shape*
        is not, because the two axes are scaled by different factors and the
        polygon lands somewhere the road never was.

        That is not a hypothetical. The calibration shipped here was drawn on a
        1920x1080 camera; played against a 608x1080 phone clip of a motorway,
        the stretched polygon discarded 6595 of 11330 detections and took the
        emergency-vehicle count from six to nought, silently — the vehicles
        were found, and then thrown away before anything looked at them.

        So an aspect ratio that disagrees with the calibration by more than a
        few per cent means the polygon is not describing this picture, and the
        honest response is to monitor the whole frame and say so.
        """
        if self._roi is None or self._calib_size is None:
            return self._roi is not None
        height, width = frame_shape[:2]
        if not height or not width:
            return False
        calibrated = self._calib_size[0] / max(self._calib_size[1], 1)
        live = width / height
        return abs(calibrated - live) <= 0.05 * calibrated

    def contains(self, point: Sequence[float], frame_shape: tuple[int, int]) -> bool:
        """Return whether a point falls inside the region of interest."""
        if self._roi is None or not self.roi_applies_to(frame_shape):
            return True
        x, y = self._rescale(point, frame_shape)
        return cv2.pointPolygonTest(self._roi, (float(x), float(y)), False) >= 0

    def roi_polygon(self, frame_shape: tuple[int, int]) -> np.ndarray | None:
        """Return the ROI polygon in live-frame pixel coordinates, if any.

        Returns ``None`` when the polygon does not apply to this frame shape,
        so the overlay never draws a boundary that nothing is being tested
        against — a drawn region that is not enforced is worse than none.
        """
        if self._roi is None or not self.roi_applies_to(frame_shape):
            return None
        if self._calib_size is None:
            return self._roi.astype(np.int32)
        height, width = frame_shape[:2]
        scaled = self._roi.copy()
        scaled[:, 0] *= width / max(self._calib_size[0], 1)
        scaled[:, 1] *= height / max(self._calib_size[1], 1)
        return scaled.astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# SPEED ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _TrackSpeedState:
    """Per-track bookkeeping used to derive a stable km/h reading."""

    samples: deque = field(default_factory=lambda: deque(maxlen=90))  # (t, x, y, wx, wy)
    scales: deque = field(default_factory=lambda: deque(maxlen=30))   # metres per pixel
    ema_kmh: float | None = None
    max_kmh: float = 0.0
    #: Running total of every accepted instantaneous measurement, which is what
    #: makes a *mean* speed over the track available. Reporting the EMA's final
    #: value instead made "average" and "top" speed the same number for any
    #: vehicle that did not slow down before leaving, because the EMA warms up
    #: from zero and ``max_kmh`` tracks the EMA rather than the measurements.
    speed_sum: float = 0.0
    speed_count: int = 0
    last_seen: float = 0.0
    first_seen: float = 0.0
    distance_m: float = 0.0
    readings: int = 0


class SpeedEstimator:
    """Turns tracked boxes into per-vehicle speeds in km/h."""

    def __init__(self, ground: GroundPlane, config: VisionConfig) -> None:
        """Initialise with a ground plane and pipeline configuration."""
        self._ground = ground
        self._config = config
        self._tracks: dict[int, _TrackSpeedState] = {}

    @property
    def source(self) -> str:
        """Return which calibration mode is producing the speeds."""
        return self._ground.mode

    def set_ground(self, ground: GroundPlane) -> None:
        """Swap in a new calibration without losing track history."""
        self._ground = ground

    def track_summary(self, track_id: int) -> dict[str, float] | None:
        """Return distance/duration/top-speed stats for one track."""
        state = self._tracks.get(track_id)
        if state is None:
            return None
        mean_kmh = (state.speed_sum / state.speed_count
                    if state.speed_count else (state.ema_kmh or 0.0))
        return {
            "distance_m": round(state.distance_m, 1),
            "duration_s": round(state.last_seen - state.first_seen, 1),
            "max_kmh": round(state.max_kmh, 1),
            "avg_kmh": round(mean_kmh, 1),
        }

    def update(self, track_id: int, bbox: Sequence[float], label: str,
               now: float, frame_shape: tuple[int, int]) -> float | None:
        """Feed one observation of a track and return its current speed in km/h.

        Returns ``None`` until enough motion history exists to measure.
        """
        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackSpeedState(first_seen=now)
            self._tracks[track_id] = state
        state.last_seen = now

        # Ground-contact point: the middle of the bottom edge is where the
        # vehicle actually touches the road, so it is the only point whose
        # motion is genuinely planar.
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        by = float(bbox[3])

        world = self._ground.image_to_world((cx, by), frame_shape)
        if world is None:
            scale = self._ground.local_scale(bbox, label)
            if scale is not None:
                state.scales.append(scale)

        previous = state.samples[-1] if state.samples else None
        state.samples.append((now, cx, by, world[0] if world else None,
                              world[1] if world else None))
        if previous is not None:
            state.distance_m += self._step_metres(previous, state.samples[-1], state)

        speed = self._measure(state)
        if speed is None:
            return state.ema_kmh

        alpha = self._config.speed_ema_alpha
        state.ema_kmh = speed if state.ema_kmh is None else (
            alpha * speed + (1.0 - alpha) * state.ema_kmh
        )
        state.max_kmh = max(state.max_kmh, state.ema_kmh)
        state.speed_sum += speed
        state.speed_count += 1
        state.readings += 1
        return state.ema_kmh

    def _measure(self, state: _TrackSpeedState) -> float | None:
        """Compute an instantaneous speed over the configured time window."""
        cfg = self._config
        if len(state.samples) < cfg.speed_min_samples:
            return None

        latest = state.samples[-1]
        window_start = latest[0] - cfg.speed_window_seconds

        # Oldest sample still inside the window (samples are time-ordered).
        oldest = None
        for sample in state.samples:
            if sample[0] >= window_start:
                oldest = sample
                break
        if oldest is None:
            return None

        dt = latest[0] - oldest[0]
        if dt < cfg.speed_min_window_seconds:
            return None

        if latest[3] is not None and oldest[3] is not None:
            # Calibrated: displacement is already in metres.
            metres = math.hypot(latest[3] - oldest[3], latest[4] - oldest[4])
        else:
            scale = float(np.median(state.scales)) if state.scales else None
            if scale is None:
                return None
            metres = math.hypot(latest[1] - oldest[1], latest[2] - oldest[2]) * scale

        kmh = (metres / dt) * 3.6
        if kmh > cfg.speed_max_plausible_kmh:
            return None  # physically impossible → almost certainly an ID switch
        return kmh

    @staticmethod
    def _step_metres(previous: tuple, current: tuple, state: _TrackSpeedState) -> float:
        """Distance in metres travelled between two consecutive observations."""
        if previous[3] is not None and current[3] is not None:
            return math.hypot(current[3] - previous[3], current[4] - previous[4])
        if not state.scales:
            return 0.0
        pixels = math.hypot(current[1] - previous[1], current[2] - previous[2])
        return pixels * float(np.median(state.scales))

    def prune(self, now: float, force: bool = False) -> list[tuple[int, dict[str, float]]]:
        """Drop tracks unseen past the TTL; return ``(id, summary)`` for each.

        Without this the dictionaries grow unbounded over a long session,
        which was a real leak in the original implementation. ``force`` retires
        every track regardless of age, for use at shutdown.
        """
        ttl = self._config.track_ttl_seconds
        stale = [tid for tid, st in self._tracks.items()
                 if force or now - st.last_seen > ttl]
        retired = [(tid, self.track_summary(tid) or {}) for tid in stale]
        for tid in stale:
            del self._tracks[tid]
        return retired


# ═══════════════════════════════════════════════════════════════════════════
# EMERGENCY VEHICLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════
#
# The colour-ratio light-bar heuristic that used to live here has been replaced
# by ``siren_vision.SirenDetector``. It asked whether a vehicle's roof carried
# saturated red or blue pixels whose ratio varied over time, and on real night
# footage that describes almost every car on the road: tail lights are
# saturated red, and the ratio varies whenever the vehicle moves through a pool
# of street light. The replacement measures the one thing that actually
# distinguishes a beacon — that it *switches*, repeatedly, between fully lit and
# fully dark — and reads the vehicle type from the colour that is doing the
# switching. See ``siren_vision.py`` for the reasoning and the measurements.



# ═══════════════════════════════════════════════════════════════════════════
# CONGESTION
# ═══════════════════════════════════════════════════════════════════════════


class CongestionEngine:
    """Derives a 0–1 congestion index and a stable level from live telemetry."""

    def __init__(self, config: VisionConfig) -> None:
        """Initialise with pipeline configuration."""
        self._config = config
        self._level = "FREE"
        self._level_since = time.time()

    def compute_index(self, avg_speed_kmh: float | None, weighted_density: float,
                      vehicle_count: int) -> float:
        """Return the congestion index in ``[0, 1]``.

        Two guards keep this honest, both of which the original version got
        wrong:

        * An empty road is free-flowing by definition. The original fell back
          to a stale speed value and could report heavy congestion with zero
          vehicles in frame.
        * *One* slow vehicle on an otherwise empty road is not congestion — it
          is one slow driver. The speed term is therefore faded in with the
          number of vehicles present, so low speed only counts as congestion
          once there is actually traffic to be congested.
        """
        cfg = self._config
        if vehicle_count <= 0:
            return 0.0

        density_factor = _clamp01(weighted_density / cfg.max_capacity_pce)
        if avg_speed_kmh is None:
            # Vehicles present but not yet measurable: density alone decides.
            return round(density_factor, 3)

        presence = _clamp01(vehicle_count / max(cfg.speed_relevance_vehicles, 1))
        speed_factor = _clamp01(1.0 - (avg_speed_kmh / cfg.free_flow_speed_kmh))
        index = (cfg.speed_weight * speed_factor * presence
                 + cfg.density_weight * density_factor)
        return round(_clamp01(index), 3)

    def level_for(self, index: float, now: float | None = None) -> str:
        """Map an index onto FREE/MODERATE/HEAVY with hysteresis and dwell time.

        Without this the level flickers between two states on every frame when
        the index sits on a threshold, which makes the LEDs and servo chatter.
        """
        cfg = self._config
        now = now if now is not None else time.time()
        candidate = self._raw_level(index)

        if candidate == self._level:
            return self._level

        # Require the index to clear the threshold by a margin before promoting
        # or demoting, and require a minimum dwell in the current level.
        margin = cfg.level_hysteresis
        if self._level == "FREE" and index < cfg.threshold_low + margin:
            return self._level
        if self._level == "MODERATE":
            if cfg.threshold_low - margin <= index < cfg.threshold_medium + margin:
                return self._level
        if self._level == "HEAVY" and index >= cfg.threshold_medium - margin:
            return self._level
        if now - self._level_since < cfg.level_min_dwell_seconds:
            return self._level

        self._level = candidate
        self._level_since = now
        return self._level

    def reset(self, now: float | None = None) -> None:
        """Return to FREE and re-anchor the dwell clock to ``now``.

        The dwell timer is a comparison against the clock it was set by, so a
        source switch that moves the clock (a video carries its own time) must
        move this with it — otherwise the level would sit frozen waiting out a
        dwell that has already elapsed, or one that never will.
        """
        self._level = "FREE"
        self._level_since = now if now is not None else time.time()

    def _raw_level(self, index: float) -> str:
        """Return the level implied by the index, ignoring hysteresis."""
        if index < self._config.threshold_low:
            return "FREE"
        if index < self._config.threshold_medium:
            return "MODERATE"
        return "HEAVY"


def _clamp01(value: float) -> float:
    """Clamp a float into the ``[0, 1]`` range."""
    return max(0.0, min(1.0, float(value)))


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR
# ═══════════════════════════════════════════════════════════════════════════


class VehicleDetector:
    """Thin, accuracy-tuned wrapper around a YOLO tracking model."""

    def __init__(self, config: VisionConfig) -> None:
        """Load the model and warm it up so the first real frame isn't slow."""
        if not ULTRALYTICS_AVAILABLE:  # pragma: no cover
            raise RuntimeError("ultralytics is not installed — run: pip install ultralytics")

        self._config = config
        model_path = Path(config.model_path)
        if not model_path.is_absolute():
            candidate = ROOT_DIR / model_path
            if candidate.exists():
                model_path = candidate

        self.device = config.resolved_device()
        self.half = config.use_half()
        print(f"[YOLO] Loading {model_path.name} on {self.device.upper()}"
              f"{' (FP16)' if self.half else ''} @ imgsz={config.imgsz}")
        self.model = YOLO(str(model_path))
        self.names: dict[int, str] = self.model.names

        # Only track classes the model actually knows about.
        self.class_ids = [i for i, n in self.names.items() if n in CLASS_WEIGHTS]
        if not self.class_ids:
            self.class_ids = list(COCO_VEHICLE_IDS)

        self._tracker_path = self._resolve_tracker(config.tracker)
        self._warmup()

    def _resolve_tracker(self, tracker: str) -> str:
        """Return a usable tracker config path, falling back to the stock one."""
        path = Path(tracker)
        if not path.is_absolute():
            path = ROOT_DIR / tracker
        if path.exists():
            return str(path)
        print(f"[YOLO] Tracker config '{tracker}' not found — using botsort.yaml")
        return "botsort.yaml"

    def _warmup(self) -> None:
        """Run one throwaway inference so CUDA kernels are compiled up front."""
        blank = np.zeros((self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(blank, imgsz=self._config.imgsz, device=self.device,
                               half=self.half, verbose=False)
            print("[YOLO] Warm-up complete.")
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort
            print(f"[YOLO] Warm-up skipped: {exc}")

    def track(self, frame: np.ndarray) -> tuple[list[tuple], float]:
        """Detect and track vehicles in a frame.

        Returns ``([(bbox, track_id, label, confidence), ...], inference_ms)``.
        Detections without a track id are dropped, since every downstream
        metric (speed, emergency evidence) needs identity over time.
        """
        cfg = self._config
        started = time.perf_counter()
        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_path,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            classes=self.class_ids,
            agnostic_nms=cfg.agnostic_nms,
            augment=cfg.augment,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        output: list[tuple] = []
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return output, elapsed_ms

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        for box, track_id, cls_idx, conf in zip(xyxy, ids, clss, confs):
            label = self.names.get(int(cls_idx), str(cls_idx))
            if label not in CLASS_WEIGHTS:
                continue
            output.append((box, int(track_id), label, float(conf)))
        return output, elapsed_ms

    def reset_tracker(self) -> None:
        """Wipe the tracker's memory so a new source starts from nothing.

        Ultralytics keeps one tracker alive across calls (``persist=True``),
        which is exactly right for a continuous feed and exactly wrong the
        moment the feed changes: without this it would spend several frames
        trying to match the vehicles of the new source onto the last frame of
        the old one, inventing enormous displacements as it went.
        """
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception as exc:  # noqa: BLE001 - very old ultralytics
                print(f"[YOLO] Tracker reset unavailable ({exc}); "
                      "ids will continue from the previous source.")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class TrafficVisionPipeline:
    """Detection → tracking → speed → congestion, for one camera."""

    def __init__(self, config: VisionConfig | None = None,
                 detector: VehicleDetector | None = None) -> None:
        """Build the pipeline; pass ``detector`` to inject a fake in tests."""
        self.config = config or VisionConfig()
        self.ground = GroundPlane.from_file(
            ROOT_DIR / self.config.calibration_path
            if not Path(self.config.calibration_path).is_absolute()
            else self.config.calibration_path,
            self.config.auto_scale_correction,
        )
        self.detector = detector if detector is not None else VehicleDetector(self.config)
        self.speed = SpeedEstimator(self.ground, self.config)
        self.siren = SirenDetector(SirenConfig(
            enabled=self.config.siren_detection,
            min_box_px=self.config.siren_min_box_px,
            window_seconds=self.config.siren_window_seconds,
            min_flashes=self.config.siren_min_flashes,
            hold_seconds=self.config.siren_hold_seconds,
            blue_type=self.config.siren_blue_type,
            red_type=self.config.siren_red_type,
            both_type=self.config.siren_both_type,
            track_ttl_seconds=max(self.config.track_ttl_seconds, 5.0),
        ))
        self.appearance = EmergencyAppearanceClassifier(EmergencyConfig(
            enabled=self.config.emergency_appearance,
            backend=self.config.emergency_backend,
            model_name=self.config.emergency_clip_model,
            model_path=self.config.emergency_model_path,
            device=self.config.device,
            check_interval_frames=self.config.emergency_check_interval,
            track_ttl_seconds=max(self.config.track_ttl_seconds, 5.0),
        ))
        self.congestion = CongestionEngine(self.config)

        self._ages: dict[int, int] = {}
        self._class_votes: dict[int, deque] = {}
        self._labels: dict[int, str] = {}
        self._seen_ids: set[int] = set()
        #: Last world-Y seen per track, and the tracks already counted, so a
        #: vehicle idling on the counting line is not counted over and over.
        self._last_world_y: dict[int, float] = {}
        self._counted_ids: set[int] = set()
        self._crossings = 0
        self._frame_times: deque = deque(maxlen=60)
        self._frame_index = 0
        self._completed: list[dict[str, Any]] = []
        #: Set once the mismatched-region warning has been printed, so it is
        #: said clearly at the start rather than on every frame.
        self._roi_warned = False

        print(f"[SPEED] Using {self.ground.accuracy_note}")
        if self.ground.counting_line_y is not None:
            print("[COUNT] Unique vehicles counted by line crossing "
                  f"(world Y {self.ground.counting_line_y:.1f} m)")
        else:
            print("[COUNT] Unique vehicles counted by distinct track ids — "
                  "inflated by id churn; calibrate for line-crossing counting")

    # -- vehicle counting ----------------------------------------------------

    def _note_line_crossing(
        self,
        track_id: int,
        centre_bottom: tuple[float, float],
        frame_shape: tuple[int, ...],
    ) -> None:
        """Count this track if it just crossed the counting line.

        Counted once per track id: a vehicle that stops astride the line, or
        whose box jitters back and forth across it, must not tick the counter
        on every frame.
        """
        line = self.ground.counting_line_y
        if line is None:
            return
        world = self.ground.image_to_world(centre_bottom, frame_shape)
        if world is None:
            return

        y = float(world[1])
        previous = self._last_world_y.get(track_id)
        self._last_world_y[track_id] = y
        if previous is None or track_id in self._counted_ids:
            return
        # Strict sign change, so merely touching the line is not a crossing.
        if (previous - line) * (y - line) < 0.0:
            self._crossings += 1
            self._counted_ids.add(track_id)

    def unique_vehicle_count(self) -> int:
        """Vehicles seen this session.

        Line crossings when calibrated, because that survives the tracker
        losing a vehicle and re-acquiring it under a new id. Without a
        calibration there is no ground plane to put a line on, so this falls
        back to distinct track ids — the old behaviour, inflated but the only
        thing available.
        """
        if self.ground.counting_line_y is None:
            return len(self._seen_ids)
        return self._crossings
        print(f"[EMRG] {self.siren.status}")
        print(f"[EMRG] {self.appearance.status}")
        print("[EMRG] Emergency mode triggers on "
              + ("a working siren light only"
                 if self.config.emergency_require_siren
                 else "a siren light or recognised livery"))

    # -- public API ----------------------------------------------------------

    @property
    def completed_vehicles(self) -> list[dict[str, Any]]:
        """Per-vehicle records for tracks that have left the scene."""
        return self._completed

    def process(self, frame: np.ndarray, now: float | None = None,
                wall: float | None = None) -> FrameResult:
        """Run the full pipeline on one frame and return its telemetry.

        ``now`` is the clock the *scene* runs on: for a live camera that is
        simply the wall clock, but for a recorded video it is the video's own
        timeline, so that a machine which cannot decode the file at full speed
        still reports the km/h it was filmed at. Every duration derived here —
        speed windows, congestion dwell, track TTL — is measured against it.

        ``wall`` is real elapsed time and is used for one thing only: the
        reported FPS, which must keep meaning "frames this machine processed
        per second" rather than the frame rate the file was encoded at. It
        defaults to ``now``, which is correct for any live source.
        """
        now = now if now is not None else time.time()
        cfg = self.config
        self._frame_index += 1
        self._frame_times.append(wall if wall is not None else now)

        raw, inference_ms = self.detector.track(frame)

        if (not self._roi_warned and self.ground.has_roi
                and not self.ground.roi_applies_to(frame.shape)):
            self._roi_warned = True
            height, width = frame.shape[:2]
            calibrated = self.ground.calibration_size
            print(f"[ROI] Region of interest ignored: it was drawn on a "
                  f"{calibrated[0]}x{calibrated[1]} picture and this source is "
                  f"{width}x{height}, a different shape. Stretching it across "
                  f"would exclude an arbitrary part of the road. Monitoring the "
                  f"whole frame — re-run calibrate_speed.py on this source to "
                  f"restore it.")

        # Split by the monitored region, keeping a tally of what fell outside
        # so the exclusion is visible. Both halves go on to be examined for
        # emergency vehicles: the region of interest exists to say which road
        # this junction is *measuring*, and an ambulance one lane the wrong
        # side of that line is still an ambulance the operator needs to know
        # about. Only the counts, the density and the congestion index are
        # restricted to it.
        visible: list[tuple] = []
        excluded: list[tuple] = []
        outside: dict[str, int] = {}
        for bbox, track_id, raw_label, conf in raw:
            ground_point = ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))
            if self.ground.contains(ground_point, frame.shape):
                visible.append((bbox, track_id, raw_label, conf))
            else:
                excluded.append((bbox, track_id, raw_label, conf))
                outside[raw_label] = outside.get(raw_label, 0) + 1

        examined = visible + excluded
        excluded_ids = {track_id for _b, track_id, _l, _c in excluded}

        # Appearance recognition runs once per frame over all tracks at once,
        # so ambulances/police/fire trucks cost a single batched forward pass
        # rather than one call per vehicle.
        self.appearance.begin_frame()
        appearance_started = time.perf_counter()
        verdicts = self.appearance.classify_frame(
            frame, [(track_id, bbox) for bbox, track_id, _l, _c in examined], now)
        appearance_ms = (time.perf_counter() - appearance_started) * 1000.0
        self.siren.begin_frame(frame, now)

        detections: list[VehicleDetection] = []
        counts = {label: 0 for label in CLASS_WEIGHTS}
        weighted = 0.0
        speeds: list[float] = []
        emergency_ids: set[int] = set()
        priority_ids: set[int] = set()
        suspected_ids: set[int] = set()
        emergency_types: Counter = Counter()

        for bbox, track_id, raw_label, conf in examined:
            in_region = track_id not in excluded_ids
            centre_bottom = ((float(bbox[0]) + float(bbox[2])) / 2.0, float(bbox[3]))

            self._ages[track_id] = self._ages.get(track_id, 0) + 1
            # Rolling window of recent class predictions. The reported class is
            # the mode, and it is *sticky*: a challenger must strictly beat the
            # incumbent, so a van sitting on a car/truck tie stops flickering.
            votes = self._class_votes.get(track_id)
            if votes is None:
                votes = deque(maxlen=cfg.class_vote_window)
                self._class_votes[track_id] = votes
            votes.append(raw_label)
            label = self._stable_label(track_id, votes)

            speed_kmh = self.speed.update(track_id, bbox, label, now, frame.shape)

            # Two cues, fused: whether the vehicle's beacon is actually
            # flashing — which is what earns a cleared lane — and what the
            # vehicle looks like, which names the type the colour cannot.
            siren_verdict = self.siren.update(frame, bbox, track_id, now, label)
            verdict = fuse_emergency(
                verdicts.get(track_id, EmergencyVerdict()), siren_verdict,
                require_siren=cfg.emergency_require_siren)

            age = self._ages[track_id]
            if age < cfg.min_track_frames:
                # Unconfirmed: tracked internally but not yet reported, so a
                # one-frame ghost never reaches the counts or the hardware.
                continue

            # Traffic measurement is about the monitored road, so only vehicles
            # inside the region are counted, weighted or timed. Recognition is
            # about what is out there, so it applies to everything in shot.
            if in_region:
                self._seen_ids.add(track_id)
                self._note_line_crossing(track_id, centre_bottom, frame.shape)
                counts[label] = counts.get(label, 0) + 1
                weighted += CLASS_WEIGHTS.get(label, 1.0)
                if speed_kmh is not None:
                    speeds.append(speed_kmh)

            if verdict.is_emergency:
                emergency_ids.add(track_id)
                emergency_types[verdict.vehicle_type] += 1
            if verdict.priority:
                priority_ids.add(track_id)
            if verdict.suspected:
                suspected_ids.add(track_id)

            if not in_region and not verdict.is_emergency:
                # Outside the monitored road and unremarkable: examined, and
                # deliberately not reported, so the overlay and the counts stay
                # about the junction rather than the whole horizon.
                continue

            detections.append(VehicleDetection(
                track_id=track_id,
                label=label,
                raw_label=raw_label,
                confidence=conf,
                bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                speed_kmh=None if speed_kmh is None else round(speed_kmh, 1),
                is_emergency=verdict.is_emergency,
                emergency_confidence=verdict.confidence,
                age_frames=age,
                world_xy=self.ground.image_to_world(centre_bottom, frame.shape),
                emergency_type=verdict.vehicle_type,
                emergency_evidence=verdict.evidence,
                emergency_suspected=verdict.suspected,
                siren_colour=siren_verdict.colour,
                siren_rate_hz=siren_verdict.rate_hz,
                emergency_priority=verdict.priority,
                outside_region=not in_region,
            ))

        avg_speed = float(np.mean(speeds)) if speeds else None
        max_speed = float(np.max(speeds)) if speeds else 0.0
        moving = sum(1 for s in speeds if s >= cfg.stationary_kmh)
        total = sum(counts.values())

        index = self.congestion.compute_index(avg_speed, weighted, total)
        level = self.congestion.level_for(index, now)

        self._retire_stale(now)

        return FrameResult(
            frame_index=self._frame_index,
            timestamp=now,
            detections=detections,
            vehicle_counts=counts,
            total_vehicles=total,
            avg_speed_kmh=round(avg_speed, 1) if avg_speed is not None else 0.0,
            max_speed_kmh=round(max_speed, 1),
            moving_vehicles=moving,
            stopped_vehicles=len(speeds) - moving,
            weighted_density=round(weighted, 2),
            congestion_index=index,
            congestion_level=level,
            emergency_ids=emergency_ids,
            fps=self.fps(),
            speed_source=self.ground.mode,
            unique_vehicles_session=self.unique_vehicle_count(),
            inference_ms=round(inference_ms, 1),
            emergency_types=[name for name, _count in emergency_types.most_common()],
            emergency_ms=round(appearance_ms, 1),
            suspected_ids=suspected_ids,
            siren_sample_hz=round(self.siren.sample_rate(), 1),
            siren_undersampled=self.siren.undersampled,
            outside_roi_counts=outside,
            priority_ids=priority_ids,
            siren_pending=self.siren.pending_tracks,
            siren_tracked=self.siren.tracked,
            roi_active=self.ground.roi_applies_to(frame.shape),
        )

    def _stable_label(self, track_id: int, votes: deque) -> str:
        """Return the track's class, changing it only on a strict majority."""
        counts = Counter(votes)
        challenger = counts.most_common(1)[0][0]
        incumbent = self._labels.get(track_id)
        if incumbent is None or counts[challenger] > counts.get(incumbent, 0):
            self._labels[track_id] = challenger
        return self._labels[track_id]

    def fps(self) -> float:
        """Return the rolling processing frame rate."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    # -- internals -----------------------------------------------------------

    def reload_calibration(self, path: str | Path | None = None) -> GroundPlane:
        """Re-read the calibration file so a re-calibration applies live."""
        target = Path(path or self.config.calibration_path)
        if not target.is_absolute():
            target = ROOT_DIR / target
        self.ground = GroundPlane.from_file(target, self.config.auto_scale_correction)
        self.speed.set_ground(self.ground)
        return self.ground

    def drain_completed(self) -> list[dict[str, Any]]:
        """Return and clear per-vehicle records for vehicles that have left."""
        drained, self._completed = self._completed, []
        return drained

    def finalize(self, now: float | None = None) -> list[dict[str, Any]]:
        """Retire every remaining track at shutdown and return all records.

        Vehicles still in frame when the session ends would otherwise never be
        written to the per-vehicle log.
        """
        self._retire_stale(now if now is not None else time.time(), force=True)
        return self.drain_completed()

    def reset(self, now: float | None = None) -> list[dict[str, Any]]:
        """Forget every track, returning the records of the ones still in frame.

        Called whenever the picture jumps — a new source is selected, a looped
        file wraps round, someone scrubs the timeline — because the vehicles on
        screen after the jump have nothing to do with the ones before it.
        Carrying the old tracks across would invent a journey between two
        unrelated roads and poison the first speed readings on the new one.

        The unique-vehicle count restarts too: the tracker's ids begin again
        from 1, so keeping the old set would silently merge the two roads'
        vehicles into one undercounted total.
        """
        now = now if now is not None else time.time()
        records = self.finalize(now)   # retires in-flight tracks into the log
        self._ages.clear()
        self._class_votes.clear()
        self._labels.clear()
        self._seen_ids.clear()
        self._last_world_y.clear()
        self._counted_ids.clear()
        self._crossings = 0
        self._frame_times.clear()
        self.siren.reset()
        self.appearance.reset()
        self.congestion.reset(now)
        resetter = getattr(self.detector, "reset_tracker", None)
        if callable(resetter):    # a stubbed detector in the tests has none
            resetter()
        return records

    def _retire_stale(self, now: float, force: bool = False) -> None:
        """Archive and forget tracks that have left the scene."""
        for track_id, summary in self.speed.prune(now, force=force):
            votes = self._class_votes.pop(track_id, None)
            age = self._ages.pop(track_id, 0)
            label = self._labels.pop(track_id, None)
            if age >= self.config.min_track_frames and votes:
                # The per-vehicle log records the same verdict the junction
                # acted on, so it has to be the fused one rather than the
                # appearance guess alone.
                final = fuse_emergency(
                    self.appearance.verdict_for(track_id),
                    self.siren.verdict_for(track_id),
                    require_siren=self.config.emergency_require_siren)
                self._completed.append({
                    "track_id": track_id,
                    "class": label or Counter(votes).most_common(1)[0][0],
                    "frames": age,
                    "emergency": final.display_name or "no",
                    "emergency_evidence": final.evidence,
                    **summary,
                })
        self.siren.prune(now)
        self.appearance.prune(now)


# ═══════════════════════════════════════════════════════════════════════════
# ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════

_LEVEL_COLORS: dict[str, tuple[int, int, int]] = {
    "FREE": (0, 200, 0),
    "MODERATE": (0, 165, 255),
    "HEAVY": (0, 0, 255),
    "EMERGENCY": (255, 80, 0),
}

#: Box colour matching the beacon that triggered the flag (BGR).
_SIREN_COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (255, 120, 0),
    "red": (0, 0, 255),
    "red+blue": (255, 0, 200),
}


def annotate(frame: np.ndarray, result: FrameResult, *, mode_label: str = "AUTO",
             servo_angle: int | None = None, emergency: bool = False,
             roi: np.ndarray | None = None, hint: str | None = None,
             source_label: str | None = None,
             progress: float | None = None) -> np.ndarray:
    """Draw boxes, per-vehicle km/h and a status HUD onto a copy of the frame.

    ``source_label`` names where the frames are coming from, which matters as
    soon as that can change while the system is running. ``progress`` (0–1)
    draws a playback bar for a video file.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    if roi is not None:
        cv2.polylines(canvas, [roi], True, (255, 200, 0), 2)

    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        if det.emergency_priority:
            # Draw the box in the colour of the beacon that raised it, so the
            # operator can see at a glance whether the system read the light
            # the same way they did.
            color = _SIREN_COLORS.get(det.siren_colour, (0, 0, 255))
            thickness = 3
            rate = f" {det.siren_rate_hz:.1f}Hz" if det.siren_rate_hz else ""
            tag = f"{det.emergency_name} #{det.track_id} {det.speed_text}{rate}"
        elif det.is_emergency or det.emergency_suspected:
            # Recognised livery, no working beacon: marked, but not given
            # priority, and drawn in amber so the difference is obvious. The
            # box is still thick, because this is a recognised emergency
            # vehicle and the old thin outline read as an ordinary car.
            color, thickness = (0, 190, 255), 3
            where = " (outside region)" if det.outside_region else ""
            tag = f"{det.emergency_name} #{det.track_id} {det.speed_text}{where}"
        else:
            color, thickness = (0, 200, 0), 2
            tag = f"#{det.track_id} {det.label.upper()} {det.speed_text}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, tag, (x1 + 3, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # The HUD is sized against the frame it is drawn on. Fixed point sizes were
    # tuned for a 1280-wide landscape picture and ran off the edge of anything
    # narrower: on a 608-wide portrait clip the emergency banner rendered as
    # "!!! POLICE, AMBULANCE DETECTED - CLEA", which is a detection the operator
    # cannot read and therefore, in the only sense that matters, a detection
    # that did not happen. The floor stops a thumbnail rendering unreadably
    # small; the ceiling stops 4K text swamping the picture.
    ui = min(max(width / 1280.0, 0.55), 1.3)

    def line(text, y, size, colour, weight=2):
        """Draw one HUD line, shrunk if it would otherwise overrun the frame."""
        scale = size * ui
        (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
        margin = int(18 * ui)
        if tw > width - 2 * margin:
            scale *= (width - 2 * margin) / max(tw, 1)
        cv2.putText(canvas, text, (margin, int(y * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, colour, weight)

    # Translucent HUD panel so text stays readable over any scene.
    panel = canvas.copy()
    cv2.rectangle(panel, (0, 0), (width, int(118 * ui)), (16, 20, 28), -1)
    cv2.addWeighted(panel, 0.55, canvas, 0.45, 0, canvas)

    color = _LEVEL_COLORS.get("EMERGENCY" if emergency else result.congestion_level,
                              (255, 255, 255))
    line(f"CI {result.congestion_index:.2f}  [{result.congestion_level}]  |  {mode_label}",
         36, 0.85, color)

    servo_text = "" if servo_angle is None else f"  |  Servo {servo_angle}deg"
    line(f"Vehicles {result.total_vehicles}  |  Avg {result.avg_speed_kmh:.0f} km/h  |  "
         f"Max {result.max_speed_kmh:.0f} km/h{servo_text}",
         68, 0.62, (235, 235, 235))

    line(f"FPS {result.fps:.1f}  |  {result.inference_ms:.0f} ms/frame  |  "
         f"speed: {result.speed_source}  |  session total {result.unique_vehicles_session}",
         98, 0.55, (170, 200, 235), 1)

    kinds = ", ".join(name.upper() for name in result.emergency_types) or "EMERGENCY"
    if emergency:
        line(f"!!! {kinds} DETECTED - CLEAR RIGHT LANE !!!", 152, 0.9, (0, 0, 255), 3)
    elif result.emergency_ids:
        # Recognised by livery with no beacon running: announced, because a
        # system that has spotted an ambulance and says nothing looks exactly
        # like one that missed it — but worded so it is clear the junction is
        # not being held.
        line(f"{kinds} DETECTED - no siren, no priority", 152, 0.75, (0, 190, 255))
    elif result.suspected_ids:
        line(f"{len(result.suspected_ids)} marked vehicle(s), no siren running",
             148, 0.6, (0, 190, 255))

    # A beacon flashes 1-4 times a second; if the machine is sampling slower
    # than that, one can pass unseen. Say so rather than reporting "no
    # emergency" as though it had been checked.
    if result.siren_undersampled:
        line(f"LOW FPS ({result.siren_sample_hz:.0f}/s) - siren lights may be missed",
             176, 0.6, (0, 210, 255))
    elif (result.siren_pending and not result.emergency_ids
            and result.siren_pending >= result.siren_tracked):
        # *Every* vehicle in shot has been seen for less time than a beacon
        # needs to prove itself — a clip too short, or a road crossed too
        # quickly. Saying "no emergency vehicle" here would be a claim about
        # footage nobody could have judged. Shown only when nothing at all has
        # been judged, so it stays a real warning rather than wallpaper on a
        # busy road where some vehicles have simply just arrived.
        line(f"{result.siren_pending} vehicle(s) not watched long enough to "
             "judge sirens - clip too short?", 176, 0.55, (0, 210, 255), 1)

    if source_label:
        # Right-aligned on the top line, so it never collides with the reading
        # on the left however long either gets.
        size = 0.55 * ui
        (tw, _th), _ = cv2.getTextSize(source_label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)
        cv2.putText(canvas, source_label, (max(int(18 * ui), width - tw - int(18 * ui)),
                                           int(34 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (150, 220, 255), 1)

    if progress is not None:
        bar_y = height - 6
        cv2.rectangle(canvas, (0, bar_y), (width, height), (40, 46, 58), -1)
        filled = int(width * _clamp01(progress))
        cv2.rectangle(canvas, (0, bar_y), (filled, height), (90, 200, 255), -1)

    if hint:
        cv2.putText(canvas, hint, (18, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def fit_to_display(frame: np.ndarray, max_width: int, max_height: int,
                   interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Scale a frame to fit a window **without** distorting its aspect ratio.

    ``interpolation`` defaults to the area filter, which is the right choice
    when the result will be looked at closely. Callers on the hot path — the
    MJPEG stream and the preview window — pass ``cv2.INTER_LINEAR`` instead,
    which is several times cheaper and indistinguishable at these scales.
    """
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)),
                      interpolation=interpolation)


# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY LOGGING
# ═══════════════════════════════════════════════════════════════════════════

LOG_HEADERS: list[str] = [
    "Session", "Session Name", "Date", "Timestamp", "Congestion Index", "Level",
    "Light Phase",
    "Total Vehicles", "Cars", "Motorcycles", "Buses", "Trucks",
    "Avg Speed (km/h)", "Max Speed (km/h)", "Moving", "Stopped",
    "Weighted Density (PCE)", "Servo Angle (deg)",
    "Emergency", "Emergency Type", "Emergency Vehicles", "Siren Colour",
    "Marked No Siren", "Unique Vehicles",
    "FPS", "Speed Source",
]
LOG_COL_WIDTHS = [18, 18, 12, 11, 15, 11, 12, 13, 7, 12, 7, 8, 15, 15, 9, 9,
                  20, 15, 11, 16, 17, 13, 16, 15, 8, 13]

#: Columns that must come back from CSV as numbers rather than strings.
NUMERIC_LOG_COLUMNS: frozenset[str] = frozenset({
    "Congestion Index", "Total Vehicles", "Cars", "Motorcycles", "Buses", "Trucks",
    "Avg Speed (km/h)", "Max Speed (km/h)", "Moving", "Stopped",
    "Weighted Density (PCE)", "Servo Angle (deg)", "Emergency Vehicles",
    "Marked No Siren", "Unique Vehicles", "FPS",
})

VEHICLE_HEADERS: list[str] = [
    "Track ID", "Class", "Emergency", "Evidence", "Frames Tracked", "Duration (s)",
    "Distance (m)", "Avg Speed (km/h)", "Top Speed (km/h)",
]
VEHICLE_COL_WIDTHS = [10, 12, 13, 18, 15, 13, 13, 16, 17]

#: The same per-vehicle facts as ``VEHICLE_HEADERS``, stamped with the session
#: and wall-clock time so rows from every run can share one file and still be
#: told apart — exactly what ``Session``/``Session Name`` do for ``LOG_HEADERS``.
VEHICLE_LOG_HEADERS: list[str] = [
    "Session", "Session Name", "Date", "Time", *VEHICLE_HEADERS,
]
VEHICLE_LOG_COL_WIDTHS = [18, 18, 12, 11, *VEHICLE_COL_WIDTHS]

#: One row per vehicle, kept beside the telemetry files.
DEFAULT_VEHICLE_CSV = "traffic_vehicles.csv"
DEFAULT_MASTER_VEHICLE_CSV = "traffic_all_vehicles.csv"


def build_vehicle_row(vehicle: dict[str, Any]) -> dict[str, Any]:
    """Flatten one completed-vehicle record into a CSV row.

    ``Session``/``Session Name`` are left to the logger, which is the only
    component that knows which session is open — the same split ``build_log_row``
    uses.
    """
    stamp = datetime.now()
    return {
        "Date": stamp.strftime("%Y-%m-%d"),
        "Time": stamp.strftime("%H:%M:%S"),
        "Track ID": vehicle.get("track_id"),
        "Class": str(vehicle.get("class", "")).capitalize(),
        "Emergency": vehicle.get("emergency", "no"),
        "Evidence": vehicle.get("emergency_evidence", "none"),
        "Frames Tracked": vehicle.get("frames"),
        "Duration (s)": vehicle.get("duration_s"),
        "Distance (m)": vehicle.get("distance_m"),
        "Avg Speed (km/h)": vehicle.get("avg_kmh"),
        "Top Speed (km/h)": vehicle.get("max_kmh"),
    }


def build_log_row(result: FrameResult, servo_angle: int, emergency: bool,
                  level_override: str | None = None,
                  light_phase: str = "") -> dict[str, Any]:
    """Flatten a :class:`FrameResult` into one telemetry log row.

    ``Session`` and ``Session Name`` are stamped on by the logger, which is
    the only component that knows which session is open.
    """
    counts = result.vehicle_counts
    stamp = datetime.fromtimestamp(result.timestamp)
    return {
        "Date": stamp.strftime("%Y-%m-%d"),
        "Timestamp": stamp.strftime("%H:%M:%S"),
        "Congestion Index": result.congestion_index,
        "Level": level_override or result.congestion_level,
        "Light Phase": light_phase,
        "Total Vehicles": result.total_vehicles,
        "Cars": counts.get("car", 0),
        "Motorcycles": counts.get("motorcycle", 0),
        "Buses": counts.get("bus", 0),
        "Trucks": counts.get("truck", 0),
        "Avg Speed (km/h)": result.avg_speed_kmh,
        "Max Speed (km/h)": result.max_speed_kmh,
        "Moving": result.moving_vehicles,
        "Stopped": result.stopped_vehicles,
        "Weighted Density (PCE)": result.weighted_density,
        "Servo Angle (deg)": servo_angle,
        "Emergency": "YES" if emergency else "no",
        "Emergency Type": ", ".join(result.emergency_types) if emergency else "none",
        "Emergency Vehicles": len(result.emergency_ids),
        # Which beacon colours are flashing right now — the evidence behind the
        # verdict, so a disputed log row can be checked against the video.
        "Siren Colour": ", ".join(sorted({
            det.siren_colour for det in result.detections
            if det.is_emergency and det.siren_colour != "none"})) or "none",
        "Marked No Siren": len(result.suspected_ids),
        "Unique Vehicles": result.unique_vehicles_session,
        "FPS": round(result.fps, 1),
        "Speed Source": result.speed_source,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY LOGGING
# ═══════════════════════════════════════════════════════════════════════════

# Session ids are stamped to the second, so two sessions beginning in the same
# second need disambiguating. These are module level, not per-logger, because
# separate TelemetryLogger instances in one process collide just as easily as
# repeated sessions within one instance.
_SESSION_ID_LOCK = threading.Lock()
_LAST_SESSION_BASE: str | None = None
_SESSION_SEQ = 0

#: Filenames used alongside the per-session log when no override is given.
DEFAULT_MASTER_CSV = "traffic_all_sessions.csv"
DEFAULT_MASTER_EXCEL = "traffic_all_sessions.xlsx"
DEFAULT_SESSION_DIR = "sessions"

#: Which sheet in each workbook carries ``LOG_HEADERS`` and so decides whether
#: the file's schema is still current. The two workbooks put it in different
#: places: the session file leads with a snapshot of the same columns, but the
#: master file leads with the *session index*, whose columns are
#: ``SESSION_INDEX_HEADERS``. Comparing that index against ``LOG_HEADERS`` can
#: never match, which archived the master as ``_legacy_`` on every single run.
LOG_SHEET_SESSION = "Traffic Log"
LOG_SHEET_MASTER = "All Records"

#: Per-session summary columns shown on the master workbook's index sheet.
SESSION_INDEX_HEADERS: list[str] = [
    "Session", "Session Name", "Date", "Start", "End", "Duration (s)", "Samples",
    "Peak Vehicles", "Mean Speed (km/h)", "Peak Speed (km/h)",
    "Mean Congestion", "Peak Congestion",
    "FREE", "MODERATE", "HEAVY", "Emergency Samples",
]
SESSION_INDEX_WIDTHS = [18, 18, 12, 10, 10, 13, 10, 14, 18, 18, 16, 16, 9, 12, 9, 18]


def _as_number(value: Any) -> float:
    """Best-effort numeric coercion; anything unparseable becomes 0.0."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def summarise_sessions(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse telemetry rows into one summary record per session.

    Used for the master workbook's index sheet and for the dashboard's session
    picker, so both describe a session in exactly the same terms.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("Session", "")), []).append(row)

    summaries: list[dict[str, Any]] = []
    for session_id, session_rows in grouped.items():
        if not session_id:
            continue
        speeds = [_as_number(r.get("Avg Speed (km/h)")) for r in session_rows]
        speeds = [s for s in speeds if s > 0]
        indices = [_as_number(r.get("Congestion Index")) for r in session_rows]
        levels = Counter(str(r.get("Level", "")) for r in session_rows)
        start = str(session_rows[0].get("Timestamp", ""))
        end = str(session_rows[-1].get("Timestamp", ""))
        summaries.append({
            "Session": session_id,
            "Session Name": str(session_rows[0].get("Session Name", "")),
            "Date": str(session_rows[0].get("Date", "")),
            "Start": start,
            "End": end,
            # Sampling is 1 Hz, so the row count is the duration in seconds.
            "Duration (s)": len(session_rows),
            "Samples": len(session_rows),
            "Peak Vehicles": int(max((_as_number(r.get("Total Vehicles"))
                                      for r in session_rows), default=0)),
            "Mean Speed (km/h)": round(float(np.mean(speeds)), 1) if speeds else 0.0,
            "Peak Speed (km/h)": round(max((_as_number(r.get("Max Speed (km/h)"))
                                            for r in session_rows), default=0.0), 1),
            "Mean Congestion": round(float(np.mean(indices)), 3) if indices else 0.0,
            "Peak Congestion": round(max(indices, default=0.0), 3),
            "FREE": levels.get("FREE", 0),
            "MODERATE": levels.get("MODERATE", 0),
            "HEAVY": levels.get("HEAVY", 0),
            "Emergency Samples": sum(1 for r in session_rows
                                     if str(r.get("Emergency", "")).upper() == "YES"),
        })
    summaries.sort(key=lambda item: item["Session"])
    return summaries


class TelemetryLogger:
    """Crash-safe telemetry logging into a per-session file *and* a master file.

    Two records are kept side by side, because they answer different questions:

    * **Session files** (``traffic_history.csv`` / ``.xlsx``) hold *only the
      run that is happening now*. This is what you look at during a demo, and
      what the dashboard's export button hands you.
    * **Master files** (``traffic_all_sessions.csv`` / ``.xlsx``) accumulate
      *every session ever recorded*, each row stamped with its session id, so
      no past run is ever lost. The master workbook also carries an index
      sheet with one summary row per session.

    Durability rests on the CSVs: every row is appended to both immediately (a
    plain append that cannot corrupt earlier rows), so they are always complete
    up to the last second no matter how the process ends.

    The styled workbooks are derived from them, and are built on request rather
    than on a timer — when one is downloaded, on ``save_now()``, on
    ``stop_session()`` and on ``close()``. Each is written atomically via a
    temp file plus rename, so killing the process can never leave a
    half-written workbook.
    """

    def __init__(self, csv_path: str | Path = "traffic_history.csv",
                 excel_path: str | Path = "traffic_history.xlsx",
                 max_excel_rows: int = 50_000,
                 master_csv: str | Path | None = None,
                 master_excel: str | Path | None = None,
                 session_dir: str | Path | None = None,
                 session_name: str = "auto") -> None:
        """Open both sinks. Workbooks are built on request, not on a timer."""
        self.csv_path = Path(csv_path)
        self.excel_path = Path(excel_path)
        base = self.csv_path.parent

        # Master and archive locations default to sitting beside the session
        # file, which keeps a temp-directory test run fully self-contained.
        self.master_csv_path = Path(master_csv) if master_csv else base / DEFAULT_MASTER_CSV
        self.master_excel_path = (Path(master_excel) if master_excel
                                  else base / DEFAULT_MASTER_EXCEL)
        self.session_dir = Path(session_dir) if session_dir else base / DEFAULT_SESSION_DIR

        # Per-vehicle rows get their own pair of files rather than extra columns
        # on the telemetry ones, because the two have different shapes: telemetry
        # is one row per sampled instant, vehicles are one row per vehicle that
        # passed. Keeping them apart is what lets either be opened and filtered
        # on its own without a pivot.
        self.vehicle_csv_path = base / DEFAULT_VEHICLE_CSV
        self.master_vehicle_path = base / DEFAULT_MASTER_VEHICLE_CSV

        self.max_excel_rows = int(max_excel_rows)

        # Only the rows that can actually reach a workbook are worth keeping in
        # memory; the CSVs already hold the complete record, so anything older
        # than the workbook cap is dead weight in a process that runs for hours.
        self._rows: deque[dict[str, Any]] = deque(maxlen=self.max_excel_rows)
        self._row_total = 0
        # Bounded for the same reason as ``_rows``. This was an unbounded list,
        # which on a busy road grew for the whole run and was re-serialised in
        # full every time a workbook was built.
        self._vehicles: deque[dict[str, Any]] = deque(maxlen=self.max_excel_rows)
        self._lock = threading.Lock()
        self._dirty = False
        self.excel_error: str | None = None
        self.master_error: str | None = None

        self.session_id = ""
        self.session_name = session_name
        self.session_started = time.time()
        self.recording = False

        self._csv_file = None
        self._writer: csv.DictWriter | None = None
        self._master_file = None
        self._master_writer: csv.DictWriter | None = None
        self._vehicle_file = None
        self._vehicle_writer: csv.DictWriter | None = None
        self._master_vehicle_file = None
        self._master_vehicle_writer: csv.DictWriter | None = None

        self._migrate_legacy_files()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._archive_previous_session()
        self._open_master()
        self._open_master_vehicles()
        self._begin_session(session_name, recording=False)

        print(f"[LOG] Session  -> {self.csv_path.name} / {self.excel_path.name}"
              f"  (id {self.session_id})")
        print(f"[LOG] All-time -> {self.master_csv_path.name} /"
              f" {self.master_excel_path.name}")
        print("[LOG] CSVs written every second; workbooks built on download,"
              " on save, and on exit")

    # -- file setup ------------------------------------------------------------

    def _migrate_legacy_files(self) -> None:
        """Move aside any file whose columns no longer match the schema.

        The schema has changed twice: speeds moved from px/s to km/h, and rows
        gained ``Session``/``Light Phase`` columns. Appending new rows to an
        older file would silently misalign the columns, so a mismatched file is
        archived with a timestamp instead of being written to.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for path, expected in ((self.csv_path, LOG_HEADERS),
                               (self.master_csv_path, LOG_HEADERS),
                               (self.vehicle_csv_path, VEHICLE_LOG_HEADERS),
                               (self.master_vehicle_path, VEHICLE_LOG_HEADERS)):
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                with open(path, "r", newline="", encoding="utf-8-sig") as handle:
                    header = next(csv.reader(handle), [])
            except Exception:  # noqa: BLE001
                header = []
            if header != expected:
                archived = path.with_name(f"{path.stem}_legacy_{stamp}.csv")
                try:
                    path.rename(archived)
                    print(f"[LOG] Archived old-format CSV -> {archived.name}")
                except OSError as exc:
                    print(f"[LOG] Could not archive {path.name} ({exc}).")

        for path, sheet in ((self.excel_path, LOG_SHEET_SESSION),
                            (self.master_excel_path, LOG_SHEET_MASTER)):
            if not path.exists():
                continue
            if self._excel_headers(path, sheet) != LOG_HEADERS:
                archived = path.with_name(f"{path.stem}_legacy_{stamp}.xlsx")
                try:
                    path.rename(archived)
                    print(f"[LOG] Archived old-format workbook -> {archived.name}")
                except OSError as exc:
                    print(f"[LOG] Could not archive old workbook ({exc}); "
                          "it will be replaced.")

    @staticmethod
    def _excel_headers(path: Path, sheet: str | None = None) -> list[str]:
        """Return row 2 of ``sheet``, or of the first sheet if it is absent.

        Falling back to the first sheet is what makes a genuinely old-format
        workbook — one written before ``sheet`` existed — still compare unequal
        and get archived, which is the behaviour this check is here for.
        """
        try:
            from openpyxl import load_workbook

            workbook = load_workbook(path, read_only=True)
            name = sheet if sheet in workbook.sheetnames else workbook.sheetnames[0]
            grid = workbook[name]
            headers = [cell.value for cell in next(grid.iter_rows(min_row=2, max_row=2))]
            workbook.close()
            return [h for h in headers if h is not None]
        except Exception:  # noqa: BLE001
            return []

    def _archive_previous_session(self) -> None:
        """Keep the previous run's session files instead of overwriting them.

        Their rows are already in the master, but a standalone copy of each run
        is convenient, so they are moved into ``sessions/`` under the session id
        recorded in the file.
        """
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            return
        previous = "previous"
        try:
            with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as handle:
                first = next(csv.DictReader(handle), None)
            if first and first.get("Session"):
                previous = str(first["Session"])
        except Exception:  # noqa: BLE001
            pass

        for path in (self.csv_path, self.excel_path):
            if not path.exists():
                continue
            target = self.session_dir / f"session_{previous}{path.suffix}"
            try:
                os.replace(path, target)
            except OSError as exc:
                print(f"[LOG] Could not archive {path.name}: {exc}")

    def _open_master(self) -> None:
        """Open the all-sessions CSV for append, writing a header if new."""
        new_file = (not self.master_csv_path.exists()
                    or self.master_csv_path.stat().st_size == 0)
        try:
            self._master_file = open(self.master_csv_path, "a", newline="",
                                     encoding="utf-8-sig" if new_file else "utf-8")
        except OSError as exc:
            print(f"[LOG] Cannot open master CSV: {exc}")
            self._master_file = None
            return
        self._master_writer = csv.DictWriter(self._master_file, fieldnames=LOG_HEADERS,
                                             extrasaction="ignore")
        if new_file:
            self._master_writer.writeheader()
            self._master_file.flush()

    def _open_master_vehicles(self) -> None:
        """Open the all-time per-vehicle CSV for append, writing a header if new."""
        new_file = (not self.master_vehicle_path.exists()
                    or self.master_vehicle_path.stat().st_size == 0)
        try:
            self._master_vehicle_file = open(self.master_vehicle_path, "a", newline="",
                                             encoding="utf-8-sig" if new_file else "utf-8")
        except OSError as exc:
            print(f"[LOG] Cannot open master vehicle CSV: {exc}")
            self._master_vehicle_file = None
            return
        self._master_vehicle_writer = csv.DictWriter(
            self._master_vehicle_file, fieldnames=VEHICLE_LOG_HEADERS, extrasaction="ignore")
        if new_file:
            self._master_vehicle_writer.writeheader()
            self._master_vehicle_file.flush()

    def _next_session_id(self) -> str:
        """Return a session id that is unique even across rapid restarts.

        Two sessions starting in the same second would otherwise share an id and
        be merged into one row of the master index. The second-resolution stamp
        is therefore disambiguated two ways, because there are two ways it can
        collide:

        * a **process-wide** counter, for sessions started back to back in this
          program (a stop/start pair takes well under a second);
        * a check against the archive on disk, for a *different* process that
          happened to start within the same second as this one.
        """
        with _SESSION_ID_LOCK:
            base = datetime.now().strftime("%Y%m%d_%H%M%S")
            global _LAST_SESSION_BASE, _SESSION_SEQ
            if base == _LAST_SESSION_BASE:
                _SESSION_SEQ += 1
            else:
                _LAST_SESSION_BASE, _SESSION_SEQ = base, 1

            candidate = base if _SESSION_SEQ == 1 else f"{base}_{_SESSION_SEQ}"
            while (self.session_dir / f"session_{candidate}.csv").exists():
                _SESSION_SEQ += 1
                candidate = f"{base}_{_SESSION_SEQ}"
            return candidate

    def _begin_session(self, name: str, recording: bool) -> None:
        """Start a fresh session: new id, empty buffers, truncated session CSV."""
        self.session_id = self._next_session_id()
        self.session_name = name or "auto"
        self.session_started = time.time()
        self.recording = recording
        self._rows.clear()
        self._row_total = 0
        self._vehicles.clear()
        self._dirty = False

        # utf-8-sig writes a BOM, which is what makes Excel open the CSV with
        # the right encoding when it is double-clicked instead of imported.
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8-sig")
        # extrasaction="ignore" so an unexpected extra key can never abort a write.
        self._writer = csv.DictWriter(self._csv_file, fieldnames=LOG_HEADERS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._csv_file.flush()

        if self._vehicle_file is not None:
            try:
                self._vehicle_file.close()
            except Exception:  # noqa: BLE001
                pass
        self._vehicle_file = open(self.vehicle_csv_path, "w", newline="",
                                  encoding="utf-8-sig")
        self._vehicle_writer = csv.DictWriter(self._vehicle_file,
                                              fieldnames=VEHICLE_LOG_HEADERS,
                                              extrasaction="ignore")
        self._vehicle_writer.writeheader()
        self._vehicle_file.flush()

    # -- writing ---------------------------------------------------------------

    def log(self, row: dict[str, Any]) -> None:
        """Append one telemetry row to both the session and the master record.

        Safe to call from the capture loop: it is a bounded append plus a
        flush, never a workbook rebuild.
        """
        with self._lock:
            row = dict(row)
            row["Session"] = self.session_id
            row["Session Name"] = self.session_name
            self._rows.append(row)
            self._row_total += 1
            self._dirty = True
            for handle, writer, label in ((self._csv_file, self._writer, "session"),
                                          (self._master_file, self._master_writer, "master")):
                if handle is None or writer is None:
                    continue
                try:
                    writer.writerow(row)
                    handle.flush()
                except Exception as exc:  # noqa: BLE001 - logging must never kill the pipeline
                    print(f"[LOG] {label} CSV write error: {exc}")

    def log_vehicles(self, vehicles: Iterable[dict[str, Any]]) -> None:
        """Record per-vehicle summaries for tracks that have left the scene.

        Each record is appended to the session and all-time vehicle CSVs as it
        arrives, so a vehicle that has passed is on disk immediately and a crash
        can only ever lose the vehicles still in frame. The in-memory copy is
        kept as well, because the session workbook is rebuilt from it.
        """
        with self._lock:
            for vehicle in vehicles:
                self._vehicles.append(vehicle)
                self._dirty = True

                row = build_vehicle_row(vehicle)
                row["Session"] = self.session_id
                row["Session Name"] = self.session_name
                for handle, writer, label in (
                        (self._vehicle_file, self._vehicle_writer, "session vehicle"),
                        (self._master_vehicle_file, self._master_vehicle_writer,
                         "master vehicle")):
                    if handle is None or writer is None:
                        continue
                    try:
                        writer.writerow(row)
                        handle.flush()
                    except Exception as exc:  # noqa: BLE001 - never kill the pipeline
                        print(f"[LOG] {label} CSV write error: {exc}")

    @property
    def row_count(self) -> int:
        """Number of telemetry rows recorded in the current session."""
        with self._lock:
            return self._row_total

    # -- session control -------------------------------------------------------

    def start_session(self, name: str = "") -> dict[str, Any]:
        """Close the current session and begin a new, named recording.

        Rows never stop flowing into the master record, so nothing is lost in
        the moment between the two sessions.
        """
        self.finalize_session()
        with self._lock:
            self._begin_session(name or f"recording_{datetime.now():%H%M%S}",
                                recording=True)
        print(f"[LOG] Recording started: {self.session_name} ({self.session_id})")
        return self.session_info()

    def stop_session(self) -> dict[str, Any]:
        """Finalise the named recording and fall back to passive logging."""
        info = self.session_info()
        self.finalize_session()
        with self._lock:
            self._begin_session("auto", recording=False)
        print(f"[LOG] Recording stopped: {info['name']} ({info['rows']} rows)")
        return info

    def finalize_session(self) -> bool:
        """Write the session workbook and keep a standalone copy of the run."""
        if not self._rows:
            return False
        ok = self.write_excel()
        self.write_master_excel()
        with self._lock:
            handle, self._csv_file, self._writer = self._csv_file, None, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
        for path, suffix in ((self.csv_path, ".csv"), (self.excel_path, ".xlsx"),
                             (self.vehicle_csv_path, "_vehicles.csv")):
            if not path.exists():
                continue
            target = self.session_dir / f"session_{self.session_id}{suffix}"
            try:
                import shutil as _shutil

                _shutil.copy2(path, target)
            except OSError as exc:
                print(f"[LOG] Could not copy {path.name} into {self.session_dir.name}: {exc}")
        return ok

    def session_info(self) -> dict[str, Any]:
        """Describe the session currently being logged."""
        with self._lock:
            rows = self._row_total
        return {
            "id": self.session_id,
            "name": self.session_name,
            "recording": self.recording,
            "rows": rows,
            "started_ms": int(self.session_started * 1000),
            "elapsed_s": round(time.time() - self.session_started, 1),
        }

    def read_master_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return rows from the all-sessions CSV, newest ``limit`` rows first.

        Numeric columns come back as numbers so callers can chart them without
        re-parsing. Returns an empty list if the master file is missing.
        """
        if not self.master_csv_path.exists():
            return []
        try:
            with open(self.master_csv_path, "r", newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                # The archive grows for the life of the project, so read it as a
                # stream and keep only the tail asked for — loading the whole
                # file just to slice the end of it is what made this expensive.
                if limit is not None and limit > 0:
                    rows = list(deque(reader, maxlen=limit))
                else:
                    rows = list(reader)
        except Exception as exc:  # noqa: BLE001
            print(f"[LOG] Could not read master CSV: {exc}")
            return []
        for row in rows:
            for key in NUMERIC_LOG_COLUMNS:
                value = row.get(key)
                if value in (None, ""):
                    row[key] = 0
                    continue
                try:
                    row[key] = float(value) if "." in str(value) else int(value)
                except (TypeError, ValueError):
                    row[key] = 0
        return rows

    def session_index(self) -> list[dict[str, Any]]:
        """One summary record per session ever logged, oldest first."""
        return summarise_sessions(self.read_master_rows())

    # -- excel -----------------------------------------------------------------

    # A background thread used to rebuild both workbooks every 20s. Neither is
    # an append -- each was a full regeneration from scratch -- and openpyxl is
    # pure Python, so the rebuild held the GIL against the detection loop
    # sharing this process. Measured on an 8,600-row archive it cost 6.3s per
    # master rebuild, and both costs grow with the record: about a quarter of a
    # core an hour into a run, approaching half after three. Every workbook the
    # timer produced is now produced on demand instead -- on download, on
    # ``save_now()``, on ``stop_session()`` and on ``close()`` -- so nothing
    # that could be saved before is lost. The CSVs are unaffected: they are
    # still appended row by row every second, and remain the crash-safe record.

    def save_now(self) -> dict[str, bool]:
        """Force an immediate rebuild of both workbooks."""
        return {"session": self.write_excel(), "master": self.write_master_excel()}

    @staticmethod
    def _styles():
        """Return the shared fonts/fills/borders used by both workbooks."""
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        thin = Side(style="thin", color="CCCCCC")
        return {
            "border": Border(left=thin, right=thin, top=thin, bottom=thin),
            "header_font": Font(name="Arial", bold=True, color="FFFFFF", size=10),
            "data_font": Font(name="Arial", size=10),
            "centre": Alignment(horizontal="center", vertical="center"),
            "title_font": Font(name="Arial", bold=True, color="FFFFFF", size=13),
            "title_fill": PatternFill("solid", fgColor="1F3864"),
            "header_fill": PatternFill("solid", fgColor="2E75B6"),
            "level_fills": {
                "FREE": PatternFill("solid", fgColor="C6EFCE"),
                "MODERATE": PatternFill("solid", fgColor="FFEB9C"),
                "HEAVY": PatternFill("solid", fgColor="FFC7CE"),
                "EMERGENCY": PatternFill("solid", fgColor="BDD7EE"),
            },
            "blank": PatternFill(),
        }

    @classmethod
    def _sheet_header(cls, worksheet, title: str, headers: list[str],
                      widths: list[int], style: dict) -> None:
        """Write the banner + header band shared by every sheet.

        The sheets stream straight to disk, so a row is gone the moment it is
        appended. Everything positional — freeze panes, column widths, row
        heights — therefore has to be set before the first append, not after.
        """
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.utils import get_column_letter

        last = get_column_letter(len(headers))
        worksheet.freeze_panes = "A3"
        for col in range(1, len(headers) + 1):
            width = widths[col - 1] if col - 1 < len(widths) else 14
            worksheet.column_dimensions[get_column_letter(col)].width = width
        worksheet.row_dimensions[1].height = 28
        worksheet.row_dimensions[2].height = 26

        banner = WriteOnlyCell(worksheet, value=title)
        banner.font = style["title_font"]
        banner.fill = style["title_fill"]
        banner.alignment = style["centre"]
        worksheet.append([banner])
        worksheet.merged_cells.add(f"A1:{last}1")

        band = []
        for label in headers:
            cell = WriteOnlyCell(worksheet, value=label)
            cell.font = style["header_font"]
            cell.fill = style["header_fill"]
            cell.alignment = style["centre"]
            cell.border = style["border"]
            band.append(cell)
        worksheet.append(band)

    @classmethod
    def _write_rows(cls, worksheet, rows: Iterable[dict[str, Any]], headers: list[str],
                    style: dict, colour_by_level: bool = True) -> None:
        """Stream telemetry rows into a sheet, colour-coded by congestion level.

        Must be called immediately after ``_sheet_header`` — rows land wherever
        the sheet's write cursor currently is.

        The congestion fill is the only styling applied per cell. Font, border
        and alignment were set on every cell too, and cost more than everything
        else in the workbook combined: on an 8,600-row archive, 5.8s of the
        5.9s spent building it, against 2.7s for the fill alone and 1.1s for
        bare values. None of the three carried information — the fill is the
        one that says how busy the road was — so a row needing no fill is
        appended as plain values and skips cell construction altogether. The
        header band keeps its full styling; it is one row.
        """
        from openpyxl.cell import WriteOnlyCell

        for row in rows:
            fill = (style["level_fills"].get(str(row.get("Level")))
                    if colour_by_level else None)
            if fill is None:
                worksheet.append([row.get(key) for key in headers])
                continue
            line = []
            for key in headers:
                cell = WriteOnlyCell(worksheet, value=row.get(key))
                cell.fill = fill
                line.append(cell)
            worksheet.append(line)

    @classmethod
    def _kv_sheet(cls, workbook, sheet_title: str, banner_text: str,
                  stats: dict[str, Any], style: dict) -> None:
        """Build one of the two-column key/value summary tabs."""
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Font

        sheet = workbook.create_sheet(sheet_title)
        sheet.column_dimensions["A"].width = 30
        sheet.column_dimensions["B"].width = 26

        banner = WriteOnlyCell(sheet, value=banner_text)
        banner.font = style["title_font"]
        banner.fill = style["title_fill"]
        banner.alignment = style["centre"]
        sheet.append([banner])
        sheet.merged_cells.add("A1:B1")

        bold = Font(bold=True, size=10)
        for key, value in stats.items():
            name = WriteOnlyCell(sheet, value=key)
            name.font = bold
            data = WriteOnlyCell(sheet, value=value)
            data.font = style["data_font"]
            sheet.append([name, data])

    def _save_atomic(self, workbook, path: Path, what: str) -> bool:
        """Save via a temp file and rename, so a crash cannot truncate ``path``."""
        temp_path = path.with_suffix(".tmp.xlsx")
        try:
            workbook.save(temp_path)
            os.replace(temp_path, path)
            if what == "session":
                self.excel_error = None
            else:
                self.master_error = None
            return True
        except PermissionError:
            # Almost always "the workbook is open in Excel" — keep the CSV
            # flowing and retry on the next tick instead of dying.
            message = f"{path.name} is locked (open in Excel?)"
            if what == "session":
                self.excel_error = message
            else:
                self.master_error = message
            print(f"[LOG] {message} — data is still safe in the CSV")
            return False
        except Exception as exc:  # noqa: BLE001
            if what == "session":
                self.excel_error = str(exc)
            else:
                self.master_error = str(exc)
            print(f"[LOG] Excel write error ({path.name}): {exc}")
            return False
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def write_excel(self) -> bool:
        """Write the current session's styled workbook atomically."""
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
        except ImportError:  # pragma: no cover
            self.excel_error = "openpyxl not installed"
            return False

        with self._lock:
            rows = list(self._rows)
            vehicles = list(self._vehicles)
            self._dirty = False

        if not rows:
            return False

        style = self._styles()
        # write_only streams each row out as it is appended instead of holding
        # the whole grid as Cell objects, which is the difference between a
        # few hundred MB and a few hundred KB on a long session.
        workbook = Workbook(write_only=True)

        # ── Sheet 1: latest snapshot ────────────────────────────────────────
        live = workbook.create_sheet("Live Status")
        self._sheet_header(live, "SMART TRAFFIC ANALYTICS - LIVE STATUS",
                           LOG_HEADERS, LOG_COL_WIDTHS, style)
        self._write_rows(live, rows[-1:], LOG_HEADERS, style)

        # ── Sheet 2: this session's log ─────────────────────────────────────
        log = workbook.create_sheet("Traffic Log")
        self._sheet_header(log, f"SESSION LOG - {self.session_name} ({self.session_id})",
                           LOG_HEADERS, LOG_COL_WIDTHS, style)
        self._write_rows(log, rows, LOG_HEADERS, style)
        log.auto_filter.ref = f"A2:{get_column_letter(len(LOG_HEADERS))}{2 + len(rows)}"

        # ── Sheet 3: one row per vehicle that passed through ────────────────
        if vehicles:
            sheet = workbook.create_sheet("Vehicle Speeds")
            self._sheet_header(sheet, "SMART TRAFFIC ANALYTICS - PER-VEHICLE SPEEDS",
                               VEHICLE_HEADERS, VEHICLE_COL_WIDTHS, style)
            # Plain values, for the reason given in ``_write_rows``: styling
            # every cell here bought nothing and was the slowest part of it.
            for vehicle in vehicles:
                sheet.append([
                    vehicle.get("track_id"), str(vehicle.get("class", "")).capitalize(),
                    vehicle.get("emergency", "no"),
                    vehicle.get("emergency_evidence", "none"),
                    vehicle.get("frames"), vehicle.get("duration_s"),
                    vehicle.get("distance_m"), vehicle.get("avg_kmh"), vehicle.get("max_kmh"),
                ])

        # ── Sheet 4: session summary ────────────────────────────────────────
        self._kv_sheet(workbook, "Session Summary", "SESSION SUMMARY",
                       self._summary_stats(rows, vehicles), style)

        return self._save_atomic(workbook, self.excel_path, "session")

    def write_master_excel(self) -> bool:
        """Rebuild the all-sessions workbook from the master CSV.

        Read back from disk rather than from memory, because the master record
        spans previous runs of the program that this process never saw.
        """
        try:
            from openpyxl import Workbook
            from openpyxl.utils import get_column_letter
        except ImportError:  # pragma: no cover
            self.master_error = "openpyxl not installed"
            return False

        rows = self.read_master_rows(self.max_excel_rows)
        if not rows:
            return False

        style = self._styles()
        sessions = summarise_sessions(rows)
        workbook = Workbook(write_only=True)

        # ── Sheet 1: one row per session ever recorded ──────────────────────
        index = workbook.create_sheet("All Sessions")
        self._sheet_header(index, "ALL RECORDED SESSIONS", SESSION_INDEX_HEADERS,
                           SESSION_INDEX_WIDTHS, style)
        self._write_rows(index, sessions, SESSION_INDEX_HEADERS, style,
                         colour_by_level=False)
        index.auto_filter.ref = (
            f"A2:{get_column_letter(len(SESSION_INDEX_HEADERS))}{2 + len(sessions)}")

        # ── Sheet 2: every telemetry row from every session ─────────────────
        everything = workbook.create_sheet("All Records")
        self._sheet_header(everything,
                           f"ALL RECORDS - {len(sessions)} sessions, {len(rows)} samples",
                           LOG_HEADERS, LOG_COL_WIDTHS, style)
        self._write_rows(everything, rows, LOG_HEADERS, style)
        everything.auto_filter.ref = (
            f"A2:{get_column_letter(len(LOG_HEADERS))}{2 + len(rows)}")

        # ── Sheet 3: every vehicle ever recorded ────────────────────────────
        vehicles = self.read_master_vehicles(self.max_excel_rows)
        if vehicles:
            sheet = workbook.create_sheet("All Vehicles")
            self._sheet_header(sheet, f"ALL VEHICLES - {len(vehicles)} recorded",
                               VEHICLE_LOG_HEADERS, VEHICLE_LOG_COL_WIDTHS, style)
            self._write_rows(sheet, vehicles, VEHICLE_LOG_HEADERS, style,
                             colour_by_level=False)
            sheet.auto_filter.ref = (
                f"A2:{get_column_letter(len(VEHICLE_LOG_HEADERS))}{2 + len(vehicles)}")

        # ── Sheet 4: totals across the whole archive ────────────────────────
        stats = {"Sessions recorded": len(sessions),
                 "Vehicles recorded": len(vehicles),
                 **self._summary_stats(rows, [])}
        self._kv_sheet(workbook, "Lifetime Summary", "LIFETIME SUMMARY", stats, style)

        return self._save_atomic(workbook, self.master_excel_path, "master")

    def read_master_vehicles(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return rows from the all-time vehicle CSV, newest ``limit`` first."""
        if not self.master_vehicle_path.exists():
            return []
        try:
            with open(self.master_vehicle_path, "r", newline="",
                      encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if limit is not None and limit > 0:
                    rows = list(deque(reader, maxlen=limit))
                else:
                    rows = list(reader)
        except Exception as exc:  # noqa: BLE001
            print(f"[LOG] Could not read master vehicle CSV: {exc}")
            return []
        for row in rows:
            for key in ("Track ID", "Frames Tracked", "Duration (s)", "Distance (m)",
                        "Avg Speed (km/h)", "Top Speed (km/h)"):
                value = row.get(key)
                if value in (None, ""):
                    row[key] = 0
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                row[key] = int(number) if number.is_integer() else number
        return rows

    @staticmethod
    def _summary_stats(rows: list[dict[str, Any]],
                       vehicles: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate a set of rows into the key numbers worth reporting."""
        speeds = [_as_number(r.get("Avg Speed (km/h)")) for r in rows]
        speeds = [s for s in speeds if s]
        indices = [_as_number(r.get("Congestion Index")) for r in rows]
        levels = Counter(str(r.get("Level", "")) for r in rows)
        phases = Counter(str(r.get("Light Phase", "")) for r in rows)
        vehicle_speeds = [v.get("avg_kmh", 0) for v in vehicles if v.get("avg_kmh")]
        emergency_kinds = Counter()
        for row in rows:
            if str(row.get("Emergency", "")).upper() == "YES":
                for kind in str(row.get("Emergency Type", "")).split(","):
                    kind = kind.strip()
                    if kind and kind not in {"—", "none"}:
                        emergency_kinds[kind] += 1
        return {
            "Session start": f"{rows[0]['Date']} {rows[0]['Timestamp']}",
            "Session end": f"{rows[-1]['Date']} {rows[-1]['Timestamp']}",
            "Samples logged": len(rows),
            "Unique vehicles": int(max((_as_number(r.get("Unique Vehicles"))
                                        for r in rows), default=0)),
            "Vehicles completed": len(vehicles),
            "Peak vehicles in frame": int(max((_as_number(r.get("Total Vehicles"))
                                               for r in rows), default=0)),
            "Mean speed (km/h)": round(float(np.mean(speeds)), 1) if speeds else 0.0,
            "Peak speed (km/h)": round(max((_as_number(r.get("Max Speed (km/h)"))
                                            for r in rows), default=0.0), 1),
            "Mean per-vehicle speed (km/h)":
                round(float(np.mean(vehicle_speeds)), 1) if vehicle_speeds else 0.0,
            "Mean congestion index": round(float(np.mean(indices)), 3) if indices else 0.0,
            "Peak congestion index": round(max(indices, default=0.0), 3),
            "Time FREE (samples)": levels.get("FREE", 0),
            "Time MODERATE (samples)": levels.get("MODERATE", 0),
            "Time HEAVY (samples)": levels.get("HEAVY", 0),
            "Light GREEN (samples)": phases.get("GREEN", 0),
            "Light YELLOW (samples)": phases.get("YELLOW", 0),
            "Light RED (samples)": phases.get("RED", 0),
            "Emergency samples": sum(1 for r in rows
                                     if str(r.get("Emergency", "")).upper() == "YES"),
            "Emergency types seen": ", ".join(
                f"{kind} ({count})" for kind, count in emergency_kinds.most_common()) or "none",
            "Speed calibration": rows[-1].get("Speed Source", "unknown"),
        }

    def close(self) -> None:
        """Flush everything and write the final session and master workbooks."""
        rows = self.row_count
        ok = self.write_excel()
        master_ok = self.write_master_excel()
        self.finalize_session()
        for handle in (self._csv_file, self._master_file,
                       self._vehicle_file, self._master_vehicle_file):
            try:
                if handle is not None:
                    handle.close()
            except Exception:  # noqa: BLE001
                pass
        self._csv_file = None
        self._master_file = None
        self._vehicle_file = None
        self._master_vehicle_file = None
        self._vehicle_writer = None
        self._master_vehicle_writer = None
        print(f"[LOG] Session closed: {rows} rows"
              f" | session workbook {'written' if ok else 'NOT written (see CSV)'}"
              f" | master {'written' if master_ok else 'NOT written (see CSV)'}")
