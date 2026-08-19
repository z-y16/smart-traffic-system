"""
emergency_vision.py — recognising ambulances, police cars and fire trucks
=========================================================================

COCO has no ambulance, police-car or fire-truck class — to YOLO an ambulance is
a ``truck`` and a police car is a ``car``. This module answers **what kind of
vehicle is that**, from its livery, markings and body shape.

It does not decide whether to clear a lane. That is ``siren_vision.py``'s job,
and it turns on a *working beacon*, because an ambulance parked outside a
hospital is still an ambulance and does not need the junction held for it. What
this module contributes is the name: a red beacon belongs to both ambulances and
fire engines, and only the livery tells them apart.

That division of labour was learned the hard way. When livery *was* the trigger,
this module flagged **88 of 299 vehicles — 29% of all traffic** — on the night
footage in ``vedios/``. Two things were wrong. "A police vehicle with sirens and
blue lights" is a sentence any dark saloon with its headlamps on matches at 50
pixels across; and the score summed *all* emergency prompts, so with 14 of them
against 15 ordinary ones a CLIP that had learned nothing from a crop scored
14/29 = 0.48 — just over the 0.45 threshold calibrated on daylight photographs.
Every ambiguous vehicle came out "police" by construction. The score is now a
two-way posterior against the ordinary class, so the same uninformative crop
scores 0.25.

Two recognisers, one interface
------------------------------
Both report on the same 0–1 scale, so the thresholds and hysteresis below govern
either. ``backend="auto"`` picks the trained model when its weights are present
and falls back to CLIP when they are not.

**CLIP** (zero-shot) needs no data, no training and no labels: it was
pre-trained on 400 million image/caption pairs and already knows what an
ambulance looks like in essentially any country. The only download is the
weight file (~335 MB), fetched on first use and cached.

**Trained** (``emergency_cls.pt``, built by ``train_emergency_classifier.py``)
knows less about the world and much more about *this* camera, because part of
its training data was mined from this system's own footage. Measured head to
head on held-out crops from that camera:

    =========================  ======  =======
    over the 0.70 threshold      CLIP  Trained
    =========================  ======  =======
    ambulance recognised        53.4%    71.1%
    police car recognised       41.1%    71.4%
    fire engine recognised      39.7%    70.3%
    false alarms, this camera    0.0%     0.0%
    false alarms, ImageNet       0.3%     5.3%
    cost per crop              4.3 ms   0.7 ms
    =========================  ======  =======

That table is why the trained model used to be the default, and the reason it
no longer is has nothing to do with those numbers being wrong. Two things
changed. Livery now *reports* an emergency instead of only annotating one, so
a false positive is no longer a harmless "?"; and the system started being
pointed at footage from cameras other than this one, where the last two rows
stop applying. On a night motorway clip it had never seen, the trained model
called **six ordinary cars "fire truck" at 0.94–1.00 confidence** — red tail
lights, the exact thing it was trained to see past — and scored an
unmistakably marked daylight ambulance 0.55, below its own threshold. CLIP
scored that ambulance 0.95 and raised nothing on the same eight cars, nor on
4186 detections of daylight traffic.

So the default is now CLIP, which is the honest reading of "works anywhere".
``backend="trained"`` remains the right choice for the camera the weights were
built for, where it is both more sensitive and seven times cheaper.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:  # pragma: no cover - exercised at runtime
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════
# Several phrasings per class ("prompt ensembling") — averaging over wordings
# is markedly more stable than relying on one sentence. The negative set is
# deliberately larger than the positive ones and stocked with the vehicles
# that actually get confused for emergency vehicles: taxis (roof signs), white
# vans (ambulance-like body), buses and lorries (size), tow and utility trucks
# (amber beacons).

EMERGENCY_PROMPTS: dict[str, list[str]] = {
    "ambulance": [
        "a photo of an ambulance",
        "an emergency ambulance vehicle",
        "a paramedic ambulance with medical markings",
        "an ambulance van with a red cross painted on the side",
        "an EMS emergency medical services vehicle",
    ],
    "police": [
        "a photo of a police car",
        "a police patrol car with police markings",
        "a police cruiser painted in black and white livery",
        "a police car with the word police written on the door",
        "a highway patrol police interceptor",
    ],
    "fire truck": [
        "a photo of a fire truck",
        "a red fire engine with ladders",
        "a fire department emergency apparatus",
        "a firefighting truck with hoses and equipment",
    ],
}

# The police prompts deliberately describe *livery* — paintwork, lettering,
# body shape — and no longer mention sirens, light bars or blue lights. Asking
# CLIP "does this have emergency lights?" at night is asking it to say yes: a
# dark saloon with its headlamps on matches that sentence almost perfectly, and
# on the night footage in ``vedios/`` the earlier wording made CLIP call 29% of
# all traffic a police car. Flashing lights are now the siren detector's job,
# where they can be measured; CLIP is asked only what the vehicle *is*.

ORDINARY_PROMPTS: list[str] = [
    "a photo of an ordinary civilian car",
    "a normal passenger car on the road",
    "a regular delivery van",
    "an ordinary commercial truck",
    "a parked family car",
    "a taxi cab",
    "a city bus",
    "a pickup truck",
    "an SUV",
    "a hatchback car",
    "a white cargo van",
    "a lorry on the motorway",
    "a tow truck with an amber beacon",
    "a construction utility vehicle",
    "an empty road with no vehicle",
    # Night-time negatives. Every one of these describes something the night
    # footage is full of, and without them those images have nowhere to go but
    # into an emergency class.
    "a car at night with its headlights on",
    "a dark car driving on a road at night",
    "the red tail lights of a car in the dark",
    "a blurry low-resolution night photograph of a vehicle",
    "a car under orange street lighting at night",
    "an out-of-focus dark shape on a road",
    "a car windscreen reflecting street lights",
    "a motorcycle at night",
    "headlight glare on a wet road",
]

ORDINARY_LABEL = "ordinary"

# Short labels for the 16-character LCD on the Arduino.
LCD_NAMES: dict[str, str] = {
    "ambulance": "AMBULANCE",
    "police": "POLICE",
    "fire truck": "FIRE TRUCK",
    "unknown": "EMERGENCY",
}


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG & RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EmergencyConfig:
    """Tunable parameters for appearance-based emergency recognition."""

    enabled: bool = True
    #: ``clip`` — zero-shot, no training, works out of the box in any country.
    #: ``trained`` — the fine-tuned classifier in ``model_path``.
    #: ``auto`` — the trained model if ``model_path`` exists, else CLIP.
    #:
    #: ``clip`` is the default, and the reason is that livery now *reports* an
    #: emergency rather than only annotating one (see :func:`fuse`). That moves
    #: the false-positive rate from cosmetic to decisive, and the two backends
    #: are not close on it. Measured on night motorway footage this camera had
    #: never seen, and on a daylight ambulance from a foreign CCTV camera:
    #:
    #:     ============================  =======  =======
    #:     on unseen footage             Trained     CLIP
    #:     ============================  =======  =======
    #:     daylight ambulance found        0.55✗    0.95✓
    #:     ordinary night cars flagged      6/8      0/8
    #:     …at confidence                0.94–1.00     —
    #:     ============================  =======  =======
    #:
    #: The trained model called six ordinary cars "fire truck" with total
    #: confidence — red tail lights at night, which is exactly what it was
    #: trained to see through — while scoring a real, unmistakably marked
    #: ambulance below the threshold. Its published 0.0% false-alarm rate was
    #: measured on *this camera's own* footage, and does not survive contact
    #: with a source it was not trained on. CLIP was clean on both, and on
    #: 4186 daylight detections it raised nothing.
    #:
    #: The trained model remains the better recogniser on the camera it was
    #: built for (71% against 45%) and is seven times faster (0.7 ms per crop
    #: against 4.3), so ``trained`` stays available — but it should only be
    #: chosen for a camera it has been trained on.
    backend: str = "clip"
    model_name: str = "ViT-B/16"     # ViT-B/32 = faster, ViT-L/14 = 2x slower
    model_path: str = "emergency_cls.pt"   # used when backend == "trained"
    trained_imgsz: int = 160               # must match how the model was trained
    device: str = "auto"
    min_crop_px: int = 56            # below this a crop carries no usable livery
    crop_padding: float = 0.05       # widen the box slightly to catch markings
    max_batch: int = 16              # crops classified per forward pass

    # Scheduling — classifying every vehicle on every frame is wasteful, so
    # each track is re-checked periodically and backs off once its verdict is
    # settled.
    check_interval_frames: int = 5
    max_backoff_frames: int = 60
    confident_ordinary_score: float = 0.35   # below this, back off faster

    # Decision — hysteresis on a smoothed score, so one odd crop cannot trip
    # the priority lane and one bad frame cannot release it.
    #
    # The score is a two-way posterior, ``P(best emergency type) / (P(best
    # emergency type) + P(ordinary))``. That framing is what makes the
    # threshold mean anything. The previous score summed the probability of
    # *all* emergency prompts, and with 14 emergency prompts against 15
    # ordinary ones a completely uninformative CLIP — which is what a 50-pixel
    # dark car at night produces — scores 14/29 = 0.48, just over the 0.45
    # threshold that was calibrated on daylight photographs. Every ambiguous
    # vehicle therefore came out "police". Under the ratio form the same
    # uninformative case scores 0.25, because the larger ordinary prompt set
    # now acts as the prior it should always have been.
    score_ema_alpha: float = 0.5
    on_threshold: float = 0.70
    off_threshold: float = 0.50
    min_checks: int = 3
    track_ttl_seconds: float = 5.0

    # Quality gates. CLIP is confident and wrong on crops that carry no
    # information, so crops it cannot possibly read are not shown to it.
    min_mean_luma: float = 28.0    # near-black crop: nothing to recognise
    min_contrast: float = 12.0     # flat crop: fog, motion blur, or a shadow


@dataclass
class EmergencyVerdict:
    """The system's current belief about one tracked vehicle."""

    is_emergency: bool = False
    vehicle_type: str = ORDINARY_LABEL
    confidence: float = 0.0
    evidence: str = "none"           # siren-blue | siren+livery | appearance | none
    checks: int = 0
    #: Looks like an emergency vehicle but is not running its beacon — an
    #: ambulance returning to base, a parked patrol car. Reported so the
    #: dashboard can show it, but it does not open the priority lane.
    suspected: bool = False
    #: Whether this vehicle should actually be given priority — the junction
    #: held, the lane cleared, the hardware told. Deliberately separate from
    #: :attr:`is_emergency`, which only means "say so on screen and in the log".
    #: A marked ambulance with no beacon running is worth reporting and not
    #: worth holding a junction for, and one flag cannot express both.
    priority: bool = False

    @property
    def display_name(self) -> str:
        """Short uppercase name for on-screen and LCD display."""
        name = LCD_NAMES.get(self.vehicle_type, "EMERGENCY")
        if self.priority:
            return name
        # Recognised, but nothing is being held for it: the trailing "?" is
        # what distinguishes a responding vehicle from a parked one at a
        # glance, and on the LCD it is the only room there is to say so.
        if self.is_emergency or self.suspected:
            return f"{name}?"
        return ""


@dataclass
class _TrackBelief:
    """Accumulated appearance evidence for one tracked vehicle."""

    score: float = 0.0
    votes: Counter = field(default_factory=Counter)
    checks: int = 0
    active: bool = False
    last_seen: float = 0.0
    next_check_frame: int = 0
    backoff: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=12))


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════


class EmergencyAppearanceClassifier:
    """Zero-shot CLIP classifier that recognises emergency vehicles by sight.

    Designed to be cheap enough to run inside the live camera loop:

    * crops are classified in **batches**, one forward pass per frame at most;
    * each track is only re-checked every ``check_interval_frames``;
    * tracks that look confidently ordinary **back off** exponentially, so a
      street full of normal cars costs almost nothing after the first second;
    * everything degrades gracefully — if CLIP or torch is unavailable the
      classifier reports itself unavailable and the pipeline falls back to the
      light-bar heuristic alone.
    """

    def __init__(self, config: EmergencyConfig | None = None) -> None:
        """Load CLIP and pre-compute the text embeddings (done once)."""
        self.config = config or EmergencyConfig()
        self.available = False
        self.error: str | None = None
        self.device = "cpu"
        self._beliefs: dict[int, _TrackBelief] = {}
        self._frame_index = 0
        self._lock = threading.Lock()
        self.last_batch_ms = 0.0
        self.total_classifications = 0
        self.unreadable_skips = 0
        self.backend = config.backend if config else "auto"

        self.labels: list[str] = list(EMERGENCY_PROMPTS) + [ORDINARY_LABEL]
        self._emergency_label_count = len(EMERGENCY_PROMPTS)

        if not self.config.enabled:
            self.error = "disabled by configuration"
            return
        if not TORCH_AVAILABLE:
            self.error = "torch not installed"
            print(f"[EMRG] Appearance recognition off: {self.error}")
            return

        try:
            self._load()
            self.available = True
        except Exception as exc:  # noqa: BLE001 - never let this kill the pipeline
            self.error = str(exc)
            print(f"[EMRG] Appearance recognition unavailable: {exc}")
            print("[EMRG] Falling back to siren-light detection only.")

    # -- setup ---------------------------------------------------------------

    def _resolved_backend(self) -> str:
        """Return the backend actually in use, resolving ``auto``."""
        backend = self.config.backend
        if backend != "auto":
            return backend
        return "trained" if self._model_file() is not None else "clip"

    def _model_file(self) -> Path | None:
        """Locate the trained weights, relative to this file if need be."""
        path = Path(self.config.model_path)
        if not path.is_absolute():
            candidate = Path(__file__).resolve().parent / path
            if candidate.exists():
                return candidate
        return path if path.exists() else None

    def _load(self) -> None:
        """Load whichever recogniser the configuration asks for."""
        self.backend = self._resolved_backend()
        if self.backend == "trained":
            self._load_trained()
        else:
            self._load_clip()

    def _load_trained(self) -> None:
        """Load the fine-tuned classifier and map its classes onto ours.

        The class names come from the training folders, so ``fire_truck``
        arrives with an underscore while the rest of the system — the LCD
        strings, the logs, the dashboard — spells it ``fire truck``. The
        mapping is done here rather than by renaming a folder, so a model
        trained by someone else still slots in.
        """
        from ultralytics import YOLO

        cfg = self.config
        path = self._model_file()
        if path is None:
            raise RuntimeError(
                f"trained model '{cfg.model_path}' not found — train one with "
                "train_emergency_classifier.py, or set backend='clip'")

        self.device = (cfg.device if cfg.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[EMRG] Loading trained classifier {path.name} on "
              f"{self.device.upper()}")
        started = time.perf_counter()
        self._model = YOLO(str(path))
        names = self._model.names
        order = [names[i] for i in sorted(names)]
        self._trained_labels = [name.replace("_", " ") for name in order]
        if ORDINARY_LABEL not in self._trained_labels:
            raise RuntimeError(
                f"trained model has classes {self._trained_labels}, with no "
                f"'{ORDINARY_LABEL}' class to compare against")
        self._ordinary_index = self._trained_labels.index(ORDINARY_LABEL)
        print(f"[EMRG] Ready in {time.perf_counter() - started:.1f}s "
              f"(classes: {', '.join(self._trained_labels)})")

    def _load_clip(self) -> None:
        """Import CLIP, load the weights and embed every prompt."""
        try:
            import clip
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the 'clip' package is missing — install it with:\n"
                "    pip install git+https://github.com/ultralytics/CLIP.git"
            ) from exc

        cfg = self.config
        self.device = (cfg.device if cfg.device != "auto"
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[EMRG] Loading CLIP {cfg.model_name} on {self.device.upper()} "
              "for ambulance / police / fire-truck recognition")

        started = time.perf_counter()
        self._model, self._preprocess = clip.load(cfg.model_name, device=self.device)
        self._model.eval()

        prompts: list[str] = []
        owners: list[int] = []
        for index, label in enumerate(EMERGENCY_PROMPTS):
            for prompt in EMERGENCY_PROMPTS[label]:
                prompts.append(prompt)
                owners.append(index)
        for prompt in ORDINARY_PROMPTS:
            prompts.append(prompt)
            owners.append(len(EMERGENCY_PROMPTS))

        with torch.no_grad():
            features = self._model.encode_text(clip.tokenize(prompts).to(self.device)).float()
            features /= features.norm(dim=-1, keepdim=True)
        self._text_features = features
        self._owners = torch.tensor(owners, device=self.device)

        print(f"[EMRG] Ready in {time.perf_counter() - started:.1f}s "
              f"({len(prompts)} prompts, {len(self.labels)} classes)")

    # -- per-frame API -------------------------------------------------------

    def begin_frame(self) -> None:
        """Advance the internal frame counter (call once per processed frame)."""
        self._frame_index += 1

    def _due(self, track_id: int, box_width: int, box_height: int) -> bool:
        """Return whether this track should be re-classified on this frame."""
        if min(box_width, box_height) < self.config.min_crop_px:
            return False
        belief = self._beliefs.get(track_id)
        if belief is None:
            return True
        return self._frame_index >= belief.next_check_frame

    def _crop(self, frame: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
        """Cut a padded vehicle crop out of the frame."""
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in box)
        pad_x = (x2 - x1) * self.config.crop_padding
        pad_y = (y2 - y1) * self.config.crop_padding
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2]

    def classify_frame(self, frame: np.ndarray,
                       tracks: Sequence[tuple[int, Sequence[float]]],
                       now: float) -> dict[int, EmergencyVerdict]:
        """Classify the tracks that are due and return the verdict for each.

        ``tracks`` is a sequence of ``(track_id, bbox)``. Every track gets a
        verdict back — tracks that were not re-classified this frame simply
        return their existing belief, which is what makes the periodic
        scheduling invisible to the caller.
        """
        verdicts: dict[int, EmergencyVerdict] = {}
        if not tracks:
            return verdicts

        pending_ids: list[int] = []
        pending_crops: list[np.ndarray] = []

        for track_id, box in tracks:
            belief = self._beliefs.get(track_id)
            if belief is None:
                belief = _TrackBelief(last_seen=now, next_check_frame=self._frame_index)
                self._beliefs[track_id] = belief
            belief.last_seen = now

            width = int(box[2]) - int(box[0])
            height = int(box[3]) - int(box[1])
            if self.available and self._due(track_id, width, height):
                crop = self._crop(frame, box)
                if crop is not None and crop.size and self._readable(crop):
                    pending_ids.append(track_id)
                    pending_crops.append(crop)
                elif crop is not None:
                    # Unreadable this frame — try again shortly rather than
                    # backing off, since the vehicle may move into better light.
                    belief.next_check_frame = (self._frame_index
                                               + self.config.check_interval_frames)
                    self.unreadable_skips += 1

        if pending_crops:
            self._run_batch(pending_ids, pending_crops)

        for track_id, _box in tracks:
            verdicts[track_id] = self._verdict(track_id)
        return verdicts

    def _run_batch(self, track_ids: list[int], crops: list[np.ndarray]) -> None:
        """Score a batch of crops and fold the result into each track's belief."""
        cfg = self.config
        started = time.perf_counter()
        try:
            for offset in range(0, len(crops), cfg.max_batch):
                chunk_ids = track_ids[offset:offset + cfg.max_batch]
                chunk = crops[offset:offset + cfg.max_batch]
                scores = self._score(chunk)
                for track_id, (emergency_score, label) in zip(chunk_ids, scores):
                    self._update_belief(track_id, emergency_score, label)
            self.total_classifications += len(crops)
        except Exception as exc:  # noqa: BLE001
            # A single bad frame must never take the traffic system down.
            self.error = str(exc)
            print(f"[EMRG] Classification error (continuing): {exc}")
        self.last_batch_ms = (time.perf_counter() - started) * 1000.0

    def _score(self, crops: list[np.ndarray]) -> list[tuple[float, str]]:
        """Return ``(P(emergency | emergency or ordinary), best_label)`` per crop.

        The reported score deliberately compares the single best-fitting
        emergency type against the ordinary class, rather than pooling all
        three emergency types together. Pooling rewards uncertainty: a crop
        that the model finds equally reminiscent of an ambulance, a police car
        and a fire engine is a crop it does not recognise, and it should not
        score three times as highly as one that clearly resembles just one of
        them. Both backends report on this same scale, so the thresholds and
        the hysteresis below mean the same thing either way.
        """
        if self.backend == "trained":
            return self._score_trained(crops)
        return self._score_clip(crops)

    def _score_trained(self, crops: list[np.ndarray]) -> list[tuple[float, str]]:
        """Score crops with the fine-tuned classifier."""
        predictions = self._model.predict(
            crops, device=self.device, verbose=False,
            imgsz=self.config.trained_imgsz)

        results: list[tuple[float, str]] = []
        for prediction in predictions:
            probabilities = prediction.probs.data.detach().cpu().numpy()
            ordinary = float(probabilities[self._ordinary_index])
            best_index, best = -1, -1.0
            for index, probability in enumerate(probabilities):
                if index != self._ordinary_index and probability > best:
                    best_index, best = index, float(probability)
            score = best / (best + ordinary) if (best + ordinary) > 1e-9 else 0.0
            label = (self._trained_labels[best_index] if score >= 0.5
                     else ORDINARY_LABEL)
            results.append((score, label))
        return results

    def _score_clip(self, crops: list[np.ndarray]) -> list[tuple[float, str]]:
        """Score crops with zero-shot CLIP against the prompt sets."""
        from PIL import Image

        batch = torch.stack([
            self._preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            for crop in crops
        ]).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(batch).float()
            features /= features.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * features @ self._text_features.T).softmax(dim=-1)

        results: list[tuple[float, str]] = []
        for row in probabilities:
            grouped = [float(row[self._owners == index].sum())
                       for index in range(len(self.labels))]
            emergency = grouped[:self._emergency_label_count]
            ordinary = grouped[self._emergency_label_count]
            best_index = int(np.argmax(emergency))
            best = emergency[best_index]
            score = best / (best + ordinary) if (best + ordinary) > 1e-9 else 0.0
            label = (self.labels[best_index] if score >= 0.5 else ORDINARY_LABEL)
            results.append((score, label))
        return results

    def _readable(self, crop: np.ndarray) -> bool:
        """Return whether a crop carries enough signal to be worth classifying.

        A near-black or perfectly flat crop still produces a confident CLIP
        answer — it just has nothing to do with the vehicle. Declining to ask
        is more honest than averaging the guess into the belief.
        """
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return bool(grey.mean() >= self.config.min_mean_luma
                    and grey.std() >= self.config.min_contrast)

    def _update_belief(self, track_id: int, emergency_score: float, label: str) -> None:
        """Fold one classification into a track's running belief."""
        cfg = self.config
        belief = self._beliefs.setdefault(track_id, _TrackBelief())
        belief.checks += 1
        belief.history.append(emergency_score)

        alpha = cfg.score_ema_alpha
        belief.score = (emergency_score if belief.checks == 1
                        else alpha * emergency_score + (1 - alpha) * belief.score)
        if label != ORDINARY_LABEL:
            belief.votes[label] += 1

        if not belief.active and belief.score >= cfg.on_threshold \
                and belief.checks >= cfg.min_checks:
            belief.active = True
        elif belief.active and belief.score < cfg.off_threshold:
            belief.active = False

        # Vehicles that look clearly ordinary are re-checked less and less
        # often; anything ambiguous or flagged keeps the fast cadence.
        if belief.score <= cfg.confident_ordinary_score and not belief.active:
            belief.backoff = min(max(belief.backoff * 2, cfg.check_interval_frames),
                                 cfg.max_backoff_frames)
        else:
            belief.backoff = cfg.check_interval_frames
        belief.next_check_frame = self._frame_index + belief.backoff

    def _verdict(self, track_id: int) -> EmergencyVerdict:
        """Build the public verdict for a track from its belief."""
        belief = self._beliefs.get(track_id)
        if belief is None:
            return EmergencyVerdict()
        vehicle_type = (belief.votes.most_common(1)[0][0] if belief.votes
                        else (ORDINARY_LABEL if not belief.active else "unknown"))
        return EmergencyVerdict(
            is_emergency=belief.active,
            vehicle_type=vehicle_type if belief.active else ORDINARY_LABEL,
            confidence=round(belief.score, 2),
            evidence="appearance" if belief.active else "none",
            checks=belief.checks,
        )

    def verdict_for(self, track_id: int) -> EmergencyVerdict:
        """Public read-only access to a track's current verdict."""
        return self._verdict(track_id)

    def reset(self) -> None:
        """Forget every track's belief, for when the picture jumps sources."""
        self._beliefs.clear()

    def prune(self, now: float) -> None:
        """Forget tracks that have not been seen for a while."""
        ttl = self.config.track_ttl_seconds
        for track_id in [t for t, b in self._beliefs.items() if now - b.last_seen > ttl]:
            del self._beliefs[track_id]

    @property
    def status(self) -> str:
        """One-line human description of the recogniser's state."""
        if not self.available:
            return f"livery recognition off — {self.error or 'unavailable'}"
        if self.backend == "trained":
            return (f"trained classifier {Path(self.config.model_path).name} on "
                    f"{self.device} ({'/'.join(self._trained_labels)})")
        return f"CLIP {self.config.model_name} on {self.device} (ambulance/police/fire)"


# ═══════════════════════════════════════════════════════════════════════════
# FUSION
# ═══════════════════════════════════════════════════════════════════════════


#: Colours that leave the vehicle type genuinely ambiguous. Blue is decisive —
#: nothing but police carries a flashing blue lamp — but red is shared between
#: ambulances and fire engines, so there the livery is worth consulting.
_AMBIGUOUS_COLOURS = frozenset({"red"})


def fuse(appearance: EmergencyVerdict, siren, *,
         require_siren: bool = True) -> EmergencyVerdict:
    """Combine what the vehicle *looks like* with what its beacon is *doing*.

    Two questions are answered separately, because they have different
    answers: **is this an emergency vehicle** (``is_emergency``, which governs
    what is displayed and logged) and **should the junction be held for it**
    (``priority``, which governs the hardware).

    The beacon leads on the second question, deliberately. A vehicle deserves a
    lane cleared for it when it is responding to a call, and the visible
    evidence of that is a working siren light — not the livery, which is just
    as present on an ambulance parked outside a hospital. So by default a
    flashing beacon is necessary *for priority*, and appearance serves three
    purposes:

    * it names the vehicle when the beacon colour is ambiguous — red belongs to
      both ambulances and fire engines, while blue can only be police;
    * it raises the reported confidence when both cues agree;
    * on its own it still reports the vehicle, because a recognised ambulance
      that goes unmentioned looks exactly like one that was never seen — and in
      daylight, where beacons bloom white and the colour gates reject them, it
      is the only cue there is.

    With ``require_siren=False`` livery alone also grants priority, which is the
    right setting for a camera watching a hospital or station approach where
    vehicles are marked but not always running their lights.

    ``siren`` is a :class:`siren_vision.SirenVerdict`; it is untyped here so
    that this module keeps working when the siren detector is not installed.
    """
    siren_active = bool(getattr(siren, "active", False))
    siren_type = getattr(siren, "vehicle_type", ORDINARY_LABEL)
    siren_colour = getattr(siren, "colour", "none")
    siren_confidence = float(getattr(siren, "confidence", 0.0))

    if siren_active:
        # The beacon decides, except where its colour cannot distinguish an
        # ambulance from a fire engine and the livery can.
        vehicle_type = siren_type
        evidence = f"siren-{siren_colour}"
        confidence = siren_confidence
        if appearance.is_emergency:
            if siren_colour in _AMBIGUOUS_COLOURS and appearance.vehicle_type != "police":
                vehicle_type = appearance.vehicle_type
            evidence = f"siren-{siren_colour}+livery"
            confidence = max(confidence, appearance.confidence)
        return EmergencyVerdict(True, vehicle_type, round(confidence, 2),
                                evidence, appearance.checks, priority=True)

    if appearance.is_emergency:
        if require_siren:
            # Marked, but not running its lights. Reported as an emergency
            # vehicle — it is one, and a system that stays silent about a
            # recognised ambulance is indistinguishable from one that failed
            # to see it — but given no priority, because an ambulance parked
            # outside a hospital does not need the junction held.
            #
            # This is also the only cue left in daylight: a beacon bright
            # enough to read as a lamp blooms to *white* on the sensor, and
            # the colour-purity gates in siren_vision.py reject white by
            # design. On the daylight footage tested, the beacon channels
            # peaked at 0.04 while CLIP scored the livery 0.94.
            return EmergencyVerdict(True, appearance.vehicle_type,
                                    appearance.confidence, "livery-no-siren",
                                    appearance.checks, suspected=True,
                                    priority=False)
        return EmergencyVerdict(True, appearance.vehicle_type,
                                appearance.confidence, "appearance",
                                appearance.checks, priority=True)

    return EmergencyVerdict(False, ORDINARY_LABEL, appearance.confidence,
                            "none", appearance.checks)
