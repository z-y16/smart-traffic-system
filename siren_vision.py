"""
siren_vision.py — recognising an emergency vehicle by its *working* siren
=========================================================================

An ambulance with its beacons off is just a van, and the traffic light has no
business clearing a lane for it. What actually deserves priority is a vehicle
**responding to a call** — and the visible signature of that is the roof beacon
switching on and off. This module detects exactly that, and reads the vehicle
type straight off the colour of the light:

    flashing blue          → police
    flashing red           → ambulance   (fire engine if the vehicle is large)
    alternating red + blue → police      (configurable per country)

Why this replaces appearance recognition as the trigger
-------------------------------------------------------
The previous system asked CLIP "does this look like a police car?". Measured on
the night footage in ``vedios/``, that question flagged **88 of 299 ordinary
vehicles** — 29% of all traffic — because a dark car with headlights at night
is, to CLIP, an excellent match for "a police vehicle with sirens and blue
lights". Appearance is a *description*, and every dark saloon fits it.

A flashing beacon is not a description, it is an **event**: light that is
present in one frame and gone in the next, in a hue no ordinary vehicle
produces, in a place ordinary lamps do not sit. That is measurable rather than
guessable, and it is what this module measures.

What separates a beacon from every other light on the road
----------------------------------------------------------
============  =========================  ==========================================
Light         Why it looks similar       Why it is rejected here
============  =========================  ==========================================
Tail lights   Saturated red              Steady — modulation depth ≈ 0
Brake lights  Saturated red, switching   Switch once or twice, not ≥3 times in 3.4 s
Centre brake  High on the body, flashes  Stops flashing; a beacon does not, and
light         when braking in traffic    confirmation is required to persist 1.2 s
Indicators    Blink at ~1.5 Hz           Amber: fails the ``g ≤ 0.55·r`` purity gate
US rear       Blink at ~1.5 Hz, pure red  Sit low on the body — the red channel is
indicators                                sampled only in the roof band
Headlights    Very bright                White — fails both purity gates
Street lamps  Bright, sometimes amber    Steady, and usually outside the vehicle box
Red car body  Saturated red              Steady — no flashes, no matter how red
Car under     Brightens and dims as it   *Ramps*, and a lamp *switches*: the share of
street light  passes each lamp           samples caught mid-transition gives it away
Tow beacons   Genuinely flashing         Amber: fails the purity gate
============  =========================  ==========================================

So the whole design is: *be generous about what counts as coloured light, and
uncompromising about it having to switch on and off repeatedly.* Only the
temporal test can tell a beacon from a paint job, and it is the temporal test
that carries the decision.

Cost
----
Per tracked vehicle per frame: one small crop, two subtractions and two
percentile reductions over at most 192×192 pixels. The flash analysis runs on a
3.4-second ring buffer of scalars. Measured over the night footage the whole
detector costs 0.5–7 ms per frame depending on how many vehicles are in shot.
Unlike the CLIP pass it needs no GPU, no weights and no download, so it also
runs on the machines where the appearance model is switched off.

Sampling requirement
--------------------
Beacons flash at roughly 1–4 Hz, so the pipeline must see a vehicle at ≥8 fps
for the modulation to survive sampling. :meth:`SirenDetector.sample_rate`
reports the rate actually achieved, and :attr:`SirenDetector.undersampled`
turns true when it drops too low to trust — which is a real diagnostic, not a
formality, since that is precisely the condition under which a siren would be
silently missed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

#: Vehicle classes big enough that a red beacon means a fire engine rather
#: than an ambulance. Purely a naming refinement — both get priority either way.
LARGE_VEHICLE_LABELS = frozenset({"truck", "bus"})

#: Reported vehicle type when no beacon is flashing.
ORDINARY_TYPE = "ordinary"

#: Failure reasons that mean "ask again later" rather than "no beacon here".
#: A track in one of these states has not been watched long enough for the
#: flash test to have an opinion at all.
_PENDING_REASONS = frozenset({"no-data", "too-few-samples", "window-too-short"})


@dataclass
class SirenConfig:
    """Tunable parameters for beacon (siren-light) detection.

    The defaults are deliberately generous on *colour* and strict on *time*:
    a light only has to be recognisably red or blue, but it must switch on and
    off at least three times within a couple of seconds before anything is
    reported.
    """

    enabled: bool = True

    # ── Where to look ───────────────────────────────────────────────────────
    min_box_px: int = 26           # below this a beacon is a single pixel
    pad_x: float = 0.06            # widen the crop — bloom spills past the box
    pad_top: float = 0.18          # and a roof beacon glows *above* the roof
    pad_bottom: float = 0.03
    # Top share of the *padded* crop searched for red. With ``pad_top`` above,
    # 0.42 covers from a fifth of a vehicle-height above the roof down to about
    # a third of the way into the box — where a light bar sits, and above where
    # brake lights sit. It was 0.55, and that reached far enough down the body
    # to catch the tail lights of a car viewed end-on from a low angle, which
    # was the last remaining false positive on the night footage.
    roof_fraction: float = 0.42
    max_crop_px: int = 192         # downscale bigger crops; keeps cost flat

    # ── What counts as coloured light ───────────────────────────────────────
    # Purity gates in raw BGR. Amber indicators and tow beacons have a strong
    # green component and are rejected by ``red_green_max``; white headlights
    # and bloomed cores fail both gates because no channel dominates.
    #
    # ``min_channel_value`` is the gate that matters most, and it is high on
    # purpose: a beacon is a *light source*, and light sources very nearly
    # saturate the sensor. Measured on the night footage, ordinary cars produce
    # plenty of bluish-lavender bodywork and reflections at B≈130–155 — which a
    # lower floor happily reports as flashing blue once motion blur and
    # compression noise make those weak values wobble. Real beacon pixels in
    # the same clips sit at 200–255. The floor is set between the two.
    min_channel_value: float = 165.0
    #: The brightness floor fades in over this many levels instead of switching
    #: at a hard edge. That matters more than it sounds: a hard threshold turns
    #: a *smooth* signal into a switched one, because everything below the
    #: threshold is clipped to zero. Sunlight sweeping across red bodywork —
    #: through trees, past buildings — then arrives at the flash test looking
    #: exactly like a lamp, and the bimodality test that is supposed to catch it
    #: has had the evidence removed before it ever runs. Fading in preserves the
    #: shape, so a ramp still reads as a ramp.
    brightness_softness: float = 45.0
    red_green_max: float = 0.55    # g ≤ 0.55·r  → rules out amber and white
    red_blue_max: float = 0.72
    blue_red_max: float = 0.70     # r ≤ 0.70·b
    blue_green_max: float = 0.90   # blue LEDs read cyan-ish on most sensors
    ambient_gain: float = 1.15     # raise the floor in daylight
    top_pixel_fraction: float = 0.02   # score = mean of the brightest 2%…
    top_pixel_minimum: int = 6         # …but never fewer than this many pixels

    # ── What counts as flashing ─────────────────────────────────────────────
    # The window has to hold at least three flashes of the *slowest* plausible
    # beacon. At 1.2 Hz — a common rotating-beacon rate — three rising edges
    # take just over three seconds, so a shorter window would silently miss
    # every slow beacon while still admitting a driver tapping the brakes twice.
    window_seconds: float = 3.4
    min_window_seconds: float = 1.1
    min_samples: int = 10
    max_samples: int = 120
    min_peak: float = 0.25         # the light must actually be bright
    min_swing: float = 0.14        # …and the on/off difference real
    min_depth: float = 0.45        # (hi−lo)/hi — the light must nearly vanish
    on_fraction: float = 0.62      # Schmitt trigger thresholds within the swing
    off_fraction: float = 0.34
    min_flashes: int = 3           # rising edges required inside the window
    # Beacons are regulated: the slowest in common use flash around 60 times a
    # minute, and modern LED bars run two to four times that. Nothing below
    # 1 Hz is a beacon, whereas plenty of things below 1 Hz are a red vehicle
    # drifting in and out of street light — which is what 0.7 admitted.
    min_rate_hz: float = 1.0
    max_rate_hz: float = 6.0       # above this it is sensor noise, not a lamp
    min_duty: float = 0.06
    max_duty: float = 0.78
    body_dominance: float = 1.8    # red mostly *below* the roof → tail lights

    # A lamp is a *switch*: it is fully on or fully off, and essentially no
    # sample catches it in between. Bodywork drifting through a pool of street
    # light is a *ramp*, and spends most of its time part-lit. Measured over
    # the night footage this one number separates the two completely — the real
    # beacon scored 0.00 while every false positive scored 0.18 to 0.43 — and
    # it is what rejects a bright red car under changing light, which every
    # brightness- and frequency-based test in this module lets through.
    midband_low: float = 0.25      # bounds of the "neither on nor off" band,
    midband_high: float = 0.75     # as fractions of the observed swing
    max_midband_fraction: float = 0.15

    # A lamp that is on stays on for a perceptible time; single-sample spikes
    # are sensor noise. Only checked when the vehicle is being sampled densely
    # enough for the distinction to exist: at 15 fps a double-flash LED bar
    # genuinely occupies about one sample per burst, so applying this below
    # ~18 Hz would reject the very beacons it is meant to protect.
    dense_sampling_hz: float = 18.0
    min_mean_on_run: float = 1.35  # samples per "on" burst

    # ── Turning the flag on and off ─────────────────────────────────────────
    # Confirmation is required in *seconds* as well as frames, because a frame
    # count means different things on different machines: three consecutive
    # frames is a quarter of a second at 12 fps but under a tenth at 39, and at
    # the fast end that was short enough for a two-frame flicker on ordinary
    # traffic to latch the flag. Requiring the evidence to survive a real
    # fraction of a second costs nothing — a beacon that flashes once is still
    # flashing half a second later — and removes those blips entirely.
    #
    # The evidence counter decays rather than resetting on a single failed
    # frame, so a beacon momentarily hidden behind a lorry does not restart its
    # confirmation clock — which is what makes a long confirmation window safe
    # to ask for. What it excludes is the last false positive left on the night
    # footage: a car's high-mounted centre brake light, which sits in the roof
    # band and does flash when the driver brakes in traffic, but does not keep
    # flashing. A beacon does.
    #
    # What this timer actually demands is that the *verdict persists*: not that
    # three passing frames were collected, but that the flash test kept saying
    # "flashing" for a continuous stretch of real time. That is what separates
    # a beacon from dappled sunlight sweeping across red bodywork, which passes
    # the flash test in bursts as it ramps. Measured over the synthetic cases in
    # ``test_traffic_system.py`` and the footage in ``vedios/``:
    #
    #     real beacon, day or night   holds "flashing" for 4.40 s (unbroken)
    #     dappled sun on red paint    holds it for 0.53 s at most
    #     steady scarlet bodywork     never reaches it at all
    #
    # It was 1.2 s, which is far inside that gap and cost real detections: the
    # timer runs from the first detected flash, and :func:`analyse_flashes`
    # cannot report one until it already holds ``min_window_seconds`` of
    # history, so the two delays stack — 1.1 s of evidence, then 1.2 s of
    # confirmation, and nothing could be flagged in under 2.3 s. Tracks of a
    # vehicle crossing the frame at speed routinely end sooner than that, so the
    # beacon was measured, found to be flashing, and the vehicle left the shot
    # before the timer expired. At 0.8 s the false cases are still rejected with
    # half a second of margin and the total latency falls to about 1.9 s.
    min_confirmations: int = 3     # net passing frames to latch ON
    min_active_seconds: float = 0.8  # …and the verdict must hold this long
    hold_seconds: float = 3.0      # stay ON this long after the last flash
    track_ttl_seconds: float = 5.0

    # ── Colour → vehicle type (differs by country) ──────────────────────────
    blue_type: str = "police"
    red_type: str = "ambulance"
    red_large_type: str = "fire truck"
    both_type: str = "police"

    # ── Health ──────────────────────────────────────────────────────────────
    min_sample_rate_hz: float = 8.0    # below this, flashes alias away


@dataclass
class SirenVerdict:
    """The system's current belief about one vehicle's beacon."""

    active: bool = False
    vehicle_type: str = ORDINARY_TYPE  # police | ambulance | fire truck
    confidence: float = 0.0
    colour: str = "none"               # blue | red | red+blue | none
    rate_hz: float = 0.0
    flashes: int = 0
    samples: int = 0

    @property
    def evidence(self) -> str:
        """Short evidence tag for logs and the on-screen label."""
        return f"siren-{self.colour}" if self.active else "none"


@dataclass
class FlashStats:
    """Result of the temporal test on one colour channel."""

    flashing: bool = False
    peak: float = 0.0
    depth: float = 0.0
    swing: float = 0.0
    flashes: int = 0
    rate_hz: float = 0.0
    duty: float = 0.0
    mean_on_run: float = 0.0
    midband: float = 0.0           # share of samples caught mid-transition
    samples: int = 0
    span: float = 0.0
    score: float = 0.0             # 0–1 strength, set when ``flashing``
    reason: str = "no-data"        # why it failed, for diagnostics


@dataclass
class _TrackSignal:
    """Rolling photometry for one tracked vehicle."""

    times: deque = field(default_factory=deque)
    red_roof: deque = field(default_factory=deque)
    red_body: deque = field(default_factory=deque)
    blue: deque = field(default_factory=deque)

    confirmations: int = 0
    first_flash: float = 0.0
    active: bool = False
    last_flash: float = 0.0
    last_seen: float = 0.0
    #: When this vehicle was first sampled. Distinct from ``times[0]``, which
    #: only reaches back one analysis window because older samples are trimmed
    #: away — this is how long the vehicle has actually been watched.
    first_seen: float = 0.0
    verdict: SirenVerdict = field(default_factory=SirenVerdict)
    last_stats: tuple[FlashStats, FlashStats] = field(
        default_factory=lambda: (FlashStats(), FlashStats()))
    #: True while the track has not yet been watched long enough for the flash
    #: test to return any verdict at all. Distinct from "no beacon": one is an
    #: answer, the other is the absence of one.
    pending: bool = True

    def trim(self, now: float, window: float, cap: int) -> None:
        """Drop samples older than the analysis window."""
        while self.times and (now - self.times[0] > window or len(self.times) > cap):
            self.times.popleft()
            self.red_roof.popleft()
            self.red_body.popleft()
            self.blue.popleft()


# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL TEST
# ═══════════════════════════════════════════════════════════════════════════


def analyse_flashes(times: Sequence[float], values: Sequence[float],
                    config: SirenConfig) -> FlashStats:
    """Decide whether a light-intensity series is a flashing beacon.

    The test is a Schmitt trigger over the observed swing rather than a fixed
    threshold, because a beacon 80 m away and one 8 m away differ in brightness
    by an order of magnitude while flashing identically. What is required is
    that the light nearly *vanishes* between flashes (``min_depth``) and does so
    repeatedly (``min_flashes``) at a plausible rate.

    Returns a :class:`FlashStats` whose ``reason`` names the first test failed,
    which is what makes a missed siren debuggable instead of mysterious.
    """
    count = len(values)
    if count < config.min_samples:
        return FlashStats(samples=count, reason="too-few-samples")

    span = float(times[-1] - times[0])
    if span < config.min_window_seconds:
        return FlashStats(samples=count, span=span, reason="window-too-short")

    series = np.asarray(values, dtype=np.float32)
    peak = float(series.max())
    low = float(np.percentile(series, 15))
    high = float(np.percentile(series, 90))
    swing = high - low
    depth = swing / high if high > 1e-6 else 0.0

    stats = FlashStats(peak=peak, depth=round(depth, 3), swing=round(swing, 3),
                       samples=count, span=round(span, 2))

    if peak < config.min_peak:
        stats.reason = "too-dim"
        return stats
    if swing < config.min_swing:
        stats.reason = "steady-light"
        return stats
    if depth < config.min_depth:
        stats.reason = "shallow-modulation"
        return stats

    # Bimodality: a switched lamp leaves the middle of its range empty.
    stats.midband = round(float(np.count_nonzero(
        (series > low + config.midband_low * swing)
        & (series < low + config.midband_high * swing)) / count), 3)
    if stats.midband > config.max_midband_fraction:
        stats.reason = "gradual-not-switched"
        return stats

    # Schmitt trigger: a sample only counts as "lit" once the light has climbed
    # past ``on_level``, and only stops counting once it has fallen back below
    # ``off_level``. The dead band means noise hovering around one threshold
    # cannot be mistaken for a burst of flashes.
    #
    # Implemented by forward-filling the last decisive sample rather than
    # looping in Python: with a couple of dozen vehicles re-analysed on two
    # colour channels every frame, the loop was the detector's whole cost.
    on_level = low + config.on_fraction * swing
    off_level = low + config.off_fraction * swing
    decision = np.zeros(count, dtype=np.int8)
    decision[series >= on_level] = 1
    decision[series < off_level] = -1
    carried = np.where(decision != 0, np.arange(count), 0)
    np.maximum.accumulate(carried, out=carried)
    lit = decision[carried] > 0

    rising = np.flatnonzero(lit[1:] & ~lit[:-1])
    stats.flashes = int(rising.size)
    stats.rate_hz = round(stats.flashes / span, 2)
    stats.duty = round(float(lit.mean()), 2)

    # Length of each contiguous "on" burst, in samples.
    edges = np.diff(np.concatenate(([0], lit.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    stats.mean_on_run = round(float(runs.mean()) if runs.size else 0.0, 2)

    if stats.flashes < config.min_flashes:
        stats.reason = "not-repeating"
        return stats
    if not (config.min_rate_hz <= stats.rate_hz <= config.max_rate_hz):
        stats.reason = "implausible-rate"
        return stats
    if not (config.min_duty <= stats.duty <= config.max_duty):
        stats.reason = "implausible-duty"
        return stats
    if (count / span >= config.dense_sampling_hz
            and stats.mean_on_run < config.min_mean_on_run):
        stats.reason = "single-sample-spikes"
        return stats

    stats.flashing = True
    stats.reason = "flashing"
    # Confidence rises with how completely the light vanishes between flashes,
    # how bright it is when lit, and how many flashes have been seen.
    depth_term = min(max((depth - config.min_depth) / (1.0 - config.min_depth), 0.0), 1.0)
    peak_term = min(max((peak - config.min_peak) / 0.35, 0.0), 1.0)
    count_term = min(max(stats.flashes / 6.0, 0.4), 1.0)
    rate_term = 1.0 if 1.0 <= stats.rate_hz <= 5.0 else 0.75
    stats.score = round(min(1.0, rate_term * (0.45 * depth_term
                                              + 0.35 * peak_term
                                              + 0.20 * count_term)), 3)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR
# ═══════════════════════════════════════════════════════════════════════════


class SirenDetector:
    """Flags vehicles whose emergency beacon is actively flashing."""

    def __init__(self, config: SirenConfig | None = None) -> None:
        """Build a detector; state is per-track and grows with the scene."""
        self.config = config or SirenConfig()
        self._signals: dict[int, _TrackSignal] = {}
        self._frame_times: deque = deque(maxlen=45)
        self._ambient = 40.0
        #: Cumulative time spent in :meth:`update` across the session, so the
        #: detector's share of the frame budget can be measured. Named for what
        #: it is: dividing by the frame count gives the per-frame cost.
        self.total_ms = 0.0

    # -- per-frame -----------------------------------------------------------

    def begin_frame(self, frame: np.ndarray, now: float | None = None) -> None:
        """Note the frame's ambient light level and the sampling rate.

        The ambient level lifts the "this is a light source" floor in daylight,
        where a scarlet car body is as bright as a beacon is at night.
        """
        self._frame_times.append(now if now is not None else time.time())
        if frame is not None and frame.size:
            self._ambient = float(frame[::8, ::8].mean())

    def sample_rate(self) -> float:
        """Frames per second at which vehicles are currently being sampled."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    @property
    def undersampled(self) -> bool:
        """True when the frame rate is too low to see a beacon flash."""
        rate = self.sample_rate()
        return 0.0 < rate < self.config.min_sample_rate_hz

    @property
    def tracked(self) -> int:
        """Vehicles the detector currently holds evidence for."""
        return len(self._signals)

    @property
    def pending_tracks(self) -> int:
        """Vehicles still being watched, with no beacon verdict yet either way.

        A vehicle that crosses the frame in less than ``window_seconds`` is
        never conclusively judged. Counting those separately is what stops "no
        emergency vehicles" being reported about footage nobody could have
        judged — and when this equals :attr:`tracked`, nothing in shot has been
        watched long enough to have been cleared of anything.
        """
        return sum(1 for signal in self._signals.values()
                   if signal.pending and not signal.active)

    # -- photometry ----------------------------------------------------------

    def _crop(self, frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray | None:
        """Cut the beacon search region: the box, widened and raised."""
        cfg = self.config
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in bbox)
        box_w, box_h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - box_w * cfg.pad_x))
        x2 = min(width, int(x2 + box_w * cfg.pad_x))
        y1 = max(0, int(y1 - box_h * cfg.pad_top))
        y2 = min(height, int(y2 + box_h * cfg.pad_bottom))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        longest = max(crop.shape[:2])
        if longest > cfg.max_crop_px:
            scale = cfg.max_crop_px / longest
            crop = cv2.resize(crop, (max(8, int(crop.shape[1] * scale)),
                                     max(8, int(crop.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA)
        return crop

    def _colour_maps(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return per-pixel red-light and blue-light strength, 0–1.

        Pure channel *dominance* is used rather than HSV hue because a bloomed
        beacon has a white core and a coloured halo, and hue is meaningless
        where the sensor has clipped. Dominance degrades gracefully: the halo
        still scores, the core simply scores zero.
        """
        cfg = self.config
        pixels = crop.astype(np.float32)
        blue_c, green_c, red_c = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        floor = max(cfg.min_channel_value, self._ambient * cfg.ambient_gain)

        # Purity stays a hard gate — a light is either red or it is amber, and
        # there is nothing in between worth keeping. Brightness is a magnitude,
        # so it fades in: see ``brightness_softness``.
        soft = max(cfg.brightness_softness, 1.0)
        red_ok = ((green_c <= cfg.red_green_max * red_c)
                  & (blue_c <= cfg.red_blue_max * red_c))
        blue_ok = ((red_c <= cfg.blue_red_max * blue_c)
                   & (green_c <= cfg.blue_green_max * blue_c))
        red_weight = np.clip((red_c - floor + soft) / soft, 0.0, 1.0)
        blue_weight = np.clip((blue_c - floor + soft) / soft, 0.0, 1.0)

        red = np.where(red_ok, (red_c - blue_c) * red_weight, 0.0) / 255.0
        blue = np.where(blue_ok, (blue_c - red_c) * blue_weight, 0.0) / 255.0
        return red, blue

    def _brightest(self, values: np.ndarray) -> float:
        """Mean of the brightest few pixels — scale-invariant beacon strength.

        A plain mean would be diluted to nothing by a large vehicle, and a plain
        maximum would latch onto a single hot pixel. The top few per cent is
        stable across a beacon that is 400 pixels on a passing fire engine and
        9 pixels on a police car at the far end of the street.
        """
        flat = values.reshape(-1)
        if flat.size == 0:
            return 0.0
        cfg = self.config
        keep = max(cfg.top_pixel_minimum, int(cfg.top_pixel_fraction * flat.size))
        if keep >= flat.size:
            return float(flat.mean())
        return float(np.partition(flat, flat.size - keep)[flat.size - keep:].mean())

    # -- public API ----------------------------------------------------------

    def update(self, frame: np.ndarray, bbox: Sequence[float], track_id: int,
               now: float, label: str = "car") -> SirenVerdict:
        """Fold this frame into the track's evidence and return its verdict."""
        cfg = self.config
        signal = self._signals.get(track_id)
        if signal is None:
            signal = _TrackSignal(first_seen=now)
            self._signals[track_id] = signal
        signal.last_seen = now

        if not cfg.enabled:
            return signal.verdict

        started = time.perf_counter()
        box_w = float(bbox[2]) - float(bbox[0])
        box_h = float(bbox[3]) - float(bbox[1])
        crop = (self._crop(frame, bbox)
                if min(box_w, box_h) >= cfg.min_box_px else None)

        if crop is not None:
            red, blue = self._colour_maps(crop)
            split = max(1, int(crop.shape[0] * cfg.roof_fraction))
            signal.times.append(now)
            signal.red_roof.append(self._brightest(red[:split]))
            signal.red_body.append(self._brightest(red[split:])
                                   if split < crop.shape[0] else 0.0)
            signal.blue.append(self._brightest(blue))
            signal.trim(now, cfg.window_seconds, cfg.max_samples)

        verdict = self._decide(signal, now, label)
        signal.verdict = verdict
        self.total_ms += (time.perf_counter() - started) * 1000.0
        return verdict

    def _decide(self, signal: _TrackSignal, now: float, label: str) -> SirenVerdict:
        """Run the temporal test and apply latching hysteresis."""
        cfg = self.config
        red_stats = analyse_flashes(signal.times, signal.red_roof, cfg)
        blue_stats = analyse_flashes(signal.times, signal.blue, cfg)
        signal.last_stats = (red_stats, blue_stats)

        # "Not watched long enough yet" is not the same answer as "no beacon",
        # and reporting the second when the first is true is how a clip too
        # short to judge comes back looking like a clear road.
        #
        # A negative verdict only becomes final once a full ``window_seconds``
        # of history exists, because that window is sized to hold three flashes
        # of the slowest beacon in use. Judged over less than that, "not
        # repeating" cannot distinguish a brake light that flashed twice from a
        # beacon that has so far been seen twice — and the vehicle in the
        # screenshot that prompted all this was in shot for 1.4 s.
        observed = now - signal.first_seen
        signal.pending = (observed < cfg.window_seconds
                          or bool(signal.confirmations))


        # Red that lives mostly on the body rather than the roof is brake or
        # tail light spill, however convincingly it blinks.
        red_flashing = red_stats.flashing
        if red_flashing and signal.red_body:
            roof_level = float(np.percentile(signal.red_roof, 90))
            body_level = float(np.percentile(signal.red_body, 90))
            if body_level > roof_level * cfg.body_dominance:
                red_flashing = False
                red_stats.reason = "red-below-roofline"

        blue_flashing = blue_stats.flashing
        flashing_now = red_flashing or blue_flashing

        if flashing_now:
            if not signal.confirmations:
                signal.first_flash = now
            signal.confirmations += 1
            signal.last_flash = now
        else:
            signal.confirmations = max(0, signal.confirmations - 1)
            if not signal.confirmations:
                signal.first_flash = 0.0

        if not signal.active:
            # Latching needs evidence *now*, not merely a counter left over
            # from earlier: the counter decays rather than resetting, so
            # without this a flag released by the hold timer would re-arm on
            # the very next frame from its own stale history.
            #
            if (flashing_now
                    and signal.confirmations >= cfg.min_confirmations
                    and now - signal.first_flash >= cfg.min_active_seconds):
                signal.active = True
        elif now - signal.last_flash > cfg.hold_seconds:
            signal.active = False
            signal.confirmations = 0
            signal.first_flash = 0.0

        if not signal.active:
            return SirenVerdict(samples=len(signal.times))

        # Colour → type. Blue is the decisive one: no civilian vehicle in any
        # country carries a flashing blue lamp, so when both fire it is still
        # read as police unless the deployment says otherwise.
        if red_flashing and blue_flashing:
            colour, vehicle_type = "red+blue", cfg.both_type
        elif blue_flashing:
            colour, vehicle_type = "blue", cfg.blue_type
        elif red_flashing:
            colour = "red"
            vehicle_type = (cfg.red_large_type if label in LARGE_VEHICLE_LABELS
                            else cfg.red_type)
        else:
            # Held on by hysteresis between flashes — keep the last reading,
            # and say "unknown" rather than inventing a colour if there is
            # somehow nothing to keep. A guessed vehicle type would be printed
            # on the LCD and written to the log as though it had been observed.
            previous = signal.verdict
            colour = previous.colour if previous.colour != "none" else "unknown"
            vehicle_type = (previous.vehicle_type
                            if previous.vehicle_type != ORDINARY_TYPE else "unknown")

        # Report the numbers of the channels that actually fired, not the
        # loudest of the two — a rate quoted from a channel that was rejected
        # would make the logs describe a flash pattern nobody detected.
        firing = ([red_stats] if red_flashing else []) + \
                 ([blue_stats] if blue_flashing else [])
        best = max((s.score for s in firing), default=0.0)
        if red_flashing and blue_flashing:
            best = min(1.0, best + 0.10)     # both colours is near-conclusive
        confidence = best or signal.verdict.confidence or 0.5

        return SirenVerdict(
            active=True,
            vehicle_type=vehicle_type,
            confidence=round(confidence, 2),
            colour=colour,
            rate_hz=max((s.rate_hz for s in firing),
                        default=signal.verdict.rate_hz),
            flashes=max((s.flashes for s in firing),
                        default=signal.verdict.flashes),
            samples=len(signal.times),
        )

    # -- housekeeping --------------------------------------------------------

    def verdict_for(self, track_id: int) -> SirenVerdict:
        """Current verdict for a track without advancing its evidence."""
        signal = self._signals.get(track_id)
        return signal.verdict if signal else SirenVerdict()

    def diagnostics(self, track_id: int) -> dict[str, object]:
        """Per-track numbers behind the verdict, for tuning and debugging."""
        signal = self._signals.get(track_id)
        if signal is None:
            return {}
        red, blue = signal.last_stats
        return {
            "samples": len(signal.times),
            "red": red.__dict__,
            "blue": blue.__dict__,
            "confirmations": signal.confirmations,
            "active": signal.active,
        }

    def prune(self, now: float) -> None:
        """Forget tracks that have not been seen for a while."""
        ttl = self.config.track_ttl_seconds
        for track_id in [t for t, s in self._signals.items()
                         if now - s.last_seen > ttl]:
            del self._signals[track_id]

    def reset(self) -> None:
        """Discard all evidence, for when the picture jumps to a new source."""
        self._signals.clear()
        self._frame_times.clear()
        self.total_ms = 0.0

    @property
    def status(self) -> str:
        """One-line human description of the detector's state."""
        if not self.config.enabled:
            return "siren detection off"
        cfg = self.config
        return (f"siren-light detection on "
                f"(blue→{cfg.blue_type}, red→{cfg.red_type}, "
                f"≥{cfg.min_flashes} flashes @ {cfg.min_rate_hz}–{cfg.max_rate_hz} Hz)")
