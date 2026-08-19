"""
test_traffic_system.py — verification suite for the traffic vision pipeline
===========================================================================

Runs standalone (no pytest required)::

    python test_traffic_system.py            # fast tests, no GPU needed
    python test_traffic_system.py --yolo     # also load yolo12m and run it for real

The fast tests use a scripted fake detector, so the exact ground-truth speed of
every synthetic vehicle is known and can be compared against what the pipeline
reports. That is the only way to prove the km/h conversion is right — with a
real camera there is nothing to check the answer against.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import traffic_vision as tv

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record the outcome of one assertion."""
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append((name, detail))
        print(f"  [FAIL] {name}  {detail}")


def close_to(actual: float, expected: float, tolerance: float) -> bool:
    """Return whether two numbers agree within an absolute tolerance."""
    return abs(actual - expected) <= tolerance


# ═══════════════════════════════════════════════════════════════════════════
# FAKE DETECTOR — lets us script exact vehicle motion
# ═══════════════════════════════════════════════════════════════════════════


class FakeDetector:
    """Stands in for :class:`traffic_vision.VehicleDetector` in tests."""

    def __init__(self, script: list[list[tuple]]) -> None:
        """Take a per-frame list of ``(bbox, track_id, label, confidence)``."""
        self.script = script
        self.frame = 0
        self.names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def track(self, frame: np.ndarray) -> tuple[list[tuple], float]:
        """Return the scripted detections for the current frame."""
        detections = self.script[self.frame] if self.frame < len(self.script) else []
        self.frame += 1
        return detections, 5.0


HOMOGRAPHY_CALIB = {
    "mode": "homography",
    "image_points": [[100, 700], [1180, 700], [900, 300], [380, 300]],
    "world_points": [[0, 0], [10, 0], [10, 20], [0, 20]],
    "frame_size": [1280, 720],
}


# ═══════════════════════════════════════════════════════════════════════════
# CALIBRATION & SPEED
# ═══════════════════════════════════════════════════════════════════════════


def test_homography_speed() -> None:
    """A vehicle crossing a known 20 m patch must read its true km/h."""
    print("\n[1] Homography speed accuracy")
    ground = tv.GroundPlane(HOMOGRAPHY_CALIB)
    check("homography mode detected", ground.mode == "homography")
    check("calibration reports as calibrated", ground.is_calibrated)

    corner = ground.image_to_world((1180, 700), (720, 1280))
    check("corner maps to its world point", close_to(corner[0], 10.0, 0.001)
          and close_to(corner[1], 0.0, 0.001), f"got {corner}")

    matrix = cv2.getPerspectiveTransform(
        np.array(HOMOGRAPHY_CALIB["image_points"], np.float32),
        np.array(HOMOGRAPHY_CALIB["world_points"], np.float32))
    inverse = np.linalg.inv(matrix)

    for true_kmh in (18.0, 36.0, 72.0):
        estimator = tv.SpeedEstimator(ground, tv.VisionConfig())
        metres_per_second = true_kmh / 3.6
        duration = 20.0 / metres_per_second
        fps, speed = 30, None
        for index in range(int(duration * fps) + 1):
            moment = index / fps
            world_y = min(20.0, metres_per_second * moment)
            point = cv2.perspectiveTransform(
                np.array([[[5.0, world_y]]], np.float32), inverse)[0][0]
            cx, by = float(point[0]), float(point[1])
            speed = estimator.update(1, (cx - 40, by - 60, cx + 40, by), "car",
                                     moment, (720, 1280))
        check(f"{true_kmh:.0f} km/h measured correctly",
              speed is not None and close_to(speed, true_kmh, 0.5),
              f"got {speed:.2f} km/h" if speed else "no reading")


def test_auto_scale_speed() -> None:
    """Uncalibrated mode must still produce real km/h from vehicle size."""
    print("\n[2] Auto (size-based) speed calibration")
    ground = tv.GroundPlane(None)
    check("falls back to auto mode", ground.mode == "auto")
    check("auto mode is flagged as uncalibrated", not ground.is_calibrated)

    # A car 60 px tall is 1.50 m tall → 0.025 m/px.
    scale = ground.local_scale((0, 0, 80, 60), "car")
    check("scale from car height", close_to(scale, 0.025, 1e-6), f"{scale:.5f} m/px")
    check("bus uses its own real height",
          close_to(ground.local_scale((0, 0, 200, 100), "bus"), 0.032, 1e-6))
    check("tiny boxes are rejected", ground.local_scale((0, 0, 6, 5), "car") is None)

    estimator = tv.SpeedEstimator(ground, tv.VisionConfig())
    # 400 px in 2 s at 0.025 m/px = 10 m in 2 s = 5 m/s = 18 km/h.
    speed = None
    for index in range(61):
        moment = index / 30
        cx = 100 + 400 * (moment / 2.0)
        speed = estimator.update(7, (cx - 40, 440, cx + 40, 500), "car", moment, (720, 1280))
    check("auto mode measures 18 km/h", speed is not None and close_to(speed, 18.0, 0.5),
          f"got {speed:.2f} km/h" if speed else "no reading")

    summary = estimator.track_summary(7)
    check("distance travelled recorded", close_to(summary["distance_m"], 10.0, 0.2),
          f"{summary['distance_m']} m")


def test_fixed_scale_mode() -> None:
    """Two-point 'scale' calibration must convert pixels to metres."""
    print("\n[3] Fixed-scale calibration mode")
    ground = tv.GroundPlane({"mode": "scale", "meters_per_pixel": 0.05,
                             "frame_size": [1280, 720]})
    check("scale mode detected", ground.mode == "scale")
    world = ground.image_to_world((200, 100), (720, 1280))
    check("pixels convert to metres", close_to(world[0], 10.0, 1e-6), f"{world}")

    estimator = tv.SpeedEstimator(ground, tv.VisionConfig())
    speed = None
    for index in range(61):  # 200 px in 2 s = 10 m in 2 s = 18 km/h
        moment = index / 30
        cx = 100 + 200 * (moment / 2.0)
        speed = estimator.update(3, (cx - 30, 440, cx + 30, 500), "car", moment, (720, 1280))
    check("fixed scale measures 18 km/h", close_to(speed, 18.0, 0.5), f"got {speed:.2f}")


def test_frame_size_independence() -> None:
    """Calibration made at 1280x720 must still work on a 640x360 frame."""
    print("\n[4] Resolution independence")
    ground = tv.GroundPlane(HOMOGRAPHY_CALIB)
    full = ground.image_to_world((640, 500), (720, 1280))
    half = ground.image_to_world((320, 250), (360, 640))
    check("half-resolution frame maps identically",
          close_to(full[0], half[0], 0.01) and close_to(full[1], half[1], 0.01),
          f"{full} vs {half}")


def test_speed_robustness() -> None:
    """Jitter, stops and ID switches must not produce nonsense speeds."""
    print("\n[5] Speed robustness")
    config = tv.VisionConfig()
    ground = tv.GroundPlane({"mode": "scale", "meters_per_pixel": 0.05})
    estimator = tv.SpeedEstimator(ground, config)

    # A parked car with 1 px of box jitter must read near zero.
    speed = None
    rng = np.random.default_rng(0)
    for index in range(60):
        moment = index / 30
        jitter = rng.uniform(-1, 1)
        speed = estimator.update(1, (500 + jitter, 400, 560 + jitter, 460),
                                 "car", moment, (720, 1280))
    check("parked car reads < 2 km/h", speed is not None and speed < 2.0,
          f"got {speed:.2f} km/h")

    # A teleport (classic ID switch) must be rejected, not reported as 900 km/h.
    estimator2 = tv.SpeedEstimator(ground, config)
    for index in range(10):
        estimator2.update(2, (100, 400, 160, 460), "car", index / 30, (720, 1280))
    after_jump = estimator2.update(2, (5000, 400, 5060, 460), "car", 10 / 30, (720, 1280))
    check("impossible jump rejected",
          after_jump is None or after_jump < config.speed_max_plausible_kmh,
          f"got {after_jump}")

    # Too few samples must yield no reading rather than a wild guess.
    estimator3 = tv.SpeedEstimator(ground, config)
    first = estimator3.update(3, (100, 400, 160, 460), "car", 0.0, (720, 1280))
    check("no reading from a single frame", first is None)


def test_memory_pruning() -> None:
    """Track bookkeeping must not grow without bound over a long session."""
    print("\n[6] Memory pruning")
    config = tv.VisionConfig(track_ttl_seconds=1.0)
    estimator = tv.SpeedEstimator(tv.GroundPlane(None), config)
    for track_id in range(500):
        estimator.update(track_id, (0, 0, 60, 60), "car", 0.0, (720, 1280))
    check("500 tracks registered", len(estimator._tracks) == 500)
    retired = estimator.prune(5.0)
    check("all stale tracks pruned", len(estimator._tracks) == 0, f"{len(retired)} retired")
    check("retirement returns summaries", len(retired) == 500 and isinstance(retired[0][1], dict))


# ═══════════════════════════════════════════════════════════════════════════
# CONGESTION
# ═══════════════════════════════════════════════════════════════════════════


def test_congestion() -> None:
    """Congestion index and level must behave sensibly and not flap."""
    print("\n[7] Congestion engine")
    config = tv.VisionConfig(free_flow_speed_kmh=50.0, max_capacity_pce=20.0)
    engine = tv.CongestionEngine(config)

    check("empty road is never congested",
          engine.compute_index(avg_speed_kmh=0.0, weighted_density=0.0, vehicle_count=0) == 0.0)

    free = engine.compute_index(50.0, 2.0, 2)
    check("fast + sparse is free flowing", free < config.threshold_low, f"CI={free}")

    jam = engine.compute_index(3.0, 20.0, 20)
    check("slow + dense is heavy", jam > config.threshold_medium, f"CI={jam}")

    check("index stays within 0..1", 0.0 <= jam <= 1.0 and 0.0 <= free <= 1.0)

    # One crawling vehicle on an empty road is a slow driver, not a traffic jam.
    lone = engine.compute_index(2.5, 3.0, 1)
    check("a single slow vehicle is not congestion", lone < config.threshold_low,
          f"CI={lone}")
    crowd = engine.compute_index(2.5, 9.0, 6)
    check("six crawling vehicles are congestion", crowd > config.threshold_medium,
          f"CI={crowd}")

    # Hysteresis: an index hovering on the FREE/MODERATE boundary must not flip
    # the level every frame, which would make the LEDs and servo chatter.
    engine2 = tv.CongestionEngine(config)
    levels = set()
    for index in range(40):
        value = config.threshold_low + (0.005 if index % 2 else -0.005)
        levels.add(engine2.level_for(value, now=time.time() + index * 0.05))
    check("level does not flap on the threshold", len(levels) == 1, f"levels seen: {levels}")

    # A sustained genuine change must still be followed.
    engine3 = tv.CongestionEngine(config)
    final = "FREE"
    for index in range(60):
        final = engine3.level_for(0.85, now=time.time() + index * 0.2)
    check("sustained jam reaches HEAVY", final == "HEAVY", f"got {final}")


# ═══════════════════════════════════════════════════════════════════════════
# EMERGENCY DETECTION
# ═══════════════════════════════════════════════════════════════════════════


def _frame_with_roof(colour: tuple[int, int, int] | None,
                     second: tuple[int, int, int] | None = None) -> np.ndarray:
    """Build a 720p frame with a vehicle box whose roof has the given lamp(s).

    Lamp colours are near-saturated on purpose: the detector requires a beacon
    to look like a light *source* rather than paintwork, so a dull red
    rectangle is supposed to be ignored however brightly it is described.
    """
    frame = np.full((720, 1280, 3), 60, np.uint8)
    cv2.rectangle(frame, (500, 300), (700, 500), (90, 90, 90), -1)   # vehicle body
    if colour is not None:
        cv2.rectangle(frame, (520, 305), (600, 340), colour, -1)     # roof lamp
    if second is not None:
        cv2.rectangle(frame, (610, 305), (680, 340), second, -1)
    return frame


LAMP_RED = (25, 20, 255)      # BGR: a red beacon as a camera records it
LAMP_BLUE = (255, 150, 40)    # BGR: a blue beacon, slightly cyan
LAMP_AMBER = (30, 190, 255)   # BGR: an indicator or tow-truck beacon


def _run_siren(detector, frames, box=(500, 300, 700, 500), track=1,
               fps=15.0, start=0.0, label="car"):
    """Feed a sequence of frames to the detector and return the last verdict."""
    verdict = None
    for index, frame in enumerate(frames):
        now = start + index / fps
        detector.begin_frame(frame, now)
        verdict = detector.update(frame, box, track, now, label)
    return verdict


def test_siren_detection() -> None:
    """A beacon must be recognised by *switching*; paint must not be."""
    print("\n[8] Siren beacon detection")
    from siren_vision import SirenConfig, SirenDetector

    seconds, fps = 4.0, 15.0
    count = int(seconds * fps)

    def flashing(colour, rate=2.0, duty=0.35):
        return [_frame_with_roof(colour if ((i / fps) * rate) % 1.0 < duty else None)
                for i in range(count)]

    # The two things the old colour-ratio heuristic could not tell from a
    # beacon: a lamp that is simply on, and bright red paint.
    steady = _run_siren(SirenDetector(SirenConfig()),
                        [_frame_with_roof(LAMP_RED)] * count)
    check("a steadily lit red lamp does not trigger", not steady.active)

    dull = _run_siren(SirenDetector(SirenConfig()),
                      [_frame_with_roof((40, 35, 150))] * count)
    check("red paintwork does not trigger", not dull.active)

    # Two brake presses in four seconds must not read as a flashing beacon.
    brake = [_frame_with_roof(LAMP_RED if (8 <= i < 16 or 30 <= i < 40) else None)
             for i in range(count)]
    check("a driver tapping the brakes twice does not trigger",
          not _run_siren(SirenDetector(SirenConfig()), brake).active)

    blue = _run_siren(SirenDetector(SirenConfig()), flashing(LAMP_BLUE))
    check("a flashing blue beacon triggers", blue.active, blue.evidence)
    check("blue is read as police", blue.vehicle_type == "police", blue.vehicle_type)

    red = _run_siren(SirenDetector(SirenConfig()), flashing(LAMP_RED))
    check("a flashing red beacon triggers", red.active, red.evidence)
    check("red is read as ambulance", red.vehicle_type == "ambulance", red.vehicle_type)

    # …unless the vehicle is large, where a red beacon means a fire engine.
    lorry = _run_siren(SirenDetector(SirenConfig()), flashing(LAMP_RED), label="truck")
    check("red on a truck is read as a fire engine",
          lorry.vehicle_type == "fire truck", lorry.vehicle_type)

    # Colour conventions differ by country, so the mapping must be settable.
    swapped = _run_siren(
        SirenDetector(SirenConfig(blue_type="ambulance")), flashing(LAMP_BLUE))
    check("colour-to-type mapping is configurable",
          swapped.vehicle_type == "ambulance", swapped.vehicle_type)

    # Alternating red and blue: a police light bar.
    alternating = [_frame_with_roof(LAMP_RED if (i // 4) % 2 else None,
                                    LAMP_BLUE if (i // 4) % 2 == 0 else None)
                   for i in range(count)]
    both = _run_siren(SirenDetector(SirenConfig()), alternating)
    check("an alternating red/blue light bar triggers", both.active)
    check("both colours are reported", both.colour == "red+blue", both.colour)

    # Hysteresis: a beacon briefly hidden behind a lorry must not release the
    # lane the instant it disappears.
    held = SirenDetector(SirenConfig())
    _run_siren(held, flashing(LAMP_BLUE))
    still = _run_siren(held, [_frame_with_roof(None)] * 20, start=seconds, fps=fps)
    check("the flag holds briefly after the beacon stops", still.active)
    gone = _run_siren(held, [_frame_with_roof(None)] * 20,
                      start=seconds + 4.0, fps=fps)
    check("the flag clears once the beacon is gone", not gone.active)

    # Amber — indicators, hazards, tow trucks — must be ignored however
    # convincingly it flashes.
    amber = _run_siren(SirenDetector(SirenConfig()), flashing(LAMP_AMBER))
    check("an amber beacon (tow truck, indicator) does not trigger", not amber.active)

    tiny = _run_siren(SirenDetector(SirenConfig()), flashing(LAMP_BLUE),
                      box=(500, 300, 518, 318), track=9)
    check("boxes too small to hold a beacon are ignored", not tiny.active)

    pruned = SirenDetector(SirenConfig())
    _run_siren(pruned, flashing(LAMP_BLUE))
    pruned.prune(10_000.0)
    check("siren state is pruned", len(pruned._signals) == 0)

    slow = SirenDetector(SirenConfig())
    _run_siren(slow, flashing(LAMP_BLUE), fps=4.0)
    check("a frame rate too low for beacons is reported", slow.undersampled,
          f"{slow.sample_rate():.1f} Hz")


def test_siren_in_daylight() -> None:
    """The same beacon logic, on a bright scene rather than a dark one.

    Everything about this detector was tuned against night footage, so the
    daylight case deserves its own check: the scene is bright, the vehicle is
    bright, and a scarlet car in direct sun is as saturated as a beacon is at
    midnight. What must still separate them is that one switches and the other
    does not.
    """
    print("\n[8e] Siren detection in daylight")
    from siren_vision import SirenConfig, SirenDetector

    fps, count = 15.0, 60

    def daylight(colour=None, body=(185, 185, 185)):
        """A sunlit road, a bright vehicle, and optionally a lit lamp."""
        frame = np.full((720, 1280, 3), 165, np.uint8)          # bright tarmac
        cv2.rectangle(frame, (500, 300), (700, 500), body, -1)  # sunlit body
        if colour is not None:
            cv2.rectangle(frame, (520, 305), (600, 340), colour, -1)
        return frame

    beacon = [daylight(LAMP_BLUE if ((i / fps) * 2.0) % 1.0 < 0.35 else None)
              for i in range(count)]
    lit = _run_siren(SirenDetector(SirenConfig()), beacon)
    check("a beacon still triggers against a bright sky", lit.active, lit.evidence)
    check("and is still read as police", lit.vehicle_type == "police")

    # A scarlet car in direct sun: as saturated and as bright as a beacon, and
    # completely steady. This is the daylight equivalent of the red-paint case.
    scarlet = _run_siren(SirenDetector(SirenConfig()),
                         [daylight(body=(30, 25, 225))] * count)
    check("a scarlet car in sunlight does not trigger", not scarlet.active,
          scarlet.evidence)

    # Sunlight moving across bodywork — through trees, past buildings — ramps
    # rather than switches, and must be rejected on that basis alone.
    def sunlit_ramp(index):
        phase = abs(((index / fps * 1.5) % 1.0) - 0.5) * 2.0
        level = int(70 + 185 * phase)
        return daylight(body=(int(level * 0.16), int(level * 0.14), level))

    dappled = _run_siren(SirenDetector(SirenConfig()),
                         [sunlit_ramp(i) for i in range(count)])
    check("sunlight sweeping across red bodywork does not trigger",
          not dappled.active, dappled.evidence)

    # Latency, which is what decides whether a vehicle crossing the frame is
    # caught at all. Two costs stack: the flash test cannot speak until it has
    # seen ``min_flashes`` rising edges, which at a beacon's own rate takes
    # 3 / rate_hz seconds — 1.5 s for the 2 Hz lamp below — and then the verdict
    # must hold for ``min_active_seconds``. So the floor is about 2.3 s here and
    # about 1.6 s for a 4 Hz LED bar. It was 0.4 s worse before
    # ``min_active_seconds`` came down from 1.2 s to 0.8 s, which is a real
    # margin at the speeds vehicles cross a junction.
    detector = SirenDetector(SirenConfig())
    moment = 0.0
    for index in range(count):
        moment = index / fps
        frame = daylight(LAMP_BLUE if (moment * 2.0) % 1.0 < 0.35 else None)
        detector.begin_frame(frame, moment)
        verdict = detector.update(frame, (500, 300, 700, 500), 1, moment, "car")
        if verdict.active:
            break
    expected = SirenConfig().min_flashes / 2.0 + SirenConfig().min_active_seconds
    check("a 2 Hz beacon is confirmed as soon as the physics allow",
          verdict.active and moment <= expected + 0.2,
          f"{moment:.2f}s, floor {expected:.2f}s")


def test_short_clip_is_reported_as_unjudged() -> None:
    """Footage too short to judge must say so, not report a clear road.

    A beacon needs a window of history before the flash test has any opinion,
    so a clip shorter than that produces no verdict at all — and reporting "no
    emergency vehicle" about footage nobody could have judged is a claim the
    system has not earned. The screenshot that started this was a 1.4-second
    clip containing a plainly visible ambulance.
    """
    print("\n[8f] Footage too short to judge")
    from siren_vision import SirenConfig, SirenDetector

    fps = 42.0
    box = (500, 300, 700, 500)

    def watch(detector, seconds, beacon=False, start=0.0):
        """Show a vehicle to the detector for a stretch of time."""
        for index in range(int(start * fps), int(seconds * fps)):
            moment = index / fps
            frame = np.full((720, 1280, 3), 30, np.uint8)
            cv2.rectangle(frame, box[:2], box[2:], (40, 40, 45), -1)
            if beacon and (moment * 2.0) % 1.0 < 0.35:
                cv2.rectangle(frame, (520, 305), (600, 340), LAMP_RED, -1)
            detector.begin_frame(frame, moment)
            detector.update(frame, box, 1, moment, "car")
        return detector

    # Too little history for the flash test to have any opinion at all.
    brief = watch(SirenDetector(SirenConfig()), 0.9)
    check("a vehicle seen for under a second is awaiting a verdict",
          brief.pending_tracks == 1, f"{brief.pending_tracks}")

    # The case from the screenshot: the beacon *is* flashing and has been
    # measured as such, but the clip ends before the verdict can be confirmed.
    # Reporting "no emergency vehicle" here would be a claim about footage
    # nobody could have judged.
    partial = watch(SirenDetector(SirenConfig()), 1.4, beacon=True)
    check("a beacon seen too briefly to confirm is not yet flagged",
          not partial.verdict_for(1).active)
    check("…and is reported as undecided rather than cleared",
          partial.pending_tracks == 1, f"{partial.pending_tracks}")

    # Watched long enough with nothing flashing, "no beacon" becomes a real
    # answer and stops being reported as an open question.
    settled = watch(SirenDetector(SirenConfig()), 6.0)
    check("once watched long enough, 'no beacon' is an answer",
          settled.pending_tracks == 0, f"{settled.pending_tracks}")
    check("…and that vehicle is not flagged", not settled.verdict_for(1).active)


def test_flash_analysis() -> None:
    """The temporal test itself, on signals of known shape."""
    print("\n[8a] Flash signal analysis")
    from siren_vision import SirenConfig, analyse_flashes

    cfg = SirenConfig()
    fps, seconds = 15.0, 3.4
    times = [i / fps for i in range(int(fps * seconds))]

    def series(fn):
        return [fn(t) for t in times]

    beacon = analyse_flashes(times, series(
        lambda t: 0.7 if (t * 2.0) % 1.0 < 0.35 else 0.02), cfg)
    check("a 2 Hz square wave is flashing", beacon.flashing, beacon.reason)
    check("its rate is measured about right",
          1.5 <= beacon.rate_hz <= 2.5, f"{beacon.rate_hz} Hz")

    slow = analyse_flashes(times, series(
        lambda t: 0.6 if (t * 1.2) % 1.0 < 0.5 else 0.02), cfg)
    check("a slow 1.2 Hz beacon is still caught", slow.flashing, slow.reason)

    steady = analyse_flashes(times, series(lambda t: 0.75), cfg)
    check("a steady light is not flashing", not steady.flashing)
    check("and the reason says so", steady.reason == "steady-light", steady.reason)

    once = analyse_flashes(times, series(
        lambda t: 0.8 if 1.0 < t < 2.0 else 0.02), cfg)
    check("a single on/off is not flashing", not once.flashing)
    check("rejected as not repeating", once.reason == "not-repeating", once.reason)

    # The discriminator that matters most on real footage: a vehicle drifting
    # through a pool of street light ramps smoothly, a lamp switches.
    ramp = analyse_flashes(times, series(
        lambda t: 0.15 + 1.2 * abs(((t * 1.5) % 1.0) - 0.5)), cfg)
    check("a smooth brightness ramp is not flashing", not ramp.flashing)
    check("rejected as gradual rather than switched",
          ramp.reason == "gradual-not-switched", ramp.reason)

    dim = analyse_flashes(times, series(lambda t: 0.03), cfg)
    check("darkness is not flashing", not dim.flashing)


def test_emergency_fusion() -> None:
    """The siren decides; livery only names the type and adds confidence."""
    print("\n[8b] Emergency cue fusion")
    from emergency_vision import ORDINARY_LABEL, EmergencyVerdict, fuse
    from siren_vision import SirenVerdict

    ambulance = EmergencyVerdict(True, "ambulance", 0.95, "appearance", 4)
    ordinary = EmergencyVerdict(False, ORDINARY_LABEL, 0.05, "none", 4)
    blue_siren = SirenVerdict(True, "police", 0.8, "blue", 2.0, 5, 30)
    red_siren = SirenVerdict(True, "ambulance", 0.7, "red", 2.0, 5, 30)
    no_siren = SirenVerdict()

    unmarked = fuse(ordinary, blue_siren)
    check("a blue beacon alone is enough", unmarked.is_emergency)
    check("blue names the vehicle police", unmarked.vehicle_type == "police")
    check("the evidence names the colour",
          unmarked.evidence == "siren-blue", unmarked.evidence)

    both = fuse(ambulance, red_siren)
    check("beacon and livery together still fire", both.is_emergency)
    check("red plus ambulance livery reads as an ambulance",
          both.vehicle_type == "ambulance")
    check("agreement is recorded in the evidence",
          both.evidence == "siren-red+livery", both.evidence)
    check("agreement takes the higher confidence",
          both.confidence == 0.95, f"{both.confidence}")

    # A fire engine's beacon is also red; the livery is what tells them apart.
    fire = fuse(EmergencyVerdict(True, "fire truck", 0.9, "appearance", 4), red_siren)
    check("red plus fire livery reads as a fire engine",
          fire.vehicle_type == "fire truck", fire.vehicle_type)

    # Blue is decisive — nothing civilian carries one — so livery must not
    # overrule it.
    conflict = fuse(ambulance, blue_siren)
    check("a blue beacon outranks conflicting livery",
          conflict.vehicle_type == "police", conflict.vehicle_type)

    check("a beacon earns priority", unmarked.priority and both.priority)

    # A marked vehicle with its lights off is reported — it is an emergency
    # vehicle, and staying silent about one the system has recognised is
    # indistinguishable from failing to see it — but it is given no priority,
    # because an ambulance parked outside a hospital does not need the
    # junction held. The two flags exist to say exactly that.
    parked = fuse(ambulance, no_siren)
    check("a marked vehicle with its lights off is still reported",
          parked.is_emergency)
    check("…but is given no priority", not parked.priority)
    check("…and is reported as suspected", parked.suspected)
    check("…and is labelled with a question mark",
          parked.display_name == "AMBULANCE?", parked.display_name)

    allowed = fuse(ambulance, no_siren, require_siren=False)
    check("livery alone takes priority when the deployment asks for it",
          allowed.is_emergency and allowed.priority
          and allowed.evidence == "appearance")

    neither = fuse(ordinary, no_siren)
    check("neither cue means no emergency", not neither.is_emergency)
    check("neither cue means no priority", not neither.priority)
    check("ordinary vehicles report no type", neither.vehicle_type == ORDINARY_LABEL)
    check("ordinary vehicles have no display name", neither.display_name == "")
    check("a prioritised vehicle is named without a question mark",
          both.display_name == "AMBULANCE", both.display_name)
    check("unknown type still displays a warning",
          EmergencyVerdict(True, "unknown", 0.8, "siren-blue",
                           priority=True).display_name == "EMERGENCY")


def test_appearance_classifier_logic() -> None:
    """Scheduling, hysteresis and degradation of the CLIP classifier."""
    print("\n[8c] Appearance classifier logic")
    from emergency_vision import EmergencyAppearanceClassifier, EmergencyConfig

    # With recognition disabled the classifier must stay silent, not crash.
    off = EmergencyAppearanceClassifier(EmergencyConfig(enabled=False))
    check("disabled classifier reports unavailable", not off.available)
    frame = np.zeros((720, 1280, 3), np.uint8)
    verdicts = off.classify_frame(frame, [(1, (100, 100, 300, 300))], 0.0)
    check("disabled classifier still returns a verdict", 1 in verdicts)
    check("disabled classifier flags nothing", not verdicts[1].is_emergency)
    check("disabled status is explained", "off" in off.status)

    # Belief maths, exercised without loading the model.
    cfg = EmergencyConfig(enabled=False, min_checks=2, on_threshold=0.60,
                          off_threshold=0.35, check_interval_frames=5,
                          max_backoff_frames=40, score_ema_alpha=1.0,
                          confident_ordinary_score=0.15)
    engine = EmergencyAppearanceClassifier(cfg)

    engine._update_belief(7, 0.95, "ambulance")
    check("one strong observation is not enough", not engine.verdict_for(7).is_emergency)
    engine._update_belief(7, 0.95, "ambulance")
    verdict = engine.verdict_for(7)
    check("two strong observations trigger", verdict.is_emergency)
    check("type is reported", verdict.vehicle_type == "ambulance", verdict.vehicle_type)
    check("confidence is reported", verdict.confidence >= 0.9)

    engine._update_belief(7, 0.50, "ambulance")
    check("a mid-confidence frame does not release the flag",
          engine.verdict_for(7).is_emergency)
    engine._update_belief(7, 0.10, "ambulance")
    check("a clearly ordinary frame releases the flag",
          not engine.verdict_for(7).is_emergency)

    # Back-off: obvious ordinary vehicles get re-checked less and less.
    engine._frame_index = 100
    for _ in range(5):
        engine._update_belief(9, 0.02, "ordinary")
    belief = engine._beliefs[9]
    check("confident ordinary vehicles back off", belief.backoff > cfg.check_interval_frames,
          f"backoff={belief.backoff} frames")
    check("back-off is capped", belief.backoff <= cfg.max_backoff_frames)

    engine._update_belief(9, 0.90, "police")
    check("an interesting vehicle resets to the fast cadence",
          engine._beliefs[9].backoff == cfg.check_interval_frames)

    # Small crops carry no usable livery and must be skipped.
    check("tiny boxes are not scheduled", not engine._due(11, 20, 20))
    check("large boxes are scheduled", engine._due(12, 200, 200))

    engine.prune(10_000.0)
    check("beliefs pruned", len(engine._beliefs) == 0)

    # Crops with no information in them must not be shown to the model at all.
    # A confident answer about a black rectangle is worse than no answer.
    readable = EmergencyAppearanceClassifier(EmergencyConfig(enabled=False))
    dark = np.full((120, 120, 3), 6, np.uint8)
    flat = np.full((120, 120, 3), 128, np.uint8)
    textured = np.random.default_rng(0).integers(0, 255, (120, 120, 3), dtype=np.uint8)
    check("a near-black crop is refused", not readable._readable(dark))
    check("a flat grey crop is refused", not readable._readable(flat))
    check("a crop with detail is accepted", readable._readable(textured))


def test_appearance_backends() -> None:
    """Both livery backends must report on the same 0-1 scale."""
    print("\n[8d] Livery recognition backends")
    from emergency_vision import EmergencyAppearanceClassifier, EmergencyConfig

    # CLIP is the default because livery now reports an emergency rather than
    # only annotating one, and the trained model's false positives do not
    # survive footage from a camera it was not trained on: on an unseen night
    # clip it called six ordinary cars "fire truck" at 0.94-1.00 confidence.
    check("backend selection defaults to clip", EmergencyConfig().backend == "clip")
    # "auto" remains available and still prefers the trained model when its
    # weights are present, falling back to CLIP when they are not.
    picker = EmergencyAppearanceClassifier(EmergencyConfig(enabled=False,
                                                          backend="auto"))
    picker.config.model_path = "definitely_not_here.pt"
    check("auto falls back to clip with no trained weights",
          picker._resolved_backend() == "clip")

    # A missing trained model must degrade to "unavailable" with a usable
    # message, never take the pipeline down: the siren detector carries on.
    missing = EmergencyAppearanceClassifier(
        EmergencyConfig(backend="trained", model_path="no_such_model.pt"))
    check("a missing trained model does not crash", not missing.available)
    check("and says how to fix it",
          "train_emergency_classifier" in (missing.error or ""), missing.error)

    # The trained backend's score is the same two-way posterior CLIP reports,
    # so one set of thresholds governs both. Verified against the arithmetic
    # rather than a model, so it holds whether or not one has been trained.
    engine = EmergencyAppearanceClassifier(EmergencyConfig(enabled=False))
    engine._trained_labels = ["ambulance", "fire truck", "ordinary", "police"]
    engine._ordinary_index = 2

    import torch  # the real model hands back tensors, so the fake must too

    class _Probs:
        def __init__(self, values): self.data = torch.tensor(values, dtype=torch.float32)

    class _Prediction:
        def __init__(self, values): self.probs = _Probs(values)

    class _Model:
        def __init__(self, rows): self.rows = rows
        def predict(self, crops, **_kwargs):
            return [_Prediction(row) for row in self.rows]

    engine._model = _Model([
        [0.80, 0.02, 0.10, 0.08],   # clearly an ambulance
        [0.02, 0.01, 0.95, 0.02],   # clearly ordinary
        [0.25, 0.25, 0.25, 0.25],   # no idea — must not read as emergency
    ])
    engine.config.backend = "trained"
    scored = engine._score_trained([None, None, None])

    check("a confident positive scores high", scored[0][0] > 0.85, f"{scored[0][0]:.2f}")
    check("and is named", scored[0][1] == "ambulance", scored[0][1])
    check("a confident negative scores low", scored[1][0] < 0.15, f"{scored[1][0]:.2f}")
    check("and is named ordinary", scored[1][1] == "ordinary", scored[1][1])
    # This is the case that broke the old system: an uninformative answer must
    # land well below the threshold rather than just above it.
    check("an uninformative answer scores 0.5, not higher",
          abs(scored[2][0] - 0.5) < 0.01, f"{scored[2][0]:.2f}")
    check("and stays below the on-threshold",
          scored[2][0] < EmergencyConfig().on_threshold)


# ═══════════════════════════════════════════════════════════════════════════
# FULL PIPELINE (fake detector)
# ═══════════════════════════════════════════════════════════════════════════


def test_pipeline() -> None:
    """End-to-end pipeline behaviour with scripted, exactly-known motion."""
    print("\n[9] Full pipeline")
    calibration_dir = Path(tempfile.mkdtemp())
    calibration_path = calibration_dir / "cal.json"
    calibration_path.write_text(json.dumps(
        {"mode": "scale", "meters_per_pixel": 0.05, "frame_size": [1280, 720]}))

    # Two cars, 200 px over 2 s = 10 m in 2 s = 18 km/h each.
    frames, fps = [], 30
    for index in range(2 * fps + 1):
        moment = index / fps
        offset = 200 * (moment / 2.0)
        frames.append([
            (np.array([100 + offset, 400, 160 + offset, 460]), 1, "car", 0.9),
            (np.array([100 + offset, 500, 190 + offset, 560]), 2, "truck", 0.8),
        ])

    config = tv.VisionConfig(calibration_path=str(calibration_path), emergency_appearance=False)
    pipeline = tv.TrafficVisionPipeline(config, detector=FakeDetector(frames))
    frame = np.zeros((720, 1280, 3), np.uint8)

    result = None
    for index in range(len(frames)):
        result = pipeline.process(frame, index / fps)

    check("both vehicles counted", result.total_vehicles == 2, f"{result.vehicle_counts}")
    check("classes preserved",
          result.vehicle_counts["car"] == 1 and result.vehicle_counts["truck"] == 1)
    check("average speed is 18 km/h", close_to(result.avg_speed_kmh, 18.0, 0.6),
          f"{result.avg_speed_kmh} km/h")
    check("weighted density counts a truck as 3 PCE",
          close_to(result.weighted_density, 4.0, 0.01), f"{result.weighted_density}")
    check("session unique count", result.unique_vehicles_session == 2)
    check("speed source reported", result.speed_source == "scale")
    check("both vehicles counted as moving", result.moving_vehicles == 2)
    check("fps measured", result.fps > 0)

    # Vehicles still in frame at shutdown must still reach the per-vehicle log.
    check("nothing retired while both vehicles are present",
          len(pipeline.completed_vehicles) == 0)
    finalized = pipeline.finalize()
    check("shutdown retires every remaining vehicle", len(finalized) == 2,
          f"{len(finalized)} records")
    check("retired records carry a speed",
          all(record.get("avg_kmh", 0) > 0 for record in finalized),
          f"{[r.get('avg_kmh') for r in finalized]}")
    check("retired records carry the stabilised class",
          {record["class"] for record in finalized} == {"car", "truck"})

    # Ghost suppression: a detection lasting a single frame must never count.
    ghost = tv.TrafficVisionPipeline(
        tv.VisionConfig(calibration_path=str(calibration_path), emergency_appearance=False),
        detector=FakeDetector([[(np.array([10, 10, 70, 70]), 99, "car", 0.9)], [], []]))
    first = ghost.process(frame, 0.0)
    check("one-frame ghost is not counted", first.total_vehicles == 0)

    # Class-vote stabilisation: a van flickering car/truck should settle on the
    # majority class instead of alternating in the log.
    flicker = []
    for index in range(20):
        label = "truck" if index % 3 else "car"     # truck 2/3 of the time
        flicker.append([(np.array([100, 400, 200, 500]), 5, label, 0.9)])
    stable = tv.TrafficVisionPipeline(
        tv.VisionConfig(calibration_path=str(calibration_path), emergency_appearance=False),
        detector=FakeDetector(flicker))
    labels = set()
    for index in range(len(flicker)):
        out = stable.process(frame, index / 30)
        labels.update(d.label for d in out.detections)
    check("class vote stabilises the label", labels == {"truck"}, f"labels seen {labels}")

    shutil.rmtree(calibration_dir, ignore_errors=True)


def test_roi_filtering() -> None:
    """Detections outside the configured ROI must be dropped."""
    print("\n[10] Region of interest")
    directory = Path(tempfile.mkdtemp())
    path = directory / "cal.json"
    path.write_text(json.dumps({
        "mode": "scale", "meters_per_pixel": 0.05, "frame_size": [1280, 720],
        "roi": [[0, 300], [1280, 300], [1280, 720], [0, 720]],   # bottom half only
    }))

    inside = [[(np.array([100, 500, 160, 600]), 1, "car", 0.9)]] * 10
    outside = [[(np.array([100, 50, 160, 150]), 2, "car", 0.9)]] * 10
    frame = np.zeros((720, 1280, 3), np.uint8)

    config = tv.VisionConfig(calibration_path=str(path), emergency_appearance=False)
    in_pipeline = tv.TrafficVisionPipeline(config, detector=FakeDetector(inside))
    out_pipeline = tv.TrafficVisionPipeline(config, detector=FakeDetector(outside))
    for index in range(10):
        in_result = in_pipeline.process(frame, index / 30)
        out_result = out_pipeline.process(frame, index / 30)
    check("vehicle inside the ROI counts", in_result.total_vehicles == 1)
    check("vehicle outside the ROI ignored", out_result.total_vehicles == 0)
    check("ROI polygon available for drawing",
          in_pipeline.ground.roi_polygon((720, 1280)) is not None)
    check("the ROI is reported as active on its own frame shape",
          in_result.roi_active)

    # A region of interest is a polygon around a stretch of road as one camera
    # saw it. Rescaling it to a frame of the same shape is sound; stretching it
    # onto a different aspect ratio scales the two axes by different factors and
    # lands it somewhere the road never was. This is not hypothetical — a 16:9
    # calibration applied to a 608x1080 phone clip discarded 6595 of 11330
    # detections and took the emergency count from six to nought, in silence.
    ground = in_pipeline.ground
    check("a 16:9 region does not apply to a portrait frame",
          not ground.roi_applies_to((1920, 1080)))
    check("…so nothing is excluded there",
          ground.contains((5, 5), (1920, 1080)))
    check("…and no boundary is drawn that is not being enforced",
          ground.roi_polygon((1920, 1080)) is None)
    check("the same region still applies to a larger 16:9 frame",
          ground.roi_applies_to((1080, 1920)))
    check("…and still excludes what it should there",
          not ground.contains((100, 100), (1080, 1920)))

    # The whole point of the guard: the emergency vehicle in the excluded half
    # is still found, because recognition is about what is out there while the
    # counts are about the monitored road.
    portrait = np.zeros((1920, 1080, 3), np.uint8)
    portrait_pipeline = tv.TrafficVisionPipeline(
        config, detector=FakeDetector([[(np.array([100, 50, 160, 150]), 2,
                                         "car", 0.9)]] * 10))
    for index in range(10):
        portrait_result = portrait_pipeline.process(portrait, index / 30)
    check("a mismatched region stops excluding anything",
          portrait_result.total_vehicles == 1 and portrait_result.outside_roi == 0)
    check("and says so in the result", not portrait_result.roi_active)
    shutil.rmtree(directory, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════


def _sample_result(speed: float = 42.0, vehicles: int = 3) -> tv.FrameResult:
    """Build a representative FrameResult for logging tests."""
    return tv.FrameResult(
        frame_index=1, timestamp=time.time(), detections=[],
        vehicle_counts={"car": vehicles, "motorcycle": 0, "bus": 0, "truck": 0},
        total_vehicles=vehicles, avg_speed_kmh=speed, max_speed_kmh=speed + 8,
        moving_vehicles=vehicles, stopped_vehicles=0, weighted_density=float(vehicles),
        congestion_index=0.21, congestion_level="FREE", emergency_ids=set(),
        fps=24.0, speed_source="homography", unique_vehicles_session=vehicles,
    )


def test_logging() -> None:
    """CSV and Excel logging must be complete, correct and crash-safe."""
    print("\n[11] Telemetry logging")
    directory = Path(tempfile.mkdtemp())
    csv_path = directory / "traffic_history.csv"
    excel_path = directory / "traffic_history.xlsx"

    row = tv.build_log_row(_sample_result(), servo_angle=90, emergency=False)
    # Session/Session Name are stamped on by the logger, which is the only
    # component that knows which session is open, so the row builder omits them.
    stamped_by_logger = {"Session", "Session Name"}
    check("log row matches the header schema",
          list(row.keys()) == [h for h in tv.LOG_HEADERS if h not in stamped_by_logger],
          f"{set(row) ^ set(tv.LOG_HEADERS)}")
    check("speed column is km/h", "Avg Speed (km/h)" in tv.LOG_HEADERS)
    check("no pixel-per-second column survives",
          not any("px" in header for header in tv.LOG_HEADERS))
    check("schema carries the session id", "Session" in tv.LOG_HEADERS)
    check("schema records the light phase", "Light Phase" in tv.LOG_HEADERS)
    check("the light phase is logged when supplied",
          tv.build_log_row(_sample_result(), servo_angle=90, emergency=False,
                           light_phase="GREEN")["Light Phase"] == "GREEN")

    logger = tv.TelemetryLogger(csv_path, excel_path)
    for index in range(25):
        logger.log(tv.build_log_row(_sample_result(speed=40 + index, vehicles=index % 6),
                                    servo_angle=90, emergency=index == 10))
    logger.log_vehicles([
        {"track_id": 1, "class": "car", "frames": 60, "duration_s": 4.0,
         "distance_m": 55.0, "avg_kmh": 49.5, "max_kmh": 52.0},
        {"track_id": 2, "class": "bus", "frames": 90, "duration_s": 6.0,
         "distance_m": 60.0, "avg_kmh": 36.0, "max_kmh": 40.0},
    ])

    check("CSV exists during the session", csv_path.exists())
    # utf-8-sig strips the BOM the logger writes so Excel opens the file cleanly.
    lines = csv_path.read_text(encoding="utf-8-sig").strip().splitlines()
    check("CSV has a header plus every row", len(lines) == 26, f"{len(lines)} lines")
    check("CSV header is the km/h schema", lines[0] == ",".join(tv.LOG_HEADERS),
          lines[0][:60])
    check("every row carries its session id",
          all(line.startswith(logger.session_id) for line in lines[1:]),
          f"session {logger.session_id}")

    logger.close()
    check("Excel written on close", excel_path.exists())

    from openpyxl import load_workbook

    workbook = load_workbook(excel_path)
    check("all four sheets present",
          workbook.sheetnames == ["Live Status", "Traffic Log", "Vehicle Speeds",
                                  "Session Summary"], f"{workbook.sheetnames}")

    log_sheet = workbook["Traffic Log"]
    headers = [cell.value for cell in log_sheet[2]]
    check("workbook headers match schema", headers == tv.LOG_HEADERS)
    check("every logged row is present", log_sheet.max_row == 2 + 25,
          f"max_row={log_sheet.max_row}")

    speed_column = tv.LOG_HEADERS.index("Avg Speed (km/h)") + 1
    values = [log_sheet.cell(row=r, column=speed_column).value for r in range(3, 28)]
    check("speeds stored as numbers, not text", all(isinstance(v, (int, float)) for v in values))
    check("speed values round-trip", values[0] == 40 and values[-1] == 64, f"{values[:3]}")

    vehicles_sheet = workbook["Vehicle Speeds"]
    check("per-vehicle sheet populated", vehicles_sheet.max_row == 4,
          f"max_row={vehicles_sheet.max_row}")
    top_speed_column = tv.VEHICLE_HEADERS.index("Top Speed (km/h)") + 1
    check("per-vehicle top speed recorded",
          vehicles_sheet.cell(row=3, column=top_speed_column).value == 52.0,
          f"col {top_speed_column}")
    emergency_column = tv.VEHICLE_HEADERS.index("Emergency") + 1
    check("per-vehicle emergency column present",
          vehicles_sheet.cell(row=2, column=emergency_column).value == "Emergency")

    summary_sheet = workbook["Session Summary"]
    summary = {summary_sheet.cell(row=r, column=1).value:
               summary_sheet.cell(row=r, column=2).value
               for r in range(2, summary_sheet.max_row + 1)}
    check("summary counts samples", summary.get("Samples logged") == 25, f"{summary}")
    check("summary records emergency samples", summary.get("Emergency samples") == 1)
    check("summary records the calibration used",
          summary.get("Speed calibration") == "homography")
    check("summary breaks down time spent on each signal phase",
          all(f"Light {phase} (samples)" in summary
              for phase in ("GREEN", "YELLOW", "RED")), f"{sorted(summary)}")
    workbook.close()

    # ── Session file vs master file ─────────────────────────────────────────
    # The session workbook covers only the run in progress; the master carries
    # every run ever recorded. A second session must therefore reset one and
    # extend the other.
    first_session = logger.session_id
    logger2 = tv.TelemetryLogger(csv_path, excel_path)
    check("a new session starts empty, not preloaded", logger2.row_count == 0,
          f"{logger2.row_count} rows")
    check("the new session has its own id", logger2.session_id != first_session,
          f"{first_session} -> {logger2.session_id}")
    logger2.log(tv.build_log_row(_sample_result(), servo_angle=45, emergency=False,
                                 light_phase="GREEN"))
    logger2.close()

    workbook2 = load_workbook(excel_path)
    check("session workbook holds only the current session",
          workbook2["Traffic Log"].max_row == 2 + 1,
          f"max_row={workbook2['Traffic Log'].max_row}")
    workbook2.close()

    master_excel = directory / "traffic_all_sessions.xlsx"
    check("master workbook exists", master_excel.exists())
    master = load_workbook(master_excel)
    check("master has an index and a full record",
          master.sheetnames == ["All Sessions", "All Records", "All Vehicles",
                                "Lifetime Summary"],
          f"{master.sheetnames}")
    check("master is cumulative across sessions",
          master["All Records"].max_row == 2 + 26,
          f"max_row={master['All Records'].max_row}")
    check("master indexes both sessions", master["All Sessions"].max_row == 2 + 2,
          f"max_row={master['All Sessions'].max_row}")
    master.close()

    index = logger2.session_index()
    check("session index reports both runs", len(index) == 2, f"{len(index)}")
    check("session index counts each run's samples",
          sorted(entry["Samples"] for entry in index) == [1, 25],
          f"{[entry['Samples'] for entry in index]}")
    check("session index records peak congestion",
          all("Peak Congestion" in entry for entry in index))

    # ── Named recordings ────────────────────────────────────────────────────
    logger3 = tv.TelemetryLogger(csv_path, excel_path)
    check("logging starts un-armed", logger3.recording is False)
    logger3.start_session("demo run")
    check("starting a recording arms it", logger3.recording is True)
    check("the recording carries its name", logger3.session_name == "demo run")
    for _ in range(4):
        logger3.log(tv.build_log_row(_sample_result(), servo_angle=90, emergency=False,
                                     light_phase="RED"))
    check("rows land in the named recording", logger3.row_count == 4)
    stopped = logger3.stop_session()
    check("stopping reports what was captured",
          stopped["rows"] == 4 and stopped["name"] == "demo run", f"{stopped}")
    check("stopping disarms recording", logger3.recording is False)
    check("a fresh session follows the recording", logger3.row_count == 0)
    logger3.close()

    named = [entry for entry in logger3.session_index()
             if entry["Session Name"] == "demo run"]
    check("the named recording appears in the master index", len(named) == 1,
          f"{[e['Session Name'] for e in logger3.session_index()]}")
    check("the named recording kept its rows",
          bool(named) and named[0]["Samples"] == 4, f"{named}")

    archived = sorted(p.name for p in (directory / "sessions").glob("session_*.csv"))
    check("every finished session is archived individually", len(archived) >= 3,
          f"{archived}")

    # ── Per-vehicle records ─────────────────────────────────────────────────
    # These used to live only in memory and only in the session workbook, so a
    # crash lost them and no all-time record of them existed at all.
    vehicles = [
        {"track_id": 3, "class": "car", "emergency": "no",
         "emergency_evidence": "none", "frames": 40, "duration_s": 3.2,
         "distance_m": 48.0, "avg_kmh": 54.0, "max_kmh": 60.5},
        {"track_id": 4, "class": "truck", "emergency": "YES",
         "emergency_evidence": "siren-blue", "frames": 55, "duration_s": 4.4,
         "distance_m": 61.0, "avg_kmh": 49.9, "max_kmh": 55.2},
    ]
    # Earlier loggers in this directory have already written vehicles, so the
    # all-time file is measured by how much it grows, not by its absolute size.
    master_vehicle_csv = directory / "traffic_all_vehicles.csv"

    def count_master_vehicles() -> list[dict[str, str]]:
        if not master_vehicle_csv.exists():
            return []
        with open(master_vehicle_csv, "r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    before_vehicles = len(count_master_vehicles())

    logger4 = tv.TelemetryLogger(csv_path, excel_path)
    logger4.log(tv.build_log_row(_sample_result(), servo_angle=90, emergency=False))
    logger4.log_vehicles(vehicles)
    first_session = logger4.session_id

    vehicle_csv = directory / "traffic_vehicles.csv"
    with open(vehicle_csv, "r", newline="", encoding="utf-8-sig") as handle:
        live = list(csv.DictReader(handle))
    check("vehicles reach the CSV before the session ends", len(live) == 2,
          f"{len(live)}")
    check("vehicle rows carry the full schema",
          list(live[0]) == tv.VEHICLE_LOG_HEADERS, f"{list(live[0])[:5]}")
    check("vehicle rows are stamped with their session",
          live[0]["Session"] == first_session, f"{live[0]['Session']}")
    check("vehicle speeds survive the round trip",
          live[0]["Top Speed (km/h)"] == "60.5", f"{live[0]['Top Speed (km/h)']}")
    check("emergency evidence survives the round trip",
          live[1]["Evidence"] == "siren-blue", f"{live[1]['Evidence']}")
    logger4.close()

    logger5 = tv.TelemetryLogger(csv_path, excel_path)
    logger5.log(tv.build_log_row(_sample_result(), servo_angle=90, emergency=False))
    logger5.log_vehicles(vehicles[:1])
    with open(vehicle_csv, "r", newline="", encoding="utf-8-sig") as handle:
        check("a new session starts its vehicle file empty",
              len(list(csv.DictReader(handle))) == 1)
    master_vehicles = count_master_vehicles()
    check("the all-time vehicle record accumulates",
          len(master_vehicles) == before_vehicles + 3,
          f"{len(master_vehicles)}, expected {before_vehicles + 3}")
    check("the all-time record spans both sessions",
          {first_session, logger5.session_id} <= {row["Session"]
                                                  for row in master_vehicles})
    logger5.close()

    # ── The master workbook must not be archived on every single run ────────
    # Its first sheet is the session index, whose columns are deliberately not
    # LOG_HEADERS; comparing the two archived a "_legacy_" copy every launch.
    before = {p.name for p in directory.glob("*_legacy_*")}
    logger6 = tv.TelemetryLogger(csv_path, excel_path)
    logger6.close()
    after = {p.name for p in directory.glob("*_legacy_*")}
    check("reopening a current workbook archives nothing", before == after,
          f"new: {sorted(after - before)}")

    shutil.rmtree(directory, ignore_errors=True)

    # ── Session ids must be unique even when sessions start in the same second ─
    # Ids are stamped to the second, so several sessions opened back to back all
    # land on the same timestamp. If they shared an id the master index would
    # merge them into a single row and their samples would be indistinguishable.
    burst_dir = Path(tempfile.mkdtemp())
    ids: list[str] = []
    loggers = []
    for _ in range(4):
        instance = tv.TelemetryLogger(burst_dir / "traffic_history.csv",
                                      burst_dir / "traffic_history.xlsx")
        instance.log(tv.build_log_row(_sample_result(), servo_angle=90,
                                      emergency=False, light_phase="RED"))
        ids.append(instance.session_id)
        loggers.append(instance)
    # Also cover repeated sessions inside one instance, the other collision path.
    # It needs a row of its own: a session that logged nothing contributes no
    # rows to the archive and so is correctly absent from the index.
    loggers[-1].start_session("burst")
    loggers[-1].log(tv.build_log_row(_sample_result(), servo_angle=90,
                                     emergency=False, light_phase="GREEN"))
    ids.append(loggers[-1].session_id)
    for instance in loggers:
        instance.close()

    check("separate loggers opened in the same second get distinct ids",
          len(set(ids)) == len(ids), f"{ids}")
    check("the ids all share the same second, so this really was a collision test",
          len({session_id[:15] for session_id in ids}) == 1, f"{sorted(set(ids))}")

    burst_index = loggers[-1].session_index()
    check("every burst session is indexed separately",
          len(burst_index) == len(ids), f"{len(burst_index)} of {len(ids)}")
    shutil.rmtree(burst_dir, ignore_errors=True)


def test_legacy_migration() -> None:
    """An old px/s logfile must be archived, never appended to with km/h."""
    print("\n[12] Legacy file migration")
    directory = Path(tempfile.mkdtemp())
    csv_path = directory / "traffic_history.csv"
    csv_path.write_text("Timestamp,Congestion Index,Avg Speed (px/s)\n00:00:01,0.2,15\n",
                        encoding="utf-8")

    logger = tv.TelemetryLogger(csv_path, directory / "x.xlsx")
    logger.log(tv.build_log_row(_sample_result(), servo_angle=90, emergency=False))
    logger.close()

    archived = list(directory.glob("traffic_history_legacy_*.csv"))
    check("old-format CSV archived", len(archived) == 1, f"{[p.name for p in archived]}")
    check("old px/s data preserved in the archive",
          "px/s" in archived[0].read_text(encoding="utf-8-sig"))
    header = csv_path.read_text(encoding="utf-8-sig").splitlines()[0]
    check("fresh CSV uses the km/h schema", header == ",".join(tv.LOG_HEADERS))
    shutil.rmtree(directory, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════


def test_rendering() -> None:
    """Overlay drawing must not distort the frame or crash on edge cases."""
    print("\n[13] Rendering")
    frame = np.zeros((1080, 1920, 3), np.uint8)
    fitted = tv.fit_to_display(frame, 1280, 720)
    check("aspect ratio preserved when scaling down", fitted.shape[:2] == (720, 1280),
          f"{fitted.shape}")

    tall = tv.fit_to_display(np.zeros((1000, 500, 3), np.uint8), 1280, 720)
    ratio_in, ratio_out = 500 / 1000, tall.shape[1] / tall.shape[0]
    check("portrait frames keep their ratio", close_to(ratio_in, ratio_out, 0.01),
          f"{tall.shape}")

    small = tv.fit_to_display(np.zeros((100, 100, 3), np.uint8), 1280, 720)
    check("small frames are not upscaled", small.shape[:2] == (100, 100))

    result = _sample_result()
    result.detections = [
        tv.VehicleDetection(1, "car", "car", 0.9, (100, 100, 200, 200), 42.3,
                            False, 0.0, 10),
        tv.VehicleDetection(2, "bus", "bus", 0.8, (300, 100, 500, 300), None,
                            True, 0.9, 20),
    ]
    annotated = tv.annotate(np.zeros((720, 1280, 3), np.uint8), result,
                            mode_label="AUTO", servo_angle=90, emergency=True,
                            hint="q:quit")
    check("annotation returns a same-size frame", annotated.shape == (720, 1280, 3))
    check("annotation does not modify the input",
          not np.array_equal(annotated, np.zeros((720, 1280, 3), np.uint8)))
    check("unmeasured speed renders as a placeholder",
          result.detections[1].speed_text == "--")
    check("measured speed renders in km/h", result.detections[0].speed_text == "42km/h")


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════


def test_config() -> None:
    """Configuration defaults, persistence and device selection."""
    print("\n[14] Configuration")
    config = tv.VisionConfig()
    # yolo11m, not yolo12m: measured 33% faster on the night footage *and*
    # finding more vehicles, so the v12 attention blocks were pure cost here.
    check("defaults to the medium model", config.model_path == "yolo11m.pt")
    # 1280 rather than 960: at night the vehicles that matter are small and
    # dark, and resolution buys far more of them than model size does.
    check("inference size raised for small vehicles", config.imgsz >= 1280)
    check("class-agnostic NMS on by default", config.agnostic_nms)
    check("emergency mode requires a working siren by default",
          config.emergency_require_siren)
    check("siren detection is on by default", config.siren_detection)

    directory = Path(tempfile.mkdtemp())
    path = directory / "vision.json"
    config.imgsz = 1536
    config.siren_red_type = "fire truck"
    config.save(path)
    reloaded = tv.VisionConfig.load(path)
    check("config round-trips through JSON", reloaded.imgsz == 1536)
    check("siren colour mapping round-trips",
          reloaded.siren_red_type == "fire truck")

    missing = tv.VisionConfig.load(directory / "nope.json")
    check("missing config falls back to defaults",
          missing.imgsz == tv.VisionConfig().imgsz)

    cpu = tv.VisionConfig(device="cpu")
    check("FP16 disabled on CPU", not cpu.use_half())
    shutil.rmtree(directory, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# REAL MODEL (opt-in)
# ═══════════════════════════════════════════════════════════════════════════


def test_real_model() -> None:
    """Load yolo12m and run it over a synthetic pan of a real traffic photo."""
    print("\n[15] Real YOLOv12-m inference")
    source = None
    for candidate in [Path("assets/traffic_test.jpg"), Path("bus.jpg")]:
        if candidate.exists():
            source = candidate
            break
    if source is None:
        try:
            from ultralytics.utils.downloads import safe_download

            safe_download("https://ultralytics.com/images/bus.jpg", dir=Path("."))
            source = Path("bus.jpg")
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] no test image available ({exc})")
            return

    photo = cv2.imread(str(source))
    if photo is None:
        print("  [SKIP] test image unreadable")
        return

    # Pan a wide canvas across the photo so the vehicle genuinely moves at a
    # known pixel rate, which the pipeline should convert into a stable km/h.
    canvas = cv2.resize(photo, (900, 700))
    background = np.full((720, 1280, 3), 70, np.uint8)

    config = tv.VisionConfig(calibration_path="__no_such_file__.json", imgsz=640,
                             min_track_frames=2)
    started = time.perf_counter()
    pipeline = tv.TrafficVisionPipeline(config)
    print(f"  model loaded in {time.perf_counter() - started:.1f}s "
          f"on {pipeline.detector.device}")

    fps, results = 15, []
    for index in range(30):
        frame = background.copy()
        x = 40 + index * 6
        frame[10:710, x:x + 900] = canvas[:, :min(900, 1280 - x)] \
            if x + 900 <= 1280 else canvas[:, :1280 - x]
        results.append(pipeline.process(frame, index / fps))

    final = results[-1]
    detected = max(r.total_vehicles for r in results)
    check("real model detects the vehicle", detected >= 1,
          f"peak count {detected}, classes {final.vehicle_counts}")
    check("inference time recorded", final.inference_ms > 0,
          f"{final.inference_ms:.0f} ms/frame")
    check("telemetry row builds from real output",
          list(tv.build_log_row(final, 90, False).keys()) == tv.LOG_HEADERS)

    speeds = [r.avg_speed_kmh for r in results if r.avg_speed_kmh > 0]
    check("a speed in km/h was produced", len(speeds) > 0,
          f"last {final.avg_speed_kmh} km/h from {final.speed_source} mode")
    check("speeds are physically plausible",
          all(0 < s < config.speed_max_plausible_kmh for s in speeds) if speeds else True,
          f"range {min(speeds, default=0):.1f}-{max(speeds, default=0):.1f} km/h")


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════


def test_real_emergency_recognition() -> None:
    """Run the real CLIP recogniser over photographs of real vehicles.

    ``assets/emergency_test`` holds Wikimedia Commons photographs named by
    ground truth: ``amb_*`` ambulances, ``pol_*`` police cars, ``fire_*`` fire
    engines, and ``normal_*`` / ``van_*`` ordinary traffic. Each photo is fed
    through the full pipeline as a short synthetic pan, so the classifier sees
    it exactly as it would see a moving vehicle.
    """
    print("\n[16] Real emergency vehicle recognition")
    from emergency_vision import EmergencyAppearanceClassifier, EmergencyConfig

    photos = sorted(Path("assets/emergency_test").glob("*.jpg"))
    if not photos:
        print("  [SKIP] no test photographs in assets/emergency_test")
        return

    classifier = EmergencyAppearanceClassifier(EmergencyConfig(check_interval_frames=1))
    if not classifier.available:
        print(f"  [SKIP] CLIP unavailable: {classifier.error}")
        return
    check("classifier loaded", classifier.available, classifier.status)

    detector = tv.VehicleDetector(tv.VisionConfig(imgsz=640, emergency_appearance=False))
    correct = tp = fp = tn = fn = 0
    mistakes: list[str] = []

    for photo in photos:
        kind = photo.stem.split("_")[0]
        truth = kind in ("amb", "pol", "fire")
        image = cv2.imread(str(photo))
        if image is None:
            continue

        tracks, _ms = detector.track(image)
        boxes = [(index, box) for index, (box, _tid, _l, _c) in enumerate(tracks)]
        if not boxes:  # no vehicle found: classify the whole frame
            boxes = [(0, (0, 0, image.shape[1], image.shape[0]))]

        # Three passes so the min_checks hysteresis can settle, as it would
        # over consecutive video frames.
        for _ in range(3):
            classifier.begin_frame()
            verdicts = classifier.classify_frame(image, boxes, time.time())
        flagged = [v for v in verdicts.values() if v.is_emergency]
        predicted = bool(flagged)

        if truth and predicted: tp += 1
        elif truth: fn += 1; mistakes.append(f"missed {photo.name}")
        elif predicted: fp += 1; mistakes.append(
            f"false alarm {photo.name} -> {flagged[0].vehicle_type} {flagged[0].confidence}")
        else: tn += 1
        correct += int(truth == predicted)
        classifier._beliefs.clear()  # each photo is an independent scene

    total = tp + fp + tn + fn
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    print(f"  {total} photos: TP={tp} FP={fp} TN={tn} FN={fn} "
          f"recall={recall:.0%} precision={precision:.0%}")
    for mistake in mistakes:
        print(f"    - {mistake}")

    check("most emergency vehicles are recognised", recall >= 0.85, f"recall {recall:.0%}")
    check("ordinary traffic rarely false-alarms", precision >= 0.90,
          f"precision {precision:.0%}")
    check("overall accuracy is good", correct / max(total, 1) >= 0.90,
          f"{correct}/{total}")


def main() -> int:
    """Run the selected test groups and report a summary."""
    parser = argparse.ArgumentParser(description="Verify the traffic vision pipeline.")
    parser.add_argument("--yolo", action="store_true",
                        help="also load the real YOLO model (slow, needs weights)")
    args = parser.parse_args()

    print("=" * 70)
    print(" SMART TRAFFIC SYSTEM - VERIFICATION SUITE".center(70))
    print("=" * 70)

    for test in (test_homography_speed, test_auto_scale_speed, test_fixed_scale_mode,
                 test_frame_size_independence, test_speed_robustness, test_memory_pruning,
                 test_congestion, test_siren_detection, test_siren_in_daylight,
                 test_short_clip_is_reported_as_unjudged,
                 test_flash_analysis,
                 test_emergency_fusion,
                 test_appearance_classifier_logic, test_appearance_backends,
                 test_pipeline,
                 test_roi_filtering, test_logging, test_legacy_migration,
                 test_rendering, test_config):
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            import traceback

            FAILED.append((test.__name__, str(exc)))
            print(f"  [ERROR] {test.__name__}: {exc}")
            traceback.print_exc()

    if args.yolo:
        for test in (test_real_model, test_real_emergency_recognition):
            try:
                test()
            except Exception as exc:  # noqa: BLE001
                import traceback

                FAILED.append((test.__name__, str(exc)))
                print(f"  [ERROR] {test.__name__}: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f" RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for name, detail in FAILED:
        print(f"   FAILED: {name}  {detail}")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
