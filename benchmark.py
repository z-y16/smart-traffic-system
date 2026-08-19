"""
benchmark.py — measure what this system actually costs and finds
================================================================

Answers the questions you get asked about a vision pipeline and cannot answer
from the code: **how fast does it run**, **what does each stage cost**, and
**how many vehicles does it find** at each model size and resolution.

Run it before changing the model or the resolution, and again afterwards. The
numbers in ``README.md`` came from this.

    python benchmark.py --source vedios/Test2_dark_fotage.mp4 --frames 60
    python benchmark.py --source clip.mp4 --grid           # models x imgsz
    python benchmark.py --source clip.mp4 --stages         # per-stage cost

A word on what "accuracy" can and cannot mean here
--------------------------------------------------
There are no hand-drawn boxes for this footage, so there is no ground truth to
score against and **this script does not print an mAP**. Anything claiming to
would be inventing it.

What it prints instead is *vehicles found per frame*, which on fixed footage is
a genuine comparison between two settings — the road holds the same vehicles
either way, so finding more of them is finding more of them — and which is the
measurement that actually drove the resolution decision. It is a comparison,
not a score: it cannot tell you the ones everything missed.

The system's real accuracy claims are checked elsewhere, against ground truth
that does exist because it was constructed:

* **speed** — ``test_traffic_system.py`` drives a synthetic vehicle at exactly
  18 / 36 / 72 km/h and requires those answers back;
* **the media clock** — ``test_video_source.py`` drives one at 16.2 km/h
  through a real MP4 at four playback rates and requires all four to agree;
* **emergency recognition** — measured on real footage in ``README.md``, with
  the false-alarm rates that matter stated next to the hit rates.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from traffic_vision import VehicleDetector, VisionConfig  # noqa: E402


def load_frames(source: Path, count: int, start: float) -> list[np.ndarray]:
    """Read a run of **consecutive** frames, so every configuration sees the same road.

    Consecutive matters, and getting it wrong invalidates the vehicle counts
    rather than just adding noise. ``VehicleDetector.track`` runs the tracker
    with ``persist=True`` and drops any detection the tracker could not give an
    id to — which is correct for the live pipeline, where every downstream
    measurement needs identity over time. But it means that sampling every
    20th frame hands the tracker a sequence in which nothing matches anything,
    so most detections arrive with no id and are discarded before they are
    counted. Measured that way, yolo11m at 1280 appeared to find 4.4 vehicles
    per frame on night footage where consecutive frames find 20.5.

    Re-decoding per configuration is also avoided: the road must hold the same
    vehicles for each one, or the comparison measures the footage instead of
    the setting.
    """
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"error: cannot open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    frames: list[np.ndarray] = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise SystemExit(f"error: no frames read from {source}")
    return frames


def time_detector(frames: list[np.ndarray], model: str, imgsz: int,
                  half: bool = True) -> dict[str, float]:
    """Run one configuration over the frames and return its cost and yield.

    The first frame is dropped from the timing. On CUDA it carries one-off
    kernel compilation and memory allocation that no later frame pays, and
    including it makes a fast configuration look slow purely because it went
    first.
    """
    config = VisionConfig(model_path=model, imgsz=imgsz, half=half)
    detector = VehicleDetector(config)
    detector.reset_tracker()

    times: list[float] = []
    found: list[int] = []
    for position, frame in enumerate(frames):
        started = time.perf_counter()
        tracked, _ms = detector.track(frame)
        elapsed = (time.perf_counter() - started) * 1000.0
        if position:
            times.append(elapsed)
            found.append(len(tracked))

    return {
        "ms_mean": statistics.mean(times),
        "ms_median": statistics.median(times),
        "ms_p95": sorted(times)[int(len(times) * 0.95) - 1] if len(times) > 2 else max(times),
        "fps": 1000.0 / statistics.mean(times),
        "vehicles": statistics.mean(found),
    }


def run_grid(frames: list[np.ndarray], models: list[str], sizes: list[int]) -> None:
    """Time every (model, imgsz) pair and print them as one comparable table."""
    print(f"\n{'model':<14} {'imgsz':>6} {'ms/frame':>9} {'median':>8} {'p95':>8} "
          f"{'fps':>7} {'veh/frame':>10}")
    print("-" * 68)
    for model in models:
        if not (ROOT_DIR / model).exists():
            print(f"{model:<14} {'—':>6}  not present, skipped")
            continue
        for imgsz in sizes:
            try:
                result = time_detector(frames, model, imgsz)
            except Exception as exc:  # noqa: BLE001 - a missing model must not stop the sweep
                print(f"{model:<14} {imgsz:>6}  failed: {exc}")
                continue
            print(f"{model:<14} {imgsz:>6} {result['ms_mean']:>9.1f} "
                  f"{result['ms_median']:>8.1f} {result['ms_p95']:>8.1f} "
                  f"{result['fps']:>7.1f} {result['vehicles']:>10.1f}")


def run_stages(frames: list[np.ndarray], model: str, imgsz: int) -> None:
    """Break the full pipeline down into what each stage costs per frame.

    This is the table that tells you where to spend effort: if detection is 80%
    of the frame time, optimising the congestion maths buys nothing.
    """
    from emergency_vision import (EmergencyAppearanceClassifier, EmergencyConfig,
                                  EmergencyVerdict, fuse)
    from siren_vision import SirenConfig, SirenDetector

    detector = VehicleDetector(VisionConfig(model_path=model, imgsz=imgsz))
    siren = SirenDetector(SirenConfig())
    appearance = EmergencyAppearanceClassifier(EmergencyConfig(backend="clip"))

    detect_ms: list[float] = []
    livery_ms: list[float] = []
    beacon_ms: list[float] = []
    total_ms: list[float] = []

    for position, frame in enumerate(frames):
        now = position / 30.0
        frame_started = time.perf_counter()

        started = time.perf_counter()
        tracked, _ = detector.track(frame)
        detect = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        appearance.begin_frame()
        looks = appearance.classify_frame(
            frame, [(tid, box) for box, tid, _l, _c in tracked], now)
        livery = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        siren.begin_frame(frame, now)
        for box, track_id, label, _conf in tracked:
            beacon = siren.update(frame, box, track_id, now, label)
            fuse(looks.get(track_id, EmergencyVerdict()), beacon)
        beacon_cost = (time.perf_counter() - started) * 1000.0

        total = (time.perf_counter() - frame_started) * 1000.0
        if position:
            detect_ms.append(detect)
            livery_ms.append(livery)
            beacon_ms.append(beacon_cost)
            total_ms.append(total)

    total_mean = statistics.mean(total_ms)
    print(f"\n{'stage':<28} {'ms/frame':>9} {'share':>7}")
    print("-" * 46)
    for name, series in (("detection + tracking (YOLO)", detect_ms),
                         ("livery recognition (CLIP)", livery_ms),
                         ("beacon detection (siren)", beacon_ms)):
        mean = statistics.mean(series)
        print(f"{name:<28} {mean:>9.1f} {100 * mean / total_mean:>6.1f}%")
    other = total_mean - sum(statistics.mean(s) for s in (detect_ms, livery_ms, beacon_ms))
    print(f"{'everything else':<28} {other:>9.1f} {100 * other / total_mean:>6.1f}%")
    print("-" * 46)
    print(f"{'TOTAL':<28} {total_mean:>9.1f} {100.0:>6.1f}%")
    print(f"\nend-to-end: {1000.0 / total_mean:.1f} fps")


def main() -> int:
    """Run the requested benchmarks; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="video to benchmark against")
    parser.add_argument("--frames", type=int, default=60,
                        help="frames to time (the first is discarded as warm-up)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="seconds into the clip to start (pick a stretch with traffic)")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--grid", action="store_true", help="sweep models x resolutions")
    parser.add_argument("--stages", action="store_true", help="per-stage cost breakdown")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found")
        return 2

    print(f"loading {args.frames} consecutive frames from {source.name} "
          f"at t={args.start:.0f}s…")
    frames = load_frames(source, args.frames, args.start)
    height, width = frames[0].shape[:2]
    print(f"{len(frames)} frames at {width}x{height}")

    if args.grid:
        run_grid(frames,
                 ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo12s.pt", "yolo12m.pt"],
                 [640, 960, 1280])
    if args.stages:
        run_stages(frames, args.model, args.imgsz)
    if not args.grid and not args.stages:
        result = time_detector(frames, args.model, args.imgsz)
        print(f"\n{args.model} @ imgsz={args.imgsz}")
        print(f"  {result['ms_mean']:.1f} ms/frame  ({result['fps']:.1f} fps)")
        print(f"  median {result['ms_median']:.1f} ms, p95 {result['ms_p95']:.1f} ms")
        print(f"  {result['vehicles']:.1f} vehicles found per frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
