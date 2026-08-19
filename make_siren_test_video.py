"""
make_siren_test_video.py — paint a flashing beacon onto real footage
====================================================================

Testing emergency-vehicle handling honestly requires an emergency vehicle, and
one does not drive past on request. This script manufactures the next best
thing: it takes a clip you already have, picks a real vehicle out of it, and
composites a physically plausible light bar onto that vehicle's roof — the
right colour, the right flash rate, with the bloom and halo a real beacon
throws on a night camera.

The result is a video where exactly **one** vehicle is an emergency vehicle and
every other one is genuine ordinary traffic, which is precisely the material
needed to measure both halves of the problem at once: does the system catch the
one, and does it leave the other forty alone.

Usage
-----
    python make_siren_test_video.py --source "vedios/Test1_dark _long_vedio.mp4"
    python make_siren_test_video.py --source clip.mp4 --colour red --rate 1.5
    python make_siren_test_video.py --source clip.mp4 --colour redblue --seconds 25

The chosen track id and frame range are printed, so the detection run that
follows can be checked against the truth rather than eyeballed.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from traffic_vision import VehicleDetector, VisionConfig  # noqa: E402

#: BGR emission colours, chosen to match how a camera actually records each
#: lamp: a blue LED bar reads cyan-ish, a red one reads almost pure red.
BEACON_COLOURS: dict[str, tuple[int, int, int]] = {
    "blue": (255, 150, 40),
    "red": (35, 30, 255),
}


def _flash_on(elapsed: float, rate: float, duty: float, pattern: str) -> bool:
    """Return whether the lamp is lit at ``elapsed`` seconds.

    ``pattern="double"`` reproduces the two-quick-flashes-then-a-gap rhythm of
    modern LED light bars, which is what most beacons on the road actually do.
    """
    phase = (elapsed * rate) % 1.0
    if pattern == "double":
        return phase < duty * 0.45 or 0.5 <= phase < 0.5 + duty * 0.45
    return phase < duty


def _paint_beacon(frame: np.ndarray, centre: tuple[int, int], radius: float,
                  colour: tuple[int, int, int], strength: float = 1.0) -> None:
    """Add one glowing lamp to the frame, in place.

    Two superimposed Gaussians: a tight, saturating core and a wide, faint
    halo. Emission is *added* to the scene rather than drawn over it, so the
    lamp brightens whatever it lands on exactly as a real light does — and
    clips to white in the middle, which is the behaviour the detector's
    colour-dominance test has to cope with in real footage.
    """
    height, width = frame.shape[:2]
    reach = int(max(3.0, radius * 4.5))
    x0, x1 = max(0, centre[0] - reach), min(width, centre[0] + reach + 1)
    y0, y1 = max(0, centre[1] - reach), min(height, centre[1] + reach + 1)
    if x1 <= x0 or y1 <= y0:
        return

    ys, xs = np.mgrid[y0:y1, x0:x1]
    squared = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2
    core = np.exp(-squared / (2.0 * max(radius, 1.0) ** 2))
    halo = np.exp(-squared / (2.0 * max(radius * 2.6, 2.0) ** 2))
    glow = (strength * (1.35 * core + 0.30 * halo))[..., None]

    patch = frame[y0:y1, x0:x1].astype(np.float32)
    patch += glow * np.array(colour, dtype=np.float32)
    frame[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)


def _pick_track(boxes_by_frame: list[dict[int, tuple[float, float, float, float]]],
                min_size: int) -> int | None:
    """Choose the longest-lived, reasonably large track to promote."""
    spans: dict[int, int] = defaultdict(int)
    sizes: dict[int, float] = defaultdict(float)
    for boxes in boxes_by_frame:
        for track_id, box in boxes.items():
            side = min(box[2] - box[0], box[3] - box[1])
            if side >= min_size:
                spans[track_id] += 1
                sizes[track_id] = max(sizes[track_id], side)
    if not spans:
        return None
    # Longevity first — a beacon needs a few seconds on screen to be a beacon —
    # with apparent size breaking ties so the lamp lands on visible pixels.
    return max(spans, key=lambda t: (spans[t], sizes[t]))


def main() -> int:
    """Build the test clip; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="input video")
    parser.add_argument("--out", default=None, help="output video (default: <source>_siren.mp4)")
    parser.add_argument("--colour", "--color", dest="colour", default="blue",
                        choices=["blue", "red", "redblue"],
                        help="beacon colour: blue=police, red=ambulance, redblue=alternating")
    parser.add_argument("--rate", type=float, default=2.0, help="flashes per second")
    parser.add_argument("--duty", type=float, default=0.35, help="fraction of the cycle lit")
    parser.add_argument("--pattern", default="double", choices=["single", "double"],
                        help="single flash or the double-flash rhythm of an LED bar")
    parser.add_argument("--seconds", type=float, default=20.0, help="length of the clip")
    parser.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    parser.add_argument("--track", type=int, default=None,
                        help="promote this track id instead of choosing automatically")
    parser.add_argument("--min-size", type=int, default=40,
                        help="ignore vehicles smaller than this many pixels")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="lamp brightness; drop below 1 to simulate a "
                             "distant or partly hidden beacon")
    parser.add_argument("--size", type=float, default=0.075,
                        help="lamp radius as a fraction of the vehicle's width")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found")
        return 2
    out_path = Path(args.out) if args.out else source.with_name(
        f"{source.stem}_siren_{args.colour}.mp4")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"error: cannot open {source}")
        return 2
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_frame = int(args.start * fps)
    wanted = int(args.seconds * fps)

    # ── Pass 1: track, so the beacon can follow a real vehicle ──────────────
    print(f"[1/2] Tracking {source.name} …")
    detector = VehicleDetector(VisionConfig())
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    boxes_by_frame: list[dict[int, tuple[float, float, float, float]]] = []
    frames: list[np.ndarray] = []
    for index in range(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        raw, _ms = detector.track(frame)
        boxes_by_frame.append({tid: tuple(box) for box, tid, _lab, _c in raw})
        frames.append(frame)
        if index % 60 == 0:
            print(f"      frame {index}/{wanted}", end="\r")
    capture.release()
    if not frames:
        print("error: no frames read")
        return 2

    target = args.track if args.track is not None else _pick_track(boxes_by_frame,
                                                                   args.min_size)
    if target is None:
        print(f"error: no vehicle stayed larger than {args.min_size}px — "
              "try --min-size 24 or a different --start")
        return 2
    present = [i for i, b in enumerate(boxes_by_frame) if target in b]
    print(f"\n      promoting track #{target}: visible in {len(present)} of "
          f"{len(frames)} frames ({present[0]}–{present[-1]})")

    # ── Pass 2: composite the light bar ─────────────────────────────────────
    print(f"[2/2] Painting a {args.colour} beacon at {args.rate} Hz …")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    lamps = (["red", "blue"] if args.colour == "redblue" else [args.colour])
    lit_frames = 0

    for index, frame in enumerate(frames):
        box = boxes_by_frame[index].get(target)
        if box is not None:
            x1, y1, x2, y2 = box
            box_w = x2 - x1
            radius = max(1.2, box_w * args.size)
            elapsed = index / fps
            for lamp_index, lamp in enumerate(lamps):
                # Alternating bars run half a cycle out of phase, which is what
                # gives a police light bar its red-then-blue rhythm.
                offset = lamp_index / (2.0 * args.rate) if len(lamps) > 1 else 0.0
                if not _flash_on(elapsed + offset, args.rate, args.duty, args.pattern):
                    continue
                spread = 0.0 if len(lamps) == 1 else (lamp_index - 0.5) * box_w * 0.34
                centre = (int(round((x1 + x2) / 2.0 + spread)),
                          int(round(y1 + (y2 - y1) * 0.06)))
                _paint_beacon(frame, centre, radius, BEACON_COLOURS[lamp],
                              strength=args.strength)
                lit_frames += 1
        writer.write(frame)
    writer.release()

    duration = len(frames) / fps
    print(f"\nWrote {out_path}")
    print(f"  {len(frames)} frames, {duration:.1f}s at {fps:.1f} fps")
    print(f"  emergency vehicle = track #{target} (as tracked by this script; the "
          "detection run will assign its own ids)")
    print(f"  lamp lit on {lit_frames} frame-lamps, "
          f"{args.rate} Hz {args.pattern} flash, duty {args.duty}")
    print("  every other vehicle in the clip is ordinary traffic and must "
          "stay unflagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
