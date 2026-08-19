"""
check_siren_detection.py — run the emergency detector over a clip and explain
============================================================================

Answers the two questions you actually have after filming: **did it catch the
emergency vehicle**, and **did it flag anything it shouldn't have** — plus, for
every vehicle it did not flag, *why not*.

That last part is the point. "No emergency vehicles detected" is not a useful
answer when you were pointing the camera at an ambulance. This prints the
reason each vehicle was rejected, so a miss is diagnosable instead of
mysterious:

    too-dim                 no red or blue light bright enough to be a lamp
    steady-light            coloured light present, but it never switches off
    shallow-modulation      it dims but does not go out — usually a reflection
    gradual-not-switched    it ramps rather than switches — a car passing a
                            street light does this; a beacon never does
    not-repeating           it switched, but fewer than three times
    implausible-rate        flashing faster or slower than any real beacon
    red-below-roofline      red light, but on the body — brake or tail lights
    window-too-short        the vehicle was not in shot long enough to judge
    too-few-samples         same, in frames rather than seconds

Usage
-----
    python check_siren_detection.py --source "vedios/morning_test.mp4"
    python check_siren_detection.py --source clip.mp4 --save-crops out/
    python check_siren_detection.py --source clip.mp4 --start 30 --seconds 40

Two limits are worth knowing before reading the output.

**This tool only exercises the beacon detector.** The live pipeline also
recognises vehicles by their livery, and reports an emergency on that alone
when no beacon can be read — so a vehicle listed here as ``too-dim`` may still
be flagged by ``traffic_vision.py``. To see what the system as a whole would
do, run the clip through ``broadcast_server.py``.

**A daylight beacon usually cannot be read as a colour at all.** Bright enough
to register as a lamp, it blooms to white on the sensor, and white fails the
red and blue purity gates by design — measured on a daylight ambulance, the
beacon channels peaked at 0.04 while its livery scored 0.95. So ``too-dim`` in
daylight is generally the expected answer rather than a fault, and the livery
is what carries the decision there.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from siren_vision import SirenConfig, SirenDetector  # noqa: E402
from traffic_vision import VehicleDetector, VisionConfig  # noqa: E402


def main() -> int:
    """Analyse one clip; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="video file to analyse")
    parser.add_argument("--start", type=float, default=0.0, help="start offset, seconds")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="how much to analyse (0 = to the end)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-crops", default=None,
                        help="write a picture of each flagged vehicle here")
    parser.add_argument("--siren-blue", default="police")
    parser.add_argument("--siren-red", default="ambulance")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"error: {source} not found")
        return 2
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"error: cannot open {source}")
        return 2

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if args.start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    limit = int(args.seconds * fps) if args.seconds else 10 ** 9

    detector = VehicleDetector(VisionConfig(imgsz=args.imgsz, conf=args.conf))
    siren = SirenDetector(SirenConfig(blue_type=args.siren_blue,
                                      red_type=args.siren_red))
    crops_dir = Path(args.save_crops) if args.save_crops else None
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    flagged: dict[int, dict] = {}
    reasons: dict[int, Counter] = {}
    ages: Counter = Counter()
    saved: set[int] = set()
    frames = 0

    print(f"analysing {source.name} …")
    while frames < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        now = frames / fps            # the video's own clock
        tracked, _ms = detector.track(frame)
        siren.begin_frame(frame, now)

        for bbox, track_id, label, _conf in tracked:
            ages[track_id] += 1
            verdict = siren.update(frame, bbox, track_id, now, label)
            diagnostics = siren.diagnostics(track_id)
            if diagnostics:
                counter = reasons.setdefault(track_id, Counter())
                counter[diagnostics["red"]["reason"]] += 1
                counter[diagnostics["blue"]["reason"]] += 1
            if verdict.active:
                record = flagged.setdefault(track_id, {
                    "frames": 0, "colour": Counter(), "type": Counter(),
                    "confidence": 0.0, "rate": 0.0, "label": label})
                record["frames"] += 1
                record["colour"][verdict.colour] += 1
                record["type"][verdict.vehicle_type] += 1
                record["confidence"] = max(record["confidence"], verdict.confidence)
                record["rate"] = max(record["rate"], verdict.rate_hz)
                if crops_dir and track_id not in saved:
                    saved.add(track_id)
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    pad = 40
                    shot = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    if shot.size:
                        cv2.imwrite(str(crops_dir /
                                        f"id{track_id}_{verdict.colour}_f{frames}.png"),
                                    cv2.resize(shot, None, fx=3, fy=3,
                                               interpolation=cv2.INTER_NEAREST))
        siren.prune(now)
    capture.release()

    confirmed = [t for t, n in ages.items() if n >= 3]
    print(f"\n{frames} frames, {len(confirmed)} vehicles tracked, "
          f"sampled at {siren.sample_rate():.1f} Hz")
    if siren.undersampled:
        print("  WARNING: below 8 Hz a beacon's flashes can be missed entirely. "
              "Lower --imgsz, or use a smaller model.")

    print(f"\nFLAGGED AS EMERGENCY: {len(flagged)}")
    for track_id, record in sorted(flagged.items(), key=lambda kv: -kv[1]["frames"]):
        colour = record["colour"].most_common(1)[0][0]
        kind = record["type"].most_common(1)[0][0]
        print(f"  vehicle #{track_id:<5} {record['label']:<8} {kind:<11} "
              f"{colour:<9} {record['rate']:.1f} Hz  "
              f"confidence {record['confidence']:.2f}  "
              f"({record['frames']} frames)")
    if not flagged:
        print("  (none)")

    # What stopped everything else. Reported per vehicle rather than as one
    # total, because "23 vehicles were too dim" is not the same information as
    # "the one you care about was too dim".
    print("\nWHY THE REST WERE NOT FLAGGED (most common reason per vehicle)")
    summary: Counter = Counter()
    for track_id in confirmed:
        if track_id in flagged:
            continue
        counter = reasons.get(track_id)
        if not counter:
            continue
        # "flashing" as a per-channel reason means one colour did flash but the
        # vehicle still did not qualify — worth seeing separately.
        best = [(reason, n) for reason, n in counter.most_common()
                if reason != "no-data"]
        if best:
            summary[best[0][0]] += 1
    for reason, count in summary.most_common():
        print(f"  {reason:<24} {count:>4} vehicles")
    if crops_dir and saved:
        print(f"\ncrops of flagged vehicles written to {crops_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
