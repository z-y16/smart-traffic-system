"""
demo_emergency_clip.py — emergency recognition on any clip, on its own
======================================================================

Shows the emergency-vehicle half of the system working on arbitrary footage,
with everything else switched off. No region of interest, no calibration, no
speed, no congestion index, no hardware — just: *there is an ambulance in this
frame, and here is why we say so*.

That is a deliberately narrower claim than ``broadcast_server.py`` makes, and
it is the honest one to make about a clip filmed somewhere else. Speed needs a
calibrated ground plane and the region needs a junction to be drawn around;
neither exists for a video off the internet, and reporting either from an
uncalibrated source would be inventing numbers. Recognition needs neither —
in the live pipeline it already runs over *every* vehicle in shot, inside the
monitored region or not, because what is out there does not stop mattering at
the edge of the junction.

The three parts that do run are the live ones, imported rather than
reimplemented, so what is shown here is what the system would do:

    traffic_vision.VehicleDetector          YOLO detection + BoT-SORT tracking
    siren_vision.SirenDetector              is a beacon actually flashing
    emergency_vision.EmergencyAppearanceClassifier   what does the livery say
    emergency_vision.fuse                   the two cues, combined

Usage
-----
Find the emergency vehicles in a long clip, without writing any video::

    python demo_emergency_clip.py --source clip.mp4 --scan

Render an annotated segment of it::

    python demo_emergency_clip.py --source clip.mp4 --start 132 --seconds 25 \
        --out out_demo/emergency.mp4

Reading the output
------------------
Two levels of claim are drawn differently, because the system distinguishes
them and a demo that flattened them would be overstating the result:

    AMBULANCE ● PRIORITY   a beacon was seen switching on and off at a
                           beacon-like rate. This is what would hold a
                           junction.
    AMBULANCE?             recognised by livery, but no beacon could be read.
                           Reported, never given priority — an ambulance
                           parked outside a hospital is still an ambulance.

In daylight the second is the common case and is not a fault: a beacon bright
enough to register at all blooms to white on the sensor, and white fails the
red/blue purity gates by design, so the livery is what carries the decision
there. See ``check_siren_detection.py`` for the per-vehicle version of that
diagnosis.

Compilations
------------
``--scene-cuts`` (on by default) watches for hard cuts and resets the tracker,
the beacon history and the livery beliefs when it sees one. Without it, a clip
that jumps between unrelated scenes carries one vehicle's flash history onto
whatever the next shot puts in the same track id, which invents beacons that
were never there.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from emergency_vision import (  # noqa: E402
    EmergencyAppearanceClassifier,
    EmergencyConfig,
    EmergencyVerdict,
    fuse,
)
from siren_vision import SirenConfig, SirenDetector  # noqa: E402
from traffic_vision import VehicleDetector, VisionConfig  # noqa: E402

# Drawing — BGR. Priority is the loud one; everything else stays quiet so the
# eye goes to the vehicle the system is actually making a claim about.
COLOUR_PRIORITY = (60, 60, 255)      # red   — beacon confirmed, lane cleared
COLOUR_REPORTED = (0, 190, 255)      # amber — recognised, no beacon read
COLOUR_ORDINARY = (150, 150, 150)    # grey  — tracked, unremarkable
FONT = cv2.FONT_HERSHEY_DUPLEX


@dataclass
class Sighting:
    """Everything worth reporting about one emergency vehicle, over its life."""

    key: tuple[int, int]                 # (scene, track id)
    label: str = "vehicle"               # what YOLO called it
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames: int = 0
    priority_frames: int = 0
    confidence: float = 0.0
    rate_hz: float = 0.0
    types: Counter = field(default_factory=Counter)
    evidence: Counter = field(default_factory=Counter)
    #: Evidence recorded only on the frames that actually earned priority.
    #: Kept apart from :attr:`evidence` because the two answer different
    #: questions and the mode of the second is not an answer to the first: a
    #: beacon usually confirms part-way through a pass, so a vehicle can spend
    #: most of its frames on livery alone and still be given priority. Reading
    #: the overall mode there prints "PRIORITY … livery-no-siren", which states
    #: the opposite of what happened.
    priority_evidence: Counter = field(default_factory=Counter)

    @property
    def vehicle_type(self) -> str:
        return self.types.most_common(1)[0][0] if self.types else "unknown"

    @property
    def best_evidence(self) -> str:
        counter = self.priority_evidence if self.had_priority else self.evidence
        return counter.most_common(1)[0][0] if counter else "none"

    @property
    def had_priority(self) -> bool:
        return self.priority_frames > 0


def distinct_passes(sightings: list[Sighting], gap: float = 1.0) -> int:
    """Estimate how many *vehicles* the sightings represent, not track ids.

    One vehicle does not reliably produce one track. A vehicle that is briefly
    occluded, or that the tracker loses during a fast pan, comes back with a
    new id and is counted again — on the segment this was built against, one
    police car produced two overlapping detections at once. Counting ids would
    therefore overstate the result, which is the wrong direction for a number
    whose whole purpose is to be believed.

    Sightings are merged when they are the same type, in the same scene, and
    their spans overlap or fall within ``gap`` seconds of each other. That is
    an estimate and not a ground truth: two genuinely separate ambulances in
    convoy, close together in one shot, merge into one. It errs low, which is
    the honest direction to err.
    """
    total = 0
    by_group: dict[tuple[int, str], list[Sighting]] = {}
    for sighting in sightings:
        by_group.setdefault((sighting.key[0], sighting.vehicle_type), []).append(sighting)
    for group in by_group.values():
        group.sort(key=lambda s: s.first_seen)
        reach = -1e9
        for sighting in group:
            if sighting.first_seen > reach + gap:
                total += 1
            reach = max(reach, sighting.last_seen)
    return total


def timestamp(seconds: float) -> str:
    """Format a video offset as m:ss.s — how you would scrub to it."""
    return f"{int(seconds // 60):d}:{seconds % 60:04.1f}"


def scene_changed(previous: np.ndarray | None, frame: np.ndarray,
                  threshold: float) -> tuple[bool, np.ndarray]:
    """Detect a hard cut, and return the signature to compare against next.

    The signature is a hue/saturation histogram, and the test is how well it
    correlates with the previous frame's. What it must *not* do is fire on a
    pan, because the cost of a false cut is high and specific: the beacon
    history is thrown away, and a beacon needs about two seconds of it to be
    confirmed, so an oversensitive detector silently converts every siren into
    "no siren".

    A pixelwise difference cannot do this job. Measured over all 21082 sampled
    frames of the compilation this was built against, mean absolute difference
    on a grey thumbnail has its 90th percentile at 27 levels — a camera panning
    to follow a vehicle past trees moves more of the picture than most cuts do
    — so a threshold low enough to catch cuts fires thousands of times:

        ==========================  ============
        mean-abs-diff threshold      cuts called
        ==========================  ============
        22                                  4221
        40                                   308
        ==========================  ============

    against roughly one genuine cut per vehicle pass, some 40 in 11 minutes.
    Colour makes the two cases separable, because a pan keeps the palette it
    started with: over the first minute, correlation held above 0.955 for 99%
    of frames and collapsed below 0.5 on all three real cuts.

    The default sits at 0.50 rather than anywhere nearer those non-cut values
    because the gap is not uniformly wide. A *whip pan* — the camera snapping
    round to follow a vehicle, which this footage does constantly and a fixed
    junction camera never does — smears the frame to a blur and drags the
    correlation down to about 0.75. Over the 35 s segment rendered from this
    compilation, a 0.70 threshold called seven cuts where there were two, and
    the five extra ones each restarted the track ids mid-pass: one police car
    came out as two overlapping detections. At 0.50 the same segment gives
    exactly the two real cuts.
    """
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    signature = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(signature, signature)
    if previous is None:
        return False, signature
    correlation = float(cv2.compareHist(previous, signature, cv2.HISTCMP_CORREL))
    return correlation < threshold, signature


def draw_vehicle(frame: np.ndarray, bbox, verdict: EmergencyVerdict,
                 track_id: int, rate_hz: float) -> None:
    """Draw one tracked vehicle, at the loudness its verdict has earned."""
    x1, y1, x2, y2 = (int(v) for v in bbox)

    if verdict.priority:
        colour, thickness = COLOUR_PRIORITY, 3
    elif verdict.is_emergency or verdict.suspected:
        colour, thickness = COLOUR_REPORTED, 2
    else:
        # Ordinary traffic is drawn thin and unlabelled. It is here to show
        # that the system is looking at everything and flagging almost none of
        # it — a detector that boxed only the ambulance would prove nothing.
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_ORDINARY, 1)
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    name = verdict.display_name or "EMERGENCY"
    text = f"{name}  {verdict.confidence:.2f}"
    if verdict.priority:
        text = f"{name}  PRIORITY  {verdict.confidence:.2f}"
        if rate_hz:
            text += f"  {rate_hz:.1f}Hz"

    (width, height), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    top = max(0, y1 - height - 10)
    cv2.rectangle(frame, (x1, top), (x1 + width + 12, top + height + 10), colour, -1)
    cv2.putText(frame, text, (x1 + 6, top + height + 2), FONT, 0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(frame, f"#{track_id}  {verdict.evidence}",
                (x1 + 2, min(frame.shape[0] - 6, y2 + 16)), FONT, 0.45, colour, 1,
                cv2.LINE_AA)


def draw_banner(frame: np.ndarray, active: list[EmergencyVerdict], tracked: int,
                now: float, fps_now: float, backend: str) -> None:
    """Draw the running status across the top of the frame.

    The two claims are spelled out rather than abbreviated, because the whole
    point of the demo is that the system distinguishes them, and "AMBULANCE?"
    only carries that distinction to someone who already knows the convention.
    """
    height, width = frame.shape[:2]
    priority = [v for v in active if v.priority]
    reported = [v for v in active if v.is_emergency and not v.priority]

    def names(verdicts: list[EmergencyVerdict]) -> str:
        return ", ".join(sorted({v.display_name.rstrip("?") for v in verdicts}))

    if priority:
        colour = COLOUR_PRIORITY
        message = f"{names(priority)} - PRIORITY (beacon confirmed)"
    elif reported:
        colour = COLOUR_REPORTED
        message = f"{names(reported)} - RECOGNISED (no beacon read)"
    else:
        colour = (70, 70, 70)
        message = "no emergency vehicle"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 74), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (0, 0), (width, 74), colour, 2)
    cv2.putText(frame, message, (18, 34), FONT, 0.85, colour, 2, cv2.LINE_AA)
    # The vehicle count is the control: it is what shows the recogniser is
    # looking at all the traffic and singling out almost none of it.
    flagged = len(priority) + len(reported)
    cv2.putText(frame,
                f"t={timestamp(now)}   tracking {tracked} vehicles, {flagged} flagged"
                f"   livery: {backend}   {fps_now:4.1f} fps processing"
                f"   no ROI - no speed - recognition only",
                (18, 62), FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    """Run the recogniser over a clip; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="video file to analyse")
    parser.add_argument("--out", default=None,
                        help="write an annotated mp4 here (omit to only report)")
    parser.add_argument("--scan", action="store_true",
                        help="report only, no video — for finding the events in a long clip")
    parser.add_argument("--start", type=float, default=0.0, help="start offset, seconds")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="how much to analyse (0 = to the end)")
    parser.add_argument("--stride", type=int, default=2,
                        help="process every Nth frame; keep the effective rate above "
                             "8 Hz or a beacon's flashes can be missed entirely")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--backend", default="clip", choices=("clip", "trained", "auto"),
                        help="livery recogniser; clip is the one that travels")
    parser.add_argument("--require-siren", dest="require_siren",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="a flashing beacon is needed for PRIORITY (default). "
                             "--no-require-siren lets livery alone grant it")
    parser.add_argument("--scene-cuts", dest="scene_cuts",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="reset tracking and beacon history on hard cuts")
    parser.add_argument("--cut-threshold", type=float, default=0.50,
                        help="colour-histogram correlation below which a frame is "
                             "called a cut; lower is more forgiving of pans")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="resize the written video by this factor")
    parser.add_argument("--save-shots", default=None,
                        help="write one still per flagged vehicle here")
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
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.stride)
    if args.start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
    start_frame = int(args.start * fps)
    limit = int(args.seconds * fps) if args.seconds else 10 ** 9

    print(f"source   {source.name}  {width}x{height} @ {fps:.2f} fps, "
          f"{total / fps / 60:.1f} min")
    print(f"sampling every {stride} frame(s) -> {fps / stride:.1f} Hz")
    if fps / stride < 8.0:
        print("  WARNING: below 8 Hz a beacon's flashes can be missed entirely")

    detector = VehicleDetector(VisionConfig(model_path=args.model, imgsz=args.imgsz,
                                            conf=args.conf))
    siren = SirenDetector(SirenConfig())
    appearance = EmergencyAppearanceClassifier(EmergencyConfig(backend=args.backend))
    print(f"[LIVERY] {appearance.status}")
    print(f"[FUSE]   priority requires a flashing beacon: {args.require_siren}")

    writer = None
    out_path = Path(args.out) if args.out else None
    if out_path and not args.scan:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        size = (int(width * args.scale), int(height * args.scale))
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / stride, size)
        print(f"writing  {out_path}  {size[0]}x{size[1]} @ {fps / stride:.1f} fps")

    shots_dir = Path(args.save_shots) if args.save_shots else None
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)

    sightings: dict[tuple[int, int], Sighting] = {}
    shot_taken: set[tuple[int, int]] = set()
    ages: Counter = Counter()
    all_tracks: set[tuple[int, int]] = set()
    scene = 0
    cuts = 0
    previous_thumb: np.ndarray | None = None
    read = 0
    processed = 0
    started = time.perf_counter()
    recent = time.perf_counter()
    fps_now = 0.0

    while read < limit:
        ok, frame = capture.read()
        if not ok:
            break
        read += 1
        if (read - 1) % stride:
            continue
        processed += 1
        now = (start_frame + read) / fps          # the video's own clock

        if args.scene_cuts:
            cut, previous_thumb = scene_changed(previous_thumb, frame,
                                                args.cut_threshold)
            if cut:
                # A new shot is a new world: ids from the old one must not be
                # matched onto it, and neither must its beacon history.
                cuts += 1
                scene += 1
                detector.reset_tracker()
                appearance.reset()
                siren = SirenDetector(SirenConfig())

        tracked, _ms = detector.track(frame)
        appearance.begin_frame()
        looks = appearance.classify_frame(
            frame, [(track_id, bbox) for bbox, track_id, _l, _c in tracked], now)
        siren.begin_frame(frame, now)

        active: list[EmergencyVerdict] = []
        confirmed_now = 0
        for bbox, track_id, label, _conf in tracked:
            key = (scene, track_id)
            ages[key] += 1
            beacon = siren.update(frame, bbox, track_id, now, label)
            verdict = fuse(looks.get(track_id, EmergencyVerdict()), beacon,
                           require_siren=args.require_siren)
            # Same gate as the live pipeline: a track has to survive a few
            # frames before it is allowed to say anything, so a one-frame
            # ghost never reaches the report.
            if ages[key] < 3:
                continue
            all_tracks.add(key)
            confirmed_now += 1

            if verdict.is_emergency or verdict.suspected:
                active.append(verdict)
                sighting = sightings.get(key)
                if sighting is None:
                    sighting = Sighting(key=key, label=label, first_seen=now)
                    sightings[key] = sighting
                sighting.last_seen = now
                sighting.frames += 1
                sighting.priority_frames += int(verdict.priority)
                sighting.confidence = max(sighting.confidence, verdict.confidence)
                sighting.types[verdict.vehicle_type] += 1
                sighting.evidence[verdict.evidence] += 1
                if verdict.priority:
                    sighting.priority_evidence[verdict.evidence] += 1
                if beacon.active:
                    # Only a rate the detector actually accepted; the running
                    # estimate is non-zero long before it passes the gates, and
                    # printing that would imply a beacon reading that was in
                    # fact rejected.
                    sighting.rate_hz = max(sighting.rate_hz, beacon.rate_hz)
                if shots_dir and key not in shot_taken:
                    shot_taken.add(key)
                    x1, y1, x2, y2 = (int(v) for v in bbox)
                    pad = 30
                    shot = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    if shot.size:
                        cv2.imwrite(str(shots_dir / f"s{scene}_id{track_id}_"
                                                    f"{verdict.vehicle_type}.png"), shot)

            if writer is not None:
                draw_vehicle(frame, bbox, verdict, track_id, beacon.rate_hz)

        siren.prune(now)
        appearance.prune(now)

        if processed % 10 == 0:
            elapsed = time.perf_counter() - recent
            fps_now = 10.0 / elapsed if elapsed else 0.0
            recent = time.perf_counter()
        if writer is not None:
            draw_banner(frame, active, confirmed_now, now, fps_now, appearance.backend)
            if args.scale != 1.0:
                frame = cv2.resize(frame, (int(width * args.scale),
                                           int(height * args.scale)))
            writer.write(frame)

        if processed % 200 == 0:
            flagged = sum(1 for s in sightings.values() if s.frames >= 3)
            print(f"  {timestamp(now)}  {processed} frames, {len(all_tracks)} vehicles, "
                  f"{flagged} flagged, {fps_now:.1f} fps", flush=True)

    capture.release()
    if writer is not None:
        writer.release()

    # ── Report ──────────────────────────────────────────────────────────────
    # A sighting that lasted a frame or two is noise; three sampled frames at
    # this stride is a fifth of a second of agreement.
    real = [s for s in sightings.values() if s.frames >= 3]
    real.sort(key=lambda s: s.first_seen)
    with_priority = [s for s in real if s.had_priority]
    elapsed = time.perf_counter() - started

    print(f"\n{'=' * 78}")
    print(f"{processed} frames analysed in {elapsed:.0f}s "
          f"({processed / elapsed:.1f} fps), {cuts} scene cuts")
    print(f"{len(all_tracks)} vehicles tracked   "
          f"{len(real)} recognised as emergency   "
          f"{len(with_priority)} given priority")
    print(f"~{distinct_passes(real)} distinct emergency vehicles once track-id churn "
          f"is merged (see distinct_passes)")
    print(f"{'=' * 78}")

    if real:
        print(f"\n{'when':>12}  {'for':>5}  {'type':<11} {'claim':<9} "
              f"{'conf':>5}  {'rate':>6}  evidence")
        for sighting in real:
            span = sighting.last_seen - sighting.first_seen
            claim = "PRIORITY" if sighting.had_priority else "reported"
            rate = f"{sighting.rate_hz:.1f}Hz" if sighting.rate_hz else "-"
            print(f"{timestamp(sighting.first_seen):>12}  {span:4.1f}s  "
                  f"{sighting.vehicle_type:<11} {claim:<9} "
                  f"{sighting.confidence:5.2f}  {rate:>6}  {sighting.best_evidence}")
    else:
        print("\nnothing flagged.")

    kinds = Counter(s.vehicle_type for s in real)
    if kinds:
        print("\nby type: " + ", ".join(f"{n} x {k}" for k, n in kinds.most_common()))
    if shots_dir and shot_taken:
        print(f"stills written to {shots_dir}")
    if out_path and not args.scan:
        print(f"video written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
