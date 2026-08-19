"""
train_emergency_classifier.py — train a livery recogniser for this camera
=========================================================================

Builds and trains the classifier that names an emergency vehicle from its
appearance: **ambulance / police / fire truck / ordinary**. It is an
alternative to the zero-shot CLIP recogniser, not a replacement for the siren
detector — what opens the priority lane is still a working beacon
(``siren_vision.py``). This model only answers *what kind of vehicle is that*,
which matters when a red beacon could be either an ambulance or a fire engine.

Why train at all, when CLIP needs no data
-----------------------------------------
CLIP knows what an ambulance looks like in any country and costs nothing to
set up, which is why it is still the default. What it does not know is *this*
camera: a 60-pixel dark shape at 3 a.m., smeared by motion blur and chewed by
H.264. A model trained partly on crops taken from this system's own footage
does know that, and it runs in about a tenth of the time.

The four stages
---------------
``fetch``   Download the vehicle classes of ImageNet-1k from Hugging Face:
            ambulance, fire engine and police van as positives, and fourteen
            confusable ordinary classes — white vans, taxis with roof signs,
            tow trucks with amber beacons, school buses — as negatives. Public,
            ungated, no account needed.

``mine``    Run the project's own detector over the clips in ``vedios/`` and
            save real vehicle crops as *ordinary*. These are the most valuable
            training images available, because they are exactly what the
            previous system was getting wrong, and no public dataset contains
            anything like them.

``build``   Bake each source photograph into degraded variants that imitate
            this camera — resolution thrown away, brightness pulled down,
            motion blur, street-light colour cast, JPEG artefacts. **This is
            the stage that decides whether the result is worth anything.**
            Training on clean daylight photographs and deploying on night
            crops is precisely the mistake that made the old system call 29%
            of ordinary traffic "police".

``train``   Fine-tune a YOLO classification model and copy the best weights to
            ``emergency_cls.pt`` in the project root.

Usage
-----
    python train_emergency_classifier.py                  # all four stages
    python train_emergency_classifier.py --stage mine     # just re-mine crops
    python train_emergency_classifier.py --stage build --stage train
    python train_emergency_classifier.py --epochs 60 --base yolo11m-cls.pt

Then run the system with it:

    python broadcast_server.py --emergency-backend trained

Retraining after new footage
----------------------------
Daytime clips are the ones worth adding first: everything mined so far is
night, so the model has only ever seen this camera in the dark.

    python train_emergency_classifier.py --stage mine --stage build \\
        --stage train --mine-clips "vedios/morning_run.mp4"

Every clip named must contain **no emergency vehicle**. This stage labels
whatever it finds as *ordinary*, so one ambulance in the footage teaches the
model that ambulances are ordinary. With no ``--mine-clips``, the built-in
``EMERGENCY_FREE_CLIPS`` list is used.
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

PARQUET_URL = ("https://huggingface.co/datasets/{ds}/resolve/main/"
               "data/{split}-00000-of-00001.parquet")

#: ImageNet-1k classes, grouped into the labels this project uses. The
#: negatives are chosen for confusability rather than variety: a moving van has
#: an ambulance's body, a taxi has a roof sign, a tow truck carries a beacon,
#: and a school bus is a large vehicle covered in high-visibility markings.
IMAGENET_CLASSES: dict[str, list[str]] = {
    "ambulance": ["mlnomad/imnet1k_ambulance"],
    "fire truck": ["mlnomad/imnet1k_fire_engine_fire_truck"],
    "police": ["mlnomad/imnet1k_police_van_police_wagon_paddy_wagon_"
               "patrol_wagon_wagon_black_Maria"],
    "ordinary": [
        "mlnomad/imnet1k_moving_van",
        "mlnomad/imnet1k_cab_hack_taxi_taxicab",
        "mlnomad/imnet1k_tow_truck_tow_car_wrecker",
        "mlnomad/imnet1k_minivan",
        "mlnomad/imnet1k_school_bus",
        "mlnomad/imnet1k_pickup_pickup_truck",
        "mlnomad/imnet1k_garbage_truck_dustcart",
        "mlnomad/imnet1k_trailer_truck_tractor_trailer_trucking_rig_rig_"
        "articulated_lorry_semi",
        "mlnomad/imnet1k_sports_car_sport_car",
        "mlnomad/imnet1k_beach_wagon_station_wagon_wagon_estate_car_beach_"
        "waggon_station_waggon_waggon",
        "mlnomad/imnet1k_jeep_landrover",
        "mlnomad/imnet1k_minibus",
        "mlnomad/imnet1k_limousine_limo",
        "mlnomad/imnet1k_convertible",
    ],
}

#: Clips known to contain no emergency vehicle. Only these are mined for
#: "ordinary" crops — mining a clip that contains a real ambulance would label
#: the one vehicle that matters as the opposite of what it is.
EMERGENCY_FREE_CLIPS = (
    "Test1_dark _long_vedio.mp4",
    "Test2_dark_zoomed_in.mp4",
    "WhatsApp Video 2026-08-07 at 10.45.16 PM.mp4",
    "WhatsApp Video 2026-08-07 at 10.45.39 PM.mp4",
    "WhatsApp Video 2026-08-07 at 10.45.48 PM.mp4",
    "WhatsApp Video 2026-08-07 at 11.13.03 PM.mp4",
)

#: Folder name marking crops that are already in the deployment domain and so
#: need no synthetic degradation.
NATIVE_SOURCE = "own_footage"

TRAIN_SIZE = 160
VAL_FRACTION = 0.15
DEGRADED_VARIANTS = 3

#: Per-source caps, set so the finished dataset lands near 2:1 ordinary to
#: emergency. There are fourteen ordinary ImageNet classes against one per
#: emergency type, so an equal cap would bury the positives; and crops from
#: this camera are never capped, because they are the ones that teach the model
#: what its actual input looks like.
MAX_PER_EMERGENCY_SOURCE = 1500
MAX_PER_ORDINARY_SOURCE = 500


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — FETCH
# ═══════════════════════════════════════════════════════════════════════════


def _fetch_one(label: str, dataset: str, raw_dir: Path) -> tuple[str, str, int]:
    """Download one ImageNet class and write its images as small JPEGs."""
    import pandas as pd
    from PIL import Image

    slug = dataset.split("/")[-1].replace("imnet1k_", "")[:40]
    target = raw_dir / label / slug
    target.mkdir(parents=True, exist_ok=True)
    written = 0

    for split in ("train", "validation"):
        try:
            with urllib.request.urlopen(
                    PARQUET_URL.format(ds=dataset, split=split), timeout=600) as response:
                blob = response.read()
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {slug} {split}: {exc}")
            continue
        frame = pd.read_parquet(io.BytesIO(blob))
        for index, row in frame.iterrows():
            try:
                cell = row["image"]
                data = cell["bytes"] if isinstance(cell, dict) else cell
                image = Image.open(io.BytesIO(data)).convert("RGB")
                # 320 px is far more than the 40-150 px the model will ever
                # see, and keeps the working set to a few hundred megabytes.
                image.thumbnail((320, 320), Image.LANCZOS)
                image.save(target / f"{split}_{index:05d}.jpg", quality=88)
                written += 1
            except Exception:  # noqa: BLE001 - a few ImageNet files are corrupt
                continue
    return label, slug, written


def stage_fetch(raw_dir: Path, workers: int) -> None:
    """Download every ImageNet class listed in :data:`IMAGENET_CLASSES`."""
    jobs = [(label, ds) for label, group in IMAGENET_CLASSES.items() for ds in group]
    print(f"[fetch] {len(jobs)} ImageNet classes -> {raw_dir}")
    totals: Counter = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for label, slug, count in pool.map(
                lambda job: _fetch_one(job[0], job[1], raw_dir), jobs):
            totals[label] += count
            print(f"    {label:<12} {slug:<42} {count:>5} images", flush=True)
    print(f"[fetch] done: {dict(totals)}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — MINE
# ═══════════════════════════════════════════════════════════════════════════


def stage_mine(raw_dir: Path, videos_dir: Path, *, clips: list[str] | None = None,
               frame_stride: int = 2, per_track_stride: int = 11,
               min_side: int = 24, max_frames: int = 1400) -> None:
    """Save real vehicle crops from this project's own clips as *ordinary*.

    Crops are taken per *track* rather than per detection, and only one every
    ``per_track_stride`` observations, so the result is a varied set rather
    than a thousand near-identical pictures of the same three cars.

    ``clips`` overrides :data:`EMERGENCY_FREE_CLIPS`, and every clip named must
    be free of emergency vehicles — this stage labels everything it finds as
    *ordinary*, so one ambulance in the footage teaches the model that
    ambulances are ordinary.
    """
    from traffic_vision import VehicleDetector, VisionConfig

    out = raw_dir / "ordinary" / NATIVE_SOURCE
    out.mkdir(parents=True, exist_ok=True)
    detector = VehicleDetector(VisionConfig(imgsz=1280, conf=0.25))
    total = 0

    names = clips if clips else list(EMERGENCY_FREE_CLIPS)
    print(f"[mine] {len(names)} clip(s), all assumed free of emergency vehicles")
    for name in names:
        path = Path(name)
        if not path.is_absolute() and not path.exists():
            path = videos_dir / name
        if not path.exists():
            print(f"    - {name}: not found, skipped")
            continue
        detector.reset_tracker()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            print(f"    ! {name}: cannot open")
            continue

        stem = path.stem.replace(" ", "_")[:24]
        seen: Counter = Counter()
        index = used = saved = 0
        while used < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            index += 1
            if (index - 1) % frame_stride:
                continue
            used += 1
            height, width = frame.shape[:2]
            tracked, _ms = detector.track(frame)
            for bbox, track_id, label, _conf in tracked:
                seen[track_id] += 1
                if seen[track_id] % per_track_stride:
                    continue
                x1, y1, x2, y2 = (float(v) for v in bbox)
                box_w, box_h = x2 - x1, y2 - y1
                if min(box_w, box_h) < min_side:
                    continue
                x1 = max(0, int(x1 - box_w * 0.06))
                y1 = max(0, int(y1 - box_h * 0.06))
                x2 = min(width, int(x2 + box_w * 0.06))
                y2 = min(height, int(y2 + box_h * 0.06))
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    cv2.imwrite(str(out / f"{stem}_f{index:05d}_t{track_id}_{label}.jpg"),
                                crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    saved += 1
        capture.release()
        total += saved
        print(f"    {stem:<26} {saved:>5} crops from {used} frames", flush=True)

    print(f"[mine] {total} ordinary crops from this camera -> {out}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILD
# ═══════════════════════════════════════════════════════════════════════════


def _motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    """Smear the image along one direction, as a moving vehicle smears."""
    if length < 3:
        return image
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    total = float(kernel.sum())
    return cv2.filter2D(image, -1, kernel / total) if total > 0 else image


def degrade(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean photograph look like something this camera produced.

    The operations are applied in the order a real camera applies them —
    downscale, blur, expose, tint, add sensor noise, then compress — because
    doing it in any other order produces artefacts that no camera makes and
    that the model would happily learn to rely on.
    """
    height, width = image.shape[:2]

    target = rng.randint(36, 130)
    scale = target / max(height, width)
    small = cv2.resize(image, (max(8, int(width * scale)), max(8, int(height * scale))),
                       interpolation=cv2.INTER_AREA)

    if rng.random() < 0.65:
        small = _motion_blur(small, rng.choice([3, 3, 5, 5, 7]), rng.uniform(0, 180))

    work = small.astype(np.float32)
    if rng.random() < 0.75:
        # Gamma pulls the midtones down; the black lift and gain reproduce a
        # sensor's floor, so the result is dim *and* low-contrast rather than
        # merely dark, which is what night footage actually looks like.
        work = 255.0 * np.power(np.clip(work / 255.0, 0, 1), rng.uniform(1.3, 3.0))
        work += rng.uniform(4, 26)
        work *= rng.uniform(0.55, 1.05)

    if rng.random() < 0.6:
        if rng.random() < 0.5:      # sodium street lighting: warm
            work *= np.array([rng.uniform(0.72, 0.92), rng.uniform(0.90, 1.00),
                              rng.uniform(1.02, 1.22)], np.float32)
        else:                       # LED street lighting: cool
            work *= np.array([rng.uniform(1.02, 1.20), rng.uniform(0.96, 1.04),
                              rng.uniform(0.82, 0.98)], np.float32)

    if rng.random() < 0.6:
        work += np.random.normal(0, rng.uniform(2, 11), work.shape)
    out = np.clip(work, 0, 255).astype(np.uint8)

    if rng.random() < 0.8:
        ok, buffer = cv2.imencode(".jpg", out,
                                  [cv2.IMWRITE_JPEG_QUALITY, rng.randint(22, 68)])
        if ok:
            out = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return out


def readable(image: np.ndarray, min_luma: float = 28.0,
             min_contrast: float = 12.0) -> bool:
    """Return whether a crop carries enough signal to be worth classifying.

    These are the same thresholds ``EmergencyConfig`` applies at run time, and
    they are applied here for the same reason. A degraded variant so dark that
    the pipeline would refuse to classify it must not appear in the training
    set labelled "ambulance": that trains the model to answer "ambulance" from
    noise, which is precisely the failure this whole exercise is correcting.
    Train on the distribution you will actually be asked about.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return bool(grey.mean() >= min_luma and grey.std() >= min_contrast)


def degrade_readable(image: np.ndarray, rng: random.Random,
                     attempts: int = 4) -> np.ndarray | None:
    """Degrade an image, retrying until the result is still readable."""
    for _ in range(attempts):
        candidate = degrade(image, rng)
        if readable(candidate):
            return candidate
    return None


def _letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Fit into a square without distorting the vehicle's proportions."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def stage_build(raw_dir: Path, cls_dir: Path, seed: int = 11) -> None:
    """Turn the raw images into a train/val classification dataset."""
    if not raw_dir.exists():
        sys.exit(f"no raw data at {raw_dir} — run --stage fetch first")
    if cls_dir.exists():
        shutil.rmtree(cls_dir)

    rng = random.Random(seed)
    counts: Counter = Counter()
    dropped: Counter = Counter()

    for label_dir in sorted(raw_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name.replace(" ", "_")
        for source_dir in sorted(label_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            native = source_dir.name == NATIVE_SOURCE
            files = sorted(source_dir.glob("*.jpg"))
            rng.shuffle(files)
            if not native:
                files = files[:MAX_PER_ORDINARY_SOURCE if label == "ordinary"
                              else MAX_PER_EMERGENCY_SOURCE]

            for path in files:
                image = cv2.imread(str(path))
                if image is None or image.size == 0:
                    continue
                split = "val" if rng.random() < VAL_FRACTION else "train"
                target = cls_dir / split / label
                target.mkdir(parents=True, exist_ok=True)
                stem = f"{source_dir.name}_{path.stem}"

                if native:
                    # Already the right domain. Kept whatever its quality,
                    # because "ordinary" is the safe answer for an unreadable
                    # crop and the run-time gate will skip those anyway.
                    cv2.imwrite(str(target / f"{stem}_n.jpg"),
                                _letterbox(image, TRAIN_SIZE))
                    counts[(split, label)] += 1
                    if split == "train" and rng.random() < 0.5:
                        variant = degrade_readable(image, rng)
                        if variant is not None:
                            cv2.imwrite(str(target / f"{stem}_d.jpg"),
                                        _letterbox(variant, TRAIN_SIZE))
                            counts[(split, label)] += 1
                    continue

                cv2.imwrite(str(target / f"{stem}_c.jpg"), _letterbox(image, TRAIN_SIZE))
                counts[(split, label)] += 1
                # The validation split is degraded too. Validating only on
                # clean photographs would report an accuracy that has nothing
                # to do with what the camera sees.
                for k in range(DEGRADED_VARIANTS):
                    variant = degrade_readable(image, rng)
                    if variant is None:
                        dropped[label] += 1
                        continue
                    cv2.imwrite(str(target / f"{stem}_d{k}.jpg"),
                                _letterbox(variant, TRAIN_SIZE))
                    counts[(split, label)] += 1

    print(f"[build] {cls_dir}")
    print(f"    {'class':<14}{'train':>9}{'val':>9}")
    for label in sorted({label for _split, label in counts}):
        print(f"    {label:<14}{counts[('train', label)]:>9}{counts[('val', label)]:>9}")
    print(f"    {'TOTAL':<14}"
          f"{sum(v for (s, _), v in counts.items() if s == 'train'):>9}"
          f"{sum(v for (s, _), v in counts.items() if s == 'val'):>9}")
    if dropped:
        print(f"    dropped as unreadable after degrading: {dict(dropped)}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════


def stage_train(cls_dir: Path, runs_dir: Path, output: Path, *, base: str,
                epochs: int, imgsz: int, batch: int, device: str) -> None:
    """Fine-tune a YOLO classification model and export the best weights."""
    if not (cls_dir / "train").exists():
        sys.exit(f"no dataset at {cls_dir} — run --stage build first")
    from ultralytics import YOLO

    print(f"[train] {base} on {cls_dir} for {epochs} epochs @ {imgsz}px")
    model = YOLO(base)
    model.train(
        data=str(cls_dir), epochs=epochs, imgsz=imgsz, batch=batch,
        device=device, project=str(runs_dir), name="emergency_cls",
        exist_ok=True, pretrained=True, patience=12, workers=4, verbose=True,

        # Augmentation is deliberately restrained, because for this task the
        # library's defaults destroy the signal. **Colour is the feature**: a
        # fire engine is red, an ambulance is white and yellow, a patrol car is
        # blue and white. Ultralytics defaults to ``auto_augment="randaugment"``
        # with ``hsv_s=0.7``, which recolours vehicles freely — inspecting the
        # first run's training batches showed a cyan police car and a green
        # ambulance — and no amount of training recovers a feature that has been
        # augmented away. Hue is therefore left almost alone, saturation is
        # nudged rather than swung, and RandAugment is off.
        auto_augment=None,
        hsv_h=0.005, hsv_s=0.25, hsv_v=0.4,

        # Brightness variation is handled properly in the build stage, by
        # simulating exposure rather than scaling pixels, so it is not needed
        # again here.
        #
        # Erasing punches black rectangles through the image; at the library's
        # default of 0.4 it regularly covered the markings that identify the
        # vehicle. Kept small, it still teaches robustness to a partly occluded
        # vehicle, which does happen on a busy road.
        erasing=0.1,

        # Vehicles appear at every heading but always the right way up, so
        # horizontal flips are free extra data and vertical ones are nonsense.
        fliplr=0.5, flipud=0.0, degrees=6.0, translate=0.06, scale=0.25,
    )

    best = runs_dir / "emergency_cls" / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"training produced no weights at {best}")
    shutil.copy(best, output)
    print(f"[train] copied {best} -> {output}")
    print("[train] use it with: python broadcast_server.py "
          "--emergency-backend trained")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    """Run the requested stages; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append",
                        choices=["fetch", "mine", "build", "train"],
                        help="run only this stage (repeatable); default is all four")
    parser.add_argument("--work", default=str(ROOT_DIR / "emergency_training"),
                        help="working directory for the dataset and training runs")
    parser.add_argument("--videos", default=str(ROOT_DIR / "vedios"),
                        help="folder of clips to mine ordinary crops from")
    parser.add_argument("--mine-clips", action="append", metavar="CLIP",
                        help="mine this clip instead of the built-in list "
                             "(repeatable). Every clip given must contain no "
                             "emergency vehicle — this stage labels whatever it "
                             "finds as ordinary")
    parser.add_argument("--out", default=str(ROOT_DIR / "emergency_cls.pt"),
                        help="where to write the trained weights")
    parser.add_argument("--base", default="yolo11s-cls.pt",
                        help="classification model to fine-tune")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=TRAIN_SIZE,
                        help="must match EmergencyConfig.trained_imgsz")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0", help="0 | cpu")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel downloads during fetch")
    args = parser.parse_args()

    stages = args.stage or ["fetch", "mine", "build", "train"]
    work = Path(args.work)
    raw_dir, cls_dir, runs_dir = work / "raw", work / "cls", work / "runs"
    work.mkdir(parents=True, exist_ok=True)

    if "fetch" in stages:
        stage_fetch(raw_dir, args.workers)
    if "mine" in stages:
        stage_mine(raw_dir, Path(args.videos), clips=args.mine_clips)
    if "build" in stages:
        stage_build(raw_dir, cls_dir)
    if "train" in stages:
        stage_train(cls_dir, runs_dir, Path(args.out), base=args.base,
                    epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
