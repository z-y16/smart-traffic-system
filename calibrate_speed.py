"""
calibrate_speed.py — teach the system how big a pixel is, in metres
===================================================================

Speed in km/h is impossible without knowing the real-world size of the scene.
This tool captures one frame from your camera (or a video/image), lets you mark
a rectangle of known size on the road surface, and writes
``speed_calibration.json``, which ``broadcast_server.py`` picks up automatically
on its next start.

Run it once per camera position. If the camera moves, run it again.

    python calibrate_speed.py                     # webcam 0
    python calibrate_speed.py --source 1          # a different webcam
    python calibrate_speed.py --source road.mp4   # a video file
    python calibrate_speed.py --source rtsp://... # a roadside camera
    python calibrate_speed.py --source frame.jpg  # a still image
    python calibrate_speed.py --mode scale        # 2-point mode, overhead cams

Calibrate against the **same view** the node will watch: a calibration is a
statement about one camera in one position, so the one on disk is meaningless
for a different road. Give the video or the camera URL you are going to run on.

HOMOGRAPHY MODE (default, most accurate)
----------------------------------------
Click 4 points **on the road surface** that form a rectangle in the real world,
in this order:

      4 ── far-left            3 ── far-right
       \\                       /
        \\                     /
      1 ── near-left       2 ── near-right

Then type the two real distances: the width across (1→2) and the length along
the road (1→4). Good things to measure: lane width (a standard lane is ~3.5 m),
the gap between two lamp posts, the length of a dashed lane marking (in many
countries a dash + gap is a fixed known length), or just tape-measure a stretch
of kerb.

SCALE MODE (quick, for near-overhead cameras)
---------------------------------------------
Click 2 points a known distance apart and type that distance. Assumes one
constant metres-per-pixel across the whole frame — only true when the camera
looks nearly straight down.

Keys
----
  click   place a point
  u       undo last point
  r       switch to ROI mode (draw the area to monitor; optional)
  SPACE   grab a fresh frame from the camera
  v       preview the metric grid (homography mode, after 4 points)
  s       save calibration
  q       quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from video_source import SourceError, classify, open_spec

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT_DIR / "speed_calibration.json"
WINDOW = "Speed Calibration - click road corners"

POINT_PROMPTS = ["NEAR-LEFT", "NEAR-RIGHT", "FAR-RIGHT", "FAR-LEFT"]
COLORS = [(0, 255, 255), (0, 200, 255), (0, 160, 255), (0, 120, 255)]


class CalibrationUI:
    """Collects road-plane points and an optional ROI from mouse clicks."""

    def __init__(self, required_points: int) -> None:
        """Initialise the click collector for the chosen calibration mode."""
        self.required = required_points
        self.points: list[tuple[int, int]] = []
        self.roi: list[tuple[int, int]] = []
        self.roi_mode = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback: append a point on left click."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.roi_mode:
            self.roi.append((x, y))
        elif len(self.points) < self.required:
            self.points.append((x, y))

    def undo(self) -> None:
        """Remove the most recently placed point."""
        target = self.roi if self.roi_mode else self.points
        if target:
            target.pop()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the current calibration state onto a copy of the frame."""
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        if len(self.points) >= 2:
            closed = len(self.points) == self.required and self.required == 4
            cv2.polylines(canvas, [np.array(self.points, np.int32)], closed,
                          (0, 220, 255), 2)
        for index, point in enumerate(self.points):
            cv2.circle(canvas, point, 7, COLORS[index % len(COLORS)], -1)
            cv2.circle(canvas, point, 9, (20, 20, 20), 2)
            label = POINT_PROMPTS[index] if self.required == 4 else f"P{index + 1}"
            cv2.putText(canvas, f"{index + 1}. {label}", (point[0] + 12, point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.roi:
            cv2.polylines(canvas, [np.array(self.roi, np.int32)], len(self.roi) > 2,
                          (255, 200, 0), 2)
            for point in self.roi:
                cv2.circle(canvas, point, 5, (255, 200, 0), -1)

        banner = canvas.copy()
        cv2.rectangle(banner, (0, 0), (width, 74), (18, 22, 30), -1)
        cv2.addWeighted(banner, 0.65, canvas, 0.35, 0, canvas)

        if self.roi_mode:
            headline = f"ROI MODE - click the area to monitor ({len(self.roi)} points), r=back"
        elif len(self.points) < self.required:
            nxt = POINT_PROMPTS[len(self.points)] if self.required == 4 \
                else f"point {len(self.points) + 1}"
            headline = f"Click {nxt}  ({len(self.points)}/{self.required} placed)"
        else:
            headline = "All points placed - press 's' to save, 'v' to preview grid"
        cv2.putText(canvas, headline, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (255, 255, 255), 2)
        cv2.putText(canvas, "u=undo  r=ROI  SPACE=new frame  v=preview  s=save  q=quit",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 210, 240), 1)
        return canvas


def open_source(source: str) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """Open a camera/video, or load a still image. Returns ``(capture, image)``.

    Anything that is not a still goes through ``video_source``, the same layer
    the CV node uses, so an address that works there works here unchanged —
    which matters because a calibration is only valid for the exact view it
    was drawn on. RTSP over TCP and a quick, readable failure on an
    unreachable host come along with it.
    """
    if Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv2.imread(source)
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        return None, image

    try:
        capture, _meta = open_spec(classify(source), 1920, 1080)
    except SourceError as exc:
        raise SystemExit(f"{exc}") from None
    return capture, None


def ask_float(prompt: str, minimum: float = 0.01) -> float:
    """Prompt on the terminal until a valid positive number is entered."""
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            print("  Please type a number, e.g. 3.5")
            continue
        if value < minimum:
            print(f"  Must be at least {minimum}")
            continue
        return value


def preview_grid(frame: np.ndarray, image_points: list[tuple[int, int]],
                 width_m: float, length_m: float) -> np.ndarray:
    """Overlay a 1-metre grid on the road plane so the fit can be eyeballed.

    If the grid lines do not follow the road (lane markings, kerbs), the marked
    points or the entered distances are wrong.
    """
    src = np.array(image_points, np.float32)
    dst = np.array([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]], np.float32)
    world_to_image = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
    canvas = frame.copy()

    def to_image(x: float, y: float) -> tuple[int, int]:
        """Project a world-plane point (metres) back into image pixels."""
        point = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), world_to_image)
        return int(point[0][0][0]), int(point[0][0][1])

    step = 1.0
    x = 0.0
    while x <= width_m + 1e-6:
        cv2.line(canvas, to_image(x, 0), to_image(x, length_m), (60, 220, 60), 1)
        x += step
    y = 0.0
    while y <= length_m + 1e-6:
        thickness = 2 if abs(y % 5.0) < 1e-6 else 1
        cv2.line(canvas, to_image(0, y), to_image(width_m, y), (60, 220, 60), thickness)
        if thickness == 2:
            cv2.putText(canvas, f"{y:.0f}m", to_image(0, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (60, 255, 60), 2)
        y += step

    cv2.putText(canvas, "Grid lines should follow the road. Any key to continue.",
                (16, canvas.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 255, 60), 2)
    return canvas


def main() -> int:
    """Run the interactive calibration session."""
    parser = argparse.ArgumentParser(description="Calibrate pixels to metres for km/h speeds.")
    parser.add_argument("--source", default="0", help="camera index, video file, or image")
    parser.add_argument("--mode", default="homography", choices=["homography", "scale"],
                        help="homography = 4 points (accurate); scale = 2 points (overhead)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output JSON path")
    args = parser.parse_args()

    required = 4 if args.mode == "homography" else 2
    capture, still = open_source(args.source)

    frame = still
    if capture is not None:
        for _ in range(10):  # let auto-exposure settle
            ok, frame = capture.read()
        if not ok or frame is None:
            raise SystemExit("Could not grab a frame from the source.")

    ui = CalibrationUI(required)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1600, frame.shape[1]), min(900, frame.shape[0]))
    cv2.setMouseCallback(WINDOW, ui.on_mouse)

    print(__doc__.split("Keys")[0])
    print(f"Mode: {args.mode} | Source: {args.source} | "
          f"Frame: {frame.shape[1]}x{frame.shape[0]}")

    width_m = length_m = None
    while True:
        cv2.imshow(WINDOW, ui.draw(frame))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("Cancelled — nothing was written.")
            break

        if key == ord("u"):
            ui.undo()

        elif key == ord("r"):
            ui.roi_mode = not ui.roi_mode
            print(f"[ROI] {'ON — click the monitored area' if ui.roi_mode else 'OFF'}")

        elif key == 32 and capture is not None:  # SPACE
            ok, new_frame = capture.read()
            if ok:
                frame = new_frame
                print("[CAM] Grabbed a new frame.")

        elif key == ord("v") and args.mode == "homography" and len(ui.points) == 4:
            if width_m is None:
                print("\nEnter the real-world size of the marked rectangle:")
                width_m = ask_float("  Width  across the road, point 1 -> 2 (metres): ")
                length_m = ask_float("  Length along the road, point 1 -> 4 (metres): ")
            cv2.imshow(WINDOW, preview_grid(frame, ui.points, width_m, length_m))
            cv2.waitKey(0)

        elif key == ord("s"):
            if len(ui.points) < required:
                print(f"Place all {required} points first.")
                continue

            if args.mode == "homography":
                if width_m is None:
                    print("\nEnter the real-world size of the marked rectangle:")
                    width_m = ask_float("  Width  across the road, point 1 -> 2 (metres): ")
                    length_m = ask_float("  Length along the road, point 1 -> 4 (metres): ")
                calibration = {
                    "mode": "homography",
                    "image_points": [list(p) for p in ui.points],
                    "world_points": [[0.0, 0.0], [width_m, 0.0],
                                     [width_m, length_m], [0.0, length_m]],
                    "frame_size": [frame.shape[1], frame.shape[0]],
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "notes": f"{width_m} m wide x {length_m} m long road patch",
                }
            else:
                print("\nEnter the real distance between the two clicked points:")
                distance_m = ask_float("  Distance (metres): ")
                pixels = float(np.hypot(ui.points[1][0] - ui.points[0][0],
                                        ui.points[1][1] - ui.points[0][1]))
                if pixels < 5:
                    print("  Points are too close together — click further apart.")
                    continue
                calibration = {
                    "mode": "scale",
                    "meters_per_pixel": distance_m / pixels,
                    "reference_points": [list(p) for p in ui.points],
                    "frame_size": [frame.shape[1], frame.shape[0]],
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "notes": f"{distance_m} m over {pixels:.1f} px",
                }

            if len(ui.roi) >= 3:
                calibration["roi"] = [list(p) for p in ui.roi]

            output = Path(args.output)
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(calibration, handle, indent=2)

            print(f"\n[OK] Saved {output}")
            print(f"     mode: {calibration['mode']}")
            if calibration["mode"] == "scale":
                print(f"     scale: {calibration['meters_per_pixel']:.5f} m/px")
            else:
                print(f"     patch: {calibration['notes']}")
            if "roi" in calibration:
                print(f"     ROI: {len(calibration['roi'])} points")
            print("     Restart broadcast_server.py to use it.")
            break

    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
