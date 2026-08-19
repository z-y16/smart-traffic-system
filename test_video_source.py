"""Checks for the video source layer — uploads, stream URLs and playback.

The claim this file exists to prove is the one that matters: **an uploaded
video is analysed exactly as the live camera is**. Three things have to hold
for that to be true, and none of them are obvious.

1. **km/h must not depend on the machine.** A recording carries its own clock,
   so a PC that decodes 20 fps of a 30 fps file must report the same speed as
   one that races through it in a fifth of the time. Section C drives a vehicle
   scripted to travel at exactly 36 km/h through a real MP4 at four different
   playback rates and requires all four answers to agree.

2. **Nothing may leak across a change of source.** Tracks, speeds, class votes
   and the congestion dwell timer all belong to one scene; carried into the
   next they invent journeys between unrelated roads. Section D checks the
   reset, including the clocks that a video's timeline moves.

3. **A source change must never break the running one.** Section E gives the
   node addresses that cannot work and requires the feed to survive each of
   them with a message that says what was wrong.

    python test_video_source.py

No GPU, no model and no network are needed: the videos are generated here and
the detector is scripted, which is the only way to know the true speed of a
vehicle and therefore the only way to check the measured one.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import broadcast_server as bs
import traffic_vision as tv
import video_source as vs
from traffic_light import Phase, TrafficLightController

passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    """Record one assertion."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    """Print a section banner."""
    print(f"\n[{title}]")


def close_to(value: float, target: float, tolerance: float) -> bool:
    """True when ``value`` is within ``tolerance`` of ``target``."""
    return abs(value - target) <= tolerance


WORKDIR = Path(tempfile.mkdtemp(prefix="traffic_source_"))


def make_video(name: str, frames: int = 60, fps: float = 30.0,
               size: tuple[int, int] = (320, 180)) -> Path:
    """Write a small MP4 with a box sliding across it, and return its path."""
    path = WORKDIR / name
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        x = int(index * (width - 40) / max(1, frames - 1))
        cv2.rectangle(frame, (x, height // 2 - 20), (x + 30, height // 2 + 20),
                      (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


class ScriptedDetector:
    """A detector whose output depends on the frame, not on the picture.

    Speeds can only be checked against a known truth, and a real detector on
    real footage provides none. This one places a vehicle at an exact position
    on every frame, so the km/h it should produce is arithmetic.
    """

    def __init__(self, pixels_per_frame: float, box: int = 60) -> None:
        """Move a single tracked car by ``pixels_per_frame`` on each call."""
        self.step = pixels_per_frame
        self.box = box
        self.frame = 0
        self.names = {2: "car"}
        self.resets = 0

    def track(self, _frame: np.ndarray) -> tuple[list[tuple], float]:
        """Return one car, advanced by one step from the previous call."""
        x = 100.0 + self.frame * self.step
        self.frame += 1
        # A 60 px box on a car 1.5 m tall gives the auto-calibrator its scale.
        return [(np.array([x, 300.0, x + 40.0, 300.0 + self.box]), 7, "car", 0.9)], 4.0

    def reset_tracker(self) -> None:
        """Record that the pipeline asked for a clean slate."""
        self.resets += 1


# ══════════════════════════════════════════════════════════════════════════
section("A] Recognising what a source is")

cases = [
    ("2", vs.KIND_CAMERA), ("0", vs.KIND_CAMERA), ("droidcam", vs.KIND_CAMERA),
    ("camera:droidcam", vs.KIND_CAMERA), ("cam:1", vs.KIND_CAMERA),
    ("road.mp4", vs.KIND_FILE), ("C:/clips/A road.MOV", vs.KIND_FILE),
    ("file:whatever", vs.KIND_FILE), ("/home/pi/road.mkv", vs.KIND_FILE),
    ("rtsp://192.168.1.9:554/s1", vs.KIND_STREAM),
    ("http://cam.example/video", vs.KIND_STREAM),
    ("https://example.com/live.m3u8", vs.KIND_STREAM),
    ("https://youtube.com/watch?v=abc", vs.KIND_STREAM),
    ("url:192.168.0.5", vs.KIND_STREAM),
]
for text, expected in cases:
    check(f"{text!r} is a {expected}", vs.classify(text).kind == expected,
          vs.classify(text).kind)

check("a camera keeps its label", vs.classify("2").label == "Camera 2")
check("a stream is labelled by host",
      "192.168.1.9" in vs.classify("rtsp://192.168.1.9:554/s1").label,
      vs.classify("rtsp://192.168.1.9:554/s1").label)
check("a relative file resolves against the project, not the shell's cwd",
      Path(vs.classify("road.mp4").target).parent == vs.ROOT_DIR,
      vs.classify("road.mp4").target)
for bad in ("", "   "):
    try:
        vs.classify(bad)
        check("an empty source is rejected", False)
    except vs.SourceError:
        check("an empty source is rejected", True)

check("youtube is recognised as a page, not a stream",
      vs._is_page_url("https://www.youtube.com/watch?v=x")
      and not vs._is_page_url("http://cam.local/video"))


# ══════════════════════════════════════════════════════════════════════════
section("B] Opening a video file")

clip = make_video("clip.mp4", frames=60, fps=30.0)
source = vs.VideoSource(str(clip), loop=False)
status = source.status()
check("the file is opened as a file", status["is_file"] and status["kind"] == "file")
check("its frame rate is read", status["native_fps"] == 30.0, str(status["native_fps"]))
check("its length is read", close_to(status["duration_s"], 2.0, 0.05),
      f"{status['duration_s']}s")
check("its size is read", (status["width"], status["height"]) == (320, 180),
      f"{status['width']}x{status['height']}")
check("the first source is not reported as a change", not source.pop_discontinuity())

read = source.read()
check("a frame comes back", read.frame is not None and not read.ended)
check("the media clock is a real timestamp", read.time > 1_700_000_000, str(read.time))

source.pause()
frozen = source.media_time
idle = source.read()
check("a paused file yields no frame", idle.frame is None and not idle.ended)
check("a paused file stops the clock", source.media_time == frozen)
source.pause(False)
check("resuming delivers frames again", source.read().frame is not None)

after_seek = source.seek(fraction=0.5)
check("seeking moves the position", close_to(after_seek["position_s"], 1.0, 0.15),
      f"{after_seek['position_s']}s")
check("seeking is a discontinuity", source.pop_discontinuity())

empty = vs.VideoSource(None)
check("a node can start with no source at all", not empty.is_open)
check("an empty source reports itself finished, not broken", empty.read().ended)
check("an empty source still describes itself", empty.status()["kind"] == "none")
try:
    empty.pause()
    check("pausing something with no timeline is refused", False)
except vs.SourceError as exc:
    check("pausing something with no timeline is refused", True, str(exc)[:52])


# ══════════════════════════════════════════════════════════════════════════
section("C] km/h does not depend on how fast the machine is")

# A vehicle crossing 6 px per frame, in a frame where 60 px is a 1.5 m car
# roof, travels 6 * (1.5/60) = 0.15 m per frame. At 30 fps that is 4.5 m/s,
# which is 16.2 km/h. The video is played at four very different speeds; the
# answer must not move, because it is a property of the traffic and not of
# the computer watching it.
EXPECTED_KMH = 6.0 * (1.5 / 60.0) * 30.0 * 3.6

measured: dict[str, float] = {}
for label, rate, drop in (("real time", 1.0, True), ("2x", 2.0, True),
                          ("unthrottled", 0.0, False), ("half speed", 0.5, True)):
    detector = ScriptedDetector(pixels_per_frame=6.0)
    pipeline = tv.TrafficVisionPipeline(
        tv.VisionConfig(calibration_path="does_not_exist.json"), detector=detector)
    playing = vs.VideoSource(str(clip), loop=False, playback_rate=rate, drop_frames=drop)

    last = None
    while True:
        frame_read = playing.read()
        if frame_read.frame is None:
            if frame_read.ended:
                break
            continue
        last = pipeline.process(frame_read.frame, frame_read.time, wall=frame_read.time)
    playing.release()
    measured[label] = last.avg_speed_kmh
    check(f"{label}: {last.avg_speed_kmh:.1f} km/h (expected {EXPECTED_KMH:.1f})",
          close_to(last.avg_speed_kmh, EXPECTED_KMH, 1.0), f"{last.avg_speed_kmh}")

spread = max(measured.values()) - min(measured.values())
check("every playback rate agrees to within 0.5 km/h", spread <= 0.5,
      f"spread {spread:.2f} km/h across {sorted(measured.values())}")

# The same run against the wall clock is what this replaces: reading a 30 fps
# file as fast as possible would put every vehicle's whole journey inside a
# fraction of a second and report an absurd speed.
detector = ScriptedDetector(pixels_per_frame=6.0)
wall_pipeline = tv.TrafficVisionPipeline(
    tv.VisionConfig(calibration_path="does_not_exist.json"), detector=detector)
racing = vs.VideoSource(str(clip), loop=False, playback_rate=0.0, drop_frames=False)
wall_last = None
while True:
    frame_read = racing.read()
    if frame_read.frame is None:
        if frame_read.ended:
            break
        continue
    now = time.time()
    wall_last = wall_pipeline.process(frame_read.frame, now, wall=now)
racing.release()
check("timing the same file by the wall clock gets it wrong "
      f"({wall_last.avg_speed_kmh:.1f} km/h instead of {EXPECTED_KMH:.1f})",
      not close_to(wall_last.avg_speed_kmh, EXPECTED_KMH, 1.0),
      "the whole video passed in a fraction of a second of real time, so the "
      "0.8s speed window never closed — this is what the media clock prevents")


# ══════════════════════════════════════════════════════════════════════════
section("D] Nothing survives a change of source")

detector = ScriptedDetector(pixels_per_frame=6.0)
pipeline = tv.TrafficVisionPipeline(
    tv.VisionConfig(calibration_path="does_not_exist.json"), detector=detector)
blank = np.zeros((720, 1280, 3), np.uint8)

for index in range(40):
    result = pipeline.process(blank, index / 30.0, wall=index / 30.0)
check("a vehicle is being tracked before the switch", result.total_vehicles == 1)
check("it has a measured speed", result.avg_speed_kmh > 0, f"{result.avg_speed_kmh}")
check("it counts towards the session total", result.unique_vehicles_session == 1)

records = pipeline.reset(now=1000.0)
check("the vehicle in flight is written to the log, not dropped", len(records) == 1,
      f"{records}")
check("the record carries its journey",
      records[0]["distance_m"] > 0 and records[0]["class"] == "car", f"{records[0]}")
check("the tracker is told to forget its ids", detector.resets == 1)
check("track ages are cleared", not pipeline._ages)
check("class votes are cleared", not pipeline._class_votes)
check("the session's unique-vehicle set is cleared", not pipeline._seen_ids)
check("speed history is cleared", not pipeline.speed._tracks)

# The congestion dwell timer is a comparison against the clock that set it, so
# a source whose clock is elsewhere must move it too.
pipeline.congestion._level = "HEAVY"
pipeline.congestion.reset(now=1000.0)
check("congestion returns to FREE on a new source",
      pipeline.congestion._level == "FREE")
check("its dwell clock moves with the new source's clock",
      pipeline.congestion._level_since == 1000.0)

light = TrafficLightController()
light.update("HEAVY", 5000.0)
light.rebase(12.0)
state = light.update("HEAVY", 12.0)
check("the signal's dwell clock can be rebased without changing the lamp",
      state.phase is Phase.RED and state.elapsed == 0.0,
      f"{state.phase.value} elapsed={state.elapsed}")

looping = vs.VideoSource(str(make_video("short.mp4", frames=12, fps=30.0)), loop=True)
looping.pop_discontinuity()
times, wraps = [], 0
for _ in range(60):
    frame_read = looping.read()
    if looping.pop_discontinuity():
        wraps += 1
    if frame_read.frame is not None:
        times.append(frame_read.time)
check("a looping file wraps round", wraps >= 2, f"{wraps} wraps")
check("each wrap is announced as a discontinuity", wraps >= 2)
check("the clock never runs backwards across a wrap",
      all(later >= earlier for earlier, later in zip(times, times[1:])))
looping.release()


# ══════════════════════════════════════════════════════════════════════════
section("E] A bad source never disturbs a good one")

# Unpaced, because a refused remote address takes a few seconds to answer and
# this two-second clip would otherwise play itself out while we waited. On the
# server the two happen on different threads, so the feed simply carries on.
live = vs.VideoSource(str(clip), loop=False, playback_rate=0.0)
for bad, expect in (("no_such_clip.mp4", "No such video file"),
                    ("rtsp://192.0.2.1:554/nope", "answering"),
                    ("rtsp://no-such-host.invalid/live", "looked up")):
    try:
        live.switch(bad)
        check(f"{bad} is refused", False, "it was accepted")
    except vs.SourceError as exc:
        check(f"{bad} is refused with a reason", expect in str(exc), str(exc)[:70])
check("the running source is untouched by every failure",
      live.read().frame is not None)
check("and still knows what it is", live.status()["label"] == "clip.mp4")
live.release()


# ══════════════════════════════════════════════════════════════════════════
section("F] Upload names cannot escape the upload directory")

for given, expected in (("My Clip!!.mp4", "My_Clip.mp4"),
                        ("../../../etc/passwd", "passwd.mp4"),
                        ("road.MOV", "road.mov"),
                        ("script.exe", "script.mp4")):
    check(f"{given!r} becomes {expected!r}",
          vs.safe_upload_name(given) == expected, vs.safe_upload_name(given))

check("a sanitised name stays inside the upload directory",
      vs.upload_path("../../evil.mp4").parent == vs.UPLOAD_DIR,
      str(vs.upload_path("../../evil.mp4")))

# Only the last path component is ever used, so a name cannot climb out of the
# upload directory at all: a traversal attempt simply names a file that is not
# there, and is reported as missing rather than acted upon.
for escape in ("../README.md", "../../README.md", "C:/Windows/system.ini"):
    check(f"{escape!r} cannot reach outside the upload directory",
          vs.delete_upload(escape) is False)
check("README.md is still there", Path("README.md").exists())


# ══════════════════════════════════════════════════════════════════════════
section("G] The HTTP endpoints the dashboard drives")

other = make_video("other.mp4", frames=30, fps=25.0)
logger = tv.TelemetryLogger(WORKDIR / "h.csv", WORKDIR / "h.xlsx")
served = vs.VideoSource(str(clip), loop=False)
bs._register_runtime(logger, TrafficLightController(), served)
client = bs.flask_app.test_client()

try:
    body = client.get("/source").get_json()
    check("/source describes the current source",
          body["kind"] == "file" and body["label"] == "clip.mp4", f"{body['label']}")
    check("/source reports the position", "position_s" in body and "progress" in body)

    catalogue = client.get("/sources").get_json()
    check("/sources lists cameras and uploads",
          "cameras" in catalogue and "uploads" in catalogue and "current" in catalogue)

    switched = client.post("/source", json={"source": str(other)}).get_json()
    check("POST /source switches source", switched["label"] == "other.mp4",
          switched["label"])
    check("the new source's frame rate is picked up", switched["native_fps"] == 25.0)

    refused = client.post("/source", json={"source": "nowhere.mp4"})
    check("an unopenable source is a 400, not a crash", refused.status_code == 400)
    check("the error says what was wrong",
          "No such video file" in refused.get_json()["error"],
          refused.get_json()["error"][:50])
    check("and the feed is still on the old source",
          client.get("/source").get_json()["label"] == "other.mp4")
    check("POST /source with no source is a 400",
          client.post("/source", json={}).status_code == 400)

    controls = [
        ("pause", {}, lambda body: body["paused"]),
        ("resume", {}, lambda body: not body["paused"]),
        ("seek", {"fraction": 0.5}, lambda body: body["progress"] >= 0.4),
        ("loop", {"value": True}, lambda body: body["loop"]),
        ("rate", {"value": 2}, lambda body: body["playback_rate"] == 2.0),
        ("restart", {}, lambda body: body["progress"] < 0.1),
    ]
    for action, params, holds in controls:
        answer = client.post("/source/control",
                             json={"action": action, **params}).get_json()
        check(f"/source/control {action}", holds(answer),
              f"progress={answer['progress']} paused={answer['paused']}")
    check("an unknown action is a 400",
          client.post("/source/control", json={"action": "explode"}).status_code == 400)

    uploaded = client.post(
        "/source/upload",
        data={"file": (io.BytesIO(clip.read_bytes()), "Bridge Cam 3.mp4")},
        content_type="multipart/form-data").get_json()
    check("an upload is stored under a safe name",
          uploaded["stored"] == "Bridge_Cam_3.mp4", f"{uploaded.get('stored')}")
    check("an upload starts playing straight away", uploaded["playing"] is True)
    check("the pipeline is now on the uploaded file",
          uploaded["source"]["label"] == "Bridge_Cam_3.mp4")
    check("the upload is listed afterwards",
          any(entry["name"] == "Bridge_Cam_3.mp4"
              for entry in client.get("/sources").get_json()["uploads"]))

    rejected = client.post(
        "/source/upload",
        data={"file": (io.BytesIO(b"this is not a video"), "broken.mp4")},
        content_type="multipart/form-data")
    check("an undecodable upload is a 400", rejected.status_code == 400)
    check("and is not left lying on disk",
          not (vs.UPLOAD_DIR / "broken.mp4").exists())
    check("an upload with no file is a 400",
          client.post("/source/upload", data={},
                      content_type="multipart/form-data").status_code == 400)

    check("the video being watched cannot be deleted under it",
          client.delete("/source/upload/Bridge_Cam_3.mp4").status_code == 409)
    client.post("/source", json={"source": str(clip)})
    check("once free it can be deleted",
          client.delete("/source/upload/Bridge_Cam_3.mp4").status_code == 200)
    check("deleting something absent is a 404",
          client.delete("/source/upload/ghost.mp4").status_code == 404)

    check("/telemetry carries the source, so a paused video still says so",
          "source" in client.get("/telemetry").get_json())
    check("/health carries the source", "source" in client.get("/health").get_json())
    landing = client.get("/").get_data(as_text=True)
    for endpoint in ("/source", "/sources", "/source/upload", "/source/control"):
        check(f"the landing page documents {endpoint}", endpoint in landing)
finally:
    logger.close()
    served.release()
    bs._register_runtime(None, None, None)

check("/source reports unavailable before the loop runs",
      client.get("/source").status_code == 503)
check("/source/control reports unavailable before the loop runs",
      client.post("/source/control", json={"action": "pause"}).status_code == 503)
check("/sources still answers, with nothing current",
      client.get("/sources").get_json()["current"] == {})


# ══════════════════════════════════════════════════════════════════════════
section("H] Command-line defaults")

args = bs.parse_args([])
check("video files play at real speed by default", args.playback_rate == 1.0)
check("frames are dropped to stay in real time by default", not args.no_drop_frames)
check("a dropped stream is reconnected by default", not args.no_reconnect)
check("a finished video holds rather than ending the run by default",
      not args.exit_on_end)
check("the playback rate is overridable",
      bs.parse_args(["--playback-rate", "0"]).playback_rate == 0.0)
check("an upload limit is configured", bs.parse_args([]).max_upload_mb > 0)


shutil.rmtree(WORKDIR, ignore_errors=True)

print("\n" + "=" * 70)
print(f" RESULT: {passed} passed, {failed} failed")
print("=" * 70)
sys.exit(1 if failed else 0)
