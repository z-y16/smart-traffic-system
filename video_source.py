"""
video_source.py — where the frames come from
============================================

Detection, speed, congestion, the signal and the logs do not care whether a
frame arrived from the webcam on the desk, an MP4 uploaded from the dashboard
or a roadside camera's RTSP stream. This module is the one place that does,
which is what lets the source be swapped **while the server is running**
without anything downstream noticing.

Three kinds of source
---------------------
============  ==============================================================
``camera``    a local device, by index (``2``) or by name (``droidcam``)
``file``      a video on disk, including one uploaded from the dashboard
``stream``    a network URL — RTSP, an IP camera's MJPEG endpoint, an HLS
              playlist, or a page URL that yt-dlp can resolve into one
              (most public "live traffic camera" links are YouTube streams)
============  ==============================================================

Why a file is not just "a camera that reads from disk"
------------------------------------------------------
Speed in km/h is distance over **time**, and for a recorded video the only
honest clock is the video's own. Measuring against the wall clock instead
means the answer depends on the hardware: a PC that manages 20 fps on a 30 fps
recording watches every vehicle take 1.5x as long to cross the frame and
reports speeds a third too low, while one that races through the file
unthrottled reports them far too high. Neither is a property of the traffic.

:class:`VideoSource` therefore publishes a **media clock** that advances one
frame interval per frame consumed. The km/h a video reports is then the km/h
it was filmed at, whatever the machine manages — which is precisely what makes
an uploaded video behave like the live camera rather than merely look like it.

The clock is monotonic across loops and seeks and is expressed as a real epoch
timestamp, so ``datetime.fromtimestamp()`` still stamps the log sensibly.

Real-time playback
------------------
By default a file plays at its native frame rate and **drops frames to stay on
schedule** when detection cannot keep up — exactly what a live camera does,
where a frame not read in time is simply gone. Skipped frames still advance
the media clock, so dropping them costs nothing but overlay smoothness.
``playback_rate=0`` disables pacing and processes every frame as fast as the
machine allows; the media clock keeps the speeds honest either way.

Nothing here imports Flask, ultralytics or torch, so the whole module is
testable without a GPU or a web server — see ``test_video_source.py``.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent

#: Where videos uploaded from the dashboard are kept.
UPLOAD_DIR = ROOT_DIR / "uploads"

#: Where footage that was put on the machine by hand lives. Searched alongside
#: the upload directory so ``nightroad.mp4`` names the same file whether it
#: arrived through the dashboard or was simply copied in.
VIDEO_DIR = ROOT_DIR / "vedios"

#: Directories a bare filename is looked up in, and which the dashboard's
#: library lists. Uploads first: a name present in both means the dashboard
#: sent it, and that copy is the one the dashboard can also delete.
LIBRARY_DIRS: tuple[Path, ...] = (UPLOAD_DIR, VIDEO_DIR)

KIND_CAMERA = "camera"
KIND_FILE = "file"
KIND_STREAM = "stream"

#: Extensions recognised as a video file even when the path does not exist yet.
VIDEO_SUFFIXES: frozenset[str] = frozenset({
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv",
    ".mpg", ".mpeg", ".mts", ".m2ts", ".ts", ".3gp", ".ogv",
})

#: Hosts that serve *pages* rather than streams. yt-dlp turns one into a direct
#: URL OpenCV can open, which is how a YouTube live traffic camera gets in.
PAGE_HOSTS: tuple[str, ...] = (
    "youtube.com", "youtu.be", "twitch.tv", "vimeo.com",
    "dailymotion.com", "facebook.com", "bilibili.com",
)

#: Used when a file does not report a sane frame rate. Every timing in this
#: module derives from it, so a wrong value shifts km/h proportionally — but a
#: missing one would make speed measurement impossible altogether.
DEFAULT_FILE_FPS = 25.0

#: Ceiling on frames skipped in one catch-up burst. Without it, a long stall
#: (a workbook write, a laptop sleeping) would try to fast-forward minutes of
#: video in a single blocking call.
MAX_CATCHUP_FRAMES = 150

#: Largest share of the loop that may go on decoding frames nobody will look
#: at. A frame count alone cannot bound this, because the only thing that
#: matters is how long a skip *costs*: skipping 45 frames is free at 0.7 ms
#: each and ruinous at 14 ms each, and the ceiling above cannot tell the two
#: apart.
#:
#: The failure it prevents is specific and was measured on this project's own
#: AV1 footage. Staying in real time on a 60 fps file needs one frame decoded
#: every 16.7 ms; when a decode costs 14.2 ms there is 2.5 ms per frame left
#: for everything else, so each pass round the loop ends further behind than it
#: started and skips more frames to compensate — 45.7 skips per processed
#: frame, 700 ms of every 810 ms spent discarding pictures, and 1.2 fps of
#: detection on a machine that manages 14 fps on the same clip when it is not
#: trying to keep up. Catching up is arithmetically impossible there, and this
#: is what stops the loop from bankrupting itself trying: past the budget it
#: gives up on real time and lets the video run slow instead, which costs
#: nothing that matters. Speeds are measured against the media clock, so km/h,
#: counts and congestion are identical either way — only the wall-clock
#: duration of the run changes.
CATCHUP_TIME_BUDGET = 0.5

#: How long a live source may fail to deliver frames before it is reconnected
#: rather than retried. A dead stream returns False instantly, so this has to
#: be measured in seconds and not in attempts.
FAILURE_GRACE_SECONDS = 2.0
RECONNECT_ATTEMPTS = 3
RECONNECT_BACKOFF_SECONDS = 1.5

#: How long to spend deciding whether a stream's host is even reachable.
#: OpenCV's own give-up time is a compile-time 30 seconds, which is far too
#: long to leave someone waiting to be told they mistyped an address, so the
#: host is checked with a plain socket first.
STREAM_CONNECT_TIMEOUT_SECONDS = 3.0

#: Default port per scheme, for that reachability check.
SCHEME_PORTS: dict[str, int] = {
    "rtsp": 554, "rtsps": 322, "rtmp": 1935, "rtmps": 443,
    "http": 80, "https": 443,
}


class SourceError(Exception):
    """A source could not be understood, opened, or controlled."""


# ═══════════════════════════════════════════════════════════════════════════
# WHAT DID THE USER ASK FOR?
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SourceSpec:
    """A source request, classified but not yet opened."""

    kind: str
    #: What to hand to OpenCV: an index as text, a filesystem path, or a URL.
    target: str
    #: Human-readable name for the dashboard and the HUD.
    label: str
    #: Exactly what was requested, so it can be echoed back and re-opened.
    original: str

    @property
    def is_file(self) -> bool:
        """True when this source has a fixed length and can be seeked."""
        return self.kind == KIND_FILE

    @property
    def is_live(self) -> bool:
        """True for cameras and network streams — no end, nothing to seek."""
        return self.kind != KIND_FILE

    def to_dict(self) -> dict[str, Any]:
        """Return the spec as JSON-friendly values."""
        return {"kind": self.kind, "target": self.target,
                "label": self.label, "spec": self.original}


#: Explicit prefixes, for the rare case where a name is ambiguous.
_PREFIXES: tuple[tuple[str, str], ...] = (
    ("camera:", KIND_CAMERA), ("cam:", KIND_CAMERA), ("device:", KIND_CAMERA),
    ("file:", KIND_FILE), ("path:", KIND_FILE), ("video:", KIND_FILE),
    ("url:", KIND_STREAM), ("stream:", KIND_STREAM), ("rtsp:", KIND_STREAM),
)


def classify(spec: str) -> SourceSpec:
    """Work out what kind of source ``spec`` names, without opening it.

    The rules, in order: an explicit ``camera:``/``file:``/``url:`` prefix
    wins; anything containing ``://`` is a network stream; a bare number is a
    camera index; a path that exists or carries a video extension is a file;
    anything else is a camera *name* to be looked up at open time.
    """
    raw = (spec or "").strip().strip('"').strip("'")
    if not raw:
        raise SourceError("No source given.")

    lowered = raw.lower()

    if lowered.startswith("file://"):
        return _file_spec(url2pathname(urlparse(raw).path), raw)

    if lowered.startswith("rtsp:") and "://" in raw:
        return _stream_spec(raw, raw)

    for prefix, kind in _PREFIXES:
        if lowered.startswith(prefix):
            rest = raw[len(prefix):].strip()
            if not rest:
                raise SourceError(f"Nothing follows '{prefix}' in {raw!r}.")
            if kind == KIND_FILE:
                return _file_spec(rest, raw)
            if kind == KIND_STREAM:
                return _stream_spec(rest, raw)
            return _camera_spec(rest, raw)

    if "://" in raw:
        return _stream_spec(raw, raw)

    if raw.isdigit():
        return _camera_spec(raw, raw)

    if Path(raw).suffix.lower() in VIDEO_SUFFIXES or _resolve_path(raw).exists():
        return _file_spec(raw, raw)

    return _camera_spec(raw, raw)


def _resolve_path(path: str) -> Path:
    """Resolve a possibly relative path against the project directory.

    The server's working directory is wherever it was launched from, but a
    video always lives next to the code, so a bare filename has to mean the
    same thing in both places.

    A name that matches nothing directly is looked for in the library
    directories, so ``Dayroad.mp4`` finds ``vedios/Dayroad.mp4`` without
    anyone having to type the folder. Falls back to the project directory when
    it is nowhere, which keeps the "no such file" error pointing somewhere
    recognisable.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    direct = (ROOT_DIR / candidate).resolve()
    if direct.exists():
        return direct

    for directory in LIBRARY_DIRS:
        found = directory / candidate.name
        if found.is_file():
            return found.resolve()

    return direct


def _file_spec(path: str, original: str) -> SourceSpec:
    """Build a file spec, resolving the path but not requiring it to exist."""
    resolved = _resolve_path(path)
    return SourceSpec(KIND_FILE, str(resolved), resolved.name, original)


def _stream_spec(url: str, original: str) -> SourceSpec:
    """Build a network-stream spec labelled by its host."""
    host = urlparse(url).hostname or url
    scheme = (urlparse(url).scheme or "stream").upper()
    return SourceSpec(KIND_STREAM, url, f"{scheme} · {host}", original)


def _camera_spec(name: str, original: str) -> SourceSpec:
    """Build a camera spec; a name is only resolved to an index at open time."""
    label = f"Camera {name}" if name.isdigit() else f"Camera '{name}'"
    return SourceSpec(KIND_CAMERA, name, label, original)


# ═══════════════════════════════════════════════════════════════════════════
# CAMERAS
# ═══════════════════════════════════════════════════════════════════════════


def camera_names() -> dict[int, str]:
    """Map camera index to device name, e.g. ``{0: 'HP Wide Vision HD Camera'}``.

    Needs ``pygrabber`` (``pip install pygrabber``); without it, name-based
    sources are unavailable and indices must be used instead. The enumeration
    order is the one OpenCV's DirectShow backend uses, which is what makes the
    index correspondence reliable.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        return {}
    try:
        return dict(enumerate(FilterGraph().get_input_devices()))
    except Exception:  # noqa: BLE001 - enumeration is best effort
        return {}


def resolve_camera(name: str) -> str:
    """Turn a camera *name* into its index; pass an index straight through.

    Indices are assigned by Windows at runtime and shift when a device is
    replugged or a virtual-camera app starts, so ``droidcam`` is steadier than
    ``3``.
    """
    if name.isdigit():
        return name

    names = camera_names()
    if not names:
        raise SourceError(
            f"Cannot look up camera {name!r}: pygrabber is not installed. "
            "Run 'pip install pygrabber', or give a numeric index instead.")

    wanted = name.strip().lower().replace(" ", "")
    for index, device in names.items():
        if wanted in device.lower().replace(" ", ""):
            print(f"[CV] Camera {name!r} matched '{device}' at index {index}.")
            return str(index)

    available = ", ".join(f"{i}={n}" for i, n in names.items()) or "none"
    raise SourceError(f"No camera matching {name!r}. Available: {available}")


def list_cameras(width: int, height: int, highest_index: int = 5) -> int:
    """Probe camera indices and report which ones actually deliver a picture.

    Indices are assigned by Windows at runtime and shift when devices are
    replugged, so they cannot be hard-coded — this prints the current mapping.
    """
    print("Probing camera indices (this takes a few seconds each)...\n")
    usable: list[int] = []
    names = camera_names()
    if not names:
        print("  (install pygrabber to see device names: pip install pygrabber)\n")

    for index in range(highest_index + 1):
        label = f"  [{index}] {names.get(index, '')}".rstrip()
        try:
            capture, _meta = _open_camera(str(index), width, height)
        except SourceError:
            print(f"{label} — not available")
            continue
        if not capture.isOpened():
            print(f"{label} — not available")
            capture.release()
            continue

        # The first frames off a webcam are routinely black while the sensor
        # warms up, so judge brightness on a later one.
        frame = None
        for _ in range(20):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
        capture.release()

        if frame is None:
            print(f"{label} — opens but delivers no frame")
            continue

        height_px, width_px = frame.shape[:2]
        if float(frame.mean()) < 5.0:
            print(f"{label} — {width_px}x{height_px}, black frame "
                  f"(not connected, covered, or disabled)")
        else:
            print(f"{label} — {width_px}x{height_px}, live picture  <-- usable")
            usable.append(index)

    if usable:
        best = names.get(usable[0], str(usable[0])).split()[0].lower() if names else usable[0]
        print(f"\nRun with:  python broadcast_server.py --source {best}")
    else:
        print("\nNo camera delivered a picture. Check that it is plugged in and "
              "that no other app is holding it, then try again.")
    return 0


def available_cameras() -> list[dict[str, Any]]:
    """Name the connected cameras **without** probing them.

    Opening a camera to test it takes seconds and steals it from whatever is
    using it, so the dashboard's picker lists names only and lets the operator
    find out by selecting one.
    """
    return [{"index": index, "name": name, "spec": str(index)}
            for index, name in sorted(camera_names().items())]


# ═══════════════════════════════════════════════════════════════════════════
# OPENING
# ═══════════════════════════════════════════════════════════════════════════


def _open_camera(target: str, width: int, height: int) -> tuple[cv2.VideoCapture, dict]:
    """Open a local camera by index or name at the requested resolution."""
    index = int(resolve_camera(target))

    # CAP_DSHOW opens far faster than the default backend and lets most webcams
    # deliver 1080p over MJPG — but it cannot open some virtual cameras at all
    # (DroidCam among them), which Media Foundation can, so fall back rather
    # than reporting the camera as missing.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        if backend == cv2.CAP_DSHOW:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always the freshest frame
        return capture, _live_meta(capture)

    raise SourceError(
        f"Camera {target!r} could not be opened. Another app may be holding it; "
        "run with --list-cameras to see which indices deliver a picture.")


@contextmanager
def _capture_env(values: dict[str, str]):
    """Apply OpenCV's FFmpeg environment settings for one capture, then undo them.

    They are read when a capture is constructed and nowhere else, so they have
    to be set around that call — and restored, because they would otherwise
    apply to every capture opened afterwards, including local files.
    """
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, was in previous.items():
            if was is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = was


def _check_reachable(url: str) -> None:
    """Fail fast when a stream's host will not accept a connection at all.

    OpenCV's give-up time on an unreachable address is a compile-time 30
    seconds, and every one of those seconds is spent with the dashboard
    apparently hung. Every scheme used here rides on TCP, so a plain socket
    answers the same question in about one, and a typo in an address gets a
    real message instead of a long silence.
    """
    parts = urlparse(url)
    scheme = (parts.scheme or "").lower()
    if scheme not in SCHEME_PORTS or not parts.hostname:
        return   # not something a TCP connect can vouch for; let OpenCV try

    port = parts.port or SCHEME_PORTS[scheme]
    try:
        with socket.create_connection((parts.hostname, port),
                                      STREAM_CONNECT_TIMEOUT_SECONDS):
            return
    except socket.gaierror:
        raise SourceError(f"'{parts.hostname}' could not be looked up. Check the "
                          "address, and that this machine has a route to it.") from None
    except (TimeoutError, OSError) as exc:
        raise SourceError(
            f"Nothing is answering at {parts.hostname}:{port} ({exc}). Check the "
            "camera is on and reachable from this machine, that the port is "
            "right, and that a firewall is not in the way.") from None


def _open_stream(url: str) -> tuple[cv2.VideoCapture, dict]:
    """Open a network stream, resolving page URLs through yt-dlp first."""
    direct = _direct_stream_url(url)
    _check_reachable(direct)

    # RTSP over UDP drops packets on any congested network and produces the
    # smeared, half-decoded frames that ruin detection, so force TCP. The
    # timeout covers a host that accepts a connection and then says nothing.
    options = f"timeout;{int(STREAM_CONNECT_TIMEOUT_SECONDS * 2_000_000)}"
    if direct.lower().startswith("rtsp"):
        options = "rtsp_transport;tcp|" + options

    started = time.time()
    with _capture_env({"OPENCV_FFMPEG_CAPTURE_OPTIONS": options}):
        capture = cv2.VideoCapture(direct, cv2.CAP_FFMPEG)
        # Falling back to the default backend is worth it when FFmpeg is simply
        # not built in — which fails instantly — but not when it spent the
        # whole timeout failing to reach the host, where it would only double
        # the wait for the same answer.
        if not capture.isOpened() and time.time() - started < 3.0:
            capture.release()
            capture = cv2.VideoCapture(direct)

    if not capture.isOpened():
        capture.release()
        raise SourceError(
            f"Could not connect to {url}. Check the address is reachable from "
            "this machine, that any username/password is included in the URL, "
            "and that the stream is one FFmpeg can read (RTSP, MJPEG or HLS).")

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture, _live_meta(capture)


def _open_capture(target: str) -> cv2.VideoCapture:
    """Open a video file for decoding on the CPU, or on the GPU if asked.

    The GPU decoder looks like the obvious win and measures as a loss, which is
    worth writing down because it is a natural thing to reach for the moment
    decoding shows up in a profile.

    Timed in isolation it is exactly as good as it sounds: ``grab()`` on a
    1080p HEVC frame costs 5.4 ms on the CPU and 0.7 ms through D3D11VA. But
    ``grab()`` is only half of a read. ``retrieve()`` has to bring the picture
    back across the bus into system memory and convert NV12 to BGR, and that
    readback costs more than CPU decoding saves — because CPU decoding is not
    actually serial with anything: FFmpeg decodes on its own worker threads,
    which run *while* this loop is busy on the GPU with the previous frame, so
    most of it is already paid for by the time it is asked for. Measured
    end-to-end on this project's own footage:

        Test1_day (HEVC 60 fps)    CPU 20.2 fps    GPU 18.1 fps
        Test2_dark (HEVC 30 fps)   CPU 18.2 fps    GPU 16.4 fps
        emergency_road (AV1)       CPU  5.4 fps    GPU  5.3 fps

    — the AV1 file being unchanged because no hardware path exists for it in
    the FFmpeg opencv-python bundles, so it silently decodes on the CPU either
    way. Hence CPU by default. ``TRAFFIC_HW_DECODE=1`` turns the GPU path on
    for a machine where the readback is cheaper than it is here: a desktop card
    on a full-width slot, or a zero-copy VAAPI setup.
    """
    if os.environ.get("TRAFFIC_HW_DECODE", "0") == "1":
        capture = cv2.VideoCapture(target, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION,
                                    cv2.VIDEO_ACCELERATION_ANY])
        if capture.isOpened():
            return capture
        capture.release()   # some codec/driver pairs refuse the open outright
    return cv2.VideoCapture(target)


#: Codecs the FFmpeg bundled inside opencv-python decodes only on the CPU, and
#: slowly. OpenCV 4.13 still ships avcodec 58 (FFmpeg 4.x, 2021), which predates
#: the dav1d AV1 decoder and has no AV1 hardware path: measured on this
#: project's own footage, one 1080p AV1 frame costs 14.2 ms against 5.4 ms for
#: HEVC and 0.7 ms for anything the GPU will take. At 60 fps that is 850 ms of
#: every second gone before detection has looked at anything.
SLOW_CODECS = {"av01": "AV1", "vp09": "VP9"}


def _fourcc(capture: cv2.VideoCapture) -> str:
    """Return the capture's four-character codec code, lowercased."""
    raw = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
    return "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()


def _warn_if_slow_codec(capture: cv2.VideoCapture, target: Path) -> None:
    """Say so when a file is in a codec this OpenCV can only decode slowly.

    Worth a line on the console because the symptom — a low frame rate — looks
    exactly like an overloaded detector, and the remedy is not in this program
    at all. Re-encoding the file once fixes it permanently, so the line comes
    with the command that does it, ready to paste.
    """
    codec = SLOW_CODECS.get(_fourcc(capture))
    if codec is None:
        return
    print(f"[CV] '{target.name}' is {codec}, which the FFmpeg inside "
          f"opencv-python decodes on the CPU at roughly 14 ms a frame — "
          f"several times the cost of H.264/HEVC, and it cannot use the GPU. "
          f"Expect a low frame rate on a 50/60 fps file. Re-encoding it once "
          f"fixes that for good:")
    print(f"[CV]   ffmpeg -i \"{target}\" -c:v h264_nvenc -preset p4 -cq 23 "
          f"-an \"{target.with_name(target.stem + '_h264.mp4')}\"")


def _open_file(path: str) -> tuple[cv2.VideoCapture, dict]:
    """Open a video file and measure its length and frame rate."""
    target = Path(path)
    if not target.exists():
        raise SourceError(f"No such video file: {target}")

    capture = _open_capture(str(target))
    if not capture.isOpened():
        capture.release()
        raise SourceError(
            f"'{target.name}' could not be decoded. It may be a format OpenCV "
            "was not built for, or the file may be incomplete.")

    _warn_if_slow_codec(capture, target)

    reported = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    # Broken containers report 0, 1000, or NaN. Anything outside what a camera
    # could plausibly have shot is not a frame rate.
    if not (1.0 <= reported <= 240.0):
        print(f"[CV] '{target.name}' reports {reported:g} fps, which cannot be "
              f"right — assuming {DEFAULT_FILE_FPS:g} fps for speed timing.")
        reported = DEFAULT_FILE_FPS

    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta = _live_meta(capture)
    meta.update({
        "native_fps": reported,
        "frame_count": max(0, frames),
        "duration_s": round(frames / reported, 2) if frames > 0 else 0.0,
        "size_bytes": target.stat().st_size,
    })
    return capture, meta


def _live_meta(capture: cv2.VideoCapture) -> dict[str, Any]:
    """Read the frame geometry (and rate, where meaningful) off a capture."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "native_fps": fps if 1.0 <= fps <= 240.0 else 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
    }


def _is_page_url(url: str) -> bool:
    """True when a URL is a web page hosting a video, not the video itself."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == page or host.endswith("." + page) for page in PAGE_HOSTS)


def _direct_stream_url(url: str) -> str:
    """Return a URL OpenCV can open, resolving a page link through yt-dlp.

    Public "live traffic camera" links are very often YouTube streams, which
    are pages rather than video. yt-dlp turns one into the underlying HLS
    manifest; without it installed, say so plainly instead of failing with an
    opaque FFmpeg error.
    """
    if not _is_page_url(url):
        return url

    try:
        import yt_dlp  # noqa: PLC0415 - optional, only needed for page URLs
    except ImportError:
        raise SourceError(
            f"{urlparse(url).hostname} serves a web page, not a video stream. "
            "Install yt-dlp to use links like this: pip install yt-dlp") from None

    options = {"quiet": True, "no_warnings": True, "skip_download": True,
               "format": "best[protocol^=m3u8]/best"}
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - any yt-dlp failure is the same to us
        raise SourceError(f"yt-dlp could not resolve {url}: {exc}") from exc

    if "url" not in info and info.get("entries"):
        info = info["entries"][0]
    direct = info.get("url")
    if not direct:
        raise SourceError(f"yt-dlp found no playable stream at {url}.")
    print(f"[CV] Resolved {urlparse(url).hostname} to a direct stream via yt-dlp.")
    return str(direct)


def open_spec(spec: SourceSpec, width: int, height: int) -> tuple[cv2.VideoCapture, dict]:
    """Open any classified source, returning the capture and its metadata."""
    if spec.kind == KIND_CAMERA:
        return _open_camera(spec.target, width, height)
    if spec.kind == KIND_STREAM:
        return _open_stream(spec.target)
    return _open_file(spec.target)


def probe(spec: str, width: int = 1280, height: int = 720) -> dict[str, Any]:
    """Open a source, read one frame, close it, and report what was found.

    Used to validate an upload or a pasted URL before committing the live
    pipeline to it, so a bad address fails with a message instead of a black
    feed.
    """
    parsed = classify(spec)
    capture, meta = open_spec(parsed, width, height)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise SourceError(
                f"{parsed.label} opened but delivered no frame. A camera app "
                "may need to be connected first, or the stream may be offline.")
        meta["height"], meta["width"] = frame.shape[:2]
    finally:
        capture.release()
    return {**parsed.to_dict(), **meta}


# ═══════════════════════════════════════════════════════════════════════════
# THE LIVE SOURCE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Read:
    """The outcome of one attempt to get a frame.

    ``frame is None`` with ``ended`` false means "nothing right now, ask
    again" — the source is paused, or a live stream hiccupped. ``ended`` means
    the source is finished and will not produce more frames until it is
    replaced or restarted.
    """

    frame: np.ndarray | None
    #: Media-clock timestamp for this frame, in epoch seconds.
    time: float
    ended: bool = False


class VideoSource:
    """A swappable supply of frames with a media clock attached.

    One instance lives for the whole run; :meth:`switch` replaces what it is
    reading from without disturbing anything holding a reference to it. Every
    method is safe to call from the HTTP thread while the capture loop is
    reading, which is the point: the dashboard changes the source, the capture
    loop never learns that anything happened beyond a single discontinuity.
    """

    def __init__(self, spec: str | None, width: int = 1920, height: int = 1080, *,
                 loop: bool = False, playback_rate: float = 1.0,
                 drop_frames: bool = True, reconnect: bool = True) -> None:
        """Open ``spec`` immediately, raising :class:`SourceError` if it fails.

        Pass ``None`` to start with nothing open. The server does that when the
        configured camera is missing: an empty source is a better failure than
        a dead process, because the operator can then upload a video or point
        the node at a stream without touching the machine it runs on.
        """
        self._lock = threading.RLock()
        self._width = width
        self._height = height
        self._loop = bool(loop)
        self._rate = max(0.0, float(playback_rate))
        self._drop = bool(drop_frames)
        self._reconnect = bool(reconnect)

        self._capture: cv2.VideoCapture | None = None
        self._spec: SourceSpec | None = None
        self._meta: dict[str, Any] = {}

        # -- the media clock ------------------------------------------------
        self._clock_base = time.time()   # epoch the current source started at
        self._elapsed = 0.0              # media seconds since that base
        self._wall_open = time.time()    # wall clock at open, for live sources
        self._frame_interval = 0.0       # 0 for live sources

        # -- position and pacing --------------------------------------------
        self._pos_frames = 0             # where we are *in the file*
        self._pace_wall = time.time()
        self._pace_media = 0.0
        #: When the last frame was handed over, so the time the caller spent on
        #: it can be measured. That is the yardstick the skip budget uses.
        self._handed_over = time.perf_counter()
        self._caller_ms = 0.0
        #: True once a file has been found impossible to decode in real time.
        #: Reported rather than merely endured, because "the video is playing
        #: slowly" is otherwise indistinguishable from "the detector is slow".
        self._behind_realtime = False

        # -- counters --------------------------------------------------------
        self._frames_read = 0
        self._frames_skipped = 0
        self._reconnects = 0
        self._discontinuities = 0
        self._pending_discontinuity = False
        self._failures = 0
        self._first_failure = 0.0

        self._paused = False
        self._finished = False

        if spec:
            self.switch(spec)
        # The first source is not a *change* of source; nothing has been built
        # from an earlier one that would need discarding.
        self._pending_discontinuity = False
        self._discontinuities = 0

    # -- properties ---------------------------------------------------------

    @property
    def media_time(self) -> float:
        """The current media-clock reading, in epoch seconds.

        Advances with the video for files and with the wall clock for live
        sources, and never runs backwards within one source.
        """
        return self._clock_base + self._elapsed

    @property
    def spec(self) -> SourceSpec | None:
        """The source currently being read, or ``None`` if nothing is open."""
        return self._spec

    @property
    def is_open(self) -> bool:
        """True when there is something to read frames from."""
        return self._capture is not None

    @property
    def is_file(self) -> bool:
        """True when the current source can be paused, seeked and looped."""
        return self._spec is not None and self._spec.is_file

    @property
    def finished(self) -> bool:
        """True once a file has played out, or a live source gave up."""
        return self._finished

    # -- reading ------------------------------------------------------------

    def read(self) -> Read:
        """Return the next frame, pacing and reconnecting as needed."""
        with self._lock:
            capture = self._capture
            if capture is None:
                return Read(None, self.media_time, ended=True)
            if self._paused or self._finished:
                return Read(None, self.media_time, ended=self._finished)
            if self.is_file:
                return self._read_file(capture)
            return self._read_live(capture)

    def _read_file(self, capture: cv2.VideoCapture) -> Read:
        """Read one frame from a file, on schedule."""
        # How long the caller had the last frame — the work this source exists
        # to feed. :meth:`_pace` spends it as the budget for skipping.
        self._caller_ms = (time.perf_counter() - self._handed_over) * 1000.0
        self._pace(capture)

        ok, frame = capture.read()
        if ok and frame is not None:
            self._advance(1)
            self._frames_read += 1
            self._handed_over = time.perf_counter()
            return Read(frame, self.media_time)

        # Out of frames.
        if self._loop and self._restart(capture):
            return Read(None, self.media_time)
        self._finished = True
        return Read(None, self.media_time, ended=True)

    def _read_live(self, capture: cv2.VideoCapture) -> Read:
        """Read one frame from a camera or stream, recovering from dropouts."""
        ok, frame = capture.read()
        self._elapsed = time.time() - self._wall_open

        if ok and frame is not None:
            self._failures = 0
            self._frames_read += 1
            return Read(frame, self.media_time)

        now = time.time()
        self._failures += 1
        if self._failures == 1:
            self._first_failure = now

        # A dead stream returns False instantly, so a plain attempt count would
        # burn through in microseconds. Give it a few seconds of real time.
        if now - self._first_failure < FAILURE_GRACE_SECONDS:
            time.sleep(0.02)
            return Read(None, self.media_time)

        if self._reconnect and self._reopen():
            return Read(None, self.media_time)

        self._finished = True
        return Read(None, self.media_time, ended=True)

    def _pace(self, capture: cv2.VideoCapture) -> None:
        """Hold a file to real time, skipping frames when we have fallen behind.

        Sleeping when early keeps the video at its filmed speed. Skipping when
        late is what a live camera does implicitly — a frame not read in time
        is simply gone — and because skipped frames still advance the media
        clock, the km/h readings do not care either way.
        """
        if self._rate <= 0.0 or self._frame_interval <= 0.0:
            return

        due = self._pace_wall + (self._elapsed - self._pace_media) / self._rate
        slack = due - time.time()
        if slack > 0.0:
            time.sleep(min(slack, 1.0))
            return
        if not self._drop:
            return

        behind = int((-slack) * self._rate / self._frame_interval)
        if behind > MAX_CATCHUP_FRAMES:
            # Not a hiccup — the process was stalled (a long workbook write, a
            # laptop asleep, a remote source taking half a minute to refuse a
            # connection). Fast-forwarding through minutes of road to "catch
            # up" would throw away the very footage being watched, so take the
            # lost time as lost and carry on from here.
            self._resync_pace()
            return

        # Never spend longer throwing frames away than the caller spent using
        # the last one. Skipping is overhead — it exists so the *next* frame is
        # the right one, not because anything looks at what it discards — and
        # once the overhead outgrows the work it serves, staying in real time
        # has stopped being worth its price.
        budget_ms = self._caller_ms * CATCHUP_TIME_BUDGET / (1.0 - CATCHUP_TIME_BUDGET)
        started = time.perf_counter()
        skipped = 0
        for _ in range(behind):
            if not capture.grab():
                break
            skipped += 1
            if (time.perf_counter() - started) * 1000.0 >= budget_ms:
                break
        if skipped:
            self._advance(skipped)
            self._frames_skipped += skipped

        if skipped < behind:
            # Out of budget with the debt unpaid. Forgive it rather than carry
            # it: left on the books it only grows, and every pass round the
            # loop would spend longer skipping than the one before until the
            # ceiling above finally caught it — the runaway this budget is here
            # to prevent. Giving up on real time costs nothing measured, since
            # every reading is timed against the media clock; the video simply
            # takes longer to play than it does to watch.
            self._resync_pace()
            self._note_behind_realtime()

    def _note_behind_realtime(self) -> None:
        """Say once that this source cannot be decoded as fast as it was shot."""
        if self._behind_realtime:
            return
        self._behind_realtime = True
        name = self._spec.label if self._spec else "this source"
        print(f"[CV] {name} cannot be decoded at {1.0 / self._frame_interval:.0f} "
              f"fps on this machine, so it is playing slower than real time "
              f"rather than dropping most of its frames. Speeds, counts and "
              f"congestion are unaffected — they are measured against the "
              f"video's own clock — but the run takes longer than the footage.")

    def _advance(self, frames: int) -> None:
        """Move the media clock and the file position on by ``frames`` frames."""
        self._elapsed += frames * self._frame_interval
        self._pos_frames += frames

    def _restart(self, capture: cv2.VideoCapture) -> bool:
        """Seek a looping file back to the start. Returns False if it will not."""
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok = capture.grab()
        if not ok:
            return False
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._pos_frames = 0
        self._resync_pace()
        # The content jumps; every track in flight belongs to the old pass.
        self._mark_discontinuity()
        return True

    def _reopen(self) -> bool:
        """Reconnect to the current live source. Assumes the lock is held."""
        assert self._spec is not None
        for attempt in range(1, RECONNECT_ATTEMPTS + 1):
            print(f"[CV] {self._spec.label} dropped out — reconnecting "
                  f"({attempt}/{RECONNECT_ATTEMPTS})...")
            time.sleep(RECONNECT_BACKOFF_SECONDS * attempt)
            try:
                capture, meta = open_spec(self._spec, self._width, self._height)
            except SourceError as exc:
                print(f"[CV] {exc}")
                continue
            self._install(capture, meta, self._spec)
            self._reconnects += 1
            print(f"[CV] {self._spec.label} reconnected.")
            return True

        print(f"[CV] {self._spec.label} could not be reconnected.")
        return False

    # -- switching ----------------------------------------------------------

    def switch(self, spec: str, *, loop: bool | None = None) -> dict[str, Any]:
        """Replace the source being read. Returns the new source's status.

        The new source is opened **before** the lock is taken, so the capture
        loop keeps serving frames from the old one for as long as the new one
        takes to connect — which for a remote camera can be several seconds.
        Raises :class:`SourceError` without disturbing the current source if
        the new one cannot be opened.
        """
        parsed = classify(spec)
        capture, meta = open_spec(parsed, self._width, self._height)
        with self._lock:
            if loop is not None:
                self._loop = bool(loop)
            self._install(capture, meta, parsed)
            print(f"[CV] Source is now {parsed.label}"
                  f"{self._file_note()}.")
            return self.status()

    def _install(self, capture: cv2.VideoCapture, meta: dict[str, Any],
                 spec: SourceSpec) -> None:
        """Adopt an already-opened capture. Assumes the lock is held."""
        if self._capture is not None:
            self._capture.release()

        self._capture = capture
        self._meta = meta
        self._spec = spec

        native = float(meta.get("native_fps") or 0.0)
        self._frame_interval = 1.0 / native if (spec.is_file and native > 0) else 0.0

        # A new source starts a new clock. Nothing survives the switch that
        # could be confused by the jump, because the pipeline is reset with it.
        self._clock_base = time.time()
        self._elapsed = 0.0
        self._wall_open = time.time()
        self._pos_frames = 0
        self._paused = False
        self._finished = False
        self._failures = 0
        # A verdict about the last file says nothing about this one.
        self._behind_realtime = False
        self._resync_pace()
        self._mark_discontinuity()

    def _file_note(self) -> str:
        """A short description of a file's length, for the console line."""
        if not self.is_file:
            return ""
        duration = float(self._meta.get("duration_s") or 0.0)
        fps = float(self._meta.get("native_fps") or 0.0)
        if duration <= 0:
            return f" ({fps:.0f} fps)"
        return f" ({_hms(duration)} at {fps:.0f} fps)"

    def _resync_pace(self) -> None:
        """Re-anchor playback timing to now, after any jump in the timeline."""
        self._pace_wall = time.time()
        self._pace_media = self._elapsed
        # The gap either side of a pause, a seek or a switch is not time the
        # caller spent working, so it must not be spent as a skip budget.
        self._handed_over = time.perf_counter()
        self._caller_ms = 0.0

    def _mark_discontinuity(self) -> None:
        """Record that the picture jumped, so the pipeline can be reset."""
        self._pending_discontinuity = True
        self._discontinuities += 1

    def pop_discontinuity(self) -> bool:
        """Return True once after the content jumped, and clear the flag.

        A switch, a loop restart, a seek or a reconnect all mean the next frame
        has nothing to do with the last. Tracks, speeds and the class-vote
        history from before must be retired rather than carried across.
        """
        with self._lock:
            was, self._pending_discontinuity = self._pending_discontinuity, False
            return was

    # -- playback control ---------------------------------------------------

    def pause(self, paused: bool = True) -> dict[str, Any]:
        """Freeze or resume a file. Live sources cannot be paused."""
        with self._lock:
            if not self.is_file:
                raise SourceError("Only a video file can be paused — a live "
                                  "camera has nowhere to pause to.")
            self._paused = bool(paused)
            if not self._paused:
                self._resync_pace()
            return self.status()

    def toggle_pause(self) -> dict[str, Any]:
        """Flip between paused and playing."""
        with self._lock:
            return self.pause(not self._paused)

    def seek(self, *, fraction: float | None = None, seconds: float | None = None,
             delta_seconds: float | None = None) -> dict[str, Any]:
        """Jump to a position in the file, by fraction, absolute or relative time."""
        with self._lock:
            if not self.is_file or self._capture is None:
                raise SourceError("Only a video file can be seeked.")

            native = float(self._meta.get("native_fps") or DEFAULT_FILE_FPS)
            total = int(self._meta.get("frame_count") or 0)

            if fraction is not None:
                if total <= 0:
                    raise SourceError("This file does not report a length, so it "
                                      "can only be seeked by time.")
                target = int(_clamp(float(fraction), 0.0, 1.0) * total)
            elif seconds is not None:
                target = int(max(0.0, float(seconds)) * native)
            elif delta_seconds is not None:
                target = int(self._pos_frames + float(delta_seconds) * native)
            else:
                raise SourceError("Give a fraction, a time, or an offset to seek to.")

            if total > 0:
                target = int(_clamp(target, 0, max(0, total - 1)))
            target = max(0, target)

            self._capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            # Most codecs land on the nearest keyframe rather than the exact
            # frame, so believe the capture rather than the request.
            landed = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES) or target)
            self._pos_frames = max(0, landed)
            self._finished = False
            self._resync_pace()
            self._mark_discontinuity()
            return self.status()

    def restart(self) -> dict[str, Any]:
        """Play a file again from the beginning."""
        return self.seek(fraction=0.0)

    def set_loop(self, loop: bool) -> dict[str, Any]:
        """Choose whether a file restarts when it reaches the end."""
        with self._lock:
            self._loop = bool(loop)
            return self.status()

    def set_rate(self, rate: float) -> dict[str, Any]:
        """Set the playback speed multiplier; ``0`` means as fast as possible.

        This changes only how fast frames are fed in. Reported km/h are
        measured against the media clock and so are unaffected.
        """
        with self._lock:
            value = max(0.0, float(rate))
            if value > 16.0:
                raise SourceError("Playback rate is capped at 16x; use 0 for "
                                  "'as fast as this machine can manage'.")
            self._rate = value
            self._resync_pace()
            return self.status()

    def set_drop_frames(self, drop: bool) -> dict[str, Any]:
        """Choose whether frames may be skipped to keep a file on schedule."""
        with self._lock:
            self._drop = bool(drop)
            self._resync_pace()
            return self.status()

    # -- reporting ----------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Everything the dashboard needs to describe and control the source."""
        spec = self._spec
        native = float(self._meta.get("native_fps") or 0.0)
        total = int(self._meta.get("frame_count") or 0)
        duration = float(self._meta.get("duration_s") or 0.0)
        position = (self._pos_frames / native) if (self.is_file and native > 0) else 0.0

        return {
            "kind": spec.kind if spec else "none",
            "label": spec.label if spec else "none",
            "spec": spec.original if spec else "",
            "target": spec.target if spec else "",
            "is_file": self.is_file,
            "live": bool(spec and spec.is_live),
            "width": int(self._meta.get("width") or 0),
            "height": int(self._meta.get("height") or 0),
            "native_fps": round(native, 2),
            "frame_count": total,
            "duration_s": round(duration, 2),
            "duration_text": _hms(duration) if duration > 0 else "",
            "position_s": round(position, 2),
            "position_text": _hms(position) if self.is_file else "",
            "progress": round(_clamp(position / duration, 0.0, 1.0), 4) if duration > 0 else 0.0,
            "paused": self._paused,
            "finished": self._finished,
            "loop": self._loop,
            "playback_rate": self._rate,
            "drop_frames": self._drop,
            # True when this machine cannot decode the file as fast as it was
            # shot, so it is deliberately playing slower than real time. The
            # readings stay correct; only the wall clock stretches.
            "behind_realtime": self._behind_realtime,
            "frames_read": self._frames_read,
            "frames_skipped": self._frames_skipped,
            "reconnects": self._reconnects,
            "switches": self._discontinuities,
            "media_time": round(self.media_time, 3),
            "media_ahead_s": round(self.media_time - time.time(), 2),
        }

    def hud_text(self) -> str:
        """A one-line description of the source for the video overlay."""
        spec = self._spec
        if spec is None:
            return "no source"
        if not self.is_file:
            return spec.label
        status = self.status()
        state = "PAUSED" if status["paused"] else ("END" if status["finished"] else "PLAY")
        if status["duration_s"] > 0:
            return (f"{spec.label}  {status['position_text']}/"
                    f"{status['duration_text']}  [{state}]")
        return f"{spec.label}  [{state}]"

    def release(self) -> None:
        """Close the capture. The object stays usable via :meth:`switch`."""
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None


def _clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def _hms(seconds: float) -> str:
    """Format a duration as ``m:ss`` or ``h:mm:ss``."""
    total = int(max(0.0, seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# ═══════════════════════════════════════════════════════════════════════════
# UPLOADS
# ═══════════════════════════════════════════════════════════════════════════

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_upload_name(filename: str) -> str:
    """Reduce a submitted filename to something safe to write.

    Everything up to the last separator is discarded and the rest is stripped
    to letters, digits, dots, dashes and underscores, so a name cannot climb
    out of the upload directory or collide with a device path.
    """
    base = Path(str(filename or "").replace("\\", "/")).name
    cleaned = _UNSAFE.sub("_", base).strip("._") or "upload"
    stem = Path(cleaned).stem.strip("._")[:60] or "upload"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"
    return f"{stem}{suffix}"


def upload_path(filename: str) -> Path:
    """Return a free path in the upload directory for ``filename``."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_upload_name(filename)
    target = UPLOAD_DIR / name
    if not target.exists():
        return target

    stem, suffix = target.stem, target.suffix
    for index in range(2, 1000):
        candidate = UPLOAD_DIR / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    return UPLOAD_DIR / f"{stem}_{int(time.time())}{suffix}"


def _digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's contents, read a megabyte at a time.

    Uploads here are whole videos — some of them gigabytes — so the file is
    never read into memory in one piece.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def find_stored_copy(path: Path) -> Path | None:
    """Return an already-stored video byte-identical to ``path``, or ``None``.

    Size is checked first because it rules out almost every candidate for the
    cost of a stat, so at most the handful of files that are the same length as
    ``path`` are ever hashed.
    """
    if not UPLOAD_DIR.exists():
        return None
    size = path.stat().st_size
    candidates = [
        other for other in UPLOAD_DIR.iterdir()
        if other != path and other.is_file()
        and other.suffix.lower() in VIDEO_SUFFIXES
        and other.stat().st_size == size
    ]
    if not candidates:
        return None
    incoming = _digest(path)
    for other in candidates:
        if _digest(other) == incoming:
            return other
    return None


def store_upload(save: Callable[[str], Any], filename: str) -> tuple[Path, bool]:
    """Store an uploaded video once, reusing an identical copy if we have one.

    ``save`` is called with a path to stream the upload onto disk — for a Flask
    upload that is ``FileStorage.save``. The bytes land on a temporary file
    first so they can be compared against what is already stored; if the same
    video is here under any name, the temporary file is dropped and the
    existing path is returned instead of a second copy.

    Returns ``(path, reused)``. Uploading the same video twice used to leave
    ``clip.mp4`` and ``clip_2.mp4`` side by side, which is how the upload
    directory grew to tens of gigabytes of identical footage. A *different*
    video that happens to share a name still gets the ``_2`` suffix, because
    the check is on content, not on the name.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    incoming = UPLOAD_DIR / f".incoming_{os.getpid()}_{time.monotonic_ns()}.part"
    try:
        save(str(incoming))
        existing = find_stored_copy(incoming)
        if existing is not None:
            return existing, True
        target = upload_path(filename)
        incoming.replace(target)
        return target, False
    finally:
        incoming.unlink(missing_ok=True)


def list_uploads() -> list[dict[str, Any]]:
    """List every video in the library directories, newest first.

    Covers footage copied onto the machine by hand as well as dashboard
    uploads, so a file dropped into ``vedios/`` is offered by the picker
    instead of having to be re-uploaded through the browser to become
    selectable.

    ``removable`` says whether the dashboard may delete it: uploads it made
    itself, yes; source footage it did not put there, no. A name appearing in
    both directories is listed once, as the upload.
    """
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for directory in LIBRARY_DIRS:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            if path.name in seen:
                continue
            seen.add(path.name)
            stat = path.stat()
            entries.append({
                "name": path.name,
                "spec": str(path),
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M",
                                             time.localtime(stat.st_mtime)),
                "modified": stat.st_mtime,
                "folder": directory.name,
                "removable": directory == UPLOAD_DIR,
            })
    return sorted(entries, key=lambda entry: entry["modified"], reverse=True)


def delete_upload(name: str) -> bool:
    """Delete one uploaded video by name. Returns False if it was not there.

    Only the last component of ``name`` is used, so nothing a caller sends can
    address a file outside the upload directory — a traversal attempt just
    names something that is not there and is reported as missing.
    """
    target = UPLOAD_DIR / Path(str(name).replace("\\", "/")).name
    if not target.is_file():
        return False
    target.unlink()
    return True
