"""Control over what the CV node is watching.

The detection pipeline does not care whether frames come from the camera on
the desk, a video someone recorded on a bridge, or a roadside camera's RTSP
stream — so the dashboard should not have to either. This service is the door
to all three: list what is available, upload a video, point the node at a URL,
and drive playback of a file that is running.

Every call degrades gracefully. If the CV node is unreachable the methods
return an ``{"error": ...}`` dict rather than raising, so the Live Camera page
still renders and can say what is wrong.
"""

from __future__ import annotations

from typing import Any, BinaryIO

import requests

from backend.services.system_status_service import (
    REQUEST_TIMEOUT_SECONDS,
    SOURCE_CONTROL_URL,
    SOURCE_TIMEOUT_SECONDS,
    SOURCE_UPLOAD_URL,
    SOURCE_URL,
    SOURCES_URL,
    UPLOAD_TIMEOUT_SECONDS,
)


class VideoSourceService:
    """Read and change the CV node's video source."""

    def __init__(self) -> None:
        """Initialise with no recorded error."""
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        """The most recent connection error, or ``None`` if the last call worked."""
        return self._last_error

    # -- reading ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return what the node is watching right now, and where it has got to."""
        return self._get(SOURCE_URL, default={})

    def catalogue(self) -> dict[str, Any]:
        """Return the cameras this node can see and the videos already uploaded."""
        return self._get(SOURCES_URL, default={"cameras": [], "uploads": [], "current": {}})

    # -- changing --------------------------------------------------------------

    def use(self, spec: str, loop: bool | None = None) -> dict[str, Any]:
        """Point the node at a camera, a video file or a stream URL.

        The node opens the new source before dropping the old one, so a URL
        that turns out to be unreachable comes back as an error here while the
        feed carries on undisturbed.
        """
        payload: dict[str, Any] = {"source": spec}
        if loop is not None:
            payload["loop"] = bool(loop)
        return self._post(SOURCE_URL, payload, timeout=SOURCE_TIMEOUT_SECONDS)

    def upload(self, filename: str, data: BinaryIO | bytes,
               play: bool = True, loop: bool = False) -> dict[str, Any]:
        """Send a video to the node, which stores it and starts playing it.

        The node decodes one frame before switching to it, so an unplayable
        file is reported here rather than becoming a black feed.
        """
        try:
            response = requests.post(
                SOURCE_UPLOAD_URL,
                files={"file": (filename, data, "application/octet-stream")},
                data={"play": str(bool(play)).lower(), "loop": str(bool(loop)).lower()},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
            body = response.json()
            self._last_error = body.get("error")
            return body
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return {"error": str(exc)}

    def delete(self, name: str) -> dict[str, Any]:
        """Delete one uploaded video from the node."""
        try:
            response = requests.delete(f"{SOURCE_UPLOAD_URL}/{name}",
                                       timeout=SOURCE_TIMEOUT_SECONDS)
            body = response.json()
            self._last_error = body.get("error")
            return body
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return {"error": str(exc)}

    # -- playback --------------------------------------------------------------

    def control(self, action: str, **params: Any) -> dict[str, Any]:
        """Send one playback command: pause, resume, toggle, restart, seek, loop, rate."""
        return self._post(SOURCE_CONTROL_URL, {"action": action, **params})

    def pause(self) -> dict[str, Any]:
        """Freeze the video on the current frame."""
        return self.control("pause")

    def resume(self) -> dict[str, Any]:
        """Resume a paused video."""
        return self.control("resume")

    def restart(self) -> dict[str, Any]:
        """Play the video again from the beginning."""
        return self.control("restart")

    def seek(self, fraction: float) -> dict[str, Any]:
        """Jump to a position in the video, as a fraction of its length."""
        return self.control("seek", fraction=float(fraction))

    def nudge(self, seconds: float) -> dict[str, Any]:
        """Jump forward (or, with a negative value, back) by some seconds."""
        return self.control("seek", delta=float(seconds))

    def set_loop(self, loop: bool) -> dict[str, Any]:
        """Choose whether the video restarts when it reaches the end."""
        return self.control("loop", value=bool(loop))

    def set_rate(self, rate: float) -> dict[str, Any]:
        """Set playback speed. ``0`` means as fast as the node can manage.

        This changes only how quickly frames are fed to the detector. Speeds
        are measured against the video's own clock, so the km/h reported do
        not change with it.
        """
        return self.control("rate", value=float(rate))

    # -- transport -------------------------------------------------------------

    def _get(self, url: str, default: Any,
             timeout: float = REQUEST_TIMEOUT_SECONDS) -> Any:
        """GET JSON, returning ``default`` and recording the error on failure."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            self._last_error = None
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return default

    def _post(self, url: str, payload: dict[str, Any],
              timeout: float = SOURCE_TIMEOUT_SECONDS) -> dict[str, Any]:
        """POST JSON, returning the node's reply or an ``{"error": ...}`` dict.

        A rejected request answers with an error message describing what was
        wrong with it, which is worth surfacing verbatim — "nothing is
        answering at 192.168.1.9:554" tells the operator far more than a
        status code would.
        """
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            body = response.json()
            self._last_error = body.get("error")
            return body
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return {"error": str(exc)}
