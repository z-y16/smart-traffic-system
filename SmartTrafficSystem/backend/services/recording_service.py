"""Session recording control and access to the all-sessions archive.

The CV node keeps two records side by side:

* the **current session** — only the run happening right now, which is what
  the export button on most pages hands you;
* the **master archive** — every session ever recorded, each row stamped with
  its session id, so nothing from a past run is ever lost.

This service is the dashboard's door to both. Every call degrades gracefully:
if the broadcast server is unreachable the methods return empty results rather
than raising, so a page can still render while the CV node is down.
"""

from __future__ import annotations

import pandas as pd
import requests

from backend.services.system_status_service import (
    ARCHIVE_TIMEOUT_SECONDS,
    EXPORT_ALL_CSV_URL,
    EXPORT_ALL_URL,
    LIGHT_URL,
    RECORD_SAVE_URL,
    RECORD_START_URL,
    RECORD_STATUS_URL,
    RECORD_STOP_URL,
    RECORDS_URL,
    REQUEST_TIMEOUT_SECONDS,
    SESSIONS_URL,
)


class RecordingService:
    """Start/stop session recordings and read the all-sessions archive."""

    def __init__(self) -> None:
        """Initialise with an empty archive cache."""
        self._records_cache: pd.DataFrame | None = None
        self._cache_key: tuple[str, int] | None = None
        self._sessions_cache: pd.DataFrame | None = None
        self._last_error: str | None = None

    @property
    def last_error(self) -> str | None:
        """The most recent connection error, or ``None`` if the last call worked."""
        return self._last_error

    # -- recording control -----------------------------------------------------

    def status(self) -> dict:
        """Return the current session's id, name, row count and armed state."""
        return self._get_json(RECORD_STATUS_URL, default={
            "recording": False, "rows": 0, "name": "", "id": "", "elapsed_s": 0.0,
        })

    def start(self, name: str = "") -> dict:
        """Begin a new named recording, closing whatever session was open."""
        self.invalidate()
        return self._post_json(RECORD_START_URL, payload={"name": name})

    def stop(self) -> dict:
        """Finish the current recording and return what it captured."""
        self.invalidate()
        return self._post_json(RECORD_STOP_URL)

    def save_now(self) -> dict:
        """Force the CV node to write both workbooks immediately."""
        return self._post_json(RECORD_SAVE_URL)

    def is_reachable(self) -> bool:
        """Return whether the broadcast server answered the last status call."""
        self.status()
        return self._last_error is None

    # -- signal ----------------------------------------------------------------

    def light(self) -> dict:
        """Return the live traffic signal state."""
        return self._get_json(LIGHT_URL, default={})

    def set_light(self, phase: str | None) -> dict:
        """Force a signal phase, or pass ``None``/``"AUTO"`` to release control."""
        return self._post_json(LIGHT_URL, payload={"phase": phase or "AUTO"})

    # -- archive ---------------------------------------------------------------

    def sessions(self) -> pd.DataFrame:
        """Return one summary row per session ever recorded, newest first.

        Cached like ``records()``, and dropped by the same ``invalidate()``.
        Answering this costs the node a full read and regroup of the archive,
        and the Analytics page asks on every rerun — so every keystroke in its
        sidebar re-read every row ever logged.
        """
        if self._sessions_cache is not None:
            return self._sessions_cache

        rows = self._get_json(SESSIONS_URL, default=[],
                              timeout=ARCHIVE_TIMEOUT_SECONDS)
        if not rows:
            # Not cached: an unreachable node is a transient state, and caching
            # the empty frame would hide the session list until a save or a
            # recording happened to clear it.
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        if "Session" in frame.columns:
            frame = frame.sort_values("Session", ascending=False).reset_index(drop=True)
        self._sessions_cache = frame
        return frame

    def records(self, session: str | None = None, limit: int = 20000) -> pd.DataFrame:
        """Return raw telemetry rows from the archive.

        Pass a session id to restrict the result to one run, or leave it out
        for every row across every session. Results are cached per
        (session, limit) so switching tabs does not re-download the archive.
        """
        key = (session or "__all__", limit)
        if self._records_cache is not None and self._cache_key == key:
            return self._records_cache

        params: dict[str, object] = {"limit": limit}
        if session:
            params["session"] = session
        rows = self._get_json(RECORDS_URL, default=[], params=params,
                              timeout=ARCHIVE_TIMEOUT_SECONDS)
        frame = pd.DataFrame(rows) if rows else pd.DataFrame()
        self._records_cache = frame
        self._cache_key = key
        return frame

    def invalidate(self) -> None:
        """Drop the cached archive so the next read re-fetches it."""
        self._records_cache = None
        self._cache_key = None
        self._sessions_cache = None

    def fetch_master_excel(self) -> bytes | None:
        """Download the all-sessions workbook, or ``None`` if unreachable."""
        return self._get_bytes(EXPORT_ALL_URL)

    def fetch_master_csv(self) -> bytes | None:
        """Download the all-sessions CSV, or ``None`` if unreachable."""
        return self._get_bytes(EXPORT_ALL_CSV_URL)

    # -- transport -------------------------------------------------------------

    def _get_json(self, url: str, default, params: dict | None = None,
                  timeout: float = REQUEST_TIMEOUT_SECONDS):
        """GET JSON, returning ``default`` and recording the error on failure."""
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            self._last_error = None
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return default

    def _post_json(self, url: str, payload: dict | None = None) -> dict:
        """POST JSON, returning an ``{"error": ...}`` dict on failure."""
        try:
            response = requests.post(url, json=payload or {},
                                     timeout=ARCHIVE_TIMEOUT_SECONDS)
            response.raise_for_status()
            self._last_error = None
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._last_error = str(exc)
            return {"error": str(exc)}

    def _get_bytes(self, url: str) -> bytes | None:
        """GET raw bytes, returning ``None`` on failure."""
        try:
            response = requests.get(url, timeout=ARCHIVE_TIMEOUT_SECONDS)
            response.raise_for_status()
            self._last_error = None
            return response.content
        except requests.exceptions.RequestException as exc:
            self._last_error = str(exc)
            return None
