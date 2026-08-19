"""Global application settings and constants."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
ASSETS_DIR: Path = ROOT_DIR / "assets"
LOGS_DIR: Path = ROOT_DIR / "logs"
DATABASE_PATH: Path = DATA_DIR / "traffic_system.db"
SETTINGS_PATH: Path = DATA_DIR / "settings.json"


@dataclass
class Settings:
    """Runtime configuration for the Smart Traffic Management System."""

    app_title: str = "Smart Traffic Management System"
    app_icon: str = "🚦"
    # Live by default: ``start.py`` brings the CV node up alongside this
    # dashboard, and the node holds the board open, so the hardware pages have
    # something real to read and drive. In simulation mode the buttons on those
    # pages stop short of the wire, which looks identical on screen -- a demo
    # that quietly drove nothing was the failure worth defaulting away from.
    # Every page still falls back to simulated values when the node is not
    # answering, so this is safe with no hardware attached.
    simulation_mode: bool = False
    theme: str = "dark"
    detection_threshold: float = 0.65
    # Reported for reference only. The dashboard never opens a serial port --
    # the CV node owns it (``broadcast_server.py --arduino-port``), because on
    # Windows a second process cannot open the same COM port at all. COM6 is
    # the ESP32, the one board the PC is wired to; the Unos hang off its UARTs.
    serial_port: str = "COM6"
    baud_rate: int = 115200
    camera_index: int = 0
    refresh_interval_ms: int = 2000
    # Where broadcast_server.py is reachable. Loopback by default: the node and
    # the dashboard normally run on the same machine, and a LAN address here is
    # a DHCP lease, not a fact -- when the router hands out a different one the
    # dashboard cannot reach the node, falls back to the simulator, and reports
    # zero vehicles at zero km/h with no error. Loopback cannot go stale.
    # Set this to the node's LAN address only when it really is another machine.
    stream_host: str = "127.0.0.1"
    stream_port: int = 8502
    lanes: list[str] = field(default_factory=lambda: ["Lane A", "Lane B", "Lane C", "Lane D"])

    def to_dict(self) -> dict[str, Any]:
        """Return settings as a plain dictionary."""
        return {
            "app_title": self.app_title,
            "simulation_mode": self.simulation_mode,
            "theme": self.theme,
            "detection_threshold": self.detection_threshold,
            "serial_port": self.serial_port,
            "baud_rate": self.baud_rate,
            "camera_index": self.camera_index,
            "refresh_interval_ms": self.refresh_interval_ms,
            "stream_host": self.stream_host,
            "stream_port": self.stream_port,
            "lanes": self.lanes,
        }

    def save(self) -> None:
        """Save settings to JSON file."""
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from JSON file or return defaults.

        Unknown keys are ignored so a settings file written by an older build
        cannot break startup.
        """
        data = _read_settings_file()
        if data is None:
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


#: Parsed contents of ``settings.json``, kept against the file's
#: (mtime, size) so a save on the Settings page is still picked up at once.
_FILE_CACHE: tuple[tuple[float, int], dict[str, Any]] | None = None


def _read_settings_file() -> dict[str, Any] | None:
    """Return the parsed settings file, re-reading it only when it changes.

    Every page and every service calls ``get_settings()``, several times per
    rerun between them, and each call used to open and parse the file afresh.
    """
    global _FILE_CACHE

    try:
        stat = SETTINGS_PATH.stat()
    except OSError:
        _FILE_CACHE = None
        return None

    stamp = (stat.st_mtime, stat.st_size)
    if _FILE_CACHE is not None and _FILE_CACHE[0] == stamp:
        return _FILE_CACHE[1]

    try:
        with open(SETTINGS_PATH, "r") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    _FILE_CACHE = (stamp, data)
    return data


def get_settings() -> Settings:
    """Return the active application settings instance.

    A fresh instance every time, deliberately: pages write to the object they
    are handed — the Live Camera page assigns the confidence slider onto it —
    and handing out one shared instance would leak those edits into every
    other browser session. Only the file read behind it is cached.
    """
    return Settings.load()
