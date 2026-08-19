"""Service for application settings management."""

from __future__ import annotations

from config.settings import Settings, get_settings


class SettingsService:
    """Manages application settings persistence and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service with optional settings."""
        self._settings = settings or get_settings()

    def get_settings(self) -> Settings:
        """Return current settings."""
        return self._settings

    def update_settings(self, **kwargs) -> Settings:
        """Update settings with provided values."""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        return self._settings

    def save_settings(self) -> None:
        """Save current settings to file."""
        self._settings.save()

    def reset_to_defaults(self) -> Settings:
        """Reset settings to default values.

        Updates in place rather than rebinding, because the running services
        hold a reference to this same instance.
        """
        defaults = Settings()
        for field in Settings.__dataclass_fields__:
            setattr(self._settings, field, getattr(defaults, field))
        return self._settings

    def get_available_serial_ports(self) -> list[str]:
        """Return selectable serial ports, always including the configured one.

        pyserial reports an empty list on a machine with no adapter attached,
        which would leave the Settings dropdown empty and save ``None`` over a
        perfectly valid port. The saved port and the usual COM names are always
        offered so the field stays editable with no hardware present.
        """
        detected: list[str] = []
        try:
            import serial.tools.list_ports

            detected = [port.device for port in serial.tools.list_ports.comports()]
        except Exception:
            detected = []

        ports = list(detected)
        for fallback in [self._settings.serial_port,
                         "COM1", "COM2", "COM3", "COM4", "COM5", "COM6"]:
            if fallback and fallback not in ports:
                ports.append(fallback)
        return ports

    def get_available_baud_rates(self) -> list[int]:
        """Return list of common baud rates."""
        return [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

    def get_available_cameras(self, scan: bool = False) -> list[int]:
        """Return selectable camera indices.

        Probing devices costs seconds and prints driver errors, so it only runs
        when ``scan`` is set (the Settings page's "Rescan cameras" button).
        Otherwise the configured index is offered without touching hardware.
        """
        if not scan:
            return sorted({0, self._settings.camera_index})

        try:
            import cv2
        except Exception:
            return sorted({0, self._settings.camera_index})

        # CAP_DSHOW on Windows avoids the noisy obsensor/MSMF probe path, and a
        # camera that opens but yields no frame is not actually usable.
        backend = getattr(cv2, "CAP_DSHOW", 0)
        available: list[int] = []
        for index in range(5):
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened() and capture.read()[0]:
                    available.append(index)
            finally:
                capture.release()

        return available or sorted({0, self._settings.camera_index})
