"""Service for system log management and filtering."""

from __future__ import annotations

from backend.models.system_state import LogEntry, LogLevel
from backend.simulation.log_simulator import LogSimulator
from config.settings import Settings, get_settings


class SystemLogService:
    """Manages system log storage, retrieval, and filtering."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service with optional settings."""
        self._settings = settings or get_settings()
        self._simulator = LogSimulator(max_logs=1000)
        self._simulator.generate_initial_logs(count=100)

    def get_logs(
        self,
        level_filter: LogLevel | None = None,
        source_filter: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Return filtered logs."""
        return self._simulator.get_logs(level_filter, source_filter, search_query, limit)

    def get_log_count_by_level(self) -> dict[LogLevel, int]:
        """Return count of logs by level."""
        return self._simulator.get_log_count_by_level()

    def add_log(
        self,
        level: LogLevel,
        message: str,
        source: str,
        module: str,
    ) -> LogEntry:
        """Add a new log entry."""
        from datetime import datetime

        log = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            source=source,
            module=module,
        )
        self._simulator._logs.insert(0, log)
        if len(self._simulator._logs) > self._simulator._max_logs:
            self._simulator._logs.pop()
        return log

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._simulator.clear_logs()

    def get_available_sources(self) -> list[str]:
        """Return list of available log sources."""
        return list(self._simulator.LOG_SOURCES)

    def get_available_levels(self) -> list[LogLevel]:
        """Return list of available log levels."""
        return list(LogLevel)

    def add_random_log(self) -> LogEntry:
        """Add a random log entry for simulation purposes."""
        return self._simulator.add_random_log()
