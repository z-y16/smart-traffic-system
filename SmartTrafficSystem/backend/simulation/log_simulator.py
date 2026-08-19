"""Simulated system log generation for demo and development mode."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from backend.models.system_state import LogEntry, LogLevel


class LogSimulator:
    """Generates realistic system logs in simulation mode."""

    LOG_SOURCES: tuple[str, ...] = (
        "Camera",
        "YOLO",
        "ESP32",
        "Arduino",
        "Serial",
        "Database",
        "System",
        "Emergency",
    )

    LOG_MODULES: tuple[str, ...] = (
        "detection",
        "tracking",
        "communication",
        "hardware",
        "database",
        "analytics",
        "ui",
    )

    INFO_MESSAGES: tuple[str, ...] = (
        "System initialized successfully",
        "Camera frame captured",
        "Detection pipeline started",
        "Vehicle tracking updated",
        "Serial connection established",
        "Database query completed",
        "Emergency mode activated",
        "Emergency mode deactivated",
        "Lane opened for priority vehicle",
        "Speed bump lowered",
        "LED color updated",
        "LCD message sent",
        "Gate opened",
        "Gate closed",
        "WiFi signal strong",
        "Heartbeat received",
    )

    WARNING_MESSAGES: tuple[str, ...] = (
        "High CPU usage detected",
        "Memory usage above threshold",
        "Camera frame rate dropped",
        "Detection confidence low",
        "Serial communication delay",
        "WiFi signal weak",
        "Voltage slightly low",
        "Current draw elevated",
        "Database query slow",
        "Tracking ID conflict",
    )

    ERROR_MESSAGES: tuple[str, ...] = (
        "Camera connection lost",
        "YOLO model failed to load",
        "Serial connection failed",
        "ESP32 not responding",
        "Arduino communication error",
        "Database connection failed",
        "Emergency detection timeout",
        "Hardware command failed",
        "Gate operation failed",
        "Servo motor stuck",
        "LED strip not responding",
        "LCD display error",
    )

    CRITICAL_MESSAGES: tuple[str, ...] = (
        "System crash imminent",
        "Power supply failure",
        "Critical hardware failure",
        "Database corruption detected",
        "Multiple subsystems offline",
        "Emergency system failure",
    )

    def __init__(self, max_logs: int = 1000) -> None:
        """Initialize simulator with optional max log count."""
        self._max_logs = max_logs
        self._logs: list[LogEntry] = []
        self._seed_time = datetime.now()

    def _generate_log_entry(self, offset_seconds: int) -> LogEntry:
        """Generate a single realistic log entry."""
        timestamp = self._seed_time - timedelta(seconds=offset_seconds)
        level_roll = random.random()

        if level_roll < 0.05:
            level = LogLevel.CRITICAL
            message = random.choice(self.CRITICAL_MESSAGES)
        elif level_roll < 0.15:
            level = LogLevel.ERROR
            message = random.choice(self.ERROR_MESSAGES)
        elif level_roll < 0.35:
            level = LogLevel.WARNING
            message = random.choice(self.WARNING_MESSAGES)
        else:
            level = LogLevel.INFO
            message = random.choice(self.INFO_MESSAGES)

        source = random.choice(self.LOG_SOURCES)
        module = random.choice(self.LOG_MODULES)

        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            source=source,
            module=module,
        )

    def generate_initial_logs(self, count: int = 100) -> list[LogEntry]:
        """Generate initial batch of logs."""
        self._logs = []
        for i in range(count):
            offset = i * random.randint(5, 60)
            log = self._generate_log_entry(offset)
            self._logs.append(log)
        self._logs.sort(key=lambda x: x.timestamp, reverse=True)
        return self._logs

    def add_random_log(self) -> LogEntry:
        """Add a new random log entry."""
        from datetime import datetime

        timestamp = datetime.now()
        level_roll = random.random()

        if level_roll < 0.05:
            level = LogLevel.CRITICAL
            message = random.choice(self.CRITICAL_MESSAGES)
        elif level_roll < 0.15:
            level = LogLevel.ERROR
            message = random.choice(self.ERROR_MESSAGES)
        elif level_roll < 0.35:
            level = LogLevel.WARNING
            message = random.choice(self.WARNING_MESSAGES)
        else:
            level = LogLevel.INFO
            message = random.choice(self.INFO_MESSAGES)

        source = random.choice(self.LOG_SOURCES)
        module = random.choice(self.LOG_MODULES)

        log = LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            source=source,
            module=module,
        )
        self._logs.insert(0, log)
        if len(self._logs) > self._max_logs:
            self._logs.pop()
        return log

    def get_logs(
        self,
        level_filter: LogLevel | None = None,
        source_filter: str | None = None,
        search_query: str | None = None,
        limit: int = 100,
    ) -> list[LogEntry]:
        """Return filtered logs."""
        filtered = self._logs

        if level_filter:
            filtered = [log for log in filtered if log.level == level_filter]

        if source_filter:
            filtered = [log for log in filtered if log.source == source_filter]

        if search_query:
            search_lower = search_query.lower()
            filtered = [
                log
                for log in filtered
                if search_lower in log.message.lower()
                or search_lower in log.source.lower()
                or search_lower in log.module.lower()
            ]

        return filtered[:limit]

    def get_log_count_by_level(self) -> dict[LogLevel, int]:
        """Return count of logs by level."""
        counts = {level: 0 for level in LogLevel}
        for log in self._logs:
            counts[log.level] += 1
        return counts

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._logs = []
