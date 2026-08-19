"""SQLite database management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import DATABASE_PATH, DATA_DIR


def ensure_data_directories() -> None:
    """Create data and logs directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    ensure_data_directories()
    path = db_path or DATABASE_PATH
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path | None = None) -> None:
    """Create database tables if they are missing."""
    ensure_data_directories()
    connection = get_connection(db_path)
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_class TEXT,
            tracking_id INTEGER,
            confidence REAL,
            lane TEXT
        );

        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            command TEXT NOT NULL,
            status TEXT,
            response TEXT
        );

        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            source TEXT,
            message TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS traffic_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_count INTEGER,
            density TEXT,
            average_speed REAL
        );

        CREATE TABLE IF NOT EXISTS emergency_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lane TEXT,
            confidence REAL,
            status TEXT
        );
        """
    )
    connection.commit()
    connection.close()
