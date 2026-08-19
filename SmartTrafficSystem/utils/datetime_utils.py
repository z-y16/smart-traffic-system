"""Date and time formatting utilities."""

from datetime import datetime


def format_current_time() -> str:
    """Return the current time formatted for display."""
    return datetime.now().strftime("%H:%M:%S")


def format_current_date() -> str:
    """Return the current date formatted for display."""
    return datetime.now().strftime("%a, %d %b %Y")


def format_timestamp(value: datetime) -> str:
    """Return a datetime value formatted for dashboard display."""
    return value.strftime("%Y-%m-%d %H:%M:%S")
