"""Shared time helpers. All stored datetimes are tz-aware UTC; display is London local time."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
LONDON = ZoneInfo("Europe/London")
SLOT = timedelta(minutes=30)  # planning resolution (Agile half-hours)


def to_utc(dt: datetime) -> datetime:
    """Normalise to tz-aware UTC. Naive values (e.g. read back from SQLite) are taken as UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def fmt_local(dt: datetime | None, fmt: str = "%H:%M") -> str:
    """Format a datetime in London local time; empty string for None."""
    return to_utc(dt).astimezone(LONDON).strftime(fmt) if dt else ""
