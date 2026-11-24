"""Clock abstraction. EVERY module must take time from a ``Clock``.

Never call ``datetime.now()`` directly in trading code: the REPLAY mode swaps in a
``SimulatedClock`` and the data layer only returns candles that closed at or before
``clock.now()``. That is how look-ahead bias is prevented (section 16).
"""

from __future__ import annotations

import threading
from datetime import date, datetime, time, timedelta
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@runtime_checkable
class Clock(Protocol):
    """Source of the current IST time."""

    def now(self) -> datetime:
        """Return the current timezone-aware IST datetime."""
        ...


class SystemClock:
    """Wall-clock time in IST."""

    def now(self) -> datetime:
        return datetime.now(IST)


class SimulatedClock:
    """Settable clock used by tests and REPLAY mode. Thread-safe."""

    def __init__(self, start: datetime) -> None:
        self._now = to_ist(start)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            return self._now

    def set(self, dt: datetime) -> None:
        """Move the clock to ``dt``. Time may never go backwards."""
        dt = to_ist(dt)
        with self._lock:
            if dt < self._now:
                raise ValueError(f"SimulatedClock cannot go backwards: {dt} < {self._now}")
            self._now = dt

    def advance(self, delta: timedelta) -> datetime:
        """Advance the clock by ``delta`` and return the new time."""
        with self._lock:
            if delta < timedelta(0):
                raise ValueError("SimulatedClock cannot go backwards")
            self._now = self._now + delta
            return self._now


def to_ist(dt: datetime) -> datetime:
    """Convert ``dt`` to IST. Naive datetimes are assumed to already be IST."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def ist_datetime(d: date, t: time | str) -> datetime:
    """Combine a date and an IST wall time (``time`` or ``"HH:MM"``)."""
    if isinstance(t, str):
        t = parse_hhmm(t)
    return datetime.combine(d, t, tzinfo=IST)


def parse_hhmm(value: str) -> time:
    """Parse ``"HH:MM"`` (or ``"HH:MM:SS"``) into a ``time``."""
    parts = [int(p) for p in value.strip().split(":")]
    if len(parts) == 2:
        return time(parts[0], parts[1])
    if len(parts) == 3:
        return time(parts[0], parts[1], parts[2])
    raise ValueError(f"Invalid HH:MM time: {value!r}")
