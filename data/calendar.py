"""NSE trading calendar, market hours and bar-close semantics (no look-ahead).

Rules
-----
* Weekends are non-trading unless listed as a special session.
* Weekday exchange holidays come from ``data/static/nse_holidays.yaml`` (2024-2027), optionally
  merged with the NSE holiday-master API (``fetch=True``), then ``ScheduleSettings.holiday_overrides``
  (extra non-trading days) and ``ScheduleSettings.special_sessions`` (trading on otherwise
  non-trading days) are applied - overrides always win.
* Regular session 09:15-15:30 IST.

Bar close times (candle contract, CONTRACTS section 1):
* intraday bar closes at ``min(open + timeframe, 15:30 of that day)``;
* a daily bar closes at 15:30 of its date;
* a weekly bar closes at 15:30 of the last trading day of its (Mon-Sun) week.

Only bars with ``close_time <= now`` may be used for decisions (``closed_bars``).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from core.clock import IST, to_ist
from core.enums import TIMEFRAME_MINUTES, Timeframe

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import ScheduleSettings

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOLIDAY_FILE = STATIC_DIR / "nse_holidays.yaml"

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
SESSION_MINUTES = 375

#: Hard bound for calendar walks so a corrupt holiday set can never loop forever.
_MAX_WALK_DAYS = 3660

INTRADAY_TIMEFRAMES: frozenset[Timeframe] = frozenset(
    {Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1}
)


def as_timeframe(tf: Timeframe | str) -> Timeframe:
    """Coerce ``"5m"`` / ``Timeframe.M5`` (and common aliases) into a ``Timeframe``."""
    if isinstance(tf, Timeframe):
        return tf
    key = str(tf).strip().lower()
    aliases = {"1d": Timeframe.D1, "d": Timeframe.D1, "day": Timeframe.D1, "daily": Timeframe.D1,
               "1w": Timeframe.W1, "1wk": Timeframe.W1, "w": Timeframe.W1, "week": Timeframe.W1,
               "weekly": Timeframe.W1, "60m": Timeframe.H1, "1h": Timeframe.H1}
    if key in aliases:
        return aliases[key]
    return Timeframe(key)


class NSECalendar:
    """NSE equity-segment trading calendar.

    Args:
        holidays: weekday exchange holidays (weekend dates are harmless but redundant).
        special_sessions: dates that ARE trading days even if they fall on a weekend/holiday.
        holiday_names: optional ``date -> description`` for display / HOLIDAY events.
    """

    def __init__(
        self,
        holidays: Iterable[date] = (),
        special_sessions: Iterable[date] = (),
        holiday_names: dict[date, str] | None = None,
    ) -> None:
        self._special: frozenset[date] = frozenset(special_sessions)
        self._holidays: frozenset[date] = frozenset(d for d in holidays if d not in self._special)
        self._names: dict[date, str] = dict(holiday_names or {})

    # ------------------------------------------------------------------ properties
    @property
    def holidays(self) -> frozenset[date]:
        """Exchange holidays (excluding special sessions)."""
        return self._holidays

    @property
    def special_sessions(self) -> frozenset[date]:
        """Dates forced to be trading days."""
        return self._special

    def holiday_name(self, d: date) -> str | None:
        """Description of a holiday, ``"Weekend"`` for weekends, ``None`` for trading days."""
        if self.is_trading_day(d):
            return None
        if d in self._holidays:
            return self._names.get(d, "NSE holiday")
        return "Weekend"

    # ------------------------------------------------------------------ day rules
    def is_trading_day(self, d: date) -> bool:
        """True if NSE holds a regular session on ``d``."""
        d = _as_date(d)
        if d in self._special:
            return True
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def is_holiday(self, d: date) -> bool:
        """True for a WEEKDAY exchange holiday (plain weekends are not 'holidays')."""
        d = _as_date(d)
        return d.weekday() < 5 and not self.is_trading_day(d)

    def next_trading_day(self, d: date) -> date:
        """First trading day strictly after ``d``."""
        d = _as_date(d)
        for _ in range(_MAX_WALK_DAYS):
            d += timedelta(days=1)
            if self.is_trading_day(d):
                return d
        raise ValueError(f"No trading day within {_MAX_WALK_DAYS} days after {d}")

    def prev_trading_day(self, d: date) -> date:
        """Last trading day strictly before ``d``."""
        d = _as_date(d)
        for _ in range(_MAX_WALK_DAYS):
            d -= timedelta(days=1)
            if self.is_trading_day(d):
                return d
        raise ValueError(f"No trading day within {_MAX_WALK_DAYS} days before {d}")

    def add_trading_days(self, d: date, n: int) -> date:
        """Move ``n`` trading days from ``d`` (negative = backwards).

        ``n == 0`` returns ``d`` if it is a trading day, else the next trading day.
        ``add_trading_days(friday, 1)`` is the following Monday (if trading).
        """
        d = _as_date(d)
        if n == 0:
            return d if self.is_trading_day(d) else self.next_trading_day(d)
        step = self.next_trading_day if n > 0 else self.prev_trading_day
        for _ in range(abs(n)):
            d = step(d)
        return d

    def trading_days_between(self, start: date, end: date) -> list[date]:
        """All trading days in ``[start, end]`` (inclusive). Empty if ``end < start``."""
        start, end = _as_date(start), _as_date(end)
        out: list[date] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    def count_trading_days(self, start_exclusive: date, end_inclusive: date) -> int:
        """Number of trading days in ``(start_exclusive, end_inclusive]`` (holding-day counting)."""
        start_exclusive, end_inclusive = _as_date(start_exclusive), _as_date(end_inclusive)
        if end_inclusive <= start_exclusive:
            return 0
        return len(self.trading_days_between(start_exclusive + timedelta(days=1), end_inclusive))

    def holidays_between(self, start: date, end: date) -> list[date]:
        """Weekday exchange holidays in ``[start, end]`` (plain weekends excluded)."""
        start, end = _as_date(start), _as_date(end)
        out: list[date] = []
        d = start
        while d <= end:
            if self.is_holiday(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    # ------------------------------------------------------------------ session times
    def market_open_dt(self, d: date) -> datetime:
        """09:15 IST on ``d``."""
        return datetime.combine(_as_date(d), MARKET_OPEN, tzinfo=IST)

    def market_close_dt(self, d: date) -> datetime:
        """15:30 IST on ``d``."""
        return datetime.combine(_as_date(d), MARKET_CLOSE, tzinfo=IST)

    def is_market_open(self, dt: datetime) -> bool:
        """True if the regular session is live at ``dt`` (09:15 <= t < 15:30 on a trading day)."""
        dt = to_ist(dt)
        if not self.is_trading_day(dt.date()):
            return False
        return MARKET_OPEN <= dt.timetz().replace(tzinfo=None) < MARKET_CLOSE

    def week_first_trading_day(self, d: date) -> date | None:
        """First trading day of the Monday-Sunday week containing ``d`` (None if none)."""
        monday = _as_date(d) - timedelta(days=_as_date(d).weekday())
        days = self.trading_days_between(monday, monday + timedelta(days=6))
        return days[0] if days else None

    def week_last_trading_day(self, d: date) -> date | None:
        """Last trading day of the Monday-Sunday week containing ``d`` (None if the week has none)."""
        monday = _as_date(d) - timedelta(days=_as_date(d).weekday())
        days = self.trading_days_between(monday, monday + timedelta(days=6))
        return days[-1] if days else None

    def last_closed_session(self, now: datetime) -> date:
        """Most recent trading day whose session has fully closed at ``now``."""
        now = to_ist(now)
        today = now.date()
        if self.is_trading_day(today) and now >= self.market_close_dt(today):
            return today
        return self.prev_trading_day(today)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"NSECalendar(holidays={len(self._holidays)}, special_sessions={len(self._special)})"


# =========================================================================== helpers


def _as_date(d: date | datetime) -> date:
    if isinstance(d, datetime):
        return to_ist(d).date()
    return d


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


@functools.lru_cache(maxsize=8)
def _read_static_file(path_str: str, mtime: float) -> tuple[dict[date, str], tuple[date, ...]]:
    """Parse the static holiday YAML (cached per path+mtime)."""
    del mtime  # cache key only
    raw = yaml.safe_load(Path(path_str).read_text(encoding="utf-8")) or {}
    holidays: dict[date, str] = {}
    for item in raw.get("holidays") or []:
        if isinstance(item, dict):
            holidays[_parse_date(item["date"])] = str(item.get("name", "NSE holiday"))
        else:
            holidays[_parse_date(item)] = "NSE holiday"
    specials: list[date] = []
    for item in raw.get("special_sessions") or []:
        specials.append(_parse_date(item["date"] if isinstance(item, dict) else item))
    return holidays, tuple(specials)


def load_static_holidays(path: Path = HOLIDAY_FILE) -> tuple[dict[date, str], set[date]]:
    """Read ``nse_holidays.yaml`` -> (``{date: name}``, special session dates).

    Returns empty structures (and logs) if the file is missing or unreadable.
    """
    try:
        holidays, specials = _read_static_file(str(path), path.stat().st_mtime)
        return dict(holidays), set(specials)
    except (OSError, yaml.YAMLError, KeyError, ValueError) as exc:
        logger.error(f"[calendar] cannot read holiday file {path}: {exc} - weekends only")
        return {}, set()


def load_calendar(
    schedule: "ScheduleSettings | None" = None,
    cache_dir: Path | None = None,
    fetch: bool = False,
    *,
    holiday_file: Path = HOLIDAY_FILE,
) -> NSECalendar:
    """Build the NSE calendar.

    Args:
        schedule: ``ScheduleSettings`` (holiday_overrides / special_sessions); optional.
        cache_dir: cache directory for the fetched NSE list (``runtime/cache``).
        fetch: also merge NSE's holiday-master API for the current and next year (network,
            failure-safe: logs and keeps the static file).
        holiday_file: static YAML path (tests may point elsewhere).
    """
    names, specials = load_static_holidays(holiday_file)
    if fetch:
        try:
            from data.nse import fetch_nse_holidays

            fetched = fetch_nse_holidays(cache_dir)
            added = {d: n for d, n in fetched.items() if d not in names and d.weekday() < 5}
            names.update(added)
            if added:
                logger.info(f"[calendar] merged {len(added)} holidays from NSE: {sorted(added)}")
        except Exception as exc:  # noqa: BLE001 - network/parse failure must not break startup
            logger.warning(f"[calendar] NSE holiday fetch failed, using static file only: {exc}")
    if schedule is not None:
        for d in schedule.holiday_overrides:
            names[d] = "Holiday override (dashboard)"
        specials |= set(schedule.special_sessions)
    for d in specials:
        names.pop(d, None)
    return NSECalendar(set(names), specials, names)


# =========================================================================== bar close


def bar_close_time(open_ts: datetime | pd.Timestamp, timeframe: Timeframe | str, calendar: NSECalendar) -> datetime:
    """Close time of the bar that opened at ``open_ts`` (see module docstring)."""
    tf = as_timeframe(timeframe)
    ts = to_ist(open_ts.to_pydatetime() if isinstance(open_ts, pd.Timestamp) else open_ts)
    day = ts.date()
    if tf in INTRADAY_TIMEFRAMES:
        return min(ts + timedelta(minutes=TIMEFRAME_MINUTES[tf]), calendar.market_close_dt(day))
    if tf == Timeframe.D1:
        return calendar.market_close_dt(day)
    last = calendar.week_last_trading_day(day) or day
    return calendar.market_close_dt(max(last, day))


def bar_close_times(index: pd.DatetimeIndex, timeframe: Timeframe | str, calendar: NSECalendar) -> pd.DatetimeIndex:
    """Vectorised ``bar_close_time`` for a whole (IST) index."""
    tf = as_timeframe(timeframe)
    if len(index) == 0:
        return pd.DatetimeIndex([], tz=IST)
    idx = index.tz_convert(IST) if index.tz is not None else index.tz_localize(IST)
    session_close = idx.normalize() + pd.Timedelta(hours=15, minutes=30)
    if tf in INTRADAY_TIMEFRAMES:
        end = idx + pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf])
        return pd.DatetimeIndex(end.where(end <= session_close, session_close))
    if tf == Timeframe.D1:
        return pd.DatetimeIndex(session_close)
    cache: dict[date, datetime] = {}
    out: list[datetime] = []
    for ts in idx:
        d = ts.date()
        if d not in cache:
            cache[d] = bar_close_time(ts.to_pydatetime(), tf, calendar)
        out.append(cache[d])
    return pd.DatetimeIndex(out).tz_convert(IST)


def closed_bars(
    df: pd.DataFrame, timeframe: Timeframe | str, now: datetime, calendar: NSECalendar
) -> pd.DataFrame:
    """Return only the bars of ``df`` that have CLOSED at ``now`` (``close_time <= now``)."""
    if df is None or df.empty:
        return df
    close = bar_close_times(pd.DatetimeIndex(df.index), timeframe, calendar)
    mask = np.asarray(close <= pd.Timestamp(to_ist(now)))
    return df.loc[mask]
