"""Event calendar: corporate actions, results / board meetings, macro events, holidays.

* ``refresh(start, end)`` pulls NSE corporate actions + event calendar (``NSEClient``) and the
  static macro file (``data/static/macro_events.yaml``) and UPSERTS them into ``market_events``
  (unique on ``symbol, event_type, event_date``).
* ``events_for(symbol, start, end)`` = that symbol's events + market-wide MACRO events (DB rows
  plus the static macro file, de-duplicated).
* ``events_in_window(...)`` is what the Event & Overnight Risk agent and the risk manager use:
  events in ``[entry_day, entry_day + holding_days trading days]`` plus a ``HOLIDAY`` event for
  every weekday market holiday inside that window (plain weekends are not flagged).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger
from sqlalchemy import and_, or_, select

from core.clock import Clock
from core.enums import EventType, Mode
from core.logging import to_jsonable
from core.types import MarketEvent
from storage.models import MarketEventRow

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.orm import Session, sessionmaker

    from data.calendar import NSECalendar
    from data.nse import NSEClient

STATIC_DIR = Path(__file__).resolve().parent / "static"
MACRO_FILE = "macro_events.yaml"

#: Event types that change price/quantity or entitlement on the ex-date.
CORPORATE_ACTION_TYPES: frozenset[EventType] = frozenset(
    {EventType.EX_DIVIDEND, EventType.EX_DATE, EventType.SPLIT, EventType.BONUS, EventType.RIGHTS}
)


@functools.lru_cache(maxsize=4)
def _read_macro_file(path_str: str, mtime: float) -> tuple[MarketEvent, ...]:
    del mtime
    raw = yaml.safe_load(Path(path_str).read_text(encoding="utf-8")) or {}
    out: list[MarketEvent] = []
    for item in raw.get("events") or []:
        d = item["date"] if isinstance(item["date"], date) else date.fromisoformat(str(item["date"]))
        details = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in item.items() if k not in ("date", "name")}
        out.append(MarketEvent(symbol=None, event_type=EventType.MACRO, event_date=d,
                               title=str(item.get("name", "Macro event")), details=details, source="static"))
    return tuple(sorted(out, key=lambda e: e.event_date))


def load_macro_events(static_dir: Path = STATIC_DIR) -> list[MarketEvent]:
    """Static macro events (RBI MPC, Union Budget, FOMC). Empty list (logged) if unreadable."""
    path = Path(static_dir) / MACRO_FILE
    try:
        return list(_read_macro_file(str(path), path.stat().st_mtime))
    except (OSError, yaml.YAMLError, KeyError, ValueError) as exc:
        logger.error(f"[events] cannot read macro events {path}: {exc}")
        return []


def _row_to_event(row: MarketEventRow) -> MarketEvent:
    return MarketEvent(
        symbol=row.symbol,
        event_type=EventType(row.event_type),
        event_date=row.event_date,
        title=row.title or "",
        details=dict(row.details_json or {}),
        ratio=row.ratio,
        amount=row.amount,
        source=row.source or "unknown",
    )


def _key(ev: MarketEvent) -> tuple[str | None, str, date]:
    return (ev.symbol, ev.event_type.value, ev.event_date)


class EventCalendar:
    """DB-backed event calendar.

    Args:
        session_factory: SQLAlchemy sessionmaker (``market_events`` table).
        nse: ``NSEClient`` or None (then only static macro events are refreshed).
        clock: time source (``fetched_at``).
        static_dir: folder containing ``macro_events.yaml``.
    """

    def __init__(
        self,
        session_factory: "sessionmaker[Session]",
        nse: "NSEClient | None",
        clock: Clock,
        static_dir: Path = STATIC_DIR,
    ) -> None:
        self._sf = session_factory
        self._nse = nse
        self._clock = clock
        self._static_dir = Path(static_dir)

    # ------------------------------------------------------------------ write
    def upsert_events(self, events: Iterable[MarketEvent]) -> int:
        """Insert or update events (unique on symbol, event_type, event_date). Returns rows written."""
        now = self._clock.now()
        dedup: dict[tuple[str | None, str, date], MarketEvent] = {}
        for ev in events:
            sym = ev.symbol.strip().upper() if ev.symbol else None
            ev = ev if ev.symbol == sym else ev.model_copy(update={"symbol": sym})
            dedup[_key(ev)] = ev
        if not dedup:
            return 0
        written = 0
        session = self._sf()
        try:
            for (sym, etype, edate), ev in dedup.items():
                cond = MarketEventRow.symbol.is_(None) if sym is None else MarketEventRow.symbol == sym
                row = session.execute(
                    select(MarketEventRow).where(cond, MarketEventRow.event_type == etype,
                                                 MarketEventRow.event_date == edate)
                ).scalar_one_or_none()
                if row is None:
                    row = MarketEventRow(symbol=sym, event_type=etype, event_date=edate)
                    session.add(row)
                row.title = ev.title
                row.ratio = ev.ratio
                row.amount = ev.amount
                row.source = ev.source[:40]
                row.details_json = to_jsonable(ev.details)
                row.fetched_at = now
                written += 1
            session.commit()
        except Exception as exc:  # noqa: BLE001 - log, roll back, report 0
            session.rollback()
            logger.error(f"[events] upsert failed: {exc}")
            try:
                from storage.audit import record_error

                record_error(None, "events", exc, {"count": len(dedup)}, clock=self._clock)
            except Exception as inner:  # noqa: BLE001
                logger.error(f"[events] could not record error: {inner}")
            return 0
        finally:
            session.close()
        return written

    def refresh(self, start: date, end: date) -> int:
        """Fetch NSE corporate actions + board meetings + static macro events in ``[start, end]``
        and upsert them. Returns the number of rows written (0 on total failure)."""
        events: list[MarketEvent] = [e for e in load_macro_events(self._static_dir) if start <= e.event_date <= end]
        if self._nse is not None:
            for name, fn in (("corporate_actions", self._nse.corporate_actions),
                             ("event_calendar", self._nse.event_calendar)):
                try:
                    events.extend(fn(start, end))
                except Exception as exc:  # noqa: BLE001 - NSEClient is failure-safe; belt and braces
                    logger.warning(f"[events] {name} refresh failed: {exc}")
        n = self.upsert_events(events)
        logger.info(f"[events] refreshed {start}..{end}: {n} events upserted")
        return n

    # ------------------------------------------------------------------ read
    def _query(self, where: Any) -> list[MarketEvent]:
        session = self._sf()
        try:
            rows = session.execute(select(MarketEventRow).where(where)).scalars().all()
            return [_row_to_event(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[events] query failed: {exc}")
            return []
        finally:
            session.close()

    def macro_events(self, start: date, end: date) -> list[MarketEvent]:
        """Market-wide MACRO events in ``[start, end]`` (DB + static file)."""
        db = self._query(and_(MarketEventRow.symbol.is_(None), MarketEventRow.event_type == EventType.MACRO.value,
                              MarketEventRow.event_date >= start, MarketEventRow.event_date <= end))
        static = [e for e in load_macro_events(self._static_dir) if start <= e.event_date <= end]
        merged = {_key(e): e for e in static}
        merged.update({_key(e): e for e in db})
        return sorted(merged.values(), key=lambda e: (e.event_date, e.event_type.value))

    def events_for(self, symbol: str, start: date, end: date) -> list[MarketEvent]:
        """Events of ``symbol`` plus market-wide MACRO events in ``[start, end]``."""
        sym = symbol.strip().upper()
        own = self._query(and_(MarketEventRow.symbol == sym, MarketEventRow.event_date >= start,
                               MarketEventRow.event_date <= end))
        merged = {_key(e): e for e in own}
        for e in self.macro_events(start, end):
            merged.setdefault(_key(e), e)
        return sorted(merged.values(), key=lambda e: (e.event_date, e.symbol is None, e.event_type.value))

    def window_end(self, mode: Mode, entry_day: date, holding_days: int, calendar: "NSECalendar") -> date:
        """Last day of the holding window: entry day for INTRADAY, else entry + N trading days."""
        if mode == Mode.INTRADAY or holding_days <= 0:
            return entry_day
        return calendar.add_trading_days(entry_day, holding_days)

    def events_in_window(
        self,
        symbol: str,
        mode: Mode,
        entry_day: date,
        holding_days: int,
        calendar: "NSECalendar",
    ) -> list[MarketEvent]:
        """Symbol + macro events in ``[entry_day, entry_day + holding_days trading days]``, plus a
        ``HOLIDAY`` event for every weekday market holiday inside the window (not plain weekends).

        INTRADAY (or ``holding_days <= 0``) uses the entry day only.
        """
        end = self.window_end(mode, entry_day, holding_days, calendar)
        events = self.events_for(symbol, entry_day, end)
        for d in calendar.holidays_between(entry_day, end):
            events.append(MarketEvent(symbol=None, event_type=EventType.HOLIDAY, event_date=d,
                                      title=f"Market holiday: {calendar.holiday_name(d) or 'NSE holiday'}",
                                      details={"window_start": entry_day.isoformat(), "window_end": end.isoformat()},
                                      source="calendar"))
        return sorted(events, key=lambda e: (e.event_date, e.event_type.value))

    def corporate_actions_on(self, day: date) -> list[MarketEvent]:
        """Ex-date corporate actions (dividend, split, bonus, rights, generic ex-date) on ``day``."""
        rows = self._query(and_(MarketEventRow.event_date == day,
                                MarketEventRow.event_type.in_([t.value for t in CORPORATE_ACTION_TYPES]),
                                MarketEventRow.symbol.is_not(None)))
        return sorted(rows, key=lambda e: (e.symbol or "", e.event_type.value))

    def upcoming(self, symbols: Iterable[str], start: date, days: int = 14) -> list[MarketEvent]:
        """Events of several symbols in the next ``days`` calendar days (dashboard helper)."""
        syms = [s.strip().upper() for s in symbols]
        end = start + timedelta(days=days)
        if not syms:
            return []
        return sorted(self._query(and_(or_(*[MarketEventRow.symbol == s for s in syms]),
                                       MarketEventRow.event_date >= start, MarketEventRow.event_date <= end)),
                      key=lambda e: (e.event_date, e.symbol or ""))
