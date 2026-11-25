"""EventCalendar: upsert into market_events, events_for (+ macro), events_in_window (+ HOLIDAY markers)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from core.clock import SimulatedClock
from core.enums import EventType, Mode
from core.types import MarketEvent
from data.calendar import NSECalendar, load_calendar
from data.events import EventCalendar, load_macro_events
from storage.models import MarketEventRow
from tests.helpers import ist

MACRO_YAML = """events:
  - {date: 2026-11-26, name: "RBI MPC policy decision", kind: RBI_MPC, verified: false}
  - {date: 2026-12-10, us_date: 2026-12-09, name: "US FOMC rate decision", kind: FOMC}
  - {date: 2027-02-01, name: "Union Budget", kind: BUDGET}
"""
CAL = NSECalendar(holidays={date(2026, 11, 24)}, holiday_names={date(2026, 11, 24): "Guru Nanak Jayanti"})


def _ev(symbol: str | None, etype: EventType, d: date, **kw: object) -> MarketEvent:
    return MarketEvent(symbol=symbol, event_type=etype, event_date=d, title=f"{symbol} {etype.value}",
                       source="NSE", **kw)  # type: ignore[arg-type]


class FakeNSE:
    def __init__(self, ca: list[MarketEvent] | None = None, ev: list[MarketEvent] | None = None,
                 fail: bool = False) -> None:
        self.ca, self.ev, self.fail = ca or [], ev or [], fail

    def corporate_actions(self, start: date, end: date) -> list[MarketEvent]:
        if self.fail:
            raise RuntimeError("nse down")
        return [e for e in self.ca if start <= e.event_date <= end]

    def event_calendar(self, start: date, end: date) -> list[MarketEvent]:
        if self.fail:
            raise RuntimeError("nse down")
        return [e for e in self.ev if start <= e.event_date <= end]


@pytest.fixture
def macro_dir(tmp_path: Path) -> Path:
    (tmp_path / "macro_events.yaml").write_text(MACRO_YAML, encoding="utf-8")
    return tmp_path


@pytest.fixture
def nse() -> FakeNSE:
    return FakeNSE(
        ca=[_ev("RELIANCE", EventType.EX_DIVIDEND, date(2026, 11, 26), amount=5.5),
            _ev("tcs", EventType.SPLIT, date(2026, 11, 27), ratio=5.0)],
        ev=[_ev("INFY", EventType.RESULTS, date(2026, 11, 25))],
    )


@pytest.fixture
def events(session_factory: sessionmaker[Session], nse: FakeNSE, macro_dir: Path) -> EventCalendar:
    return EventCalendar(session_factory, nse, SimulatedClock(ist(2026, 11, 23, 8, 30)), macro_dir)  # type: ignore[arg-type]


def _rows(sf: sessionmaker[Session]) -> int:
    with sf() as s:
        return int(s.execute(select(func.count()).select_from(MarketEventRow)).scalar_one())


def test_refresh_upserts_and_is_idempotent(events: EventCalendar, nse: FakeNSE,
                                           session_factory: sessionmaker[Session]) -> None:
    assert events.refresh(date(2026, 11, 20), date(2026, 11, 30)) == 4          # 3 NSE + 1 macro in range
    assert _rows(session_factory) == 4
    nse.ca[0] = _ev("RELIANCE", EventType.EX_DIVIDEND, date(2026, 11, 26), amount=6.0)
    assert events.refresh(date(2026, 11, 20), date(2026, 11, 30)) == 4
    assert _rows(session_factory) == 4                                          # updated, not duplicated
    with session_factory() as s:
        row = s.execute(select(MarketEventRow).where(MarketEventRow.symbol == "RELIANCE")).scalar_one()
        assert row.amount == 6.0 and row.event_type == "EX_DIVIDEND" and row.fetched_at.tzinfo is not None
        assert s.execute(select(MarketEventRow).where(MarketEventRow.symbol == "TCS")).scalar_one().ratio == 5.0


def test_market_wide_events_are_not_duplicated(events: EventCalendar, session_factory: sessionmaker[Session]) -> None:
    macro = MarketEvent(symbol=None, event_type=EventType.MACRO, event_date=date(2026, 12, 4), title="RBI")
    assert events.upsert_events([macro]) == 1
    assert events.upsert_events([macro.model_copy(update={"title": "RBI MPC"})]) == 1
    assert _rows(session_factory) == 1
    assert events.upsert_events([]) == 0


def test_events_for_symbol_plus_macro(events: EventCalendar) -> None:
    events.refresh(date(2026, 11, 20), date(2026, 11, 30))
    got = events.events_for("reliance", date(2026, 11, 20), date(2026, 12, 31))
    kinds = [(e.symbol, e.event_type, e.event_date) for e in got]
    assert kinds == [("RELIANCE", EventType.EX_DIVIDEND, date(2026, 11, 26)),
                     (None, EventType.MACRO, date(2026, 11, 26)),
                     (None, EventType.MACRO, date(2026, 12, 10))]              # Dec FOMC from the static file
    fomc = got[-1]
    assert fomc.details["kind"] == "FOMC" and fomc.details["us_date"] == "2026-12-09"
    assert all(e.symbol in (None, "RELIANCE") for e in got)


def test_static_macro_events_without_refresh(session_factory: sessionmaker[Session], macro_dir: Path) -> None:
    cal = EventCalendar(session_factory, None, SimulatedClock(ist(2026, 11, 23)), macro_dir)
    got = cal.events_for("ABC", date(2026, 11, 1), date(2027, 3, 1))
    assert [e.title for e in got] == ["RBI MPC policy decision", "US FOMC rate decision", "Union Budget"]
    assert cal.refresh(date(2026, 11, 1), date(2026, 11, 30)) == 1             # nse=None: macro only
    assert load_macro_events(macro_dir / "missing") == []


def test_refresh_survives_nse_failure(session_factory: sessionmaker[Session], macro_dir: Path) -> None:
    cal = EventCalendar(session_factory, FakeNSE(fail=True), SimulatedClock(ist(2026, 11, 23)), macro_dir)  # type: ignore[arg-type]
    assert cal.refresh(date(2026, 11, 1), date(2026, 11, 30)) == 1


def test_events_in_window_flags_weekday_holidays(events: EventCalendar) -> None:
    events.refresh(date(2026, 11, 20), date(2026, 11, 30))
    btst = events.events_in_window("INFY", Mode.BTST, date(2026, 11, 23), 1, CAL)   # Mon -> Wed (Tue holiday)
    assert [(e.event_type, e.event_date) for e in btst] == [(EventType.HOLIDAY, date(2026, 11, 24)),
                                                            (EventType.RESULTS, date(2026, 11, 25))]
    holiday = btst[0]
    assert holiday.symbol is None and "Guru Nanak Jayanti" in holiday.title
    assert holiday.details == {"window_start": "2026-11-23", "window_end": "2026-11-25"}
    intraday = events.events_in_window("INFY", Mode.INTRADAY, date(2026, 11, 23), 3, CAL)
    assert intraday == []                                                         # entry day only
    weekend = events.events_in_window("RELIANCE", Mode.SWING, date(2026, 11, 26), 3, CAL)  # Thu -> Tue
    assert [e.event_type for e in weekend] == [EventType.EX_DIVIDEND, EventType.MACRO]
    assert all(e.event_type != EventType.HOLIDAY for e in weekend)              # plain weekends not flagged
    assert events.window_end(Mode.SWING, date(2026, 11, 26), 3, CAL) == date(2026, 12, 1)


def test_events_in_window_with_real_static_files(session_factory: sessionmaker[Session]) -> None:
    cal = EventCalendar(session_factory, None, SimulatedClock(ist(2026, 11, 23)))
    nse_cal = load_calendar()
    got = cal.events_in_window("ABC", Mode.SWING, date(2026, 12, 21), 5, nse_cal)  # spans Christmas
    assert (EventType.HOLIDAY, date(2026, 12, 25)) in [(e.event_type, e.event_date) for e in got]
    mpc = cal.events_in_window("ABC", Mode.POSITIONAL, date(2026, 12, 1), 5, nse_cal)
    assert any(e.event_type == EventType.MACRO and "RBI" in e.title for e in mpc)


def test_corporate_actions_on_and_upcoming(events: EventCalendar) -> None:
    events.refresh(date(2026, 11, 20), date(2026, 11, 30))
    on_26 = events.corporate_actions_on(date(2026, 11, 26))
    assert [(e.symbol, e.event_type, e.amount) for e in on_26] == [("RELIANCE", EventType.EX_DIVIDEND, 5.5)]
    assert events.corporate_actions_on(date(2026, 11, 25)) == []                 # results are not CAs
    up = events.upcoming(["tcs", "INFY"], date(2026, 11, 20), days=14)
    assert [(e.symbol, e.event_type) for e in up] == [("INFY", EventType.RESULTS), ("TCS", EventType.SPLIT)]
    assert events.upcoming([], date(2026, 11, 20)) == []
