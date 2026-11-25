"""NSECalendar, load_calendar, bar_close_time and closed_bars (no look-ahead)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from config.schema import Settings
from core.clock import IST
from core.enums import Timeframe
from core.types import empty_ohlcv
from data.calendar import (
    NSECalendar,
    as_timeframe,
    bar_close_time,
    bar_close_times,
    closed_bars,
    load_calendar,
)
from tests.helpers import ist, make_daily_candles, make_intraday_candles

TUE_HOLIDAY = date(2026, 11, 24)     # Guru Nanak Jayanti 2026 (Tuesday)
FRI_HOLIDAY = date(2026, 12, 25)     # Christmas 2026 (Friday)
SAT_SPECIAL = date(2026, 11, 28)     # pretend Saturday special session


@pytest.fixture
def cal() -> NSECalendar:
    return NSECalendar(holidays={TUE_HOLIDAY, FRI_HOLIDAY}, special_sessions={SAT_SPECIAL},
                       holiday_names={TUE_HOLIDAY: "Guru Nanak Jayanti"})


# ------------------------------------------------------------------ day rules


def test_weekends_holidays_and_special_sessions(cal: NSECalendar) -> None:
    assert cal.is_trading_day(date(2026, 11, 23))          # Monday
    assert not cal.is_trading_day(TUE_HOLIDAY)
    assert not cal.is_trading_day(date(2026, 11, 22))      # Sunday
    assert not cal.is_trading_day(date(2026, 11, 29))      # Sunday
    assert cal.is_trading_day(SAT_SPECIAL)                 # special Saturday session
    assert cal.is_holiday(TUE_HOLIDAY) and not cal.is_holiday(date(2026, 11, 22))
    assert cal.holiday_name(TUE_HOLIDAY) == "Guru Nanak Jayanti"
    assert cal.holiday_name(date(2026, 11, 22)) == "Weekend"
    assert cal.holiday_name(date(2026, 11, 23)) is None
    # datetimes are accepted and converted to IST first (23:00 UTC on 23-Nov == 24-Nov IST)
    assert not cal.is_trading_day(datetime(2026, 11, 23, 23, 0, tzinfo=timezone.utc))


def test_special_session_overrides_holiday() -> None:
    c = NSECalendar(holidays={TUE_HOLIDAY}, special_sessions={TUE_HOLIDAY})
    assert c.is_trading_day(TUE_HOLIDAY)
    assert TUE_HOLIDAY not in c.holidays


def test_next_prev_trading_day(cal: NSECalendar) -> None:
    assert cal.next_trading_day(date(2026, 11, 23)) == date(2026, 11, 25)   # skips Tue holiday
    assert cal.prev_trading_day(date(2026, 11, 25)) == date(2026, 11, 23)
    assert cal.next_trading_day(date(2026, 11, 27)) == SAT_SPECIAL          # Fri -> special Sat
    assert cal.next_trading_day(SAT_SPECIAL) == date(2026, 11, 30)          # Sat -> Mon
    assert cal.prev_trading_day(date(2026, 11, 30)) == SAT_SPECIAL
    assert cal.next_trading_day(date(2026, 12, 24)) == date(2026, 12, 28)   # Thu -> Mon (Fri holiday)


def test_add_trading_days(cal: NSECalendar) -> None:
    assert cal.add_trading_days(date(2026, 11, 23), 1) == date(2026, 11, 25)
    assert cal.add_trading_days(date(2026, 11, 23), 3) == date(2026, 11, 27)
    assert cal.add_trading_days(date(2026, 11, 23), 4) == SAT_SPECIAL
    assert cal.add_trading_days(date(2026, 11, 30), -2) == date(2026, 11, 27)
    assert cal.add_trading_days(date(2026, 11, 25), -1) == date(2026, 11, 23)
    assert cal.add_trading_days(date(2026, 11, 23), 0) == date(2026, 11, 23)
    assert cal.add_trading_days(TUE_HOLIDAY, 0) == date(2026, 11, 25)       # 0 on a holiday -> next
    assert cal.add_trading_days(date(2026, 11, 22), 1) == date(2026, 11, 23)  # from a Sunday


def test_trading_days_between_and_count(cal: NSECalendar) -> None:
    days = cal.trading_days_between(date(2026, 11, 23), date(2026, 11, 30))
    assert days == [date(2026, 11, 23), date(2026, 11, 25), date(2026, 11, 26), date(2026, 11, 27),
                    SAT_SPECIAL, date(2026, 11, 30)]
    assert cal.trading_days_between(date(2026, 11, 30), date(2026, 11, 23)) == []
    # (start_exclusive, end_inclusive]
    assert cal.count_trading_days(date(2026, 11, 23), date(2026, 11, 25)) == 1
    assert cal.count_trading_days(date(2026, 11, 23), date(2026, 11, 30)) == 5
    assert cal.count_trading_days(date(2026, 11, 23), date(2026, 11, 23)) == 0
    assert cal.count_trading_days(date(2026, 11, 25), date(2026, 11, 23)) == 0
    assert cal.holidays_between(date(2026, 11, 20), date(2026, 12, 31)) == [TUE_HOLIDAY, FRI_HOLIDAY]


def test_market_hours(cal: NSECalendar) -> None:
    d = date(2026, 11, 23)
    assert cal.market_open_dt(d) == ist(2026, 11, 23, 9, 15)
    assert cal.market_close_dt(d) == ist(2026, 11, 23, 15, 30)
    assert not cal.is_market_open(ist(2026, 11, 23, 9, 14))
    assert cal.is_market_open(ist(2026, 11, 23, 9, 15))
    assert cal.is_market_open(ist(2026, 11, 23, 15, 29))
    assert not cal.is_market_open(ist(2026, 11, 23, 15, 30))
    assert not cal.is_market_open(ist(2026, 11, 24, 11, 0))                 # holiday
    assert not cal.is_market_open(ist(2026, 11, 22, 11, 0))                 # Sunday
    assert cal.is_market_open(datetime(2026, 11, 28, 11, 0, tzinfo=IST))     # special session
    assert cal.is_market_open(datetime(2026, 11, 23, 4, 0, tzinfo=timezone.utc))  # 09:30 IST
    assert cal.is_market_open(datetime(2026, 11, 23, 10, 0))                 # naive == IST


def test_week_first_last_trading_day_and_last_closed_session(cal: NSECalendar) -> None:
    assert cal.week_last_trading_day(date(2026, 12, 21)) == date(2026, 12, 24)   # Fri 25 holiday
    assert cal.week_last_trading_day(date(2026, 11, 23)) == SAT_SPECIAL
    assert cal.week_first_trading_day(date(2026, 11, 27)) == date(2026, 11, 23)
    c = NSECalendar(holidays={date(2026, 11, 30)})
    assert c.week_first_trading_day(date(2026, 12, 3)) == date(2026, 12, 1)
    none_week = NSECalendar(holidays={date(2026, 11, 23) + timedelta(days=i) for i in range(5)})
    assert none_week.week_last_trading_day(date(2026, 11, 25)) is None
    assert cal.last_closed_session(ist(2026, 11, 23, 15, 29)) == date(2026, 11, 20)
    assert cal.last_closed_session(ist(2026, 11, 23, 15, 30)) == date(2026, 11, 23)
    assert cal.last_closed_session(ist(2026, 11, 24, 12, 0)) == date(2026, 11, 23)


def test_as_timeframe_aliases() -> None:
    assert as_timeframe("5m") is Timeframe.M5
    assert as_timeframe("60m") is Timeframe.H1
    assert as_timeframe("daily") is Timeframe.D1
    assert as_timeframe("1w") is Timeframe.W1
    assert as_timeframe(Timeframe.M15) is Timeframe.M15
    with pytest.raises(ValueError):
        as_timeframe("3m")


# ------------------------------------------------------------------ load_calendar


def test_load_calendar_static_file() -> None:
    c = load_calendar()
    assert not c.is_trading_day(date(2026, 1, 26))          # Republic Day
    assert not c.is_trading_day(date(2025, 12, 25))
    assert c.is_trading_day(date(2024, 1, 20))              # Saturday special session (2024)
    assert c.is_trading_day(date(2025, 2, 1))               # Budget Saturday
    assert c.is_trading_day(date(2026, 11, 23))
    years = {d.year for d in c.holidays}
    assert {2024, 2025, 2026, 2027} <= years


def test_load_calendar_schedule_overrides_win(settings: Settings) -> None:
    schedule = settings.schedule.model_copy(update={
        "holiday_overrides": [date(2026, 11, 25)],
        "special_sessions": [date(2026, 1, 26), date(2026, 11, 29)],
    })
    c = load_calendar(schedule)
    assert not c.is_trading_day(date(2026, 11, 25))
    assert c.holiday_name(date(2026, 11, 25)) == "Holiday override (dashboard)"
    assert c.is_trading_day(date(2026, 1, 26))
    assert c.is_trading_day(date(2026, 11, 29))             # Sunday special session


def test_load_calendar_fetch_merges_and_failure_is_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import data.nse as nse

    monkeypatch.setattr(nse, "fetch_nse_holidays",
                        lambda cache_dir=None: {date(2026, 12, 1): "Ad-hoc holiday", date(2026, 12, 5): "Sat"})
    c = load_calendar(None, tmp_path, fetch=True)
    assert not c.is_trading_day(date(2026, 12, 1))
    assert c.holiday_name(date(2026, 12, 1)) == "Ad-hoc holiday"

    def boom(cache_dir: Path | None = None) -> dict[date, str]:
        raise ConnectionError("offline")

    monkeypatch.setattr(nse, "fetch_nse_holidays", boom)
    c2 = load_calendar(None, tmp_path, fetch=True)
    assert c2.is_trading_day(date(2026, 12, 1))
    assert not c2.is_trading_day(date(2026, 1, 26))


def test_load_calendar_missing_or_custom_file(tmp_path: Path) -> None:
    c = load_calendar(holiday_file=tmp_path / "missing.yaml")
    assert c.holidays == frozenset()
    assert not c.is_trading_day(date(2026, 11, 22))
    f = tmp_path / "h.yaml"
    f.write_text("holidays:\n  - {date: 2026-11-23, name: X}\n  - 2026-11-25\nspecial_sessions:\n  - 2026-11-22\n",
                 encoding="utf-8")
    c2 = load_calendar(holiday_file=f)
    assert not c2.is_trading_day(date(2026, 11, 23)) and not c2.is_trading_day(date(2026, 11, 25))
    assert c2.is_trading_day(date(2026, 11, 22))
    assert c2.holiday_name(date(2026, 11, 23)) == "X"


# ------------------------------------------------------------------ bar close times


def test_bar_close_time_intraday(cal: NSECalendar) -> None:
    assert bar_close_time(ist(2026, 11, 23, 9, 55), Timeframe.M5, cal) == ist(2026, 11, 23, 10, 0)
    assert bar_close_time(ist(2026, 11, 23, 9, 15), "15m", cal) == ist(2026, 11, 23, 9, 30)
    assert bar_close_time(ist(2026, 11, 23, 14, 15), "1h", cal) == ist(2026, 11, 23, 15, 15)
    assert bar_close_time(ist(2026, 11, 23, 15, 15), "1h", cal) == ist(2026, 11, 23, 15, 30)  # capped
    assert bar_close_time(ist(2026, 11, 23, 15, 29), "1m", cal) == ist(2026, 11, 23, 15, 30)
    assert bar_close_time(pd.Timestamp("2026-11-23 09:55", tz=IST), "5m", cal) == ist(2026, 11, 23, 10, 0)
    utc_open = datetime(2026, 11, 23, 4, 25, tzinfo=timezone.utc)                             # 09:55 IST
    assert bar_close_time(utc_open, "5m", cal) == ist(2026, 11, 23, 10, 0)


def test_bar_close_time_daily_weekly(cal: NSECalendar) -> None:
    assert bar_close_time(ist(2026, 11, 23), Timeframe.D1, cal) == ist(2026, 11, 23, 15, 30)
    assert bar_close_time(ist(2026, 11, 23), Timeframe.W1, cal) == ist(2026, 11, 28, 15, 30)  # special Sat
    assert bar_close_time(ist(2026, 12, 21), Timeframe.W1, cal) == ist(2026, 12, 24, 15, 30)  # Fri holiday
    assert bar_close_time(ist(2026, 11, 30), Timeframe.W1, cal) == ist(2026, 12, 4, 15, 30)


def test_bar_close_times_vectorised_matches_scalar(cal: NSECalendar) -> None:
    for tf, freq in ((Timeframe.M5, 5), (Timeframe.H1, 60)):
        df = make_intraday_candles([date(2026, 11, 23), date(2026, 11, 25)], freq)
        vec = bar_close_times(pd.DatetimeIndex(df.index), tf, cal)
        assert list(vec) == [bar_close_time(ts, tf, cal) for ts in df.index]
    daily = make_daily_candles(date(2026, 12, 31), 40)
    for tf in (Timeframe.D1, Timeframe.W1):
        vec = bar_close_times(pd.DatetimeIndex(daily.index), tf, cal)
        assert list(vec) == [bar_close_time(ts, tf, cal) for ts in daily.index]
    assert len(bar_close_times(pd.DatetimeIndex([], tz=IST), Timeframe.M5, cal)) == 0


def test_closed_bars_5m_visible_exactly_at_close(cal: NSECalendar) -> None:
    df = make_intraday_candles([date(2026, 11, 23)], 5)
    before = closed_bars(df, Timeframe.M5, ist(2026, 11, 23, 9, 59) + timedelta(seconds=59), cal)
    assert before.index[-1] == ist(2026, 11, 23, 9, 50)
    at = closed_bars(df, Timeframe.M5, ist(2026, 11, 23, 10, 0), cal)
    assert at.index[-1] == ist(2026, 11, 23, 9, 55)
    assert len(at) == 9
    assert closed_bars(df, "5m", ist(2026, 11, 23, 9, 15), cal).empty
    assert len(closed_bars(df, "5m", ist(2026, 11, 23, 16, 0), cal)) == len(df)


def test_closed_bars_last_hourly_bar_closes_at_1530(cal: NSECalendar) -> None:
    df = make_intraday_candles([date(2026, 11, 23)], 60)
    assert df.index[-1] == ist(2026, 11, 23, 15, 15)
    assert closed_bars(df, "1h", ist(2026, 11, 23, 15, 29), cal).index[-1] == ist(2026, 11, 23, 14, 15)
    assert closed_bars(df, "1h", ist(2026, 11, 23, 15, 30), cal).index[-1] == ist(2026, 11, 23, 15, 15)


def test_closed_bars_daily_invisible_until_1530(cal: NSECalendar) -> None:
    daily = make_daily_candles(date(2026, 11, 23), 10)
    assert daily.index[-1] == ist(2026, 11, 23)
    assert closed_bars(daily, "1d", ist(2026, 11, 23, 15, 29), cal).index[-1] == ist(2026, 11, 20)
    assert closed_bars(daily, "1d", ist(2026, 11, 23, 15, 30), cal).index[-1] == ist(2026, 11, 23)
    assert closed_bars(daily, "1d", ist(2026, 11, 23, 9, 15), cal).index[-1] == ist(2026, 11, 20)


def test_closed_bars_weekly_invisible_until_week_last_trading_day(cal: NSECalendar) -> None:
    weekly = pd.DataFrame({"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0],
                           "volume": [1.0, 1.0]},
                          index=pd.DatetimeIndex([ist(2026, 12, 14), ist(2026, 12, 21)], name="ts"))
    # the week of 21-Dec ends Thursday 24-Dec (Christmas Friday)
    assert closed_bars(weekly, "1wk", ist(2026, 12, 24, 15, 29), cal).index[-1] == ist(2026, 12, 14)
    assert closed_bars(weekly, "1wk", ist(2026, 12, 24, 15, 30), cal).index[-1] == ist(2026, 12, 21)
    assert closed_bars(weekly, "1wk", ist(2026, 12, 18, 15, 29), cal).empty
    assert len(closed_bars(weekly, "1wk", ist(2026, 12, 18, 15, 30), cal)) == 1


def test_closed_bars_empty_input(cal: NSECalendar) -> None:
    assert closed_bars(empty_ohlcv(), "5m", ist(2026, 11, 23, 10, 0), cal).empty
