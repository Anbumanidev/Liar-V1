"""ReplayDataProvider: strict no-look-ahead for candles AND quotes while a SimulatedClock advances."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.clock import SimulatedClock
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.types import empty_ohlcv, validate_ohlcv
from data.bars import resample_ohlcv
from data.base import DataProvider
from data.calendar import NSECalendar, bar_close_time, bar_close_times
from data.providers.replay_provider import ReplayDataProvider
from tests.helpers import ist, make_intraday_candles, trading_days

CAL = NSECalendar(holidays={date(2026, 11, 24)})
DAYS = [d for d in trading_days(date(2026, 11, 16), 8) if CAL.is_trading_day(d)]   # 16..25 minus 24
FAR_PAST = ist(2026, 1, 1)
FAR_FUTURE = ist(2027, 12, 31)


@pytest.fixture
def m5() -> pd.DataFrame:
    return make_intraday_candles(DAYS, 5, seed=11)


def _closed_reference(df: pd.DataFrame, tf: Timeframe, now: datetime) -> pd.DataFrame:
    close = bar_close_times(pd.DatetimeIndex(df.index), tf, CAL)
    return df.loc[close <= pd.Timestamp(now)]


ALL_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.D1, Timeframe.W1)
INTRADAY_TFS = (Timeframe.M5, Timeframe.M15, Timeframe.H1)


def _probes() -> list[tuple[datetime, tuple[Timeframe, ...]]]:
    """Monotonic (instant, timeframes) probes: 1-minute sweeps across bar / session / week
    boundaries plus a coarse multi-day grid crossing the 24-Nov holiday."""
    week_close = [(ist(2026, 11, 20, 15, 28) + timedelta(minutes=i), ALL_TFS) for i in range(4)]
    opening = [(ist(2026, 11, 23, 9, 13) + timedelta(minutes=i), INTRADAY_TFS) for i in range(68)]
    session_close = [(ist(2026, 11, 23, 15, 24) + timedelta(minutes=i), ALL_TFS) for i in range(8)]
    coarse = [(ist(2026, 11, 23, 16, 0) + timedelta(minutes=211 * i), ALL_TFS) for i in range(15)]
    return week_close + opening + session_close + coarse


def _window_start(tf: Timeframe, now: datetime) -> datetime:
    return now - timedelta(days=1) if tf in (Timeframe.M15, Timeframe.H1) else FAR_PAST


def test_candles_and_quotes_never_look_ahead_while_clock_advances(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 20, 9, 0))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    full = {Timeframe.M5: m5}
    full.update({tf: resample_ohlcv(m5, tf) for tf in ALL_TFS[1:]})        # complete history, resampled once
    ref_close = {tf: bar_close_times(pd.DatetimeIndex(df.index), tf, CAL) for tf, df in full.items()}
    probes = _probes()
    for now, tfs in probes:
        clock.set(now)
        stamp = pd.Timestamp(now)
        for tf in tfs:
            start = _window_start(tf, now)
            got = rp.get_candles("ABC", tf, start, FAR_FUTURE)
            ref = full[tf].loc[(ref_close[tf] <= stamp) & (full[tf].index >= pd.Timestamp(start))]
            # identical bars AND values to the complete history restricted to bars closed at `now`
            assert list(got.index) == list(ref.index), f"{tf} at {now}"
            assert np.allclose(got.to_numpy(), ref.to_numpy()), f"{tf} values at {now}"
        q = rp.get_quote("ABC")
        last_closed = m5.loc[ref_close[Timeframe.M5] <= stamp]
        assert q is not None and q.timestamp <= now
        assert q.ltp == last_closed["close"].iloc[-1]
        assert q.timestamp == bar_close_time(last_closed.index[-1], Timeframe.M5, CAL)
    assert len(probes) > 90


def test_returned_frames_meet_the_candle_contract(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 25, 12, 7))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    for tf in ALL_TFS:
        got = rp.get_candles("ABC", tf, FAR_PAST, FAR_FUTURE)
        validate_ohlcv(got)
        assert (bar_close_times(pd.DatetimeIndex(got.index), tf, CAL) <= pd.Timestamp(clock.now())).all()


def test_resampled_hour_bar_appears_only_when_complete(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 14))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    h = rp.get_candles("ABC", Timeframe.H1, FAR_PAST, FAR_FUTURE)
    assert h.index[-1] == ist(2026, 11, 20, 15, 15)                 # 09:15 bar still forming
    clock.set(ist(2026, 11, 23, 10, 15))
    h = rp.get_candles("ABC", Timeframe.H1, FAR_PAST, FAR_FUTURE)
    assert h.index[-1] == ist(2026, 11, 23, 9, 15)
    hour = m5.loc[ist(2026, 11, 23, 9, 15):ist(2026, 11, 23, 10, 10)]
    assert len(hour) == 12
    assert h.iloc[-1]["high"] == hour["high"].max() and h.iloc[-1]["close"] == hour["close"].iloc[-1]


def test_daily_and_weekly_visibility(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 20, 15, 29))               # Friday, before the close
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    assert rp.get_candles("ABC", Timeframe.D1, FAR_PAST, FAR_FUTURE).index[-1] == ist(2026, 11, 19)
    assert rp.get_candles("ABC", Timeframe.W1, FAR_PAST, FAR_FUTURE).empty
    clock.set(ist(2026, 11, 20, 15, 30))
    assert rp.get_candles("ABC", Timeframe.D1, FAR_PAST, FAR_FUTURE).index[-1] == ist(2026, 11, 20)
    weekly = rp.get_candles("ABC", Timeframe.W1, FAR_PAST, FAR_FUTURE)
    assert list(weekly.index) == [ist(2026, 11, 16)]


def test_end_is_clamped_and_start_filters(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 11, 0))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    got = rp.get_candles("ABC", Timeframe.M5, ist(2026, 11, 23, 10, 0), FAR_FUTURE)
    assert got.index[0] == ist(2026, 11, 23, 10, 0) and got.index[-1] == ist(2026, 11, 23, 10, 55)
    daily = rp.get_candles("ABC", Timeframe.D1, ist(2026, 11, 19, 12, 0), FAR_FUTURE)
    assert list(daily.index) == [ist(2026, 11, 19), ist(2026, 11, 20)]    # D1 start = whole day


def test_preloaded_daily_frame_and_quote_fields(m5: pd.DataFrame) -> None:
    d1 = resample_ohlcv(m5, Timeframe.D1)
    clock = SimulatedClock(ist(2026, 11, 23, 12, 0))
    rp = ReplayDataProvider({("ABC", Timeframe.D1): d1}, clock, CAL)
    q = rp.get_quote("abc")
    assert q is not None and q.symbol == "ABC" and q.source == "replay"
    assert q.ltp == d1.loc[ist(2026, 11, 20), "close"]                    # today's bar not closed yet
    assert q.timestamp == ist(2026, 11, 20, 15, 30)
    assert q.prev_close == d1.loc[ist(2026, 11, 19), "close"]
    assert Timeframe.W1 in rp.supported_timeframes and Timeframe.M5 not in rp.supported_timeframes
    with pytest.raises(DataUnavailableError):
        rp.get_candles("ABC", Timeframe.M5, FAR_PAST, FAR_FUTURE)          # cannot build finer bars


def test_intraday_quote_aggregates_today(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 0))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    q = rp.get_quote("ABC")
    today = m5.loc[ist(2026, 11, 23, 9, 15):ist(2026, 11, 23, 9, 55)]
    assert q is not None
    assert q.open == today["open"].iloc[0] and q.high == today["high"].max() and q.low == today["low"].min()
    assert q.volume == pytest.approx(today["volume"].sum())
    assert q.prev_close == m5.loc[ist(2026, 11, 20, 15, 25), "close"]


def test_quote_prefers_most_recent_bar_over_stale_finer_frame(m5: pd.DataFrame) -> None:
    gap = m5.loc[m5.index.date < date(2026, 11, 23)]                          # 5m data missing 23-Nov
    d1 = resample_ohlcv(m5, Timeframe.D1)
    clock = SimulatedClock(ist(2026, 11, 23, 16, 0))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): gap, ("ABC", Timeframe.D1): d1}, clock, CAL)
    q = rp.get_quote("ABC")
    assert q is not None and q.ltp == d1.loc[ist(2026, 11, 23), "close"]
    assert q.timestamp == ist(2026, 11, 23, 15, 30)
    clock.set(ist(2026, 11, 25, 15, 30))                                     # both current: finest wins
    full = ReplayDataProvider({("ABC", Timeframe.M5): m5, ("ABC", Timeframe.D1): d1}, clock, CAL)
    q2 = full.get_quote("ABC")
    assert q2 is not None and q2.open == m5.loc[ist(2026, 11, 25, 9, 15), "open"]   # intraday aggregation


def test_unknown_symbol_and_before_any_data(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 16, 9, 15))
    rp = ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL)
    assert rp.get_quote("ABC") is None                                       # nothing closed yet
    assert rp.get_candles("ABC", Timeframe.M5, FAR_PAST, FAR_FUTURE).empty
    assert rp.get_quote("XYZ") is None
    with pytest.raises(DataUnavailableError):
        rp.get_candles("XYZ", Timeframe.M5, FAR_PAST, FAR_FUTURE)
    assert rp.healthcheck()
    assert not ReplayDataProvider({}, clock, CAL).healthcheck()


class _FakeLive(DataProvider):
    name = "fake"
    supported_timeframes = frozenset({Timeframe.M5, Timeframe.D1})

    def __init__(self, frames: dict[tuple[str, Timeframe], pd.DataFrame], fail: set[str]) -> None:
        self.frames, self.fail, self.calls = frames, fail, []

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((symbol, timeframe))
        if symbol in self.fail:
            raise DataUnavailableError("boom")
        return self.frames.get((symbol, timeframe), empty_ohlcv())

    def get_quote(self, symbol: str) -> None:
        return None

    def healthcheck(self) -> bool:
        return True


def test_from_provider_preloads_and_resamples_weekly(m5: pd.DataFrame) -> None:
    d1 = resample_ohlcv(m5, Timeframe.D1)
    live = _FakeLive({("ABC", Timeframe.M5): m5, ("ABC", Timeframe.D1): d1}, fail={"BAD"})
    clock = SimulatedClock(ist(2026, 11, 23, 9, 15))
    rp = ReplayDataProvider.from_provider(live, ["abc", "BAD", "NONE"], [Timeframe.M5, Timeframe.D1, Timeframe.W1],
                                          FAR_PAST, FAR_FUTURE, clock, CAL)
    assert ("ABC", Timeframe.D1) in live.calls                              # symbols normalised
    assert ("ABC", Timeframe.W1) not in live.calls                          # weekly fetched as daily
    assert rp.symbols == {"ABC"}
    assert rp.timeframes_for("ABC") == [Timeframe.M5, Timeframe.D1, Timeframe.W1]
    weekly = rp.get_candles("ABC", Timeframe.W1, FAR_PAST, FAR_FUTURE)
    assert list(weekly.index) == [ist(2026, 11, 16)]
