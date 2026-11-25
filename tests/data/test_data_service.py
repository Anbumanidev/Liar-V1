"""MarketDataService: closed-bars-only, fallback, caching, resampling, quotes, failure handling."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from config.schema import Settings
from core.clock import Clock, SimulatedClock, SystemClock
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.types import Quote, empty_ohlcv, validate_ohlcv
from data.base import DataProvider
from data.calendar import NSECalendar, bar_close_times
from data.providers.replay_provider import ReplayDataProvider
from data.providers.yfinance_provider import YFinanceProvider
from data.service import MarketDataService, build_market_data_service, build_provider
from storage.models import ErrorLog
from tests.helpers import ist, make_daily_candles, make_intraday_candles

CAL = NSECalendar(holidays={date(2026, 11, 24)})
DAYS = [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)]


class FakeProvider(DataProvider):
    """Vendor-like fake: returns every bar that OPENED in [start, end] (including a forming one)."""

    def __init__(
        self,
        name: str,
        frames: dict[tuple[str, Timeframe], pd.DataFrame] | None = None,
        supported: Iterable[Timeframe] = (Timeframe.M5, Timeframe.D1),
        fail: Exception | None = None,
        quotes: dict[str, Quote] | None = None,
        raw: bool = False,
    ) -> None:
        self.name = name
        self.raw = raw
        self.frames = frames or {}
        self.supported_timeframes = frozenset(supported)
        self.fail = fail
        self.quote_map = quotes or {}
        self.calls: list[tuple[str, Timeframe]] = []
        self.quote_calls: list[str] = []

    def get_candles(self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((symbol, timeframe))
        if self.fail is not None:
            raise self.fail
        df = self.frames.get((symbol, timeframe))
        if df is None:
            return empty_ohlcv()
        if self.raw:  # malformed vendor output: returned as-is
            return df
        lo = pd.Timestamp(start).normalize() if timeframe == Timeframe.D1 else pd.Timestamp(start)
        return df.loc[(df.index >= lo) & (df.index <= pd.Timestamp(end))]

    def get_quote(self, symbol: str) -> Quote | None:
        self.quote_calls.append(symbol)
        if self.fail is not None:
            raise self.fail
        return self.quote_map.get(symbol)

    def healthcheck(self) -> bool:
        return self.fail is None


@pytest.fixture
def m5() -> pd.DataFrame:
    return make_intraday_candles(DAYS, 5, seed=21)


@pytest.fixture
def d1() -> pd.DataFrame:
    return make_daily_candles(date(2026, 11, 23), 60, seed=22)


def _svc(provider: DataProvider, clock: Clock, *fallbacks: DataProvider, **kw: object) -> MarketDataService:
    return MarketDataService(provider, clock, CAL, fallbacks, **kw)  # type: ignore[arg-type]


# ------------------------------------------------------------------ closed bars only


def test_candles_drop_forming_bars(m5: pd.DataFrame, d1: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    p = FakeProvider("vendor", {("ABC", Timeframe.M5): m5, ("ABC", Timeframe.D1): d1})
    svc = _svc(p, clock)
    bars = svc.candles("abc", "5m", lookback_bars=300)
    validate_ohlcv(bars)
    assert bars.index[-1] == ist(2026, 11, 23, 9, 55)              # 10:00 bar is still forming
    assert (bar_close_times(pd.DatetimeIndex(bars.index), Timeframe.M5, CAL) <= pd.Timestamp(clock.now())).all()
    daily = svc.candles("ABC", Timeframe.D1, lookback_bars=10)
    assert daily.index[-1] == ist(2026, 11, 20)                     # today's daily bar closes at 15:30
    assert len(daily) == 10
    assert len(svc.candles("ABC", "5m", lookback_bars=5)) == 5
    since = svc.candles("ABC", "5m", start=ist(2026, 11, 23, 9, 15))
    assert since.index[0] == ist(2026, 11, 23, 9, 15) and len(since) == 9
    assert svc.healthy() and svc.last_error is None


def test_resamples_timeframes_the_provider_lacks(m5: pd.DataFrame, d1: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 11, 20))
    p = FakeProvider("vendor", {("ABC", Timeframe.M5): m5, ("ABC", Timeframe.D1): d1})
    svc = _svc(p, clock)
    hourly = svc.candles("ABC", "1h", lookback_bars=50)
    assert hourly.index[-1] == ist(2026, 11, 23, 10, 15)            # 11:15 bar forming
    assert hourly.loc[ist(2026, 11, 23, 9, 15), "high"] == \
        m5.loc[ist(2026, 11, 23, 9, 15):ist(2026, 11, 23, 10, 10), "high"].max()
    assert ("ABC", Timeframe.M5) in p.calls and ("ABC", Timeframe.H1) not in p.calls
    weekly = svc.candles("ABC", "1wk", lookback_bars=4)
    assert weekly.index[-1] == ist(2026, 11, 16)                    # week of 23-Nov not closed
    assert ("ABC", Timeframe.W1) not in p.calls


# ------------------------------------------------------------------ fallback / failure


def test_fallback_when_primary_raises_or_is_empty(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    bad = FakeProvider("primary", fail=RuntimeError("vendor down"))
    good = FakeProvider("backup", {("ABC", Timeframe.M5): m5})
    svc = _svc(bad, clock, good)
    out = svc.candles("ABC", "5m")
    assert not out.empty and svc.last_source[("ABC", Timeframe.M5)] == "backup"
    empty_primary = FakeProvider("primary", {})
    svc2 = _svc(empty_primary, clock, good)
    assert not svc2.candles("ABC", "5m").empty
    assert svc2.last_source[("ABC", Timeframe.M5)] == "backup"
    assert svc2.providers == (empty_primary, good)


def test_total_failure_returns_empty_frame_and_records_error(
    m5: pd.DataFrame, session_factory: sessionmaker[Session]
) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    p1 = FakeProvider("a", fail=DataUnavailableError("a down"))
    p2 = FakeProvider("b", fail=ValueError("b broken"))
    svc = _svc(p1, clock, p2, session_factory=session_factory)
    out = svc.candles("ABC", "5m")
    validate_ohlcv(out)
    assert out.empty
    assert not svc.healthy()
    assert svc.last_error is not None and "a down" in svc.last_error and "b broken" in svc.last_error
    with session_factory() as s:
        assert s.execute(select(func.count()).select_from(ErrorLog)).scalar_one() >= 1
    p1.fail = None
    p1.frames = {("ABC", Timeframe.M5): m5}
    assert not svc.candles("ABC", "5m").empty
    assert svc.healthy() and svc.last_error is None


def test_malformed_provider_output_never_raises(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    raw = m5.rename(columns=str.capitalize)
    raw.index = raw.index.tz_localize(None)                         # naive, vendor-style columns
    svc = _svc(FakeProvider("vendor", {("ABC", Timeframe.M5): raw}, raw=True), clock)
    out = svc.candles("ABC", "5m")
    validate_ohlcv(out)
    assert out.index[-1] == ist(2026, 11, 23, 9, 55)
    junk = pd.DataFrame({"open": [1.0]}, index=pd.Index(["not-a-time"]))
    svc2 = _svc(FakeProvider("junk", {("ABC", Timeframe.M5): junk}, raw=True), clock)
    assert svc2.candles("ABC", "5m").empty and not svc2.healthy()


# ------------------------------------------------------------------ caching


def test_cache_until_next_bar_boundary(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    p = FakeProvider("vendor", {("ABC", Timeframe.M5): m5})
    svc = _svc(p, clock)
    svc.candles("ABC", "5m")
    svc.candles("ABC", "5m", lookback_bars=10)
    assert len(p.calls) == 1
    clock.set(ist(2026, 11, 23, 10, 4) + timedelta(seconds=59))
    assert svc.candles("ABC", "5m").index[-1] == ist(2026, 11, 23, 9, 55)
    assert len(p.calls) == 1
    clock.set(ist(2026, 11, 23, 10, 5))                             # a new 5m bar closed
    assert svc.candles("ABC", "5m").index[-1] == ist(2026, 11, 23, 10, 0)
    assert len(p.calls) == 2
    svc.candles("ABC", "5m", start=ist(2026, 11, 2, 9, 15))          # earlier start than cached (16-Nov)
    assert len(p.calls) == 3
    svc.clear_cache()
    svc.candles("ABC", "5m")
    assert len(p.calls) == 4


def test_lagging_vendor_is_retried_quickly(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 2))
    lagging = m5.loc[m5.index < ist(2026, 11, 23, 9, 55)]            # 09:55 bar not published yet
    p = FakeProvider("vendor", {("ABC", Timeframe.M5): lagging})
    svc = _svc(p, clock)
    assert svc.candles("ABC", "5m").index[-1] == ist(2026, 11, 23, 9, 50)
    p.frames[("ABC", Timeframe.M5)] = m5
    clock.advance(timedelta(seconds=10))
    svc.candles("ABC", "5m")
    assert len(p.calls) == 1                                        # still inside the 15 s retry window
    clock.advance(timedelta(seconds=6))
    assert svc.candles("ABC", "5m").index[-1] == ist(2026, 11, 23, 9, 55)
    assert len(p.calls) == 2


def test_daily_history_disk_cache_survives_restart(d1: pd.DataFrame, tmp_path: Path) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 11, 0))
    p = FakeProvider("vendor", {("ABC", Timeframe.D1): d1})
    first = _svc(p, clock, cache_dir=tmp_path).candles("ABC", "1d", lookback_bars=30)
    assert len(p.calls) == 1
    assert list((tmp_path / "candles" / "vendor").glob("ABC_1d_20261120_*.parquet"))
    p2 = FakeProvider("vendor", {("ABC", Timeframe.D1): d1})
    second = _svc(p2, clock, cache_dir=tmp_path).candles("ABC", "1d", lookback_bars=30)
    assert p2.calls == []                                           # served from parquet
    assert list(second.index) == list(first.index)
    assert (second["close"].to_numpy() == first["close"].to_numpy()).all()


# ------------------------------------------------------------------ quotes


def _q(symbol: str, ts: datetime, ltp: float = 100.0, source: str = "fake") -> Quote:
    return Quote(symbol=symbol, ltp=ltp, timestamp=ts, source=source)


def test_future_quotes_rejected_and_fallback_used() -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 0))
    ahead = FakeProvider("ahead", quotes={"ABC": _q("ABC", ist(2026, 11, 23, 10, 1))})
    ok = FakeProvider("ok", quotes={"ABC": _q("ABC", ist(2026, 11, 23, 9, 59), 99.0)})
    assert _svc(ahead, clock).quote("ABC") is None
    q = _svc(ahead, clock, ok).quote("abc")
    assert q is not None and q.ltp == 99.0


def test_live_quote_stamped_after_now_is_accepted() -> None:
    """Regression: with a wall clock the provider stamps its quote AFTER the service read `now`."""
    class Live(FakeProvider):
        def get_quote(self, symbol: str) -> Quote | None:
            return _q(symbol, SystemClock().now())

        def get_quotes(self, symbols: Iterable[str]) -> dict[str, Quote]:
            return {s: _q(s, SystemClock().now()) for s in symbols}

    svc = _svc(Live("live"), SystemClock())
    assert svc.quote("ABC") is not None
    assert set(svc.quotes(["ABC", "XYZ"])) == {"ABC", "XYZ"}
    assert svc.healthy()


def test_quote_cache_ttl() -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 0))
    p = FakeProvider("p", quotes={"ABC": _q("ABC", ist(2026, 11, 23, 10, 0))})
    svc = _svc(p, clock)
    svc.quote("ABC")
    clock.advance(timedelta(seconds=4))
    svc.quote("ABC")
    assert p.quote_calls == ["ABC"]
    clock.advance(timedelta(seconds=2))
    svc.quote("ABC")
    assert p.quote_calls == ["ABC", "ABC"]


def test_bulk_quotes_retry_missing_on_fallback_and_total_failure() -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 10, 0))
    t = ist(2026, 11, 23, 9, 59)
    primary = FakeProvider("p", quotes={"AAA": _q("AAA", t, 1.0)})
    backup = FakeProvider("b", quotes={"AAA": _q("AAA", t, 2.0), "BBB": _q("BBB", t, 3.0)})
    svc = _svc(primary, clock, backup)
    got = svc.quotes(["aaa", "BBB", "CCC"])
    assert {k: v.ltp for k, v in got.items()} == {"AAA": 1.0, "BBB": 3.0}
    assert backup.quote_calls == ["BBB", "CCC"]
    dead = _svc(FakeProvider("x", fail=RuntimeError("down")), clock)
    assert dead.quotes(["AAA"]) == {} and not dead.healthy()
    assert dead.quote("AAA") is None


# ------------------------------------------------------------------ replay + factories


def test_service_over_replay_provider_is_look_ahead_free(m5: pd.DataFrame) -> None:
    clock = SimulatedClock(ist(2026, 11, 23, 9, 14))
    svc = _svc(ReplayDataProvider({("ABC", Timeframe.M5): m5}, clock, CAL), clock)
    for minutes in range(0, 90, 3):
        now = ist(2026, 11, 23, 9, 14) + timedelta(minutes=minutes)
        clock.set(now)
        for tf in ("5m", "15m", "1h", "1d"):
            bars = svc.candles("ABC", tf, lookback_bars=50)
            closes = bar_close_times(pd.DatetimeIndex(bars.index), tf, CAL)
            assert (closes <= pd.Timestamp(now)).all(), (tf, now)
        q = svc.quote("ABC")
        assert q is not None and q.timestamp <= now


def test_build_provider_and_service(settings: Settings) -> None:
    assert isinstance(build_provider("yfinance"), YFinanceProvider)
    with pytest.raises(ValueError):
        build_provider("replay")
    with pytest.raises(ValueError):
        build_provider("bloomberg")
    clock = SimulatedClock(ist(2026, 11, 23, 10, 0))
    svc = build_market_data_service(settings, None, clock, CAL)
    assert svc.provider.name == settings.engine.data_provider == "yfinance"
    assert svc.fallbacks == ()                                       # primary is not its own fallback
    replay = ReplayDataProvider({}, clock, CAL)
    svc2 = build_market_data_service(settings, None, clock, CAL, primary=replay)
    assert svc2.provider is replay and svc2.fallbacks == ()          # never mix live data into replay
    multi = settings.model_copy(update={"engine": settings.engine.model_copy(
        update={"data_provider": "upstox", "fallback_providers": ["angelone", "yfinance", "yfinance", "replay"]})})
    svc3 = build_market_data_service(multi, None, clock, CAL)
    assert [p.name for p in svc3.providers] == ["upstox", "angelone", "yfinance"]
