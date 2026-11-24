"""Candlestick patterns (TA-Lib and pure-python fallback) and price-action helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.enums import Signal
from features import patterns as pt
from tests.helpers import ist, make_daily_candles, make_intraday_candles, trading_days

BOTH = [pytest.param(True, id="talib"), pytest.param(False, id="fallback")]


def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:15", periods=len(bars), freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx).assign(volume=1000.0)


def _downtrend(n: int = 12, start: float = 120.0) -> tuple[list[tuple[float, float, float, float]], float]:
    bars, p = [], start
    for _ in range(n):
        o, c = p, p - 1.5
        bars.append((o, o + 0.3, c - 0.3, c))
        p = c
    return bars, p


def _uptrend(n: int = 12, start: float = 100.0) -> tuple[list[tuple[float, float, float, float]], float]:
    bars, p = [], start
    for _ in range(n):
        o, c = p, p + 1.5
        bars.append((o, c + 0.3, o - 0.3, c))
        p = c
    return bars, p


def _names(hits: list) -> dict[str, tuple[Signal, int]]:
    return {h.name: (h.direction, h.strength) for h in hits}


@pytest.fixture(autouse=True)
def _need_talib_for_talib_param(request: pytest.FixtureRequest) -> None:
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None and callspec.params.get("use_talib") is True and not pt.TALIB_AVAILABLE:
        pytest.skip("TA-Lib not installed")


@pytest.mark.parametrize("use_talib", BOTH)
def test_bullish_engulfing(use_talib: bool) -> None:
    bars, p = _downtrend()
    bars += [(p + 0.2, p + 0.5, p - 1.2, p - 1.0), (p - 1.3, p + 1.2, p - 1.4, p + 1.0)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["engulfing"] == (Signal.BULLISH, 100)


@pytest.mark.parametrize("use_talib", BOTH)
def test_bearish_engulfing(use_talib: bool) -> None:
    bars, p = _uptrend()
    bars += [(p - 0.2, p + 1.2, p - 0.5, p + 1.0), (p + 1.3, p + 1.4, p - 1.4, p - 1.0)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["engulfing"] == (Signal.BEARISH, -100)


@pytest.mark.parametrize("use_talib", BOTH)
def test_hammer_after_downtrend(use_talib: bool) -> None:
    bars, p = _downtrend()
    bars += [(p - 0.6, p - 0.1, p - 3.0, p - 0.15)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["hammer"][0] == Signal.BULLISH
    assert hits["hammer"][1] > 0


@pytest.mark.parametrize("use_talib", BOTH)
def test_shooting_star_after_uptrend(use_talib: bool) -> None:
    bars, p = _uptrend()
    bars += [(p + 0.1, p + 3.0, p, p + 0.05)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["shooting_star"] == (Signal.BEARISH, -100)


@pytest.mark.parametrize("use_talib", BOTH)
def test_doji_is_neutral(use_talib: bool) -> None:
    bars, p = _downtrend()
    bars += [(p, p + 1.0, p - 1.0, p + 0.02)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["doji"] == (Signal.NEUTRAL, 0)


@pytest.mark.parametrize("use_talib", BOTH)
def test_morning_star(use_talib: bool) -> None:
    bars, p = _downtrend()
    bars += [(p, p + 0.2, p - 3.2, p - 3.0), (p - 3.4, p - 3.2, p - 3.9, p - 3.5), (p - 3.3, p - 0.5, p - 3.4, p - 0.8)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["morning_star"] == (Signal.BULLISH, 100)


@pytest.mark.parametrize("use_talib", BOTH)
def test_evening_star(use_talib: bool) -> None:
    bars, p = _uptrend()
    bars += [(p, p + 3.2, p - 0.2, p + 3.0), (p + 3.4, p + 3.9, p + 3.2, p + 3.5), (p + 3.3, p + 3.4, p + 0.5, p + 0.8)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["evening_star"] == (Signal.BEARISH, -100)


@pytest.mark.parametrize("use_talib", BOTH)
def test_bearish_harami_and_inside_bar(use_talib: bool) -> None:
    bars, p = _uptrend()
    bars += [(p, p + 3.2, p - 0.1, p + 3.0), (p + 2.0, p + 2.2, p + 1.4, p + 1.6)]
    hits = _names(pt.detect_patterns(_frame(bars), "1d", 1, use_talib=use_talib))
    assert hits["harami"] == (Signal.BEARISH, -100)
    assert hits["inside_bar"] == (Signal.NEUTRAL, 0)


def test_lookback_limits_reported_bars() -> None:
    bars, p = _downtrend()
    bars += [(p + 0.2, p + 0.5, p - 1.2, p - 1.0), (p - 1.3, p + 1.2, p - 1.4, p + 1.0)]
    # add 3 plain bars after the engulfing so it falls outside lookback=3
    q = p + 1.0
    for _ in range(3):
        bars.append((q, q + 2.0, q - 0.1, q + 1.9))
        q += 1.9
    df = _frame(bars)
    assert "engulfing" not in _names(pt.detect_patterns(df, "1d", 3, use_talib=False))
    hits = pt.detect_patterns(df, "1d", 4, use_talib=False)
    eng = [h for h in hits if h.name == "engulfing"]
    assert eng and eng[0].bar_time == df.index[-4].to_pydatetime()
    assert [h.bar_time for h in hits] == sorted(h.bar_time for h in hits)


def test_hits_have_timeframe_and_ist_time() -> None:
    df = make_intraday_candles(trading_days(date(2026, 11, 2), 3), 5, seed=7)
    hits = pt.detect_patterns(df, "5m", lookback=20)
    for h in hits:
        assert h.timeframe == "5m"
        assert h.bar_time.utcoffset().total_seconds() == 5.5 * 3600
        assert -100 <= h.strength <= 100


def test_empty_input() -> None:
    assert pt.detect_patterns(make_daily_candles(date(2026, 11, 24), 5).iloc[:0], "1d") == []


def test_fallback_patterns_shapes() -> None:
    df = make_daily_candles(date(2026, 11, 24), 50, seed=3)
    res = pt.fallback_patterns(df["open"], df["high"], df["low"], df["close"])
    assert set(res) == {"doji", "hammer", "shooting_star", "engulfing", "harami", "morning_star", "evening_star"}
    for arr in res.values():
        assert len(arr) == 50
        assert set(np.unique(arr)).issubset({-100, 0, 100})


# --------------------------------------------------------------------------- price action


def test_opening_range_complete_window() -> None:
    day = date(2026, 11, 24)
    df = make_intraday_candles([day], 5, seed=2)
    closed = df[df.index < ist(2026, 11, 24, 9, 30)]  # bars 09:15, 09:20, 09:25 (all closed by 09:30)
    rng = pt.opening_range(closed, day)
    assert rng == (closed["high"].max(), closed["low"].min())


def test_opening_range_incomplete_is_none() -> None:
    day = date(2026, 11, 24)
    df = make_intraday_candles([day], 5, seed=2)
    partial = df[df.index < ist(2026, 11, 24, 9, 25)]  # 09:25 bar not closed yet
    assert pt.opening_range(partial, day) is None


def test_opening_range_bar_bigger_than_window_is_none() -> None:
    day = date(2026, 11, 24)
    df = make_intraday_candles([day], 60, seed=2)
    assert pt.opening_range(df, day) is None


def test_opening_range_single_15m_bar_with_hint() -> None:
    day = date(2026, 11, 24)
    df = make_intraday_candles([day], 15, seed=2).iloc[:1]
    assert pt.opening_range(df, day) is None  # bar size not inferable
    rng = pt.opening_range(df, day, bar_minutes=15)
    assert rng == (float(df["high"].iloc[0]), float(df["low"].iloc[0]))


def test_opening_range_no_data_for_day() -> None:
    df = make_intraday_candles([date(2026, 11, 23)], 5)
    assert pt.opening_range(df, date(2026, 11, 24)) is None


def test_gap_pct_from_daily_and_intraday() -> None:
    daily = make_daily_candles(date(2026, 11, 23), 10, seed=1)
    prev_close = float(daily["close"].iloc[-1])
    intra = make_intraday_candles([date(2026, 11, 24)], 5, start_price=prev_close * 1.02, seed=4)
    first_open = float(intra["open"].iloc[0])
    assert pt.gap_pct(daily, intra) == pytest.approx((first_open - prev_close) / prev_close * 100)
    # daily-only: last bar's open vs previous close
    exp = (daily["open"].iloc[-1] - daily["close"].iloc[-2]) / daily["close"].iloc[-2] * 100
    assert pt.gap_pct(daily) == pytest.approx(exp)
    assert pt.gap_pct(daily.iloc[:0]) is None


def test_gap_pct_intraday_only_uses_prior_session_close() -> None:
    intra = make_intraday_candles([date(2026, 11, 23), date(2026, 11, 24)], 15, seed=4)
    day1 = intra[intra.index.date == date(2026, 11, 23)]
    day2 = intra[intra.index.date == date(2026, 11, 24)]
    exp = (day2["open"].iloc[0] - day1["close"].iloc[-1]) / day1["close"].iloc[-1] * 100
    assert pt.gap_pct(None, intra) == pytest.approx(exp)


def test_inside_bar_helper() -> None:
    df = _frame([(10, 12, 8, 11), (11, 11.5, 9, 10)])
    assert pt.inside_bar(df)
    df2 = _frame([(10, 12, 8, 11), (11, 12.5, 9, 10)])
    assert not pt.inside_bar(df2)
    assert not pt.inside_bar(df.iloc[:1])


def test_trend_structure_helper() -> None:
    up = [10, 12, 14, 12, 11, 13, 16, 14, 13, 15, 18, 16, 15, 17, 20, 18, 17]
    df_up = _frame([(x, x + 0.2, x - 0.5, x) for x in up])
    df_dn = _frame([(30 - x, 30 - x + 0.5, 30 - x - 0.2, 30 - x) for x in up])
    assert pt.trend_structure(df_up, 2) == 1
    assert pt.trend_structure(df_dn, 2) == -1
    flat = _frame([(10, 10.5, 9.5, 10)] * 10)
    assert pt.trend_structure(flat, 2) == 0
