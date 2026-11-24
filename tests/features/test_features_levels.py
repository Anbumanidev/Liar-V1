"""Pivots / CPR math, swing points, support/resistance and reference periods."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.types import Levels
from features import levels as lv
from tests.helpers import make_daily_candles, make_intraday_candles, trading_days


def test_pivots_classic_math() -> None:
    h, lo, c = 110.0, 90.0, 105.0
    L = lv.pivots(h, lo, c)
    p = (h + lo + c) / 3
    assert L.pivot == pytest.approx(p)
    assert L.r1 == pytest.approx(2 * p - lo)
    assert L.s1 == pytest.approx(2 * p - h)
    assert L.r2 == pytest.approx(p + (h - lo))
    assert L.s2 == pytest.approx(p - (h - lo))
    assert L.r3 == pytest.approx(h + 2 * (p - lo))
    assert L.s3 == pytest.approx(lo - 2 * (h - p))
    assert L.s3 < L.s2 < L.s1 < L.pivot < L.r1 < L.r2 < L.r3
    assert (L.prev_high, L.prev_low, L.prev_close) == (h, lo, c)


def test_cpr_ordering_and_width() -> None:
    # close above midpoint: raw tc > bc
    a = lv.pivots(110, 90, 108)
    bc_raw = 100.0
    p = (110 + 90 + 108) / 3
    tc_raw = 2 * p - bc_raw
    assert a.cpr_bc == pytest.approx(min(bc_raw, tc_raw))
    assert a.cpr_tc == pytest.approx(max(bc_raw, tc_raw))
    assert a.cpr_width_pct == pytest.approx(abs(tc_raw - bc_raw) / p * 100)
    # close below midpoint: raw tc < bc -> still ordered
    b = lv.pivots(110, 90, 92)
    assert b.cpr_bc <= b.cpr_tc
    assert b.cpr_tc == pytest.approx(100.0)


def test_pivots_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        lv.pivots(90, 110, 100)


def _frame(highs: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:15", periods=len(highs), freq="D", tz="Asia/Kolkata")
    h = pd.Series(highs, index=idx, dtype="float64")
    return pd.DataFrame({"open": h - 0.5, "high": h, "low": h - 1.0, "close": h - 0.3, "volume": 1000.0})


def test_swing_points_confirmed_only() -> None:
    df = _frame([1, 2, 5, 2, 1, 3, 4, 3, 2, 9])
    highs, lows = lv.swing_points(df, 2, 2)
    assert list(highs.to_numpy()) == [5.0, 4.0]
    assert list(highs.index) == [df.index[2], df.index[6]]
    assert list(lows.to_numpy()) == [0.0]
    # bar 9 (9.0) is the highest but cannot be a swing yet (no right arm)
    assert df.index[9] not in highs.index


def test_cluster_levels() -> None:
    out = lv.cluster_levels([100.0, 100.3, 104.0, 104.2, 110.0], cluster_pct=0.5)
    assert [n for _, n in out] == [2, 2, 1]
    assert out[0][0] == pytest.approx(100.15)


def test_support_resistance_nearest_first() -> None:
    # swings at 10, 20, 30, 40 (highs) and lows -1 each
    seq = [5, 6, 10, 6, 5, 6, 20, 6, 5, 6, 30, 6, 5, 6, 40, 6, 5]
    df = _frame(seq)
    sup, res = lv.support_resistance(df, price=25.0, n=3, cluster_pct=0.1)
    assert sup == sorted(sup, reverse=True)
    assert res == sorted(res)
    assert sup[0] == pytest.approx(20.0)
    assert res[0] == pytest.approx(30.0)
    assert all(s < 25 for s in sup) and all(r > 25 for r in res)
    assert len(sup) <= 3 and len(res) <= 3


def test_compute_levels_intraday_uses_prior_session() -> None:
    daily = make_daily_candles(date(2026, 11, 24), 30, seed=3)
    intra = make_intraday_candles([date(2026, 11, 24)], 5, start_price=float(daily["close"].iloc[-2]), seed=3)
    L = lv.compute_levels(daily, intra, float(intra["close"].iloc[-1]), timeframe="5m",
                          session_date=date(2026, 11, 24))
    prior = daily.iloc[-2]  # 2026-11-23 (daily includes 11-24, must be skipped)
    assert L.prev_high == pytest.approx(prior["high"])
    assert L.prev_low == pytest.approx(prior["low"])
    assert L.prev_close == pytest.approx(prior["close"])
    assert L.swing_highs and L.swing_lows
    assert len(L.swing_highs) <= lv.DEFAULT_MAX_SWINGS


def test_compute_levels_daily_uses_prior_week() -> None:
    daily = make_daily_candles(date(2026, 11, 25), 60, seed=4)  # Wed 2026-11-25
    L = lv.compute_levels(daily, daily, float(daily["close"].iloc[-1]), timeframe="1d",
                          session_date=date(2026, 11, 25))
    wk = daily[(daily.index.date >= date(2026, 11, 16)) & (daily.index.date <= date(2026, 11, 20))]
    assert L.prev_high == pytest.approx(wk["high"].max())
    assert L.prev_low == pytest.approx(wk["low"].min())
    assert L.prev_close == pytest.approx(wk["close"].iloc[-1])


def test_compute_levels_weekend_rolls_to_next_week() -> None:
    daily = make_daily_candles(date(2026, 11, 27), 60, seed=4)  # Fri 2026-11-27
    L = lv.compute_levels(daily, daily, 100.0, timeframe="1d", session_date=date(2026, 11, 28))  # Saturday
    wk = daily[(daily.index.date >= date(2026, 11, 23)) & (daily.index.date <= date(2026, 11, 27))]
    assert L.prev_high == pytest.approx(wk["high"].max())


def test_compute_levels_weekly_uses_prior_month() -> None:
    daily = make_daily_candles(date(2026, 11, 24), 120, seed=5)
    L = lv.compute_levels(daily, daily, 100.0, timeframe="1wk", session_date=date(2026, 11, 24))
    oct_ = daily[(daily.index.month == 10) & (daily.index.year == 2026)]
    assert L.prev_high == pytest.approx(oct_["high"].max())
    assert L.prev_close == pytest.approx(oct_["close"].iloc[-1])


def test_compute_levels_aggregates_intraday_when_no_daily() -> None:
    days = trading_days(date(2026, 11, 23), 2)
    intra = make_intraday_candles(days, 15, seed=6)
    L = lv.compute_levels(None, intra, float(intra["close"].iloc[-1]), timeframe="15m")
    d1 = intra[intra.index.date == days[0]]
    assert L.prev_high == pytest.approx(d1["high"].max())
    assert L.prev_close == pytest.approx(d1["close"].iloc[-1])


def test_compute_levels_without_reference_period() -> None:
    daily = make_daily_candles(date(2026, 11, 24), 2, seed=1)  # Mon + Tue only
    L = lv.compute_levels(daily, daily, 100.0, timeframe="1d", session_date=date(2026, 11, 24))
    assert L.pivot is None  # prior week not in data
    assert isinstance(L, Levels)


def test_inside_cpr() -> None:
    L = lv.pivots(110, 90, 108)
    mid = (L.cpr_bc + L.cpr_tc) / 2
    assert lv.inside_cpr(mid, L)
    assert not lv.inside_cpr(L.cpr_tc + 1, L)
    assert not lv.inside_cpr(None, L)
    assert not lv.inside_cpr(100.0, Levels())
