"""Indicator correctness (vs TA-Lib) and custom-rule behaviour."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from features import indicators as ind
from features.swings import fractal_flags, rsi_divergence_series, trend_structure_series
from tests.helpers import make_daily_candles, make_intraday_candles, trading_days

talib = pytest.importorskip("talib")

TOL = 1e-8


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return make_daily_candles(date(2026, 11, 24), 400, seed=11)


def _assert_close(ours: pd.Series | np.ndarray, ref: np.ndarray, start: int, tol: float = TOL) -> None:
    a = np.asarray(ours, dtype="float64")[start:]
    b = np.asarray(ref, dtype="float64")[start:]
    assert not np.isnan(a).any(), "our indicator has NaN after warm-up"
    assert not np.isnan(b).any()
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)


def test_rsi_matches_talib(daily: pd.DataFrame) -> None:
    ours = ind.rsi(daily["close"], 14)
    _assert_close(ours, talib.RSI(daily["close"].to_numpy(), 14), 14)
    assert ours.iloc[:14].isna().all()


@pytest.mark.parametrize("period", [9, 20, 21, 50, 200])
def test_ema_matches_talib(daily: pd.DataFrame, period: int) -> None:
    ours = ind.ema(daily["close"], period)
    _assert_close(ours, talib.EMA(daily["close"].to_numpy(), period), period - 1)
    assert ours.iloc[: period - 1].isna().all()


def test_sma_matches_talib(daily: pd.DataFrame) -> None:
    _assert_close(ind.sma(daily["close"], 20), talib.SMA(daily["close"].to_numpy(), 20), 19)


def test_atr_matches_talib(daily: pd.DataFrame) -> None:
    h, lo, c = (daily[k].to_numpy() for k in ("high", "low", "close"))
    _assert_close(ind.atr(daily, 14), talib.ATR(h, lo, c, 14), 14)


def test_adx_and_di_match_talib(daily: pd.DataFrame) -> None:
    h, lo, c = (daily[k].to_numpy() for k in ("high", "low", "close"))
    out = ind.adx(daily, 14)
    _assert_close(out["adx"], talib.ADX(h, lo, c, 14), 27)
    _assert_close(out["plus_di"], talib.PLUS_DI(h, lo, c, 14), 14)
    _assert_close(out["minus_di"], talib.MINUS_DI(h, lo, c, 14), 14)


def test_macd_matches_talib(daily: pd.DataFrame) -> None:
    m = ind.macd(daily["close"])
    ref = talib.MACD(daily["close"].to_numpy(), 12, 26, 9)
    for col, r in zip(("macd", "macd_signal", "macd_hist"), ref):
        _assert_close(m[col], r, 33)


def test_bollinger_matches_talib(daily: pd.DataFrame) -> None:
    bb = ind.bollinger(daily["close"], 20, 2.0)
    up, mid, low = talib.BBANDS(daily["close"].to_numpy(), 20, 2.0, 2.0, 0)
    _assert_close(bb["bb_upper"], up, 19)
    _assert_close(bb["bb_mid"], mid, 19)
    _assert_close(bb["bb_lower"], low, 19)
    expected_width = (up - low) / mid * 100
    _assert_close(bb["bb_width_pct"], expected_width, 19)


def test_intraday_indicators_match_talib() -> None:
    df = make_intraday_candles(trading_days(date(2026, 11, 2), 6), 5, seed=5)
    h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    _assert_close(ind.rsi(df["close"]), talib.RSI(c, 14), 14)
    _assert_close(ind.atr(df), talib.ATR(h, lo, c, 14), 14)
    _assert_close(ind.adx(df)["adx"], talib.ADX(h, lo, c, 14), 27)


def test_obv_matches_talib(daily: pd.DataFrame) -> None:
    ours = ind.obv(daily).to_numpy()
    ref = talib.OBV(daily["close"].to_numpy(), daily["volume"].to_numpy())
    # TA-Lib seeds OBV with the first volume; ours starts at 0 -> constant offset
    np.testing.assert_allclose(ours - ours[0], ref - ref[0], atol=1e-6)


def test_rsi_flat_series_is_50() -> None:
    s = pd.Series([100.0] * 30)
    assert ind.rsi(s).iloc[-1] == pytest.approx(50.0)


# --------------------------------------------------------------------------- VWAP


def test_vwap_resets_each_session() -> None:
    df = make_intraday_candles(trading_days(date(2026, 11, 23), 2), 5, seed=3)
    vw = ind.vwap(df)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day2 = df.index.normalize() == df.index.normalize()[-1]
    first_idx = np.flatnonzero(day2)[0]
    # first bar of the new session: VWAP == its typical price (reset)
    assert vw.iloc[first_idx] == pytest.approx(tp.iloc[first_idx])
    assert vw.iloc[0] == pytest.approx(tp.iloc[0])
    # manual cumulative check at the end of day 2
    d2 = df[day2]
    tp2 = tp[day2]
    expected = float((tp2 * d2["volume"]).sum() / d2["volume"].sum())
    assert vw.iloc[-1] == pytest.approx(expected)
    # and it differs from a non-reset VWAP over both days
    non_reset = float((tp * df["volume"]).sum() / df["volume"].sum())
    assert abs(vw.iloc[-1] - non_reset) > 1e-9


def test_vwap_none_for_daily_frames(daily: pd.DataFrame) -> None:
    vals = ind.latest_values(ind.compute_indicators(daily, "1d"))
    assert vals["vwap"] is None
    assert vals["vwap_dev_pct"] is None


def test_vwap_present_for_intraday() -> None:
    df = make_intraday_candles(trading_days(date(2026, 11, 2), 5), 15, seed=2)
    vals = ind.latest_values(ind.compute_indicators(df, "15m"))
    assert vals["vwap"] is not None
    assert vals["vwap_dev_pct"] == pytest.approx((vals["close"] - vals["vwap"]) / vals["vwap"] * 100)


# --------------------------------------------------------------------------- supertrend


def _frame_from_closes(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:15", periods=len(closes), freq="D", tz="Asia/Kolkata")
    c = np.asarray(closes, dtype="float64")
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": np.full(len(c), 1000.0)}, index=idx)


def test_supertrend_direction_flips() -> None:
    up = list(np.linspace(100, 140, 40))
    down = list(np.linspace(139, 90, 40))
    df = _frame_from_closes(up + down)
    st = ind.supertrend(df, 10, 3.0)
    d = st["supertrend_dir"]
    assert d.iloc[39] == 1.0
    assert d.iloc[-1] == -1.0
    flips = d.dropna().diff().fillna(0) != 0
    assert flips.sum() >= 1
    # line is below price in uptrend, above in downtrend
    assert st["supertrend"].iloc[39] < df["close"].iloc[39]
    assert st["supertrend"].iloc[-1] > df["close"].iloc[-1]
    assert st["supertrend"].iloc[:10].isna().all()


def test_supertrend_lower_band_never_falls_in_uptrend() -> None:
    df = _frame_from_closes(list(np.linspace(100, 160, 60)))
    st = ind.supertrend(df)
    line = st["supertrend"].dropna()
    assert (st["supertrend_dir"].dropna() == 1.0).all()
    assert (line.diff().dropna() >= -1e-12).all()


# --------------------------------------------------------------------------- volume / squeeze


def test_rvol_intraday_uses_same_time_of_day() -> None:
    days = trading_days(date(2026, 11, 2), 4)
    df = make_intraday_candles(days, 15, seed=1)
    df["volume"] = 1000.0
    last_day = df.index.normalize() == df.index.normalize()[-1]
    first_bar_last_day = np.flatnonzero(last_day)[0]
    df.iloc[first_bar_last_day, df.columns.get_loc("volume")] = 3000.0
    rv = ind.rvol(df, intraday=True, sessions=10, min_sessions=2)
    assert rv.iloc[first_bar_last_day] == pytest.approx(3.0)
    assert rv.iloc[-1] == pytest.approx(1.0)
    # first session has no prior same-time history -> NaN
    assert rv.iloc[0] != rv.iloc[0]


def test_rvol_daily_vs_prior_20_bars() -> None:
    df = make_daily_candles(date(2026, 11, 24), 40, seed=1)
    df["volume"] = 500.0
    df.iloc[-1, df.columns.get_loc("volume")] = 1000.0
    rv = ind.rvol(df, intraday=False)
    assert rv.iloc[-1] == pytest.approx(2.0)
    assert rv.iloc[:20].isna().all()


def test_bollinger_squeeze_on_contraction() -> None:
    rng = np.random.default_rng(0)
    wide = 100 + np.cumsum(rng.normal(0, 2.0, 150))
    tight = wide[-1] + np.cumsum(rng.normal(0, 0.05, 40))
    df = _frame_from_closes(list(wide) + list(tight))
    out = ind.compute_indicators(df, "1d")
    assert out["bb_squeeze"].iloc[-1] == 1.0
    assert out["bb_squeeze"].iloc[: ind.SQUEEZE_MIN_PERIODS].isna().all()
    # absolute threshold can flag a squeeze before the percentile window is ready
    sq = ind.bollinger_squeeze(out["bb_width_pct"], threshold_pct=1000.0)
    assert sq.iloc[25] == 1.0


def test_atr_expansion_and_pct(daily: pd.DataFrame) -> None:
    out = ind.compute_indicators(daily, "1d")
    a = out["atr_14"]
    expected = a.iloc[-1] / a.iloc[-20:].mean()
    assert out["atr_expansion"].iloc[-1] == pytest.approx(expected)
    assert out["atr_pct"].iloc[-1] == pytest.approx(a.iloc[-1] / daily["close"].iloc[-1] * 100)


# --------------------------------------------------------------------------- compute / latest


def test_compute_indicators_columns_exact(daily: pd.DataFrame) -> None:
    out = ind.compute_indicators(daily, "1d")
    assert tuple(out.columns) == ind.VALUE_KEYS
    assert out.index.equals(daily.index)


def test_compute_indicators_is_causal(daily: pd.DataFrame) -> None:
    full = ind.compute_indicators(daily, "1d")
    part = ind.compute_indicators(daily.iloc[:300], "1d")
    cols = [c for c in ind.VALUE_KEYS]
    pd.testing.assert_frame_equal(full.iloc[:300][cols], part[cols], check_exact=False, atol=1e-9)


def test_latest_values_converts_nan_to_none() -> None:
    df = make_daily_candles(date(2026, 11, 24), 30, seed=2)
    vals = ind.latest_values(ind.compute_indicators(df, "1d"))
    assert set(vals) == set(ind.VALUE_KEYS)
    assert vals["ema_200"] is None  # not enough bars
    assert vals["ema_9"] is not None
    assert all(v is None or isinstance(v, float) for v in vals.values())
    assert all(v is None or np.isfinite(v) for v in vals.values())


def test_empty_frame() -> None:
    empty = make_daily_candles(date(2026, 11, 24), 5).iloc[:0]
    out = ind.compute_indicators(empty, "1d")
    assert list(out.columns) == list(ind.VALUE_KEYS)
    assert ind.latest_values(out) == {k: None for k in ind.VALUE_KEYS}


def test_inside_bar_column() -> None:
    df = _frame_from_closes([100, 105, 103], spread=0.0)
    df.loc[df.index[2], ["high", "low"]] = [104.0, 101.0]
    out = ind.compute_indicators(df, "1d")
    assert out["inside_bar"].iloc[2] == 1.0
    assert out["inside_bar"].iloc[1] == 0.0
    assert np.isnan(out["inside_bar"].iloc[0])


def test_intraday_inference() -> None:
    intr = make_intraday_candles(trading_days(date(2026, 11, 2), 1), 5)
    assert ind.infer_intraday(intr)
    assert not ind.infer_intraday(make_daily_candles(date(2026, 11, 24), 10))
    assert ind.is_intraday_timeframe("1h") is True
    assert ind.is_intraday_timeframe("1wk") is False
    assert ind.is_intraday_timeframe("weird") is None


# --------------------------------------------------------------------------- swings / structure / divergence


def test_fractal_flags_basic() -> None:
    high = np.array([1, 2, 5, 2, 1, 3, 4, 3, 2], dtype=float)
    low = high - 1
    is_h, is_l = fractal_flags(high, low, 2, 2)
    assert list(np.flatnonzero(is_h)) == [2, 6]
    assert list(np.flatnonzero(is_l)) == [4]
    # last two bars can never be swings (not confirmed)
    assert not is_h[-2:].any() and not is_l[-2:].any()


def test_trend_structure_series_up_and_down() -> None:
    # zig-zag with higher highs and higher lows
    up = [10, 12, 14, 12, 11, 13, 16, 14, 13, 15, 18, 16, 15, 17, 20, 18, 17]
    h = np.array(up, dtype=float)
    ts = trend_structure_series(h, h - 0.5, 2, 2)
    assert ts[-1] == 1
    down = [30 - x for x in up]
    hd = np.array(down, dtype=float)
    ts_d = trend_structure_series(hd, hd - 0.5, 2, 2)
    assert ts_d[-1] == -1


def test_rsi_divergence_rule() -> None:
    lows = np.array([10, 9, 8, 9, 10, 11, 10, 9, 7, 9, 10, 11], dtype=float)
    highs = lows + 1
    rsi = np.array([50, 40, 30, 40, 50, 55, 50, 45, 35, 45, 50, 55], dtype=float)
    bull, bear = rsi_divergence_series(highs, lows, rsi, 2, 2, max_age=5)
    # swing lows at 2 (8, rsi 30) and 8 (7, rsi 35): lower low in price, higher low in RSI
    assert bull[10] == 1.0  # confirmed at bar 10 = 8 + right
    assert bull[9] == 0.0   # not confirmed yet at bar 9 (no look-ahead)
    assert not bear.any()
    rsi_conf = rsi.copy()
    rsi_conf[8] = 25.0  # RSI confirms the lower low -> no divergence
    bull2, _ = rsi_divergence_series(highs, lows, rsi_conf, 2, 2)
    assert bull2[10] == 0.0
