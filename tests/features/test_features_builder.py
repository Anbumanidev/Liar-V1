"""FeatureBuilder / build_from_frames on synthetic multi-timeframe data."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from config.schema import Settings
from core.clock import IST
from core.enums import Mode, Regime, Signal, Timeframe
from core.errors import DataUnavailableError
from core.types import MarketContext, StockFeatures
from features.builder import (
    FeatureBuilder,
    bar_close_times,
    build_from_frames,
    closed_only,
    relative_strength,
)
from features.indicators import VALUE_KEYS
from tests.helpers import ist, make_daily_candles, make_intraday_candles, trading_days

TODAY = date(2026, 11, 24)
AS_OF = ist(2026, 11, 24, 11, 0)


def aggregate(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Session-aligned (09:15) aggregation of intraday candles."""
    mins = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 15)
    bucket = df.index.normalize() + pd.to_timedelta(9 * 60 + 15 + (mins // minutes) * minutes, unit="min")
    g = df.groupby(bucket)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    out.index = pd.DatetimeIndex(out.index, name="ts")
    return out


@pytest.fixture(scope="module")
def data() -> dict[str, pd.DataFrame]:
    days = trading_days(date(2026, 10, 26), 22)  # ends 2026-11-24
    assert days[-1] == TODAY
    m5 = make_intraday_candles(days, 5, start_price=100.0, seed=21)  # includes FUTURE bars after 11:00
    daily = make_daily_candles(TODAY, 300, start_price=80.0, seed=22)  # includes today's (unclosed) bar
    nifty = make_daily_candles(TODAY, 300, start_price=20000.0, seed=23, vol=0.008)
    return {"5m": m5, "15m": aggregate(m5, 15), "1h": aggregate(m5, 60), "1d": daily, "nifty": nifty}


@pytest.fixture(scope="module")
def feats(data: dict[str, pd.DataFrame]) -> StockFeatures:
    return build_from_frames("TEST", Mode.INTRADAY, {"5m": data["5m"], "15m": data["15m"], "1h": data["1h"]},
                             data["1d"], AS_OF, "5m", index_daily=data["nifty"])


def _no_nan(values: dict[str, float | None]) -> None:
    assert set(values) == set(VALUE_KEYS)
    for k, v in values.items():
        assert v is None or (isinstance(v, float) and math.isfinite(v)), k


def test_values_complete_and_nan_free(feats: StockFeatures) -> None:
    assert set(feats.timeframes) == {"5m", "15m", "1h"}
    for tff in feats.timeframes.values():
        _no_nan(tff.values)
        assert tff.levels is not None
    assert feats.timeframes["5m"].values["vwap"] is not None
    # JSON round trip works (no NaN leaking into the DTO)
    assert StockFeatures.model_validate_json(feats.model_dump_json()).ltp == feats.ltp


def test_only_closed_bars_used(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    m5 = data["5m"]
    assert feats.timeframes["5m"].as_of == ist(2026, 11, 24, 10, 55)
    assert feats.timeframes["15m"].as_of == ist(2026, 11, 24, 10, 45)
    assert feats.timeframes["1h"].as_of == ist(2026, 11, 24, 9, 15)  # 10:15 bar closes 11:15
    assert feats.ltp == pytest.approx(float(m5.loc[ist(2026, 11, 24, 10, 55), "close"]))
    assert feats.extra["ltp_bar_time"] == ist(2026, 11, 24, 10, 55).isoformat()


def test_no_lookahead_future_bars_ignored(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    cut = {k: v[v.index < AS_OF] for k, v in data.items() if k in ("5m", "15m", "1h")}
    cut["1h"] = cut["1h"][cut["1h"].index <= ist(2026, 11, 24, 9, 15)]
    daily_cut = data["1d"][data["1d"].index.date < TODAY]
    f2 = build_from_frames("TEST", Mode.INTRADAY, cut, daily_cut, AS_OF, "5m", index_daily=data["nifty"])
    assert f2.model_dump() == feats.model_dump()


def test_session_fields(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    m5 = data["5m"]
    daily = data["1d"]
    prior = daily[daily.index.date < TODAY].iloc[-1]
    today = m5[(m5.index.date == TODAY) & (m5.index < AS_OF)]
    assert feats.prev_close == pytest.approx(prior["close"])
    assert feats.extra["prev_day_high"] == pytest.approx(prior["high"])
    assert feats.extra["prev_day_low"] == pytest.approx(prior["low"])
    assert feats.gap_pct == pytest.approx((today["open"].iloc[0] - prior["close"]) / prior["close"] * 100)
    orw = today[today.index < ist(2026, 11, 24, 9, 30)]
    assert feats.opening_range_high == pytest.approx(orw["high"].max())
    assert feats.opening_range_low == pytest.approx(orw["low"].min())
    assert feats.day_high == pytest.approx(today["high"].max())
    assert feats.day_low == pytest.approx(today["low"].min())


def test_daily_aggregates(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    daily = data["1d"]
    closed = daily[daily.index.date < TODAY]
    last20 = closed.iloc[-20:]
    assert feats.avg_daily_volume_20 == pytest.approx(last20["volume"].mean())
    assert feats.avg_daily_turnover_20 == pytest.approx((last20["close"] * last20["volume"]).mean())
    year = closed[closed.index >= AS_OF - timedelta(days=365)]
    assert feats.week52_high == pytest.approx(year["high"].max())
    assert feats.week52_low == pytest.approx(year["low"].min())


def test_rs_vs_nifty(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    s = data["1d"][data["1d"].index.date < TODAY]["close"]
    n = data["nifty"][data["nifty"].index.date < TODAY]["close"]
    exp = ((s.iloc[-1] / s.iloc[-21] - 1) - (n.iloc[-1] / n.iloc[-21] - 1)) * 100
    assert feats.extra["rs_lookback"] == 20
    assert feats.rs_vs_nifty == pytest.approx(exp)


def test_primary_derived_fields(feats: StockFeatures) -> None:
    prim = feats.timeframes["5m"]
    assert feats.primary_timeframe == "5m"
    assert feats.atr == prim.values["atr_14"]
    assert feats.rvol == prim.values["rvol"]
    assert feats.regime == prim.regime
    assert feats.regime in set(Regime)


def test_intraday_levels_use_prior_session(feats: StockFeatures, data: dict[str, pd.DataFrame]) -> None:
    prior = data["1d"][data["1d"].index.date < TODAY].iloc[-1]
    lv5 = feats.timeframes["5m"].levels
    assert lv5 is not None and lv5.prev_high == pytest.approx(prior["high"])
    assert lv5.cpr_bc is not None and lv5.cpr_bc <= lv5.cpr_tc


def test_swing_mode_daily_primary(data: dict[str, pd.DataFrame]) -> None:
    after_close = ist(2026, 11, 24, 16, 0)
    f = build_from_frames("TEST", Mode.SWING, {"1h": data["1h"], "1d": data["1d"]}, None, after_close, "1d",
                          index_daily=data["nifty"])
    d = f.timeframes["1d"]
    assert d.as_of == ist(2026, 11, 24, 9, 15)  # today's daily bar closed at 15:30
    assert d.values["vwap"] is None
    assert d.values["ema_200"] is not None
    assert f.atr == d.values["atr_14"]
    # weekly pivots for daily tf: previous ISO week (16-20 Nov)
    daily = data["1d"]
    wk = daily[(daily.index.date >= date(2026, 11, 16)) & (daily.index.date <= date(2026, 11, 20))]
    assert d.levels is not None and d.levels.prev_high == pytest.approx(wk["high"].max())
    # prev close = Monday's close (today's daily bar exists)
    assert f.prev_close == pytest.approx(daily.iloc[-2]["close"])


def test_positional_rs_lookback(data: dict[str, pd.DataFrame]) -> None:
    f = build_from_frames("TEST", Mode.POSITIONAL, {"1d": data["1d"]}, None, ist(2026, 11, 24, 16, 0), "1d",
                          index_daily=data["nifty"])
    assert f.extra["rs_lookback"] == 55
    s, n = data["1d"]["close"], data["nifty"]["close"]
    exp = ((s.iloc[-1] / s.iloc[-56] - 1) - (n.iloc[-1] / n.iloc[-56] - 1)) * 100
    assert f.rs_vs_nifty == pytest.approx(exp)


def test_missing_timeframe_is_graceful(data: dict[str, pd.DataFrame]) -> None:
    empty = data["1h"].iloc[:0]
    f = build_from_frames("TEST", Mode.INTRADAY, {"5m": data["5m"], "1h": empty}, data["1d"], AS_OF, "5m")
    assert f.timeframes["1h"].bars == 0
    assert all(v is None for v in f.timeframes["1h"].values.values())
    assert "1h" in f.extra["missing_timeframes"]
    assert f.rs_vs_nifty is None  # no index data supplied


def test_premarket_has_no_session_fields(data: dict[str, pd.DataFrame]) -> None:
    pre = ist(2026, 11, 24, 8, 30)
    f = build_from_frames("TEST", Mode.INTRADAY, {"5m": data["5m"]}, data["1d"], pre, "5m")
    assert f.gap_pct is None
    assert f.opening_range_high is None and f.day_high is None
    assert f.timeframes["5m"].as_of.date() == date(2026, 11, 23)


def test_no_data_raises() -> None:
    empty = make_daily_candles(TODAY, 5).iloc[:0]
    with pytest.raises(DataUnavailableError):
        build_from_frames("X", Mode.SWING, {"1d": empty}, empty, AS_OF, "1d")


def test_market_context_feeds_regime(data: dict[str, pd.DataFrame]) -> None:
    mkt = MarketContext(as_of=AS_OF, nifty_trend=Signal.BULLISH, vix_elevated=True)
    f = build_from_frames("TEST", Mode.INTRADAY, {"5m": data["5m"]}, data["1d"], AS_OF, "5m", market=mkt)
    assert f.regime == Regime.HIGH_VOLATILITY
    assert all(t.regime == Regime.HIGH_VOLATILITY for t in f.timeframes.values() if t.bars)


def test_bar_close_times() -> None:
    idx = pd.DatetimeIndex([ist(2026, 11, 24, 15, 15), ist(2026, 11, 24, 9, 15)])
    assert bar_close_times(idx[:1], "1h") == [ist(2026, 11, 24, 15, 30)]  # capped at session end
    assert bar_close_times(idx[1:], "5m") == [ist(2026, 11, 24, 9, 20)]
    assert bar_close_times(idx[1:], "1d") == [ist(2026, 11, 24, 15, 30)]
    monday = pd.DatetimeIndex([ist(2026, 11, 23, 9, 15)])
    assert bar_close_times(monday, "1wk") == [ist(2026, 11, 27, 15, 30)]

    class Cal:
        def week_last_trading_day(self, d: date) -> date:
            return date(2026, 11, 26)

    assert bar_close_times(monday, "1wk", Cal()) == [ist(2026, 11, 26, 15, 30)]


def test_closed_only_sorts_and_dedupes() -> None:
    df = make_daily_candles(TODAY, 5)
    shuffled = pd.concat([df.iloc[::-1], df.iloc[:1]])
    out = closed_only(shuffled, "1d", ist(2026, 11, 24, 12, 0))
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert out.index[-1].date() == date(2026, 11, 23)


def test_relative_strength_insufficient() -> None:
    s = make_daily_candles(TODAY, 10)["close"]
    assert relative_strength(s, s, 20) is None
    assert relative_strength(s, s, 5) == pytest.approx(0.0)


# --------------------------------------------------------------------------- FeatureBuilder (service wrapper)


class FakeData:
    """Minimal stand-in for MarketDataService.candles."""

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame], fail: set[tuple[str, str]] | None = None) -> None:
        self.frames = frames
        self.fail = fail or set()
        self.calls: list[tuple[str, str, int]] = []

    def candles(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 300,
                start: datetime | None = None) -> pd.DataFrame:
        self.calls.append((symbol, timeframe.value, lookback_bars))
        if (symbol, timeframe.value) in self.fail:
            raise RuntimeError("provider down")
        df = self.frames.get((symbol, timeframe.value))
        if df is None:
            return make_daily_candles(TODAY, 1).iloc[:0]
        return df.tail(lookback_bars)


def test_feature_builder_build(settings: Settings, data: dict[str, pd.DataFrame]) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    frames = {("TEST", "5m"): data["5m"], ("TEST", "15m"): data["15m"], ("TEST", "1h"): data["1h"],
              ("TEST", "1d"): data["1d"], ("NIFTY50", "1d"): data["nifty"]}
    fake = FakeData(frames)
    fb = FeatureBuilder(fake)
    f = fb.build("TEST", Mode.INTRADAY, cfg, AS_OF)
    fetched = {(s, tf) for s, tf, _ in fake.calls}
    assert {("TEST", tf.value) for tf in cfg.timeframes} <= fetched
    assert ("NIFTY50", "1d") in fetched and ("TEST", "1d") in fetched
    assert f.primary_timeframe == cfg.primary_timeframe.value
    assert set(f.timeframes) == {tf.value for tf in cfg.timeframes}
    assert f.rs_vs_nifty is not None
    lookbacks = {tf: n for s, tf, n in fake.calls if s == "TEST"}
    assert lookbacks["5m"] == fb.lookback_bars["5m"] == 1500


def test_feature_builder_degrades_on_fetch_failure(settings: Settings, data: dict[str, pd.DataFrame]) -> None:
    cfg = settings.mode(Mode.INTRADAY)
    frames = {("TEST", "5m"): data["5m"], ("TEST", "15m"): data["15m"], ("TEST", "1h"): data["1h"],
              ("TEST", "1d"): data["1d"]}
    fake = FakeData(frames, fail={("TEST", "15m"), ("NIFTY50", "1d")})
    f = FeatureBuilder(fake).build("TEST", Mode.INTRADAY, cfg, AS_OF)
    assert "15m" in f.extra["missing_timeframes"]
    assert f.timeframes["15m"].bars == 0
    assert f.rs_vs_nifty is None
    assert f.ltp > 0


def test_feature_builder_btst_fetches_5m_base(settings: Settings, data: dict[str, pd.DataFrame]) -> None:
    cfg = settings.mode(Mode.BTST)
    frames = {("TEST", "5m"): data["5m"], ("TEST", "15m"): data["15m"], ("TEST", "1h"): data["1h"],
              ("TEST", "1d"): data["1d"], ("NIFTY50", "1d"): data["nifty"]}
    fake = FakeData(frames)
    f = FeatureBuilder(fake).build("TEST", Mode.BTST, cfg, ist(2026, 11, 24, 15, 0))
    assert ("TEST", "5m") in {(s, tf) for s, tf, _ in fake.calls}
    or_bars = data["5m"][(data["5m"].index >= ist(2026, 11, 24, 9, 15)) & (data["5m"].index < ist(2026, 11, 24, 9, 30))]
    assert f.opening_range_high == pytest.approx(or_bars["high"].max())
    assert "5m" not in f.timeframes


def test_instance_build_from_frames_alias(data: dict[str, pd.DataFrame]) -> None:
    fb = FeatureBuilder(FakeData({}))
    f = fb.build_from_frames("TEST", Mode.INTRADAY, {"5m": data["5m"]}, data["1d"], AS_OF, "5m")
    assert f.ltp > 0
    assert np.isfinite(f.ltp)
    assert f.as_of.tzinfo is not None and f.as_of.utcoffset() == IST.utcoffset(f.as_of)
