"""Regime classification (spec 7.1)."""

from __future__ import annotations

from datetime import date

import pytest

from core.enums import Regime, Signal
from core.types import TimeframeFeatures
from features.indicators import compute_indicators, latest_values
from features.regime import detect_regime
from tests.helpers import ist, make_daily_candles


def _tf(**values: float | None) -> TimeframeFeatures:
    base: dict[str, float | None] = {"adx_14": 22.0, "close": 100.0, "ema_21": 100.0, "vwap": None,
                                     "atr_expansion": 1.0, "bb_squeeze": 0.0}
    base.update(values)
    return TimeframeFeatures(timeframe="5m", as_of=ist(2026, 11, 24, 10, 0), bars=100, values=base)


def test_trending_up() -> None:
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100)) == Regime.TRENDING_UP


def test_trending_up_requires_above_vwap_when_present() -> None:
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100, vwap=106)) == Regime.UNDEFINED
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100, vwap=104)) == Regime.TRENDING_UP


def test_trending_down() -> None:
    assert detect_regime(_tf(adx_14=30, close=95, ema_21=100, vwap=96)) == Regime.TRENDING_DOWN


def test_index_opposition_blocks_trend() -> None:
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100), index_trend=Signal.BEARISH) == Regime.UNDEFINED
    assert detect_regime(_tf(adx_14=30, close=95, ema_21=100), index_trend=Signal.BULLISH) == Regime.UNDEFINED
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100), index_trend=Signal.NEUTRAL) == Regime.TRENDING_UP


def test_high_volatility_overrides() -> None:
    assert detect_regime(_tf(adx_14=30, close=105, ema_21=100), vix_elevated=True) == Regime.HIGH_VOLATILITY
    assert detect_regime(_tf(atr_expansion=1.5)) == Regime.HIGH_VOLATILITY
    assert detect_regime(_tf(atr_expansion=1.49)) == Regime.UNDEFINED


def test_ranging_conditions() -> None:
    assert detect_regime(_tf(adx_14=15)) == Regime.RANGING
    assert detect_regime(_tf(adx_14=22, bb_squeeze=1.0)) == Regime.RANGING
    assert detect_regime(_tf(adx_14=22), inside_cpr=True) == Regime.RANGING
    # strong ADX but price conflicting with EMA21 and no other range evidence -> UNDEFINED
    assert detect_regime(_tf(adx_14=30, close=100, ema_21=100)) == Regime.UNDEFINED


def test_boundaries_and_missing() -> None:
    assert detect_regime(_tf(adx_14=25, close=105, ema_21=100)) == Regime.UNDEFINED  # needs > 25
    assert detect_regime(_tf(adx_14=20)) == Regime.UNDEFINED  # needs < 20
    empty = TimeframeFeatures(timeframe="1d", as_of=ist(2026, 11, 24), bars=0, values={})
    assert detect_regime(empty) == Regime.UNDEFINED


def test_custom_thresholds() -> None:
    assert detect_regime(_tf(adx_14=22, close=105, ema_21=100), adx_trending=21) == Regime.TRENDING_UP
    assert detect_regime(_tf(atr_expansion=1.3), atr_expansion_high_vol=1.2) == Regime.HIGH_VOLATILITY


@pytest.mark.parametrize("drift,expected", [(0.01, Regime.TRENDING_UP), (-0.01, Regime.TRENDING_DOWN)])
def test_on_real_indicators(drift: float, expected: Regime) -> None:
    df = make_daily_candles(date(2026, 11, 24), 250, seed=1, drift=drift, vol=0.004)
    vals = latest_values(compute_indicators(df, "1d"))
    tf = TimeframeFeatures(timeframe="1d", as_of=ist(2026, 11, 24), bars=len(df), values=vals)
    assert detect_regime(tf) == expected
