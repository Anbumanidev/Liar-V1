"""Market context from a fake data service."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from core.enums import Regime, Signal, Timeframe
from core.types import Quote
from features.market import DEFAULT_GLOBAL_CUES, build_market_context, daily_change_pct, index_trend
from tests.helpers import ist, make_daily_candles

TODAY = date(2026, 11, 24)
AS_OF = ist(2026, 11, 24, 10, 0)


def _trend_frame(direction: int, n: int = 120, start: float = 100.0) -> pd.DataFrame:
    df = make_daily_candles(date(2026, 11, 23), n, start_price=start, seed=1, drift=0.0, vol=0.001)
    closes = start * np.exp(np.cumsum(np.full(n, 0.01 * direction)))
    df["close"] = closes
    df["open"] = np.concatenate([[closes[0]], closes[:-1]])
    df["high"] = np.maximum(df["open"], df["close"]) * 1.002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.998
    return df


class FakeData:
    def __init__(self, frames: dict[str, pd.DataFrame], quotes: dict[str, Quote] | None = None,
                 fail: set[str] | None = None) -> None:
        self.frames = frames
        self.quotes = quotes or {}
        self.fail = fail or set()

    def candles(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 300,
                start: datetime | None = None) -> pd.DataFrame:
        assert timeframe == Timeframe.D1
        if symbol in self.fail:
            raise ConnectionError(f"{symbol} down")
        df = self.frames.get(symbol)
        if df is None:
            return make_daily_candles(TODAY, 1).iloc[:0]
        return df.tail(lookback_bars)

    def quote(self, symbol: str) -> Quote | None:
        return self.quotes.get(symbol)


class FakeNSE:
    def advance_decline(self) -> float | None:
        return 1.8

    def fii_dii(self) -> dict:
        return {"fii_net_cr": -1234.5, "dii_net_cr": "987.0"}


def _vix(last: float, base: float = 12.0) -> pd.DataFrame:
    df = make_daily_candles(date(2026, 11, 23), 40, start_price=base, seed=3, drift=0.0, vol=0.0)
    df["close"] = base
    df.iloc[-1, df.columns.get_loc("close")] = last
    return df


def test_index_trend() -> None:
    assert index_trend(_trend_frame(+1)) == Signal.BULLISH
    assert index_trend(_trend_frame(-1)) == Signal.BEARISH
    assert index_trend(_trend_frame(+1).iloc[:30]) == Signal.NEUTRAL  # not enough bars
    assert index_trend(None) == Signal.NEUTRAL


def test_daily_change_pct() -> None:
    df = _trend_frame(+1, 5)
    exp = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
    assert daily_change_pct(df) == pytest.approx(exp)
    assert daily_change_pct(df.iloc[:1]) is None


def test_full_context() -> None:
    frames = {"NIFTY50": _trend_frame(+1, start=20000), "BANKNIFTY": _trend_frame(-1, start=45000),
              "INDIAVIX": _vix(15.0), "NIFTYIT": _trend_frame(+1, 10), "SP500": _trend_frame(-1, 10)}
    ctx = build_market_context(FakeData(frames), AS_OF, nse=FakeNSE(), sector_indices={"IT": "NIFTYIT"},
                               vix_elevated_pct=10.0, global_cues=("SP500",))
    assert ctx.nifty_trend == Signal.BULLISH
    assert ctx.banknifty_trend == Signal.BEARISH
    assert ctx.nifty_change_pct == pytest.approx(daily_change_pct(frames["NIFTY50"]))
    assert ctx.india_vix == pytest.approx(15.0)
    vix_avg = (12.0 * 19 + 15.0) / 20
    assert ctx.vix_20d_avg == pytest.approx(vix_avg)
    assert ctx.vix_elevated is True  # 15 > 12.15 * 1.1
    assert ctx.regime == Regime.HIGH_VOLATILITY
    assert ctx.sector_strength["IT"] == pytest.approx(daily_change_pct(frames["NIFTYIT"]), abs=1e-4)
    assert ctx.global_cues["SP500"] < 0
    assert ctx.advance_decline_ratio == pytest.approx(1.8)
    assert ctx.fii_net_cr == pytest.approx(-1234.5)
    assert ctx.dii_net_cr == pytest.approx(987.0)


def test_vix_not_elevated_and_trend_regime() -> None:
    frames = {"NIFTY50": _trend_frame(+1, start=20000), "INDIAVIX": _vix(12.5)}
    ctx = build_market_context(FakeData(frames), AS_OF, sector_indices={}, global_cues=())
    assert ctx.vix_elevated is False  # 12.5 < 12.025 * 1.1
    assert ctx.regime == Regime.TRENDING_UP


def test_quotes_preferred_for_live_values() -> None:
    frames = {"NIFTY50": _trend_frame(+1, start=20000), "INDIAVIX": _vix(12.0)}
    quotes = {
        "NIFTY50": Quote(symbol="NIFTY50", ltp=101.0, prev_close=100.0, timestamp=AS_OF),
        "INDIAVIX": Quote(symbol="INDIAVIX", ltp=20.0, timestamp=AS_OF),
    }
    ctx = build_market_context(FakeData(frames, quotes), AS_OF, sector_indices={}, global_cues=())
    assert ctx.nifty_change_pct == pytest.approx(1.0)
    assert ctx.india_vix == pytest.approx(20.0)
    assert ctx.vix_elevated is True


def test_failures_degrade_gracefully() -> None:
    fake = FakeData({}, fail={"NIFTY50", "INDIAVIX", "NIFTYIT"})

    class BadNSE:
        def advance_decline(self) -> float:
            raise RuntimeError("nse 403")

        def fii_dii(self) -> dict:
            raise RuntimeError("nse 403")

    ctx = build_market_context(fake, AS_OF, nse=BadNSE(), sector_indices={"IT": "NIFTYIT"})
    assert ctx.nifty_trend == Signal.NEUTRAL
    assert ctx.india_vix is None and ctx.vix_elevated is False
    assert ctx.regime == Regime.UNDEFINED
    assert ctx.sector_strength == {} and ctx.global_cues == {}
    assert ctx.advance_decline_ratio is None and ctx.fii_net_cr is None
    assert any("NIFTY50" in n for n in ctx.notes)
    assert any("FII/DII" in n for n in ctx.notes)


def test_default_global_cues_list() -> None:
    assert set(DEFAULT_GLOBAL_CUES) == {"SP500", "NASDAQ", "DOWJONES", "NIKKEI", "HANGSENG", "CRUDE", "USDINR"}
