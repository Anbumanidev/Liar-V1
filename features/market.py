"""Broad market context (Market Context Agent input) - deterministic.

* NIFTY50 / BANKNIFTY trend from daily closes: BULLISH when ``close > EMA20 > EMA50``,
  BEARISH when ``close < EMA20 < EMA50``, else NEUTRAL (NEUTRAL with < 50 daily bars).
* Index % change: live quote's change vs previous close when a quote is available, else the
  last closed daily bar vs the one before.
* India VIX: latest value (quote, else last daily close) vs the mean of the last 20 daily
  closes; ``vix_elevated`` when ``vix > avg * (1 + vix_elevated_pct / 100)``.
* Sector strength: % change of each sector index (``sector -> index symbol`` map).
* Global cues: % change of the last closed daily bar of SP500, NASDAQ, DOWJONES, NIKKEI,
  HANGSENG, CRUDE, USDINR.
* Optional NSE client: ``advance_decline()`` ratio and ``fii_dii()`` net flows (crore).
* Market regime: :func:`features.regime.detect_regime` on NIFTY50 daily features with the
  VIX flag.

Every fetch is isolated: a failure is logged, noted in ``MarketContext.notes`` and the field
is left at its neutral default (graceful degradation, never raises).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger

from core.clock import to_ist
from core.enums import Regime, Signal, Timeframe
from core.types import MarketContext, Quote, TimeframeFeatures
from features.builder import closed_only
from features.indicators import compute_indicators, ema, latest_values
from features.regime import detect_regime
from storage.audit import record_error

NIFTY = "NIFTY50"
BANKNIFTY = "BANKNIFTY"
VIX = "INDIAVIX"
VIX_AVG_BARS = 20
INDEX_LOOKBACK_BARS = 300

#: sector label -> NSE sector index pseudo-symbol (labels are upper-case short names).
DEFAULT_SECTOR_INDICES: dict[str, str] = {
    "IT": "NIFTYIT",
    "PHARMA": "NIFTYPHARMA",
    "FMCG": "NIFTYFMCG",
    "AUTO": "NIFTYAUTO",
    "METAL": "NIFTYMETAL",
    "REALTY": "NIFTYREALTY",
    "ENERGY": "NIFTYENERGY",
    "PSU_BANK": "NIFTYPSUBANK",
    "PVT_BANK": "NIFTYPVTBANK",
    "FIN_SERVICE": "NIFTYFINSERVICE",
    "MEDIA": "NIFTYMEDIA",
    "BANK": "BANKNIFTY",
}

DEFAULT_GLOBAL_CUES: tuple[str, ...] = ("SP500", "NASDAQ", "DOWJONES", "NIKKEI", "HANGSENG", "CRUDE", "USDINR")


class MarketDataSource(Protocol):
    """Subset of ``data.service.MarketDataService`` used here."""

    def candles(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 300,
                start: datetime | None = None) -> pd.DataFrame: ...

    def quote(self, symbol: str) -> Quote | None: ...


class NSEFlowsSource(Protocol):
    """Subset of ``data.nse.NSEClient`` used here."""

    def advance_decline(self) -> float | None: ...

    def fii_dii(self) -> dict: ...


def index_trend(daily_df: pd.DataFrame | None) -> Signal:
    """Trend of an index from daily closes (see module docstring)."""
    if daily_df is None or len(daily_df) < 50:
        return Signal.NEUTRAL
    close = daily_df["close"].astype("float64")
    e20 = ema(close, 20).iloc[-1]
    e50 = ema(close, 50).iloc[-1]
    c = close.iloc[-1]
    if any(pd.isna(x) for x in (c, e20, e50)):
        return Signal.NEUTRAL
    if c > e20 > e50:
        return Signal.BULLISH
    if c < e20 < e50:
        return Signal.BEARISH
    return Signal.NEUTRAL


def daily_change_pct(daily_df: pd.DataFrame | None) -> float | None:
    """% change of the last daily close vs the previous one."""
    if daily_df is None or len(daily_df) < 2:
        return None
    prev, last = float(daily_df["close"].iloc[-2]), float(daily_df["close"].iloc[-1])
    if prev == 0 or not np.isfinite(prev) or not np.isfinite(last):
        return None
    return (last - prev) / prev * 100.0


def _parse_flow(data: dict, *keys: str) -> float | None:
    for k in keys:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except (TypeError, ValueError):
                logger.warning(f"fii_dii field {k!r} not numeric: {data[k]!r}")
    return None


class _Ctx:
    """Helper bundling the data source and the notes list for isolated fetches."""

    def __init__(self, data: MarketDataSource, as_of: datetime) -> None:
        self.data = data
        self.as_of = as_of
        self.notes: list[str] = []

    def candles(self, symbol: str, bars: int = INDEX_LOOKBACK_BARS) -> pd.DataFrame | None:
        try:
            df = self.data.candles(symbol, Timeframe.D1, lookback_bars=bars)
        except Exception as exc:  # isolate data failures
            record_error(None, "features.market", exc, {"symbol": symbol})
            self.notes.append(f"{symbol}: candles unavailable ({exc})")
            return None
        if df is None or len(df) == 0:
            self.notes.append(f"{symbol}: no daily candles")
            return None
        return closed_only(df, Timeframe.D1.value, self.as_of)

    def quote(self, symbol: str) -> Quote | None:
        getter = getattr(self.data, "quote", None)
        if getter is None:
            return None
        try:
            return getter(symbol)
        except Exception as exc:  # isolate data failures
            record_error(None, "features.market", exc, {"symbol": symbol, "what": "quote"})
            self.notes.append(f"{symbol}: quote unavailable ({exc})")
            return None


def _change(ctx: _Ctx, symbol: str, daily: pd.DataFrame | None, use_quote: bool = True) -> float | None:
    if use_quote:
        q = ctx.quote(symbol)
        if q is not None and q.change_pct is not None:
            return float(q.change_pct)
    return daily_change_pct(daily)


def build_market_context(
    data: MarketDataSource,
    as_of: datetime,
    nse: NSEFlowsSource | Any | None = None,
    sector_indices: dict[str, str] | None = None,
    vix_elevated_pct: float = 10.0,
    *,
    global_cues: tuple[str, ...] | list[str] = DEFAULT_GLOBAL_CUES,
    use_quotes: bool = True,
) -> MarketContext:
    """Assemble :class:`MarketContext` (see module docstring). Never raises."""
    as_of = to_ist(as_of)
    ctx = _Ctx(data, as_of)
    sectors = DEFAULT_SECTOR_INDICES if sector_indices is None else sector_indices

    nifty = ctx.candles(NIFTY)
    bank = ctx.candles(BANKNIFTY)
    vix_df = ctx.candles(VIX, bars=60)

    india_vix: float | None = None
    vix_avg: float | None = None
    vix_elevated = False
    if vix_df is not None and len(vix_df):
        closes = vix_df["close"].astype("float64").dropna()
        if len(closes) >= VIX_AVG_BARS:
            vix_avg = float(closes.iloc[-VIX_AVG_BARS:].mean())
        india_vix = float(closes.iloc[-1]) if len(closes) else None
    if use_quotes:
        q = ctx.quote(VIX)
        if q is not None and q.ltp and q.ltp > 0:
            india_vix = float(q.ltp)
    if india_vix is not None and vix_avg is not None and vix_avg > 0:
        vix_elevated = india_vix > vix_avg * (1.0 + vix_elevated_pct / 100.0)
    elif india_vix is None:
        ctx.notes.append("India VIX unavailable")

    sector_strength: dict[str, float] = {}
    for sector, sym in sectors.items():
        chg = _change(ctx, sym, ctx.candles(sym, bars=5), use_quotes)
        if chg is not None:
            sector_strength[sector] = round(chg, 4)

    cues: dict[str, float] = {}
    for sym in global_cues:
        chg = daily_change_pct(ctx.candles(sym, bars=5))
        if chg is not None:
            cues[sym] = round(chg, 4)

    ad_ratio = fii = dii = None
    if nse is not None:
        try:
            ad_ratio = nse.advance_decline()
        except Exception as exc:  # NSE endpoints are flaky; degrade
            record_error(None, "features.market", exc, {"what": "advance_decline"})
            ctx.notes.append(f"advance/decline unavailable ({exc})")
        try:
            flows = nse.fii_dii() or {}
            fii = _parse_flow(flows, "fii_net_cr", "fii_net", "FII", "fii")
            dii = _parse_flow(flows, "dii_net_cr", "dii_net", "DII", "dii")
        except Exception as exc:  # NSE endpoints are flaky; degrade
            record_error(None, "features.market", exc, {"what": "fii_dii"})
            ctx.notes.append(f"FII/DII unavailable ({exc})")

    regime = Regime.UNDEFINED
    nifty_trend = index_trend(nifty)
    if nifty is not None and len(nifty):
        try:
            values = latest_values(compute_indicators(nifty, Timeframe.D1))
            tff = TimeframeFeatures(timeframe=Timeframe.D1.value, as_of=as_of, bars=len(nifty), values=values)
            regime = detect_regime(tff, index_trend=nifty_trend, vix_elevated=vix_elevated)
        except Exception as exc:  # never let the regime calc kill the context
            record_error(None, "features.market", exc, {"what": "regime"})
            ctx.notes.append(f"market regime unavailable ({exc})")
    elif vix_elevated:
        regime = Regime.HIGH_VOLATILITY

    return MarketContext(
        as_of=as_of,
        nifty_trend=nifty_trend,
        nifty_change_pct=_change(ctx, NIFTY, nifty, use_quotes),
        banknifty_trend=index_trend(bank),
        banknifty_change_pct=_change(ctx, BANKNIFTY, bank, use_quotes),
        india_vix=india_vix,
        vix_20d_avg=vix_avg,
        vix_elevated=vix_elevated,
        advance_decline_ratio=float(ad_ratio) if ad_ratio is not None else None,
        fii_net_cr=fii,
        dii_net_cr=dii,
        sector_strength=sector_strength,
        global_cues=cues,
        regime=regime,
        notes=ctx.notes,
    )
