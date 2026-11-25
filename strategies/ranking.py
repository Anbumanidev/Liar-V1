"""'Stocks in play' ranking for the intraday watchlist (advanced selection).

Instead of rotating blindly through the watchlist, each INTRADAY cycle analyses the symbols
most likely to move today. Deterministic score (0-100) from closed bars only:

* relative volume today vs the same time of day over prior sessions (35 %),
* absolute gap / move from the previous close (25 %),
* intraday range expansion vs ATR (20 %),
* relative strength vs NIFTY today, either direction (20 %).

Liquidity and price filters are already applied when the watchlist is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from core.clock import to_ist
from core.enums import Timeframe


@dataclass(frozen=True)
class InPlay:
    symbol: str
    score: float
    rvol: float | None
    move_pct: float | None
    range_atr: float | None
    rs_pct: float | None


def _clip(x: float | None, lo: float, hi: float) -> float:
    if x is None or x != x:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _day_stats(intraday: pd.DataFrame, daily: pd.DataFrame, now: datetime) -> tuple[float | None, ...]:
    today = to_ist(now).date()
    tod = intraday[intraday.index.date == today]
    prior = daily[daily.index.date < today]
    if tod.empty or prior.empty:
        return None, None, None, None
    prev_close = float(prior["close"].iloc[-1])
    last = float(tod["close"].iloc[-1])
    move = (last - prev_close) / prev_close * 100.0 if prev_close else None
    # relative volume vs the same number of bars on previous sessions
    n = len(tod)
    vols = []
    for d in sorted({x for x in intraday.index.date if x < today})[-10:]:
        day = intraday[intraday.index.date == d]
        if len(day) >= n:
            vols.append(float(day["volume"].iloc[:n].sum()))
    rvol = float(tod["volume"].sum()) / (sum(vols) / len(vols)) if vols and sum(vols) > 0 else None
    tr = (prior["high"] - prior["low"]).tail(14)
    atr = float(tr.mean()) if len(tr) else None
    rng = float(tod["high"].max() - tod["low"].min())
    range_atr = rng / atr if atr else None
    return rvol, move, range_atr, prev_close


def rank_in_play(symbols: list[str], data: Any, now: datetime, top_n: int | None = None,
                 index_symbol: str = "NIFTY50") -> list[InPlay]:
    """Score and sort ``symbols`` (highest first). Symbols without data score 0 (kept last)."""
    idx_move = None
    try:
        idx_i = data.candles(index_symbol, Timeframe.M5, lookback_bars=400)
        idx_d = data.candles(index_symbol, Timeframe.D1, lookback_bars=30)
        _, idx_move, _, _ = _day_stats(idx_i, idx_d, now) if len(idx_i) and len(idx_d) else (None,) * 4
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[ranking] index data unavailable: {exc}")
    out: list[InPlay] = []
    for sym in symbols:
        try:
            intr = data.candles(sym, Timeframe.M5, lookback_bars=400)
            daily = data.candles(sym, Timeframe.D1, lookback_bars=30)
            rvol, move, range_atr, _ = _day_stats(intr, daily, now) if len(intr) and len(daily) else (None,) * 4
        except Exception as exc:  # noqa: BLE001 - one symbol never breaks the ranking
            logger.warning(f"[ranking] {sym}: {exc}")
            rvol = move = range_atr = None
        rs = (move - idx_move) if (move is not None and idx_move is not None) else None
        score = 100.0 * (0.35 * _clip(rvol, 0.8, 3.0) + 0.25 * _clip(abs(move) if move is not None else None, 0.3, 4.0)
                         + 0.20 * _clip(range_atr, 0.3, 1.5) + 0.20 * _clip(abs(rs) if rs is not None else None,
                                                                            0.3, 3.0))
        out.append(InPlay(sym, round(score, 2), rvol, move, range_atr, rs))
    out.sort(key=lambda x: (-x.score, x.symbol))
    return out[:top_n] if top_n else out
