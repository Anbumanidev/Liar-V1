"""Price behaviour before an entry: where in the day's range, how stretched, and how this stock behaved.

Owner's finding (2026-11-25): all three losing trades were bought high in the day's range -
ITC at 86 % of a range already 168 % of its normal daily range and 3.4 above VWAP; BEL twice at
72-73 % of a range already 112 % of normal, just under the previous day's high. This module
measures that for every decision (and for the post-mortem), deterministically:

* ``range_pos``      0 = at the day's low, 1 = at the day's high (live price included)
* ``range_vs_adr``   today's range so far / 14-day average daily range (>= 1 = a full day's move done)
* ``vwap_ext_atr``   (price - session VWAP) / ATR(14) of 5-minute bars, signed
* gap vs the previous close, previous day high / low / close
* **behaviour history** - the stock's own last N sessions: on each day, the first time a 5-minute
  bar closed in the top 15 % (bottom 15 % for shorts) of a range already >= 0.7 x ADR between
  10:00 and 14:30, did a trade from there reach +1.5R before -1R (R = 1.5 x 5m ATR)? Neither
  by the close -> it continued if the close was beyond the entry. ``continuation_rate`` =
  continued / samples. A low rate means "moves to the top of the range usually fade here".

Only CLOSED bars before ``now`` are used (no look-ahead), so the same function serves live
decisions, approvals and post-mortems. Never raises: returns None without enough data.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

import numpy as np
import pandas as pd

from core.clock import to_ist
from features.indicators import atr as atr_fn

ZONE = 0.85                 # "top of the range" for the history study (bottom = 1 - ZONE)
MIN_RANGE_ADR = 0.7         # the day must already have moved this much of its ADR
WINDOW = (time(10, 0), time(14, 30))
R_ATR = 1.5                 # R = 1.5 x 5m ATR for the history outcome
WIN_R = 1.5                 # "continued" = reached +1.5R before -1R (the INTRADAY minimum reward:risk)


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _frame(data: Any, symbol: str, timeframe: str, **kw: Any) -> pd.DataFrame:
    try:
        df = data.candles(symbol, timeframe, **kw)
    except Exception:  # noqa: BLE001 - behaviour is advisory input; missing data -> None
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _history(m5: pd.DataFrame, atr5: pd.Series, adr: float, today: Any, sessions: int,
             long_: bool) -> tuple[int, int]:
    """(continued, samples) of zone touches over the last ``sessions`` days before ``today``."""
    days = sorted({ts.date() for ts in m5.index if ts.date() < today})[-sessions:]
    cont = samples = 0
    for d in days:
        day = m5.loc[[ts.date() == d for ts in m5.index]]
        if len(day) < 20:
            continue
        hi, lo = -np.inf, np.inf
        for i, (ts, bar) in enumerate(day.iterrows()):
            hi, lo = max(hi, float(bar["high"])), min(lo, float(bar["low"]))
            if not (WINDOW[0] <= ts.time() <= WINDOW[1]) or hi - lo < MIN_RANGE_ADR * adr or hi <= lo:
                continue
            pos = (float(bar["close"]) - lo) / (hi - lo)
            if (long_ and pos < ZONE) or (not long_ and pos > 1 - ZONE):
                continue
            a = float(atr5.get(ts, np.nan))
            if not np.isfinite(a) or a <= 0:
                break
            entry, r = float(bar["close"]), R_ATR * a
            won: bool | None = None
            for _, nb in day.iloc[i + 1:].iterrows():
                hi_b, lo_b = float(nb["high"]), float(nb["low"])
                if long_:
                    if lo_b <= entry - r:            # stop first (also when both in one bar: conservative)
                        won = False
                        break
                    if hi_b >= entry + WIN_R * r:
                        won = True
                        break
                else:
                    if hi_b >= entry + r:
                        won = False
                        break
                    if lo_b <= entry - WIN_R * r:
                        won = True
                        break
            if won is None:
                last = float(day["close"].iloc[-1])
                won = last > entry if long_ else last < entry
            samples += 1
            cont += int(won)
            break   # one event per day
    return cont, samples


def compute(data: Any, symbol: str, now: datetime, *, direction: str = "LONG", live_price: float | None = None,
            sessions: int = 20, thresholds: Any = None) -> dict[str, Any] | None:
    """Price-behaviour snapshot of ``symbol`` at ``now`` (see module docstring).

    Args:
        data: MarketDataService-like object with ``candles(symbol, timeframe, lookback_bars=, start=)``.
        direction: "LONG" or "SHORT" (which side of the range counts as chasing).
        live_price: price to judge (quote / planned fill); default = last closed 5m close.
        thresholds: optional ``settings.risk`` - adds a display ``verdict`` (the risk manager decides).
    """
    now = to_ist(now)
    long_ = str(direction).upper() != "SHORT"
    m5 = _frame(data, symbol, "5m", lookback_bars=76 * (sessions + 3))
    if m5.empty:
        return None
    m5 = m5.loc[m5.index < pd.Timestamp(now)]
    today = now.date()
    tod = m5.loc[[ts.date() == today for ts in m5.index]]
    if tod.empty:
        return None
    d1 = _frame(data, symbol, "1d", lookback_bars=30)
    if not d1.empty:
        d1 = d1.loc[[ts.date() < today for ts in d1.index]]
    if len(d1) < 5:
        return None
    adr = float((d1["high"] - d1["low"]).tail(14).mean())
    pdh, pdl, pdc = float(d1["high"].iloc[-1]), float(d1["low"].iloc[-1]), float(d1["close"].iloc[-1])
    ltp = float(live_price) if live_price else float(tod["close"].iloc[-1])
    day_high = max(float(tod["high"].max()), ltp)
    day_low = min(float(tod["low"].min()), ltp)
    day_open = float(tod["open"].iloc[0])
    rng = day_high - day_low
    range_pos = (ltp - day_low) / rng if rng > 0 else 0.5
    range_vs_adr = rng / adr if adr > 0 else None
    tp = (tod["high"] + tod["low"] + tod["close"]) / 3.0
    vol = tod["volume"].astype(float)
    vwap = float((tp * vol).sum() / vol.sum()) if float(vol.sum()) > 0 else float(tp.mean())
    atr5 = atr_fn(m5, 14)
    a = float(atr5.iloc[-1]) if len(atr5) else float("nan")
    vwap_ext = (ltp - vwap) / a if np.isfinite(a) and a > 0 else None
    cont = samples = 0
    if adr > 0 and len(m5) > 200:
        cont, samples = _history(m5, atr5, adr, today, sessions, long_)
    rate = cont / samples if samples else None

    side_pos = range_pos if long_ else 1.0 - range_pos
    notes = [f"price {ltp:.2f} is at {range_pos * 100:.0f}% of today's range {day_low:.2f}-{day_high:.2f}"]
    if range_vs_adr is not None:
        notes.append(f"the day has already moved {range_vs_adr * 100:.0f}% of its normal daily range "
                     f"(14-day ADR {adr:.2f})")
    if vwap_ext is not None:
        notes.append(f"{abs(vwap_ext):.1f} ATR {'above' if vwap_ext >= 0 else 'below'} VWAP {vwap:.2f}")
    if long_ and pdh > ltp and a > 0 and (pdh - ltp) <= 1.0 * a:
        notes.append(f"just under the previous day's high {pdh:.2f} (resistance)")
    if not long_ and pdl < ltp and a > 0 and (ltp - pdl) <= 1.0 * a:
        notes.append(f"just above the previous day's low {pdl:.2f} (support)")
    if samples:
        notes.append(f"last {samples} comparable session(s): {rate * 100:.0f}% of moves from the "
                     f"{'top' if long_ else 'bottom'} of the range continued")

    verdict = "OK"
    if thresholds is not None and range_vs_adr is not None:
        t = thresholds
        ext = (vwap_ext or 0.0) * (1 if long_ else -1)
        exhausted = range_vs_adr >= t.exhausted_range_adr
        if (side_pos >= t.max_range_pos_long and (exhausted or ext >= t.max_vwap_ext_atr)) or \
                (exhausted and side_pos >= t.exhausted_max_range_pos):
            verdict = "CHASING"
        elif rate is not None and samples >= t.behaviour_min_samples and side_pos >= 0.7 and \
                rate < t.min_continuation_rate:
            verdict = "FADES"
    return {
        "direction": "LONG" if long_ else "SHORT", "as_of": now.isoformat(), "ltp": _r(ltp, 2),
        "day_high": _r(day_high, 2), "day_low": _r(day_low, 2), "day_open": _r(day_open, 2),
        "range_pos": _r(range_pos), "range_vs_adr": _r(range_vs_adr), "adr": _r(adr, 2), "atr": _r(a, 4),
        "vwap": _r(vwap, 2), "vwap_ext_atr": _r(vwap_ext, 3), "pdh": _r(pdh, 2), "pdl": _r(pdl, 2),
        "pdc": _r(pdc, 2), "gap_pct": _r((day_open / pdc - 1) * 100 if pdc else None, 3),
        "continuation_rate": _r(rate), "samples": samples, "notes": notes, "verdict": verdict,
    }


__all__ = ["compute"]
