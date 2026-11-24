"""Deterministic stop-loss candidates (input for the Stop-Loss Agent; code validates its pick).

Candidates (each only if STRICTLY on the protective side of ``entry`` - below for LONG,
above for SHORT - and positive):

* ``ATR``: ``entry -/+ k * ATR`` for k in ``ATR_MULTIPLES`` (ATR = primary timeframe ATR(14)).
* ``SWING``: the most recent confirmed swing low (LONG) / swing high (SHORT) of the primary
  timeframe that lies beyond entry.
* ``VWAP``: the primary (intraday) timeframe's session VWAP, if on the protective side.
* ``OPENING_RANGE``: opening-range low (LONG) / high (SHORT), intraday only.
* ``DAILY_STRUCTURE``: prior day low (LONG) / high (SHORT).

Prices are rounded to the exchange tick AWAY from entry (LONG -> floor, SHORT -> ceil) so
rounding never moves a stop inside its computed level. Output is sorted by distance
(tightest first); exact duplicates (same method and price) are dropped.
"""

from __future__ import annotations

import math

from core.enums import Direction, SLMethod
from core.types import SLCandidate, StockFeatures

ATR_MULTIPLES: tuple[float, ...] = (1.0, 1.5, 2.0)
TICK_SIZE = 0.05


def _round_protective(price: float, direction: Direction, tick: float) -> float:
    if tick <= 0:
        return price
    steps = price / tick
    # small epsilon so prices already on the tick grid are not pushed one tick further
    steps = math.floor(steps + 1e-9) if direction == Direction.LONG else math.ceil(steps - 1e-9)
    return round(steps * tick, 10)


def stop_loss_candidates(
    features: StockFeatures,
    direction: Direction,
    entry: float,
    *,
    atr_multiples: tuple[float, ...] = ATR_MULTIPLES,
    tick: float = TICK_SIZE,
) -> list[SLCandidate]:
    """All valid stop-loss candidates for a trade at ``entry`` (see module docstring)."""
    if entry is None or not math.isfinite(entry) or entry <= 0:
        return []
    long = direction == Direction.LONG
    atr = features.atr if features.atr is not None and features.atr > 0 else None
    raw: list[tuple[SLMethod, float, str]] = []

    if atr is not None:
        for k in atr_multiples:
            price = entry - k * atr if long else entry + k * atr
            raw.append((SLMethod.ATR, price, f"{k:g}x ATR({atr:.4g})"))

    prim = features.tf(features.primary_timeframe)
    if prim is not None and prim.levels is not None:
        swings = prim.levels.swing_lows if long else prim.levels.swing_highs
        for level in reversed(swings):  # most recent first
            if (long and level < entry) or (not long and level > entry):
                raw.append((SLMethod.SWING, level, f"latest swing {'low' if long else 'high'} ({prim.timeframe})"))
                break

    if prim is not None:
        vwap = prim.v("vwap")
        if vwap is not None:
            raw.append((SLMethod.VWAP, vwap, f"session VWAP ({prim.timeframe})"))

    or_level = features.opening_range_low if long else features.opening_range_high
    if or_level is not None:
        raw.append((SLMethod.OPENING_RANGE, or_level, f"opening range {'low' if long else 'high'}"))

    pd_key = "prev_day_low" if long else "prev_day_high"
    pd_level = features.extra.get(pd_key)
    if pd_level is None:
        # fall back to the prior-session levels of an intraday timeframe
        for tff in features.timeframes.values():
            if tff.levels is not None and tff.timeframe in ("1m", "5m", "15m", "1h"):
                pd_level = tff.levels.prev_low if long else tff.levels.prev_high
                if pd_level is not None:
                    break
    if pd_level is not None:
        raw.append((SLMethod.DAILY_STRUCTURE, float(pd_level), f"prior day {'low' if long else 'high'}"))

    out: list[SLCandidate] = []
    seen: set[tuple[SLMethod, float]] = set()
    for method, price, note in raw:
        if price is None or not math.isfinite(price):
            continue
        p = _round_protective(float(price), direction, tick)
        if p <= 0:
            continue
        if long and not p < entry:
            continue
        if not long and not p > entry:
            continue
        key = (method, p)
        if key in seen:
            continue
        seen.add(key)
        dist = abs(entry - p)
        out.append(SLCandidate(
            method=method,
            price=p,
            distance=round(dist, 6),
            distance_atr=round(dist / atr, 6) if atr else None,
            note=note,
        ))
    out.sort(key=lambda c: (c.distance, c.method.value))
    return out
