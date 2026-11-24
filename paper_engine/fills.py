"""Realistic paper fill rules (section 9).

* MARKET orders fill at the NEXT tick / candle open after the order time, plus slippage
  (default 0.05 %, always against us).
* LIMIT orders fill only if price trades THROUGH the limit (strictly beyond it); if the bar
  opens beyond the limit in our favour we get the open.
* Stops: if a bar OPENS beyond the stop (gap) the fill is the open price, not the stop
  (overnight gap-downs fill at the next-day open). Otherwise a touched stop fills at the stop
  price minus slippage (stop-market).
* Targets are resting limit orders: filled at the target (or the better open on a gap).
* All prices are rounded to the NSE tick (Rs 0.05) in the adverse direction for slippage
  fills and to the nearest tick for limit fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.enums import Direction, OrderSide

TICK: float = 0.05


def round_tick(price: float, tick: float = TICK, *, mode: str = "nearest") -> float:
    """Round ``price`` to the exchange tick. ``mode``: nearest | up | down."""
    if price <= 0:
        return round(max(price, 0.0), 2)
    units = price / tick
    if mode == "up":
        units = math.ceil(units - 1e-9)
    elif mode == "down":
        units = math.floor(units + 1e-9)
    else:
        units = math.floor(units + 0.5)
    return round(units * tick, 2)


def slipped_price(side: OrderSide, ref_price: float, slippage_pct: float) -> float:
    """Market fill price: BUY pays up, SELL receives less (rounded adversely to the tick)."""
    s = max(0.0, slippage_pct) / 100.0
    if OrderSide(side) == OrderSide.BUY:
        return round_tick(ref_price * (1 + s), mode="up")
    return max(TICK, round_tick(ref_price * (1 - s), mode="down"))


@dataclass(frozen=True)
class Bar:
    """One OHLC bar (open time ``ts`` is informational)."""

    open: float
    high: float
    low: float
    close: float


def limit_fill_price(side: OrderSide, limit: float, bar: Bar) -> float | None:
    """Fill price of a LIMIT order on ``bar`` or None.

    BUY fills only if ``low < limit`` (traded through); price = min(limit, open).
    SELL fills only if ``high > limit``; price = max(limit, open).
    """
    if OrderSide(side) == OrderSide.BUY:
        if bar.low < limit:
            return round_tick(min(limit, bar.open))
        return None
    if bar.high > limit:
        return round_tick(max(limit, bar.open))
    return None


def exit_side_for(direction: Direction) -> OrderSide:
    return OrderSide.SELL if Direction(direction) == Direction.LONG else OrderSide.BUY


def stop_fill(direction: Direction, stop: float, bar: Bar, slippage_pct: float) -> tuple[float, bool] | None:
    """Stop-loss fill on ``bar``: ``(price, gapped)`` or None if not triggered.

    ``gapped`` is True when the bar opened beyond the stop (fill at the open, no slippage).
    """
    if Direction(direction) == Direction.LONG:
        if bar.open <= stop:
            return round_tick(bar.open, mode="down"), True
        if bar.low <= stop:
            return slipped_price(OrderSide.SELL, stop, slippage_pct), False
        return None
    if bar.open >= stop:
        return round_tick(bar.open, mode="up"), True
    if bar.high >= stop:
        return slipped_price(OrderSide.BUY, stop, slippage_pct), False
    return None


def target_fill(direction: Direction, target: float, bar: Bar) -> float | None:
    """Target (resting limit) fill: at the target, or at the open if it gapped beyond."""
    if Direction(direction) == Direction.LONG:
        if bar.open >= target:
            return round_tick(bar.open)
        if bar.high >= target:
            return round_tick(target)
        return None
    if bar.open <= target:
        return round_tick(bar.open)
    if bar.low <= target:
        return round_tick(target)
    return None


def favorable_excursion(direction: Direction, bar: Bar) -> float:
    """Best price reached in the bar for ``direction`` (high for long, low for short)."""
    return bar.high if Direction(direction) == Direction.LONG else bar.low
