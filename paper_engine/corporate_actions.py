"""Corporate actions on held CNC positions (section 9).

* SPLIT / BONUS on the ex-date: quantity x ratio (fractional entitlements dropped), every price
  on the position (entry, stops, targets, extremes) / ratio. Cost basis (margin) and accrued
  charges are unchanged, so P&L and the capital ledger are unaffected.
* EX_DIVIDEND: ``qty x amount`` credited to TRADING cash on the ex-date via
  ``CapitalManager.credit_dividend`` (counted as realized P&L in the ledger invariant).
* Idempotent: one adjustment per (position, action, ex-date).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import Clock
from core.enums import Direction, EventType, PositionStatus, Product
from core.logging import log_event
from core.types import MarketEvent
from storage.models import CorporateActionAdjustment, Position

SPLIT_TYPES = {EventType.SPLIT, EventType.BONUS}


def _already_applied(session: Session, position_id: int, action: str, ex_date: date) -> bool:
    return session.execute(
        select(CorporateActionAdjustment.id).where(
            CorporateActionAdjustment.position_id == position_id,
            CorporateActionAdjustment.action_type == action,
            CorporateActionAdjustment.ex_date == ex_date,
        )
    ).first() is not None


def apply_corporate_actions(
    session: Session,
    events: list[MarketEvent],
    day: date,
    capital: Any,
    clock: Clock,
) -> list[CorporateActionAdjustment]:
    """Apply ``events`` whose ex-date is ``day`` to open CNC positions. Returns new adjustments."""
    out: list[CorporateActionAdjustment] = []
    for ev in events:
        if ev.symbol is None or ev.event_date != day:
            continue
        etype = EventType(ev.event_type)
        if etype not in SPLIT_TYPES and etype != EventType.EX_DIVIDEND:
            continue
        positions = session.execute(
            select(Position).where(Position.symbol == ev.symbol.upper(), Position.status == PositionStatus.OPEN.value,
                                   Position.product == Product.CNC.value)
        ).scalars().all()
        for pos in positions:
            if _already_applied(session, pos.id, etype.value, day):
                continue
            row = _apply_one(session, pos, ev, etype, day, capital, clock)
            if row is not None:
                out.append(row)
    session.flush()
    return out


def _apply_one(session: Session, pos: Position, ev: MarketEvent, etype: EventType, day: date, capital: Any,
               clock: Clock) -> CorporateActionAdjustment | None:
    qty_before, price_before, sl_before = pos.qty, pos.avg_entry, pos.stop_loss
    cash_credit = 0.0
    if etype in SPLIT_TYPES:
        ratio = ev.ratio
        if ratio is None or ratio <= 0 or math.isclose(ratio, 1.0):
            logger.warning(f"[corp-action] {etype.value} {ev.symbol} {day}: missing/invalid ratio {ratio}; skipped")
            return None
        pos.qty = int(math.floor(pos.qty * ratio + 1e-9))
        pos.initial_qty = int(math.floor(pos.initial_qty * ratio + 1e-9))
        for attr in ("avg_entry", "stop_loss", "initial_stop_loss", "target_1", "target_2",
                     "highest_price", "lowest_price", "last_price"):
            val = getattr(pos, attr)
            if val is not None:
                setattr(pos, attr, round(val / ratio, 4))
        hist = list(pos.sl_history_json or [])
        hist.append({"ts": clock.now().isoformat(), "old": sl_before, "new": pos.stop_loss, "by": f"CORP_{etype.value}"})
        pos.sl_history_json = hist
    else:
        amount = ev.amount
        if amount is None or amount <= 0:
            logger.warning(f"[corp-action] dividend {ev.symbol} {day}: missing amount; skipped")
            return None
        if pos.direction != Direction.LONG.value:
            return None
        cash_credit = round(pos.qty * amount, 4)
        capital.credit_dividend(session, pos.batch_id, cash_credit, position_id=pos.id, symbol=pos.symbol,
                                mode=pos.mode, note=f"dividend Rs {amount}/share x {pos.qty}")
    row = CorporateActionAdjustment(
        batch_id=pos.batch_id, position_id=pos.id, symbol=pos.symbol, action_type=etype.value, ex_date=day,
        ratio=ev.ratio, amount=ev.amount, qty_before=qty_before, qty_after=pos.qty, price_before=price_before,
        price_after=pos.avg_entry, sl_before=sl_before, sl_after=pos.stop_loss, cash_credit=cash_credit,
        created_at=clock.now(),
    )
    session.add(row)
    session.flush()
    log_event("capital", {"event": "corporate_action", "symbol": pos.symbol, "type": etype.value, "ex_date": day,
                          "qty": [qty_before, pos.qty], "price": [price_before, pos.avg_entry],
                          "cash_credit": cash_credit}, clock=clock)
    return row
