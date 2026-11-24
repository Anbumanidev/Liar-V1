"""Read-only views of open paper positions (shared by risk, engine, monitor, reports)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.enums import Direction, Mode, PositionStatus, Product
from core.types import OpenPositionView
from storage.models import Position


def position_to_view(pos: Position, ltp: float | None = None) -> OpenPositionView:
    """Convert an ORM ``Position`` into an ``OpenPositionView``.

    Args:
        pos: the position row.
        ltp: live price override; defaults to ``pos.last_price``.
    """
    return OpenPositionView(
        position_id=pos.id,
        symbol=pos.symbol,
        mode=Mode(pos.mode),
        product=Product(pos.product),
        direction=Direction(pos.direction),
        qty=pos.qty,
        avg_entry=pos.avg_entry,
        stop_loss=pos.stop_loss,
        target_1=pos.target_1,
        target_2=pos.target_2,
        ltp=ltp if ltp is not None else pos.last_price,
        margin_blocked=pos.margin_blocked,
        entry_charges=pos.entry_charges,
        sector=pos.sector,
        opened_at=pos.opened_at,
        overnight=bool(pos.overnight),
    )


def open_positions(session: Session, batch_id: int | None = None) -> list[Position]:
    """OPEN positions (optionally of one batch), oldest first."""
    stmt = select(Position).where(Position.status == PositionStatus.OPEN.value)
    if batch_id is not None:
        stmt = stmt.where(Position.batch_id == batch_id)
    return list(session.execute(stmt.order_by(Position.id)).scalars())


def open_position_views(session: Session, batch_id: int, ltps: dict[str, float] | None = None) -> list[OpenPositionView]:
    """``OpenPositionView`` list for a batch, with optional live prices by symbol."""
    ltps = ltps or {}
    return [position_to_view(p, ltps.get(p.symbol)) for p in open_positions(session, batch_id)]
