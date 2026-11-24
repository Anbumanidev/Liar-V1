"""Batch queries. Batch lifecycle orchestration (exit positions, archive, new capital
account) lives in ``control.batches``; this module only reads/writes the ``batches`` table."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.enums import BatchStatus
from storage.models import Batch

#: Statuses in which a batch is "the current batch" (one at a time).
CURRENT_STATUSES = (BatchStatus.ACTIVE.value, BatchStatus.PAUSED.value, BatchStatus.HALTED.value)


def get_current_batch(session: Session) -> Batch | None:
    """The single ACTIVE / PAUSED / HALTED batch, if any."""
    return session.execute(
        select(Batch)
        .where(Batch.status.in_(CURRENT_STATUSES), Batch.deleted.is_(False))
        .order_by(Batch.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def get_batch(session: Session, batch_id: int) -> Batch | None:
    return session.get(Batch, batch_id)


def list_batches(session: Session, include_deleted: bool = False) -> list[Batch]:
    stmt = select(Batch).order_by(Batch.id.desc())
    if not include_deleted:
        stmt = stmt.where(Batch.deleted.is_(False))
    return list(session.execute(stmt).scalars())


def closing_batches(session: Session) -> list[Batch]:
    """Old batches whose CNC positions are allowed to run to completion."""
    return list(
        session.execute(select(Batch).where(Batch.status == BatchStatus.CLOSING.value)).scalars()
    )
