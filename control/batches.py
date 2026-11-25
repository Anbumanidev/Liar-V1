"""Batch lifecycle (section 11).

* One ACTIVE (or PAUSED / HALTED) batch at a time; everything carries ``batch_id``.
* New batch: exits open positions - or, by the owner's choice, lets CNC positions run to
  completion under the old batch in CLOSING state - archives the old batch and starts fresh
  (TRADING = starting capital, RESERVE = 0) with a full settings snapshot.
* History is never deleted by a reset. Deletion (dashboard) is a soft delete.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.loader import settings_to_dict
from config.store import revert_to
from control.payloads import CloseBatchPayload, CommandResult, NewBatchPayload, ResumeHaltPayload
from core.enums import ApprovalStatus, BatchStatus, ExitReason, OrderPurpose, OrderStatus, PositionStatus, Product
from core.logging import log_event
from storage.audit import audit
from storage.batches import get_current_batch
from storage.models import ApprovalItem, Batch, Order, Position

CONFIRM = "CONFIRM"


def create_batch(session: Session, services: Any, *, name: str, starting_capital: float, actor: str) -> Batch:
    """Create an ACTIVE batch with a settings snapshot and an initialised capital account."""
    now = services.clock.now()
    batch = Batch(name=name, starting_capital=float(starting_capital), status=BatchStatus.ACTIVE.value,
                  settings_version_id=services.ctx.settings_version_id, created_at=now, started_at=now,
                  settings_snapshot=settings_to_dict(services.settings))
    session.add(batch)
    session.flush()
    services.capital.initialize_batch(session, batch.id, float(starting_capital))
    audit(session, actor, "BATCH_CREATE", f"batch:{batch.id}", {"name": name, "starting_capital": starting_capital},
          batch_id=batch.id, clock=services.clock)
    log_event("control", {"event": "batch_created", "batch_id": batch.id, "name": name}, clock=services.clock)
    return batch


def ensure_current_batch(session: Session, services: Any, *, actor: str = "system") -> Batch:
    """Return the current batch, creating 'Batch 1' (starting capital from settings) if none exists."""
    batch = get_current_batch(session)
    if batch is not None:
        return batch
    n = len(list(session.execute(select(Batch.id)).scalars())) + 1
    name = f"Batch {n}"
    while session.execute(select(Batch.id).where(Batch.name == name)).first() is not None:
        n += 1
        name = f"Batch {n}"
    return create_batch(session, services, name=name, starting_capital=services.settings.capital.starting_capital,
                        actor=actor)


def _cancel_open_work(session: Session, batch_id: int, reason: str, services: Any) -> tuple[int, int]:
    """Cancel pending entry orders and reject pending approvals of ``batch_id``."""
    orders = session.execute(select(Order).where(Order.batch_id == batch_id, Order.status == OrderStatus.PENDING.value,
                                                 Order.purpose == OrderPurpose.ENTRY.value)).scalars().all()
    for o in orders:
        services.broker.cancel_order(session, o, reason)
    items = session.execute(select(ApprovalItem).where(ApprovalItem.batch_id == batch_id,
                                                       ApprovalItem.status == ApprovalStatus.PENDING.value)).scalars().all()
    for it in items:
        it.status = ApprovalStatus.REJECTED.value
        it.decided_at = services.clock.now()
        it.decided_by = "SYSTEM"
        it.user_reason = reason
    session.flush()
    return len(orders), len(items)


def _market_open(services: Any) -> bool:
    cal = services.calendar
    return bool(cal.is_market_open(services.clock.now())) if cal is not None else True


def start_new_batch(session: Session, payload: NewBatchPayload, services: Any, *, actor: str) -> CommandResult:
    """NEW_BATCH command (typed confirmation required)."""
    current = get_current_batch(session)
    expected = {CONFIRM} | ({current.name} if current is not None else set())
    if payload.confirm_text.strip() not in expected:
        return CommandResult(ok=False, message="confirmation text must be CONFIRM or the current batch name")
    if session.execute(select(Batch.id).where(Batch.name == payload.name)).first() is not None:
        return CommandResult(ok=False, message=f"a batch named {payload.name!r} already exists")
    if payload.copy_settings_from_version_id is not None:
        try:
            revert_to(session, payload.copy_settings_from_version_id, services.caps, author=actor,
                      batch_id=current.id if current else None, clock=services.clock)
            services.reload_settings()
        except Exception as exc:  # noqa: BLE001
            return CommandResult(ok=False, message=f"could not copy settings version: {exc}")
    exits = 0
    if current is not None:
        market_open = _market_open(services)
        run_cnc = payload.open_cnc_handling == "RUN_TO_COMPLETION"
        trades = services.broker.close_positions(session, reason=ExitReason.BATCH_CLOSE, batch_id=current.id,
                                                 include_mis=True, include_cnc=not run_cnc, actor=actor,
                                                 market_open=market_open)
        exits = len(trades)
        _cancel_open_work(session, current.id, "batch closed", services)
        still_open = session.execute(select(Position.id).where(Position.batch_id == current.id,
                                                               Position.status == PositionStatus.OPEN.value)).first()
        current.status = (BatchStatus.CLOSING if (run_cnc and still_open is not None) else BatchStatus.CLOSED).value
        current.ended_at = services.clock.now()
        audit(session, actor, "BATCH_ARCHIVE", f"batch:{current.id}",
              {"status": current.status, "exits": exits, "cnc_handling": payload.open_cnc_handling},
              batch_id=current.id, clock=services.clock)
    batch = create_batch(session, services, name=payload.name, starting_capital=payload.starting_capital, actor=actor)
    services.ctx.batch_id = batch.id
    if services.notifier is not None:
        services.notifier.batch_id = batch.id
    return CommandResult(ok=True, message=f"batch {batch.name} started (old batch: {current.status if current else '-'})",
                         data={"batch_id": batch.id, "exits": exits})


def close_batch(session: Session, payload: CloseBatchPayload, services: Any, *, actor: str) -> CommandResult:
    """CLOSE_BATCH: exit every position and close the batch (no new batch is started)."""
    batch = session.get(Batch, payload.batch_id)
    if batch is None:
        return CommandResult(ok=False, message=f"batch {payload.batch_id} not found")
    if payload.confirm_text.strip() not in (CONFIRM, batch.name):
        return CommandResult(ok=False, message="confirmation text must be CONFIRM or the batch name")
    services.broker.close_positions(session, reason=ExitReason.BATCH_CLOSE, batch_id=batch.id, actor=actor,
                                    market_open=_market_open(services))
    _cancel_open_work(session, batch.id, "batch closed", services)
    open_left = session.execute(select(Position.id).where(Position.batch_id == batch.id,
                                                          Position.status == PositionStatus.OPEN.value)).first()
    batch.status = (BatchStatus.CLOSING if open_left else BatchStatus.CLOSED).value
    batch.ended_at = services.clock.now()
    audit(session, actor, "BATCH_CLOSE", f"batch:{batch.id}", {"status": batch.status}, batch_id=batch.id,
          clock=services.clock)
    return CommandResult(ok=True, message=f"batch {batch.name} {batch.status}", data={"batch_id": batch.id})


def resume_halted(session: Session, payload: ResumeHaltPayload, services: Any, *, actor: str) -> CommandResult:
    """RESUME_HALT: only from the dashboard, with a typed confirmation."""
    if payload.confirm_text.strip() != CONFIRM:
        return CommandResult(ok=False, message="type CONFIRM to resume a HALTED batch")
    batch = get_current_batch(session)
    if batch is None or batch.status != BatchStatus.HALTED.value:
        return CommandResult(ok=False, message="no HALTED batch to resume")
    services.capital.clear_halt(session, batch.id, actor, note="resumed from dashboard")
    batch.status = BatchStatus.ACTIVE.value
    batch.halt_reason = None
    audit(session, actor, "BATCH_RESUME_HALT", f"batch:{batch.id}",
          {"warning": "capital protection floors still apply; a new HALT triggers immediately if breached"},
          batch_id=batch.id, clock=services.clock)
    logger.warning(f"[batches] batch {batch.id} resumed from HALT by {actor}")
    return CommandResult(ok=True, message=f"batch {batch.name} resumed (floors still apply)",
                         data={"batch_id": batch.id})


def finalize_closing_batches(session: Session, services: Any) -> list[int]:
    """CLOSING batches whose last CNC position has exited become CLOSED."""
    done: list[int] = []
    for b in session.execute(select(Batch).where(Batch.status == BatchStatus.CLOSING.value)).scalars():
        left = session.execute(select(Position.id).where(Position.batch_id == b.id,
                                                         Position.status == PositionStatus.OPEN.value,
                                                         Position.product == Product.CNC.value)).first()
        if left is None:
            b.status = BatchStatus.CLOSED.value
            done.append(b.id)
    session.flush()
    return done
