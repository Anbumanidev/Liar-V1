"""Approval queue (sections 3, 14.3.3).

* ``enqueue`` - a risk-approved MANUAL_APPROVE trade waits for the owner (email sent).
* ``approve`` / ``modify_and_approve`` - the risk manager re-vets the (edited) trade against
  the CURRENT state: it keeps the final veto over the owner's approval and edits. If blocked,
  the item stays PENDING with ``block_rule`` set and the exact rule is returned.
* ``reject`` / ``bulk_reject`` / ``expire`` (APPROVAL_TIMEOUT) - followed by shadow tracking.

Deadlines per mode: ``approval_timeout_minutes`` or ``approval_until_next_session`` (e.g.
SWING/POSITIONAL until 09:45 of the next session).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from config.schema import Settings
from control.payloads import CommandResult, ModifyApprovePayload
from core.clock import Clock, ist_datetime, parse_hhmm, to_ist
from core.enums import ApprovalStatus, Direction, EntryType, Mode, RejectionSource, RiskRule, Timeframe
from core.logging import log_event, to_jsonable
from core.types import RiskVerdict, TradeProposal
from modes.base import is_overnight
from risk.state import build_risk_state
from storage.audit import audit
from storage.models import ApprovalItem


class ApprovalQueue:
    """Owner approval workflow over the ``approval_queue`` table."""

    def __init__(
        self,
        settings: Settings,
        clock: Clock,
        calendar: Any,
        risk: Any,
        broker: Any,
        capital: Any,
        shadow: Any,
        *,
        data: Any | None = None,
        notifier: Any | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        self.calendar = calendar
        self.risk = risk
        self.broker = broker
        self.capital = capital
        self.shadow = shadow
        self.data = data
        self.notifier = notifier

    def update_settings(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ deadlines

    def deadline(self, mode: Mode, created: datetime) -> datetime:
        """Approval deadline for an item of ``mode`` created at ``created``."""
        cfg = self.settings.mode(Mode(mode))
        created = to_ist(created)
        if cfg.approval_timeout_minutes:
            return created + timedelta(minutes=cfg.approval_timeout_minutes)
        if cfg.approval_until_next_session:
            t = parse_hhmm(cfg.approval_until_next_session)
            d = created.date()
            is_trading = self.calendar.is_trading_day(d) if self.calendar is not None else d.weekday() < 5
            if is_trading and created.time() < t:
                return ist_datetime(d, t)
            nd = self.calendar.next_trading_day(d) if self.calendar is not None else d + timedelta(days=1)
            return ist_datetime(nd, t)
        return created + timedelta(minutes=10)

    # ------------------------------------------------------------------ enqueue

    def enqueue(
        self,
        session: Session,
        proposal: TradeProposal,
        verdict: RiskVerdict,
        *,
        decision_id: int | None = None,
        summary: dict[str, Any] | None = None,
    ) -> ApprovalItem:
        """Add a risk-approved trade to the owner's queue and email the owner."""
        now = to_ist(self.clock.now())
        item = ApprovalItem(
            batch_id=proposal.batch_id, decision_id=decision_id, symbol=proposal.symbol, mode=proposal.mode.value,
            direction=proposal.direction.value, entry_type=proposal.entry_type.value,
            entry_price=proposal.entry_price, stop_loss=proposal.stop_loss, target_1=proposal.target_1,
            target_2=proposal.target_2, qty=verdict.qty, max_qty=verdict.max_qty, confidence=proposal.confidence,
            grade=proposal.grade.value if proposal.grade else None, strategy=proposal.strategy,
            proposal_json=proposal.model_dump(mode="json"), risk_json=verdict.model_dump(mode="json"),
            summary_json=to_jsonable(summary or {}), status=ApprovalStatus.PENDING.value, created_at=now,
            expires_at=self.deadline(proposal.mode, now),
        )
        session.add(item)
        session.flush()
        log_event("approvals", {"event": "queued", "id": item.id, "symbol": item.symbol, "mode": item.mode,
                                "qty": item.qty, "expires_at": item.expires_at}, clock=self.clock)
        if self.notifier is not None:
            try:
                self.notifier.pending_approval(item)
            except Exception as exc:  # noqa: BLE001 - email never blocks trading
                logger.error(f"[approvals] pending-approval email failed: {exc}")
        return item

    def pending(self, session: Session, batch_id: int | None = None) -> list[ApprovalItem]:
        stmt = select(ApprovalItem).where(ApprovalItem.status == ApprovalStatus.PENDING.value)
        if batch_id is not None:
            stmt = stmt.where(ApprovalItem.batch_id == batch_id)
        return list(session.execute(stmt.order_by(ApprovalItem.expires_at)).scalars())

    # ------------------------------------------------------------------ decisions

    def _get_pending(self, session: Session, item_id: int) -> ApprovalItem | CommandResult:
        item = session.get(ApprovalItem, item_id)
        if item is None:
            return CommandResult(ok=False, message=f"approval item {item_id} not found")
        if item.status != ApprovalStatus.PENDING.value:
            return CommandResult(ok=False, message=f"approval item {item_id} is {item.status}, not PENDING")
        if to_ist(self.clock.now()) >= to_ist(item.expires_at):
            self._timeout(session, item)
            return CommandResult(ok=False, message=f"approval item {item_id} expired (APPROVAL_TIMEOUT)")
        return item

    def _state(self, session: Session, item: ApprovalItem) -> Any:
        return build_risk_state(session, batch_id=item.batch_id, capital_manager=self.capital, clock=self.clock,
                                calendar=self.calendar, data=self.data, symbol=item.symbol,
                                correlation_lookback=self.settings.risk.correlation_lookback_days)

    def _check_timing(self, mode: Mode) -> bool:
        # CNC orders only fill inside their entry window; approval itself may happen any time.
        return not is_overnight(mode) or Mode(mode) == Mode.BTST

    def _execute(self, session: Session, item: ApprovalItem, proposal: TradeProposal, *, actor: str,
                 status: ApprovalStatus, edits: dict[str, Any] | None, reason: str | None) -> CommandResult:
        state = self._state(session, item)
        if self.data is not None and self._check_timing(proposal.mode):
            # Re-price at the live quote: the stored one is as old as the analysis + the wait.
            try:
                q = self.data.quote(proposal.symbol)
                if q is not None and q.ltp and q.ltp > 0:
                    proposal = proposal.model_copy(update={"ltp": float(q.ltp), "ltp_at": q.timestamp})
            except Exception as exc:  # noqa: BLE001 - keep the stored quote
                logger.warning(f"[approvals] live quote {proposal.symbol}: {exc}")
            # Price behaviour again at the approval price (the stock may have run to the top meanwhile).
            from agents.graph import behaviour_fields, price_behaviour_for

            beh = price_behaviour_for(self.data, proposal.symbol, proposal.mode, proposal.direction, self.clock.now(),
                                      live_price=proposal.ltp or proposal.entry_price, settings=self.settings)
            if beh is not None:
                proposal = proposal.model_copy(update=behaviour_fields(beh))
        verdict = self.risk.evaluate(proposal, state, check_timing=self._check_timing(proposal.mode))
        now = to_ist(self.clock.now())
        if not verdict.approved:
            item.block_rule = verdict.rule.value if verdict.rule else None
            item.user_edits_json = to_jsonable(edits) if edits else item.user_edits_json
            session.flush()
            audit(session, actor, "APPROVAL_BLOCKED", f"approval:{item.id}",
                  {"rule": item.block_rule, "message": verdict.message, "edits": edits}, batch_id=item.batch_id,
                  clock=self.clock)
            return CommandResult(ok=False, message=verdict.message, rule=verdict.rule,
                                 data={"approval_id": item.id, "max_qty": verdict.max_qty})
        order = self.broker.submit_entry(session, proposal, verdict, decision_id=item.decision_id, approval_id=item.id)
        item.status = status.value
        item.decided_at = now
        item.decided_by = actor
        item.order_id = order.id
        item.final_mode = proposal.mode.value
        item.block_rule = None
        item.user_reason = reason
        if edits:
            item.user_edits_json = to_jsonable(edits)
        session.flush()
        audit(session, actor, f"APPROVAL_{status.value}", f"approval:{item.id}",
              {"order_id": order.id, "qty": verdict.qty, "edits": edits}, batch_id=item.batch_id, clock=self.clock)
        log_event("approvals", {"event": status.value.lower(), "id": item.id, "order_id": order.id,
                                "qty": verdict.qty, "edits": edits}, clock=self.clock)
        return CommandResult(ok=True, message=f"{status.value}: order {order.id} for {verdict.qty} {item.symbol}",
                             data={"approval_id": item.id, "order_id": order.id, "qty": verdict.qty})

    def approve(self, session: Session, item_id: int, *, actor: str) -> CommandResult:
        """Owner approval: re-vet with the risk manager, then submit the paper order."""
        got = self._get_pending(session, item_id)
        if isinstance(got, CommandResult):
            return got
        proposal = TradeProposal.model_validate(got.proposal_json).model_copy(
            update={"requested_qty": got.qty, "approval_id": got.id, "created_at": to_ist(self.clock.now())})
        return self._execute(session, got, proposal, actor=actor, status=ApprovalStatus.APPROVED, edits=None,
                             reason=None)

    def modify_and_approve(self, session: Session, payload: ModifyApprovePayload, *, actor: str) -> CommandResult:
        """Owner edits (mode, entry, SL, targets, qty <= risk max) then approval - all re-vetted."""
        got = self._get_pending(session, payload.approval_id)
        if isinstance(got, CommandResult):
            return got
        base = TradeProposal.model_validate(got.proposal_json)
        edits: dict[str, Any] = {}
        upd: dict[str, Any] = {"approval_id": got.id, "created_at": to_ist(self.clock.now())}
        for field, attr in (("mode", "mode"), ("entry_type", "entry_type"), ("entry_price", "entry_price"),
                            ("stop_loss", "stop_loss"), ("target_1", "target_1"), ("target_2", "target_2")):
            val = getattr(payload, field)
            if val is not None and val != getattr(base, attr):
                edits[field] = {"old": getattr(base, attr), "new": val}
                upd[attr] = val
        qty = payload.qty if payload.qty is not None else got.qty
        if payload.qty is not None and payload.qty != got.qty:
            edits["qty"] = {"old": got.qty, "new": payload.qty}
        upd["requested_qty"] = qty
        new_mode = Mode(upd.get("mode", base.mode))
        if new_mode != base.mode:
            upd["max_holding_days"] = self.settings.mode(new_mode).max_holding_days or None
            atr = self._atr_for(base.symbol, new_mode)
            if atr is not None:
                upd["atr"] = atr
        proposal = base.model_copy(update=upd)
        status = ApprovalStatus.MODIFIED if edits else ApprovalStatus.APPROVED
        return self._execute(session, got, proposal, actor=actor, status=status, edits=edits or None,
                             reason=payload.reason)

    def _atr_for(self, symbol: str, mode: Mode) -> float | None:
        """ATR(14) of ``mode``'s primary timeframe (SL distance bounds after a mode change)."""
        if self.data is None:
            return None
        try:
            from features.indicators import atr as atr_fn

            tf = Timeframe(self.settings.mode(mode).primary_timeframe)
            df = self.data.candles(symbol, tf, lookback_bars=60)
            if df is None or len(df) < 15:
                return None
            val = float(atr_fn(df, 14).iloc[-1])
            return val if val == val and val > 0 else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[approvals] ATR for {symbol} {mode.value} failed: {exc}")
            return None

    def reject(self, session: Session, item_id: int, *, actor: str, reason: str | None = None) -> CommandResult:
        item = session.get(ApprovalItem, item_id)
        if item is None or item.status != ApprovalStatus.PENDING.value:
            return CommandResult(ok=False, message=f"approval item {item_id} is not pending")
        item.status = ApprovalStatus.REJECTED.value
        item.decided_at = to_ist(self.clock.now())
        item.decided_by = actor
        item.user_reason = reason
        session.flush()
        self._shadow(session, item, RejectionSource.USER, reason)
        audit(session, actor, "APPROVAL_REJECTED", f"approval:{item.id}", {"reason": reason}, batch_id=item.batch_id,
              clock=self.clock)
        log_event("approvals", {"event": "rejected", "id": item.id, "reason": reason}, clock=self.clock)
        return CommandResult(ok=True, message=f"rejected {item.symbol}", data={"approval_id": item.id})

    def bulk_reject(self, session: Session, item_ids: list[int], *, actor: str, reason: str | None = None) -> CommandResult:
        done, failed = [], []
        for i in item_ids:
            res = self.reject(session, i, actor=actor, reason=reason)
            (done if res.ok else failed).append(i)
        return CommandResult(ok=not failed, message=f"rejected {len(done)}, failed {len(failed)}",
                             data={"rejected": done, "failed": failed})

    def _timeout(self, session: Session, item: ApprovalItem) -> None:
        item.status = ApprovalStatus.TIMEOUT.value
        item.decided_at = to_ist(self.clock.now())
        item.decided_by = "SYSTEM"
        item.user_reason = "APPROVAL_TIMEOUT"
        session.flush()
        self._shadow(session, item, RejectionSource.TIMEOUT, "APPROVAL_TIMEOUT")
        log_event("approvals", {"event": "timeout", "id": item.id, "symbol": item.symbol}, clock=self.clock)

    def expire(self, session: Session, now: datetime | None = None) -> list[ApprovalItem]:
        """Auto-reject every PENDING item past its deadline as APPROVAL_TIMEOUT."""
        now = to_ist(now or self.clock.now())
        rows = session.execute(
            select(ApprovalItem).where(ApprovalItem.status == ApprovalStatus.PENDING.value,
                                       ApprovalItem.expires_at <= now)
        ).scalars().all()
        for item in rows:
            self._timeout(session, item)
        return list(rows)

    def _shadow(self, session: Session, item: ApprovalItem, source: RejectionSource, reason: str | None) -> None:
        if self.shadow is None:
            return
        try:
            self.shadow.create(
                session, batch_id=item.batch_id, symbol=item.symbol, mode=Mode(item.mode),
                direction=Direction(item.direction), entry=item.entry_price, stop_loss=item.stop_loss,
                target_1=item.target_1, target_2=item.target_2, qty=item.qty, source=source,
                rule=item.block_rule,
                reason=reason, decision_id=item.decision_id, approval_id=item.id, strategy=item.strategy,
                grade=item.grade, confidence=item.confidence, entry_type=EntryType(item.entry_type),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[approvals] shadow for item {item.id} failed: {exc}")
