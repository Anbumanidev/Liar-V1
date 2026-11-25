"""Route a vetted decision by the mode's execution policy (section 3).

The policy is ``settings.modes.effective_policy`` - the global autonomy switch (FULL_AUTO /
HYBRID / HUMAN_IN_THE_LOOP / SIGNALS_ONLY) overrides the per-mode policy unless PER_MODE.

* risk veto            -> ``RISK_REJECTED`` + shadow(RISK)
* ``SIGNALS_ONLY``     -> ``SIGNAL_ONLY`` (logged + emailed, never executed) + shadow
* ``AUTO``             -> paper order submitted -> ``EXECUTED``
* ``MANUAL_APPROVE``   -> approval queue -> ``QUEUED``
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from config.schema import Settings
from core.enums import ExecutionPolicy, RejectionSource
from core.logging import log_event
from core.types import RiskVerdict, TradeProposal
from storage.models import DecisionRow

FINAL_RISK_REJECTED = "RISK_REJECTED"
FINAL_SIGNAL_ONLY = "SIGNAL_ONLY"
FINAL_EXECUTED = "EXECUTED"
FINAL_QUEUED = "QUEUED"
FINAL_NO_TRADE = "NO_TRADE"
FINAL_FILTERED = "FILTERED"
FINAL_ANALYZED = "ANALYZED"   # Analyze-now without routing


def route_decision(
    session: Session,
    *,
    settings: Settings,
    decision: DecisionRow,
    proposal: TradeProposal,
    verdict: RiskVerdict,
    broker: Any,
    queue: Any,
    shadow: Any,
    notifier: Any | None = None,
    summary: dict[str, Any] | None = None,
    clock: Any | None = None,
) -> str:
    """Apply the execution policy; updates ``decision.final_action`` and returns it."""
    policy = settings.modes.effective_policy(proposal.mode, grade=proposal.grade, confidence=proposal.confidence)
    final: str
    if not verdict.approved:
        final = FINAL_RISK_REJECTED
        _shadow(shadow, session, decision, proposal, verdict, RejectionSource.RISK,
                verdict.rule.value if verdict.rule else None, verdict.message)
    elif policy == ExecutionPolicy.SIGNALS_ONLY:
        final = FINAL_SIGNAL_ONLY
        _shadow(shadow, session, decision, proposal, verdict, RejectionSource.SIGNALS_ONLY, None, "SIGNALS_ONLY policy")
        if notifier is not None:
            try:
                notifier.swing_scan([{**(summary or {}), "symbol": proposal.symbol, "mode": proposal.mode.value,
                                      "direction": proposal.direction.value, "entry": proposal.entry_price,
                                      "stop_loss": proposal.stop_loss, "target_1": proposal.target_1,
                                      "target_2": proposal.target_2, "qty": verdict.qty,
                                      "confidence": proposal.confidence,
                                      "grade": proposal.grade.value if proposal.grade else None,
                                      "note": "SIGNALS_ONLY - not executed"}],
                                    batch_id=proposal.batch_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[router] signal email failed: {exc}")
    elif policy == ExecutionPolicy.AUTO:
        order = broker.submit_entry(session, proposal, verdict, decision_id=decision.id)
        final = FINAL_EXECUTED
        log_event("decisions", {"event": "auto_executed", "decision_id": decision.id, "order_id": order.id},
                  clock=clock)
    else:
        item = queue.enqueue(session, proposal, verdict, decision_id=decision.id, summary=summary)
        final = FINAL_QUEUED
        log_event("decisions", {"event": "queued", "decision_id": decision.id, "approval_id": item.id}, clock=clock)
    decision.final_action = final
    ctx = dict(decision.context_json or {})
    ctx["execution_policy"] = policy.value
    ctx["autonomy"] = settings.modes.autonomy.value
    decision.context_json = ctx
    session.flush()
    return final


def _shadow(shadow: Any, session: Session, decision: DecisionRow, p: TradeProposal, v: RiskVerdict,
            source: RejectionSource, rule: str | None, reason: str | None) -> None:
    if shadow is None:
        return
    try:
        shadow.create(session, batch_id=p.batch_id, symbol=p.symbol, mode=p.mode, direction=p.direction,
                      entry=p.entry_price, stop_loss=p.stop_loss, target_1=p.target_1, target_2=p.target_2,
                      qty=v.qty or v.max_qty, source=source, rule=rule, reason=reason, decision_id=decision.id,
                      strategy=p.strategy, grade=p.grade.value if p.grade else None, confidence=p.confidence,
                      entry_type=p.entry_type, max_holding_days=p.max_holding_days)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[router] shadow create failed: {exc}")
