"""Audit log + error log helpers (rule 0.7: everything is logged)."""

from __future__ import annotations

import traceback as tb
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from core.clock import IST, Clock
from core.logging import log_event, to_jsonable
from storage.models import AuditLog, ErrorLog


def _now(clock: Clock | None) -> datetime:
    return clock.now() if clock is not None else datetime.now(IST)


def audit(
    session: Session,
    actor: str,
    action: str,
    target: str | None = None,
    details: dict[str, Any] | None = None,
    *,
    batch_id: int | None = None,
    clock: Clock | None = None,
) -> AuditLog:
    """Write an audit_log row (caller commits) and mirror it to JSONL."""
    row = AuditLog(
        ts=_now(clock),
        actor=actor,
        action=action,
        target=target,
        batch_id=batch_id,
        details_json=to_jsonable(details or {}),
    )
    session.add(row)
    log_event("audit", {"actor": actor, "action": action, "target": target, "batch_id": batch_id,
                        "details": details or {}}, clock=clock)
    return row


def record_error(
    session: Session | None,
    component: str,
    error: BaseException | str,
    context: dict[str, Any] | None = None,
    *,
    severity: str = "ERROR",
    batch_id: int | None = None,
    clock: Clock | None = None,
) -> None:
    """Log an error to loguru, JSONL and (if a session is given) the ``errors`` table.

    Never raises - error reporting must not break the trading loop.
    """
    message = str(error)
    trace = (
        "".join(tb.format_exception(type(error), error, error.__traceback__))
        if isinstance(error, BaseException)
        else None
    )
    logger.error(f"[{component}] {message}")
    log_event("errors", {"component": component, "severity": severity, "message": message,
                         "context": context or {}, "traceback": trace}, clock=clock)
    if session is None:
        return
    try:
        session.add(
            ErrorLog(
                ts=_now(clock),
                component=component,
                severity=severity,
                message=message[:4000],
                traceback=trace,
                batch_id=batch_id,
                context_json=to_jsonable(context or {}),
            )
        )
        session.flush()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Could not persist error row for {component}: {exc}")
