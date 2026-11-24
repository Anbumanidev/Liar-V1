"""Dashboard-side command queue API (section 14.2).

The dashboard ONLY writes rows here; the engine process polls ``control_commands`` every
2-5 s, executes, and writes status/results (see ``control.processor``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from control.payloads import CommandResult, validate_payload
from core.clock import IST, Clock
from core.enums import CommandStatus, CommandType
from core.logging import log_event, to_jsonable
from storage.audit import audit
from storage.models import ControlCommand


def enqueue_command(
    session: Session,
    command_type: CommandType,
    payload: dict[str, Any] | None,
    *,
    actor: str,
    batch_id: int | None = None,
    clock: Clock | None = None,
) -> ControlCommand:
    """Validate and enqueue a command for the engine. Caller commits.

    Raises:
        pydantic.ValidationError: if the payload is invalid for the command type.
    """
    model = validate_payload(command_type, payload or {})
    now = clock.now() if clock is not None else datetime.now(IST)
    row = ControlCommand(
        created_at=now,
        command_type=CommandType(command_type).value,
        payload_json=to_jsonable(model.model_dump(mode="json")),
        status=CommandStatus.PENDING.value,
        requested_by=actor,
        batch_id=batch_id,
    )
    session.add(row)
    session.flush()
    audit(session, actor, f"COMMAND_{row.command_type}", f"command:{row.id}", row.payload_json,
          batch_id=batch_id, clock=clock)
    log_event("control", {"event": "enqueued", "id": row.id, "type": row.command_type,
                          "payload": row.payload_json, "actor": actor}, clock=clock)
    return row


def get_command(session: Session, command_id: int) -> ControlCommand | None:
    return session.get(ControlCommand, command_id)


def command_result(row: ControlCommand) -> CommandResult | None:
    """Parsed result of a finished command (None while pending/running)."""
    if row.result_json is None:
        return None
    return CommandResult.model_validate(row.result_json)


def recent_commands(session: Session, limit: int = 50) -> list[ControlCommand]:
    return list(session.execute(select(ControlCommand).order_by(ControlCommand.id.desc()).limit(limit)).scalars())


def pending_commands(session: Session, limit: int = 20) -> list[ControlCommand]:
    """Oldest-first pending commands (engine side)."""
    return list(
        session.execute(
            select(ControlCommand)
            .where(ControlCommand.status == CommandStatus.PENDING.value)
            .order_by(ControlCommand.id.asc())
            .limit(limit)
        ).scalars()
    )
