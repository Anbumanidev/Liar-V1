"""Engine status row (``engine_status`` id=1): written by the engine, read by the dashboard."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from core.enums import EngineState
from core.logging import to_jsonable
from storage.models import EngineStatus


def get_status_row(session: Session) -> EngineStatus:
    row = session.get(EngineStatus, 1)
    if row is None:
        row = EngineStatus(id=1, state=EngineState.STOPPED.value)
        session.add(row)
        session.flush()
    return row


def get_engine_state(session: Session) -> EngineState:
    row = session.get(EngineStatus, 1)
    return EngineState(row.state) if row is not None else EngineState.STOPPED


def set_engine_state(session: Session, state: EngineState, *, now: datetime, activity: str | None = None) -> None:
    row = get_status_row(session)
    row.state = EngineState(state).value
    row.heartbeat_at = now
    if state == EngineState.RUNNING and row.started_at is None:
        row.started_at = now
    if activity is not None:
        row.activity = activity
    session.flush()


def heartbeat(
    session: Session,
    *,
    now: datetime,
    batch_id: int | None,
    settings_version_id: int | None,
    activity: str | None = None,
    next_cycle_at: datetime | None = None,
    market_open: bool | None = None,
    data_feed_ok: bool | None = None,
    llm_ok: bool | None = None,
    regime: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    last_error: str | None = None,
    clear_next_cycle: bool = False,
) -> None:
    """Refresh the heartbeat and the live status fields shown on the dashboard.

    ``clear_next_cycle`` empties ``next_cycle_at`` (no job left today / holiday) instead of
    leaving a stale past time that the dashboard would show as "next cycle due".
    """
    row = get_status_row(session)
    row.heartbeat_at = now
    row.pid = os.getpid()
    row.batch_id = batch_id
    row.settings_version_id = settings_version_id
    if activity is not None:
        row.activity = activity
    if next_cycle_at is not None:
        row.next_cycle_at = next_cycle_at
    elif clear_next_cycle:
        row.next_cycle_at = None
    if market_open is not None:
        row.market_open = market_open
    if data_feed_ok is not None:
        row.data_feed_ok = data_feed_ok
    if llm_ok is not None:
        row.llm_ok = llm_ok
    if regime is not None:
        row.regime_json = to_jsonable(regime)
    if status is not None:
        row.status_json = to_jsonable(status)
    if last_error is not None:
        row.last_error = last_error[:2000]
    session.flush()
