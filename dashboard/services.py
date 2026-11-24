"""DB reads/writes behind the dashboard control pages (section 14.3).

Everything here is a plain function taking a SQLAlchemy ``Session`` (or paths/values) so it
can be unit tested without Streamlit. Rules (section 14.2):

* Never import ``agents/``, ``paper_engine/``, ``risk/``, ``scheduler/`` or engine code.
* Actions on the engine go through ``control.commands.enqueue_command`` (validated payloads).
* Settings edits go through ``config.store`` (versioned, cap-validated, audited) with numeric
  inputs clamped by ``config.loader.clamp_value``; every new version enqueues
  ``RELOAD_SETTINGS``.
* Every other write is audit-logged with ``storage.audit.audit``.
* All arithmetic shown in the UI (countdowns, unrealized P&L estimates) is deterministic
  Python - no LLM ever computes a number.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config.loader import clamp_value, get_cap
from config.schema import HardCaps, Settings
from config.store import apply_changes, get_active_settings, list_versions, revert_to
from control.commands import command_result, enqueue_command
from control.payloads import CommandResult
from core.clock import Clock
from core.enums import (
    ApprovalStatus,
    BatchStatus,
    CommandStatus,
    CommandType,
    Direction,
    EngineState,
    Mode,
    PositionStatus,
)
from core.errors import HardCapViolationError
from core.logging import read_events
from modes.base import direction_allowed
from storage.audit import audit, record_error
from storage.batches import CURRENT_STATUSES
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Base,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    ControlCommand,
    DebateRow,
    DecisionRow,
    EngineStatus,
    ModeState,
    NotificationLog,
    Order,
    Position,
    PromptTemplate,
    SettingsVersion,
    StrategyState,
    Trade,
)

#: Typed confirmation for destructive actions.
CONFIRM_WORD = "CONFIRM"

#: JSONL categories offered by the log viewer (see core.logging).
LOG_CATEGORIES: tuple[str, ...] = (
    "agents", "debates", "decisions", "risk", "orders", "trades", "capital", "approvals", "control",
    "settings", "errors", "scans", "notifications", "shadow", "replay", "audit",
)

#: Tables included in a batch export (sheet/file name -> model).
EXPORT_TABLES: dict[str, type[Base]] = {
    "trades": Trade,
    "positions": Position,
    "ledger": CapitalLedgerEntry,
    "decisions": DecisionRow,
    "approvals": ApprovalItem,
    "orders": Order,
}

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-_.]{0,19}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class DashboardError(Exception):
    """User-facing error from a dashboard action (message is shown verbatim)."""


# =========================================================================== generic helpers


def row_to_dict(row: Any) -> dict[str, Any]:
    """ORM row -> plain dict of its columns."""
    return {c.key: getattr(row, c.key) for c in row.__table__.columns}


def confirmation_ok(typed: str | None, *, batch_name: str | None = None) -> bool:
    """True if ``typed`` is exactly ``CONFIRM`` or (when given) the batch name."""
    text = (typed or "").strip()
    if text == CONFIRM_WORD:
        return True
    return bool(batch_name) and text == batch_name


def format_countdown(seconds: float | None) -> str:
    """``mm:ss`` under an hour, ``Hh MMm`` above, ``EXPIRED`` at or below zero."""
    if seconds is None:
        return "-"
    secs = int(seconds)
    if secs <= 0:
        return "EXPIRED"
    if secs < 3600:
        return f"{secs // 60:02d}:{secs % 60:02d}"
    hours, rem = divmod(secs, 3600)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return f"{days}d {hours}h {rem // 60:02d}m"
    return f"{hours}h {rem // 60:02d}m"


def parse_symbols(text: str | Iterable[str]) -> tuple[list[str], list[str]]:
    """Parse comma/space/newline separated NSE symbols -> (valid upper-case unique, invalid)."""
    parts = re.split(r"[\s,;]+", text) if isinstance(text, str) else list(text)
    valid: list[str] = []
    invalid: list[str] = []
    for raw in parts:
        sym = raw.strip().upper().removesuffix(".NS")
        if not sym:
            continue
        if _SYMBOL_RE.match(sym):
            if sym not in valid:
                valid.append(sym)
        else:
            invalid.append(raw.strip())
    return sorted(valid), invalid


def parse_emails(text: str) -> tuple[list[str], list[str]]:
    """Parse a comma/newline list of e-mail addresses -> (valid, invalid)."""
    valid: list[str] = []
    invalid: list[str] = []
    for raw in re.split(r"[\s,;]+", text or ""):
        addr = raw.strip()
        if not addr:
            continue
        (valid if _EMAIL_RE.match(addr) else invalid).append(addr)
    return valid, invalid


def parse_windows(text: str) -> list[dict[str, str]]:
    """Parse ``"09:30-14:45, 15:00-15:10"`` into ``[{start, end}, ...]``.

    Raises:
        DashboardError: on malformed input or start >= end.
    """
    windows: list[dict[str, str]] = []
    for chunk in re.split(r"[,;\n]+", text or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"^(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})$", chunk)
        if not m or not _HHMM_RE.match(m.group(1)) or not _HHMM_RE.match(m.group(2)):
            raise DashboardError(f"Invalid window {chunk!r}; use HH:MM-HH:MM")
        if m.group(1) >= m.group(2):
            raise DashboardError(f"Window {chunk!r}: start must be before end")
        windows.append({"start": m.group(1), "end": m.group(2)})
    return windows


def format_windows(windows: Sequence[Any]) -> str:
    """Inverse of :func:`parse_windows` (accepts TimeWindow models or dicts)."""
    out = []
    for w in windows:
        start = w["start"] if isinstance(w, dict) else w.start
        end = w["end"] if isinstance(w, dict) else w.end
        out.append(f"{start}-{end}")
    return ", ".join(out)


def parse_dates(text: str) -> list[date]:
    """Parse ``YYYY-MM-DD`` dates separated by commas/newlines (sorted, unique)."""
    out: set[date] = set()
    for chunk in re.split(r"[\s,;]+", text or ""):
        if not chunk.strip():
            continue
        try:
            out.add(date.fromisoformat(chunk.strip()))
        except ValueError as exc:
            raise DashboardError(f"Invalid date {chunk!r}; use YYYY-MM-DD") from exc
    return sorted(out)


def validate_hhmm(value: str | None, *, allow_empty: bool = False) -> str | None:
    """Validate an ``HH:MM`` string (returns it stripped, or None when empty and allowed)."""
    v = (value or "").strip()
    if not v and allow_empty:
        return None
    if not _HHMM_RE.match(v):
        raise DashboardError(f"Invalid time {value!r}; use HH:MM (24h, IST)")
    return v


# =========================================================================== engine status


@dataclass(frozen=True)
class EngineStatusView:
    """Engine heartbeat as seen by the dashboard."""

    state: str
    heartbeat_at: datetime | None
    age_seconds: float | None
    stale: bool
    activity: str | None = None
    next_cycle_at: datetime | None = None
    started_at: datetime | None = None
    pid: int | None = None
    batch_id: int | None = None
    settings_version_id: int | None = None
    data_feed_ok: bool | None = None
    llm_ok: bool | None = None
    market_open: bool | None = None
    last_error: str | None = None
    regime: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)

    @property
    def alive(self) -> bool:
        """Engine process heartbeat is fresh (state may still be PAUSED / HALTED)."""
        return not self.stale and self.state != EngineState.STOPPED.value

    @property
    def safe_for_restore(self) -> bool:
        """DB restore is allowed only when the engine is stopped or its heartbeat is stale."""
        return self.stale or self.state == EngineState.STOPPED.value

    @property
    def label(self) -> str:
        if self.state == "UNKNOWN":
            return "NO HEARTBEAT"
        return f"{self.state} (stale)" if self.stale else self.state


def engine_status(session: Session, now: datetime, stale_seconds: int) -> EngineStatusView:
    """Read ``engine_status`` id=1. Missing row -> state UNKNOWN, stale."""
    row = session.get(EngineStatus, 1)
    if row is None:
        return EngineStatusView(state="UNKNOWN", heartbeat_at=None, age_seconds=None, stale=True)
    age = (now - row.heartbeat_at).total_seconds() if row.heartbeat_at else None
    stale = age is None or age > stale_seconds
    return EngineStatusView(
        state=row.state,
        heartbeat_at=row.heartbeat_at,
        age_seconds=age,
        stale=stale,
        activity=row.activity,
        next_cycle_at=row.next_cycle_at,
        started_at=row.started_at,
        pid=row.pid,
        batch_id=row.batch_id,
        settings_version_id=row.settings_version_id,
        data_feed_ok=row.data_feed_ok,
        llm_ok=row.llm_ok,
        market_open=row.market_open,
        last_error=row.last_error,
        regime=dict(row.regime_json or {}),
        status=dict(row.status_json or {}),
    )


# =========================================================================== batches


@dataclass(frozen=True)
class BatchOption:
    id: int
    name: str
    status: str
    starting_capital: float
    created_at: datetime | None

    @property
    def label(self) -> str:
        return f"#{self.id} {self.name} [{self.status}]"


def batch_options(session: Session, include_deleted: bool = False) -> list[BatchOption]:
    """Batches for the sidebar selector, newest first."""
    stmt = select(Batch).order_by(Batch.id.desc())
    if not include_deleted:
        stmt = stmt.where(Batch.deleted.is_(False))
    return [
        BatchOption(b.id, b.name, b.status, b.starting_capital, b.created_at)
        for b in session.execute(stmt).scalars()
    ]


def soft_delete_batch(
    session: Session,
    batch_id: int,
    *,
    confirm_text: str,
    confirm_name: str,
    actor: str,
    clock: Clock,
) -> Batch:
    """Soft-delete a non-current batch (double confirmation; audit + history retained).

    Raises:
        DashboardError: when the batch is current/closing, has open positions, is already
            deleted, or a confirmation does not match.
    """
    batch = session.get(Batch, batch_id)
    if batch is None:
        raise DashboardError(f"Batch {batch_id} not found")
    if batch.deleted:
        raise DashboardError(f"Batch {batch.name!r} is already deleted")
    if batch.status in CURRENT_STATUSES:
        raise DashboardError(f"Batch {batch.name!r} is the current batch ({batch.status}); close it first")
    if batch.status == BatchStatus.CLOSING.value:
        raise DashboardError(f"Batch {batch.name!r} is CLOSING (CNC positions still running)")
    open_count = session.execute(
        select(func.count(Position.id)).where(
            Position.batch_id == batch_id, Position.status == PositionStatus.OPEN.value
        )
    ).scalar_one()
    if open_count:
        raise DashboardError(f"Batch {batch.name!r} still has {open_count} open position(s)")
    if (confirm_text or "").strip() != CONFIRM_WORD:
        raise DashboardError("First confirmation must be the word CONFIRM")
    if (confirm_name or "").strip() != batch.name:
        raise DashboardError("Second confirmation must be the exact batch name")
    batch.deleted = True
    audit(session, actor, "BATCH_SOFT_DELETE", f"batch:{batch.id}",
          {"name": batch.name, "status": batch.status, "note": "soft delete; all rows retained"},
          batch_id=batch.id, clock=clock)
    session.flush()
    return batch


# =========================================================================== approvals & decisions


@dataclass
class ApprovalView:
    """A pending approval item with its countdown."""

    item: ApprovalItem
    seconds_left: float
    expired: bool

    @property
    def countdown(self) -> str:
        return format_countdown(self.seconds_left)

    @property
    def risk_amount(self) -> float:
        """Rupee risk at the queued qty (|entry - SL| x qty)."""
        return abs(self.item.entry_price - self.item.stop_loss) * self.item.qty

    @property
    def rr(self) -> float | None:
        """Reward:risk to target 1."""
        risk = abs(self.item.entry_price - self.item.stop_loss)
        if risk <= 0:
            return None
        return abs(self.item.target_1 - self.item.entry_price) / risk


def pending_approvals(session: Session, batch_id: int | None, now: datetime) -> list[ApprovalView]:
    """PENDING approvals (soonest expiry first) with countdowns. ``batch_id`` None = all batches."""
    stmt = select(ApprovalItem).where(ApprovalItem.status == ApprovalStatus.PENDING.value)
    if batch_id is not None:
        stmt = stmt.where(ApprovalItem.batch_id == batch_id)
    stmt = stmt.order_by(ApprovalItem.expires_at.asc(), ApprovalItem.id.asc())
    out: list[ApprovalView] = []
    for item in session.execute(stmt).scalars():
        left = (item.expires_at - now).total_seconds()
        out.append(ApprovalView(item=item, seconds_left=max(0.0, left), expired=left <= 0))
    return out


def recent_approval_history(session: Session, batch_id: int | None, limit: int = 30) -> list[ApprovalItem]:
    """Decided approvals (approved / rejected / modified / timeout), newest first."""
    stmt = select(ApprovalItem).where(ApprovalItem.status != ApprovalStatus.PENDING.value)
    if batch_id is not None:
        stmt = stmt.where(ApprovalItem.batch_id == batch_id)
    return list(session.execute(stmt.order_by(ApprovalItem.id.desc()).limit(limit)).scalars())


@dataclass
class DecisionDetails:
    """Everything that explains one decision: meta, agents, debate, confluence, risk."""

    decision: dict[str, Any] | None
    agent_reports: list[dict[str, Any]]
    debates: list[dict[str, Any]]
    risk: dict[str, Any]
    meta: dict[str, Any]
    context: dict[str, Any]


def decision_details(session: Session, decision_id: int | None) -> DecisionDetails:
    """Decision row + agent_reports + debates (by decision_id) + risk/meta/context JSON."""
    if decision_id is None:
        return DecisionDetails(None, [], [], {}, {}, {})
    row = session.get(DecisionRow, decision_id)
    reports = session.execute(
        select(AgentReportRow).where(AgentReportRow.decision_id == decision_id).order_by(AgentReportRow.agent)
    ).scalars()
    debates = session.execute(
        select(DebateRow).where(DebateRow.decision_id == decision_id).order_by(DebateRow.id)
    ).scalars()
    return DecisionDetails(
        decision=row_to_dict(row) if row is not None else None,
        agent_reports=[row_to_dict(r) for r in reports],
        debates=[row_to_dict(d) for d in debates],
        risk=dict((row.risk_json if row is not None else None) or {}),
        meta=dict((row.meta_json if row is not None else None) or {}),
        context=dict((row.context_json if row is not None else None) or {}),
    )


def approval_details(session: Session, approval_id: int) -> tuple[ApprovalItem | None, DecisionDetails]:
    """Approval row + the full reasoning of its decision. Approval ``risk_json`` wins if set."""
    item = session.get(ApprovalItem, approval_id)
    if item is None:
        return None, decision_details(session, None)
    details = decision_details(session, item.decision_id)
    if item.risk_json:
        details.risk = dict(item.risk_json)
    return item, details


def allowed_modes_for_direction(direction: str) -> list[str]:
    """Modes a trade of ``direction`` may use (shorts are INTRADAY only - hard-coded)."""
    return [m.value for m in Mode if direction_allowed(m, Direction(direction))]


# =========================================================================== commands


@dataclass(frozen=True)
class CommandView:
    id: int
    command_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None
    requested_by: str
    payload: dict[str, Any]
    result: CommandResult | None
    error: str | None

    @property
    def finished(self) -> bool:
        return self.status in (CommandStatus.DONE.value, CommandStatus.FAILED.value, CommandStatus.REJECTED.value)

    @property
    def rule(self) -> str | None:
        return self.result.rule.value if self.result is not None and self.result.rule is not None else None

    @property
    def message(self) -> str:
        if self.result is not None and self.result.message:
            return self.result.message
        return self.error or ""


def _command_view(row: ControlCommand) -> CommandView:
    try:
        result = command_result(row)
    except ValidationError as exc:
        logger.warning(f"Unparseable result_json for command {row.id}: {exc}")
        raw = row.result_json or {}
        result = CommandResult(ok=bool(raw.get("ok", False)), message=str(raw.get("message", "")))
    return CommandView(
        id=row.id,
        command_type=row.command_type,
        status=row.status,
        created_at=row.created_at,
        completed_at=row.completed_at,
        requested_by=row.requested_by,
        payload=dict(row.payload_json or {}),
        result=result,
        error=row.error,
    )


def command_status(session: Session, command_id: int) -> CommandView | None:
    """Status + parsed result of one command."""
    row = session.get(ControlCommand, command_id)
    return _command_view(row) if row is not None else None


def recent_command_views(
    session: Session, limit: int = 20, types: Iterable[CommandType | str] | None = None
) -> list[CommandView]:
    """Newest commands first, optionally filtered by type."""
    stmt = select(ControlCommand).order_by(ControlCommand.id.desc()).limit(limit)
    if types:
        stmt = stmt.where(ControlCommand.command_type.in_([CommandType(t).value for t in types]))
    return [_command_view(r) for r in session.execute(stmt).scalars()]


def send_command(
    session: Session,
    command_type: CommandType,
    payload: dict[str, Any] | None,
    *,
    actor: str,
    batch_id: int | None,
    clock: Clock,
) -> int:
    """Validate + enqueue a command (audited by enqueue_command). Caller commits.

    Raises:
        DashboardError: when the payload is invalid.
    """
    try:
        row = enqueue_command(session, command_type, payload, actor=actor, batch_id=batch_id, clock=clock)
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise DashboardError(f"Invalid {CommandType(command_type).value} request: {msgs}") from exc
    return row.id


# =========================================================================== positions / states


def open_positions(session: Session, batch_id: int | None) -> list[Position]:
    """OPEN positions of a batch (all batches when None), oldest first."""
    stmt = select(Position).where(Position.status == PositionStatus.OPEN.value)
    if batch_id is not None:
        stmt = stmt.where(Position.batch_id == batch_id)
    return list(session.execute(stmt.order_by(Position.opened_at.asc())).scalars())


def unrealized_pnl(position: Position) -> float | None:
    """MTM P&L of the open qty net of accrued entry charges (None without a last price)."""
    if position.last_price is None:
        return None
    sign = 1.0 if position.direction == Direction.LONG.value else -1.0
    return round((position.last_price - position.avg_entry) * position.qty * sign - position.entry_charges, 2)


def position_rows(positions: Sequence[Position]) -> list[dict[str, Any]]:
    """Compact table rows for open positions."""
    return [
        {
            "id": p.id, "symbol": p.symbol, "mode": p.mode, "dir": p.direction, "qty": p.qty,
            "entry": p.avg_entry, "SL": p.stop_loss, "T1": p.target_1, "T2": p.target_2,
            "ltp": p.last_price, "uPnL": unrealized_pnl(p), "days": p.holding_days,
            "trail": p.trailing_rule, "pin": p.pin_no_event_exit,
        }
        for p in positions
    ]


def validate_sl_tighten(direction: str, current_sl: float, new_sl: float, ltp: float | None) -> str | None:
    """UI-side check that a new SL is strictly tighter and on the right side of price.

    Returns an error message, or None when acceptable. The engine re-validates (hard guard).
    """
    if new_sl <= 0:
        return "Stop-loss must be positive."
    if direction == Direction.LONG.value:
        if new_sl <= current_sl:
            return f"LONG: new SL {new_sl:.2f} must be ABOVE the current SL {current_sl:.2f} (tighten only)."
        if ltp is not None and new_sl >= ltp:
            return f"LONG: new SL {new_sl:.2f} must stay below the last price {ltp:.2f}."
    else:
        if new_sl >= current_sl:
            return f"SHORT: new SL {new_sl:.2f} must be BELOW the current SL {current_sl:.2f} (tighten only)."
        if ltp is not None and new_sl <= ltp:
            return f"SHORT: new SL {new_sl:.2f} must stay above the last price {ltp:.2f}."
    return None


def validate_trade_levels(
    direction: str, entry: float | None, stop_loss: float, target_1: float, target_2: float | None
) -> list[str]:
    """UI-side sanity checks for a manual trade / modified approval (engine re-checks all rules)."""
    errors: list[str] = []
    if stop_loss <= 0 or target_1 <= 0:
        errors.append("Stop-loss and target 1 are mandatory and must be positive.")
        return errors
    if entry is None or entry <= 0:
        return errors
    if direction == Direction.LONG.value:
        if stop_loss >= entry:
            errors.append("LONG: stop-loss must be below entry.")
        if target_1 <= entry:
            errors.append("LONG: target 1 must be above entry.")
        if target_2 is not None and target_2 < target_1:
            errors.append("LONG: target 2 must be >= target 1.")
    else:
        if stop_loss <= entry:
            errors.append("SHORT: stop-loss must be above entry.")
        if target_1 >= entry:
            errors.append("SHORT: target 1 must be below entry.")
        if target_2 is not None and target_2 > target_1:
            errors.append("SHORT: target 2 must be <= target 1.")
    return errors


def mode_states(session: Session, batch_id: int | None) -> dict[str, ModeState]:
    """Mode runtime state rows keyed by mode value."""
    if batch_id is None:
        return {}
    rows = session.execute(select(ModeState).where(ModeState.batch_id == batch_id)).scalars()
    return {r.mode: r for r in rows}


def strategy_states(session: Session, batch_id: int | None) -> list[StrategyState]:
    if batch_id is None:
        return []
    return list(
        session.execute(
            select(StrategyState).where(StrategyState.batch_id == batch_id)
            .order_by(StrategyState.strategy, StrategyState.mode)
        ).scalars()
    )


def strategy_performance(session: Session, batch_id: int | None) -> pd.DataFrame:
    """Per strategy x mode: trades, win rate %, net P&L, avg R (closed legs)."""
    cols = ["strategy", "mode", "trades", "win_rate_pct", "net_pnl", "avg_r"]
    if batch_id is None:
        return pd.DataFrame(columns=cols)
    rows = session.execute(
        select(Trade.strategy, Trade.mode, Trade.net_pnl, Trade.r_multiple).where(Trade.batch_id == batch_id)
    ).all()
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["strategy", "mode", "net_pnl", "r"])
    df["strategy"] = df["strategy"].fillna("(manual/none)")
    grouped = df.groupby(["strategy", "mode"])
    out = pd.DataFrame({
        "trades": grouped.size(),
        "win_rate_pct": grouped["net_pnl"].apply(lambda s: round(100.0 * float((s > 0).mean()), 1)),
        "net_pnl": grouped["net_pnl"].sum().round(2),
        "avg_r": grouped["r"].mean().round(2),
    }).reset_index()
    return out[cols]


def capital_account(session: Session, batch_id: int | None) -> CapitalAccount | None:
    return session.get(CapitalAccount, batch_id) if batch_id is not None else None


def recent_ledger(session: Session, batch_id: int | None, limit: int = 20) -> list[CapitalLedgerEntry]:
    if batch_id is None:
        return []
    return list(
        session.execute(
            select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id)
            .order_by(CapitalLedgerEntry.id.desc()).limit(limit)
        ).scalars()
    )


def recent_notifications(session: Session, limit: int = 20) -> list[NotificationLog]:
    return list(session.execute(select(NotificationLog).order_by(NotificationLog.id.desc()).limit(limit)).scalars())


# =========================================================================== settings


@dataclass(frozen=True)
class SettingsChangeResult:
    version_id: int
    changed: bool
    adjustments: list[str]
    command_id: int | None


def load_settings(session: Session, caps: HardCaps, clock: Clock) -> tuple[int, Settings]:
    """Active settings (seeds version 1 from YAML if the table is empty). Caller commits."""
    return get_active_settings(session, caps, clock=clock)


def cap_hint(path: str, caps: HardCaps) -> str | None:
    """Human text for a capped setting, e.g. ``"hard cap: max 2.0"``."""
    cap = get_cap(path, caps)
    if cap is None:
        return None
    kind, value = cap
    return f"hard cap: {'max' if kind == 'max' else 'min'} {value:g} (config/hard_caps.yaml)"


def cap_bounds(path: str, caps: HardCaps) -> tuple[float | None, float | None]:
    """(min_value, max_value) imposed by a hard cap for a numeric input."""
    cap = get_cap(path, caps)
    if cap is None:
        return None, None
    kind, value = cap
    return (None, value) if kind == "max" else (value, None)


def clamp_changes(changes: dict[str, Any], caps: HardCaps) -> tuple[dict[str, Any], list[str]]:
    """Clamp every numeric change to its hard cap. Returns (clamped changes, adjustments)."""
    out: dict[str, Any] = {}
    adjustments: list[str] = []
    for path, value in changes.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clamped = clamp_value(path, float(value), caps)
            if clamped != float(value):
                adjustments.append(f"{path}: {value} -> {clamped:g} ({cap_hint(path, caps)})")
                value = type(value)(clamped) if isinstance(value, int) and clamped == int(clamped) else clamped
        out[path] = value
    return out, adjustments


def apply_settings_changes(
    session: Session,
    changes: dict[str, Any],
    caps: HardCaps,
    *,
    actor: str,
    note: str | None,
    batch_id: int | None,
    clock: Clock,
) -> SettingsChangeResult:
    """Clamp -> validate -> save a new settings version -> enqueue RELOAD_SETTINGS. Caller commits.

    No-op (no version, no command) when nothing changed.

    Raises:
        DashboardError: invalid value / path or hard-cap violation (message is user-facing).
    """
    if not changes:
        vid, _ = get_active_settings(session, caps, clock=clock)
        return SettingsChangeResult(vid, False, [], None)
    clamped, adjustments = clamp_changes(changes, caps)
    before_id, _ = get_active_settings(session, caps, clock=clock)
    try:
        row = apply_changes(session, clamped, caps, author=actor, note=note or None, batch_id=batch_id, clock=clock)
    except HardCapViolationError as exc:
        raise DashboardError(f"Hard cap violation: {exc}") from exc
    except ValidationError as exc:
        msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise DashboardError(f"Invalid settings: {msgs}") from exc
    except KeyError as exc:
        raise DashboardError(f"Unknown settings path {exc}") from exc
    except ValueError as exc:
        raise DashboardError(f"Invalid settings: {exc}") from exc
    if row.id == before_id:
        return SettingsChangeResult(row.id, False, adjustments, None)
    cmd_id = send_command(session, CommandType.RELOAD_SETTINGS, {}, actor=actor, batch_id=batch_id, clock=clock)
    return SettingsChangeResult(row.id, True, adjustments, cmd_id)


def revert_settings(
    session: Session, version_id: int, caps: HardCaps, *, actor: str, batch_id: int | None, clock: Clock
) -> SettingsChangeResult:
    """Create a new version restoring ``version_id`` and enqueue RELOAD_SETTINGS. Caller commits."""
    try:
        row = revert_to(session, version_id, caps, author=actor, batch_id=batch_id, clock=clock)
    except KeyError as exc:
        raise DashboardError(str(exc)) from exc
    except HardCapViolationError as exc:
        raise DashboardError(f"Version {version_id} violates the current hard caps: {exc}") from exc
    cmd_id = send_command(session, CommandType.RELOAD_SETTINGS, {}, actor=actor, batch_id=batch_id, clock=clock)
    return SettingsChangeResult(row.id, True, [], cmd_id)


def settings_history(session: Session, limit: int = 50) -> list[SettingsVersion]:
    return list_versions(session, limit=limit)


def allocation_total(allocations: dict[str, float]) -> float:
    return round(sum(float(v) for v in allocations.values()), 6)


def validate_allocations(allocations: dict[str, float]) -> str | None:
    """Error text if allocations are negative or sum above 100 %."""
    if any(float(v) < 0 for v in allocations.values()):
        return "Allocations must be >= 0 %."
    total = allocation_total(allocations)
    if total > 100.0 + 1e-9:
        return f"Allocations sum to {total:.2f}% (must be <= 100%)."
    return None


def changed_only(current: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    """Keep only entries of ``proposed`` that differ from ``current`` (same dotted paths)."""
    out: dict[str, Any] = {}
    for path, value in proposed.items():
        old = current.get(path)
        if isinstance(value, float) and isinstance(old, (int, float)) and not isinstance(old, bool):
            if abs(float(old) - value) > 1e-12:
                out[path] = value
        elif old != value:
            out[path] = value
    return out


# =========================================================================== prompt templates


def prompt_versions(session: Session, agent: str) -> list[PromptTemplate]:
    """All versions of an agent's prompt, newest first."""
    return list(
        session.execute(
            select(PromptTemplate).where(PromptTemplate.agent == agent).order_by(PromptTemplate.version.desc())
        ).scalars()
    )


def latest_prompt(session: Session, agent: str) -> PromptTemplate | None:
    versions = prompt_versions(session, agent)
    return versions[0] if versions else None


def save_prompt_version(
    session: Session,
    agent: str,
    system_prompt: str,
    user_template: str,
    *,
    author: str,
    note: str | None,
    clock: Clock,
) -> PromptTemplate:
    """Store a new prompt version (max(version)+1) and make it the active one. Caller commits."""
    if not system_prompt.strip() or not user_template.strip():
        raise DashboardError("System prompt and user template must not be empty.")
    current_max = session.execute(
        select(func.max(PromptTemplate.version)).where(PromptTemplate.agent == agent)
    ).scalar_one_or_none() or 0
    for old in session.execute(select(PromptTemplate).where(PromptTemplate.agent == agent)).scalars():
        old.active = False
    row = PromptTemplate(
        agent=agent, version=current_max + 1, system_prompt=system_prompt, user_template=user_template,
        created_at=clock.now(), author=author, note=note, active=True,
    )
    session.add(row)
    session.flush()
    audit(session, author, "PROMPT_SAVE", f"prompt:{agent}:v{row.version}",
          {"agent": agent, "version": row.version, "note": note}, clock=clock)
    return row


def revert_prompt(session: Session, agent: str, version: int, *, author: str, clock: Clock) -> PromptTemplate:
    """Copy an older version into a NEW version (history is append-only). Caller commits."""
    old = session.execute(
        select(PromptTemplate).where(PromptTemplate.agent == agent, PromptTemplate.version == version)
    ).scalar_one_or_none()
    if old is None:
        raise DashboardError(f"Prompt {agent} v{version} not found")
    row = save_prompt_version(session, agent, old.system_prompt, old.user_template, author=author,
                              note=f"Revert to v{version}", clock=clock)
    audit(session, author, "PROMPT_REVERT", f"prompt:{agent}:v{row.version}",
          {"agent": agent, "reverted_from": version, "new_version": row.version}, clock=clock)
    return row


# =========================================================================== Ollama


def installed_ollama_models(host: str, timeout: float = 2.0, client: httpx.Client | None = None) -> list[str]:
    """Model names from ``GET {host}/api/tags``. Any failure -> ``[]`` (logged)."""
    url = host.rstrip("/") + "/api/tags"
    try:
        if client is not None:
            resp = client.get(url, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as c:
                resp = c.get(url)
        resp.raise_for_status()
        data = resp.json()
        names = [str(m.get("name") or m.get("model")) for m in data.get("models", []) if isinstance(m, dict)]
        return sorted(n for n in names if n and n != "None")
    except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
        logger.warning(f"Ollama not reachable at {host}: {exc}")
        return []


def ollama_reachable(host: str, timeout: float = 2.0, client: httpx.Client | None = None) -> bool:
    """True when ``{host}/api/tags`` answers (models may be empty)."""
    url = host.rstrip("/") + "/api/tags"
    try:
        if client is not None:
            resp = client.get(url, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as c:
                resp = c.get(url)
        return resp.status_code == 200
    except httpx.HTTPError as exc:
        logger.debug(f"Ollama health check failed: {exc}")
        return False


# =========================================================================== logs


def query_logs(
    day: date,
    categories: Sequence[str],
    *,
    text: str | None = None,
    limit: int = 500,
    log_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """JSONL events of ``day`` for ``categories`` filtered by case-insensitive ``text``, newest first."""
    needle = (text or "").strip().lower()
    events: list[dict[str, Any]] = []
    for cat in categories:
        try:
            rows = read_events(day, cat, log_dir=log_dir)
        except OSError as exc:
            record_error(None, "dashboard.logs", exc, {"day": day.isoformat(), "category": cat})
            continue
        for ev in rows:
            if needle and needle not in json.dumps(ev, ensure_ascii=False, default=str).lower():
                continue
            events.append(ev)
    events.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
    return events[:limit]


# =========================================================================== DB backup / restore


def sqlite_path_from_url(url: str) -> Path | None:
    """Filesystem path of a file-based SQLite URL (None for in-memory / other engines)."""
    if not url.startswith("sqlite"):
        return None
    if url in ("sqlite://", "sqlite:///:memory:"):
        return None
    path = url.split("sqlite:///", 1)[-1]
    if not path or path == ":memory:":
        return None
    return Path(path)


def database_size_bytes(url: str) -> int | None:
    """Size of the SQLite DB file plus its WAL (None for non-file DBs)."""
    path = sqlite_path_from_url(url)
    if path is None or not path.exists():
        return None
    total = path.stat().st_size
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        total += wal.stat().st_size
    return total


def backup_database(url: str, backup_dir: Path, now: datetime, *, prefix: str = "trading") -> Path:
    """Online backup via the sqlite3 backup API -> ``backup_dir/trading_YYYYmmdd_HHMMSS.db``.

    Raises:
        DashboardError: for non-SQLite / in-memory DBs or a missing DB file.
    """
    src_path = sqlite_path_from_url(url)
    if src_path is None:
        raise DashboardError("Backup from the dashboard supports file-based SQLite only (use pg_dump for PostgreSQL).")
    if not src_path.exists():
        raise DashboardError(f"Database file not found: {src_path}")
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{prefix}_{stamp}.db"
    n = 1
    while dest.exists():
        dest = backup_dir / f"{prefix}_{stamp}_{n}.db"
        n += 1
    src = sqlite3.connect(str(src_path), timeout=30)
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    logger.info(f"Database backup written to {dest}")
    return dest


def list_backups(backup_dir: Path) -> list[Path]:
    """Backup files, newest first."""
    backup_dir = Path(backup_dir)
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)


def _validate_backup_file(path: Path) -> None:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        ok = conn.execute("PRAGMA integrity_check").fetchone()
        if not ok or ok[0] != "ok":
            raise DashboardError(f"Backup {path.name} failed integrity check: {ok}")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError as exc:
        raise DashboardError(f"{path.name} is not a valid SQLite database: {exc}") from exc
    finally:
        conn.close()
    missing = {"batches", "settings_versions", "control_commands", "audit_log"} - tables
    if missing:
        raise DashboardError(f"{path.name} is not a Liar-V1 database (missing tables: {sorted(missing)})")


def restore_database(
    url: str,
    backup_path: Path,
    *,
    engine_view: EngineStatusView,
    confirm_text: str,
    backup_dir: Path,
    now: datetime,
    sa_engine: Any | None = None,
) -> Path:
    """Restore ``backup_path`` over the live SQLite DB. Returns the safety backup of the old DB.

    Preconditions: typed ``CONFIRM`` and the engine stopped (or heartbeat stale). The current DB
    is backed up first; the copy uses the sqlite3 backup API (WAL-safe, page-by-page), and the
    SQLAlchemy pool is disposed so new connections see the restored data.
    """
    if (confirm_text or "").strip() != CONFIRM_WORD:
        raise DashboardError("Type CONFIRM to restore the database.")
    if not engine_view.safe_for_restore:
        raise DashboardError(
            f"Engine is {engine_view.state} with a fresh heartbeat. Stop the engine before restoring."
        )
    db_path = sqlite_path_from_url(url)
    if db_path is None:
        raise DashboardError("Restore supports file-based SQLite only.")
    backup_path = Path(backup_path)
    if not backup_path.exists():
        raise DashboardError(f"Backup not found: {backup_path}")
    if backup_path.resolve() == db_path.resolve():
        raise DashboardError("Cannot restore the live database onto itself.")
    _validate_backup_file(backup_path)
    safety = backup_database(url, backup_dir, now, prefix="pre_restore")
    if sa_engine is not None:
        sa_engine.dispose()
    src = sqlite3.connect(f"file:{backup_path.as_posix()}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(db_path), timeout=30)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    if sa_engine is not None:
        sa_engine.dispose()
    logger.warning(f"Database restored from {backup_path} (previous DB saved to {safety})")
    return safety


# =========================================================================== export


def _export_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None).isoformat(sep=" ")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def table_frame(session: Session, model: type[Base], batch_id: int) -> pd.DataFrame:
    """All rows of ``model`` for ``batch_id`` as a DataFrame (datetimes as IST strings, JSON as text)."""
    table = model.__table__  # type: ignore[attr-defined]
    cols = [c.key for c in table.columns]
    rows = session.execute(
        select(model).where(getattr(model, "batch_id") == batch_id).order_by(getattr(model, table.primary_key.columns.keys()[0]))
    ).scalars()
    data = [{c: _export_value(getattr(r, c)) for c in cols} for r in rows]
    return pd.DataFrame(data, columns=cols)


def export_batch(session: Session, batch_id: int, fmt: str) -> tuple[bytes, str, str]:
    """Export trades/positions/ledger/decisions/approvals/orders -> (bytes, file name, mime).

    ``fmt`` = ``"xlsx"`` (one sheet per table, openpyxl) or ``"csv"`` (zip of CSV files).
    """
    batch = session.get(Batch, batch_id)
    if batch is None:
        raise DashboardError(f"Batch {batch_id} not found")
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", batch.name).strip("_") or f"batch{batch_id}"
    frames = {name: table_frame(session, model, batch_id) for name, model in EXPORT_TABLES.items()}
    buf = io.BytesIO()
    if fmt == "xlsx":
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, df in frames.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)
        return buf.getvalue(), f"batch_{batch_id}_{safe_name}.xlsx", (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    if fmt == "csv":
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, df in frames.items():
                zf.writestr(f"{name}.csv", df.to_csv(index=False))
        return buf.getvalue(), f"batch_{batch_id}_{safe_name}_csv.zip", "application/zip"
    raise DashboardError(f"Unknown export format {fmt!r}")


# =========================================================================== health


def health_status(
    session: Session, *, db_url: str, ollama_host: str, now: datetime, stale_seconds: int,
    check_ollama: bool = True,
) -> dict[str, Any]:
    """Engine heartbeat, DB size, Ollama reachability, data feed flag (from engine_status)."""
    view = engine_status(session, now, stale_seconds)
    size = database_size_bytes(db_url)
    return {
        "engine_state": view.label,
        "heartbeat_age_s": None if view.age_seconds is None else round(view.age_seconds, 1),
        "engine_alive": view.alive,
        "db_url_kind": db_url.split(":", 1)[0],
        "db_size_mb": None if size is None else round(size / 1_048_576, 2),
        "ollama_reachable": ollama_reachable(ollama_host) if check_ollama else None,
        "llm_ok (engine)": view.llm_ok,
        "data_feed_ok (engine)": view.data_feed_ok,
        "market_open (engine)": view.market_open,
        "last_error": view.last_error,
    }


def heartbeat_stale_seconds(settings: Settings | None) -> int:
    """Stale threshold from settings (default 30 s)."""
    return int(settings.engine.heartbeat_stale_seconds) if settings is not None else 30


def ollama_host(settings: Settings, override: str | None) -> str:
    """Effective Ollama host (``.env`` OLLAMA_HOST override wins)."""
    return override or settings.models.ollama_host
