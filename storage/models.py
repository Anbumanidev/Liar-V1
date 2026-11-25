"""SQLAlchemy 2.x ORM models (section 12). SQLite by default, PostgreSQL-compatible.

Rules:
* Every trading row carries ``batch_id``. History is never deleted by a batch reset.
* Datetimes are stored as IST wall time and returned timezone-aware (``ISTDateTime``).
* Enum columns are stored as plain strings (``StrEnum`` values).
* JSON payloads (agent reports, settings snapshots, ...) use the generic ``JSON`` type.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.clock import IST


class ISTDateTime(TypeDecorator[datetime]):
    """Store aware datetimes as naive IST wall time; return them IST-aware."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(IST).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=IST)
        return value.astimezone(IST)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _str(n: int = 32) -> String:
    return String(n)


# =========================================================================== batches & settings


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    starting_capital: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(_str(), index=True)  # BatchStatus
    settings_version_id: Mapped[int | None] = mapped_column(ForeignKey("settings_versions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    started_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    halted_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    halt_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)  # soft delete; audit retained
    settings_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class SettingsVersion(Base):
    """Full config snapshot per change (section 14.2)."""

    __tablename__ = "settings_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    author: Mapped[str] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    diff_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)  # [{path, old, new}]
    reverted_from_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PromptTemplate(Base):
    """Versioned prompt templates (section 16). Editable from the dashboard with revert."""

    __tablename__ = "prompt_templates"
    __table_args__ = (UniqueConstraint("agent", "version", name="uq_prompt_agent_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent: Mapped[str] = mapped_column(String(48), index=True)
    version: Mapped[int] = mapped_column(Integer)
    system_prompt: Mapped[str] = mapped_column(Text)
    user_template: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    author: Mapped[str] = mapped_column(String(64))
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


# =========================================================================== runs & analysis


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    run_type: Mapped[str] = mapped_column(_str())  # RunType
    mode: Mapped[str | None] = mapped_column(_str(), nullable=True)
    started_at: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    status: Mapped[str] = mapped_column(_str(), default="RUNNING")  # RUNNING | DONE | FAILED
    n_symbols: Mapped[int] = mapped_column(Integer, default=0)
    settings_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_replay: Mapped[bool] = mapped_column(Boolean, default=False)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class StockAnalyzed(Base):
    """Every stock considered (universe filter, watchlist, analysis) and why (section 10/7.5)."""

    __tablename__ = "stocks_analyzed"
    __table_args__ = (Index("ix_stocks_batch_time", "batch_id", "as_of"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    as_of: Mapped[datetime] = mapped_column(ISTDateTime)
    stage: Mapped[str] = mapped_column(_str())  # UNIVERSE | WATCHLIST | ANALYZED | SCAN
    included: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    regime: Mapped[str | None] = mapped_column(_str(), nullable=True)
    confluence: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    features_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class AgentReportRow(Base):
    __tablename__ = "agent_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    agent: Mapped[str] = mapped_column(String(48), index=True)
    signal: Mapped[str] = mapped_column(_str())
    confidence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(_str())  # AgentStatus
    model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    prompt_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime, index=True)


class DebateRow(Base):
    __tablename__ = "debates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    rounds: Mapped[int] = mapped_column(Integer)
    bull_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    bear_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    transcript_json: Mapped[list[Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(_str())
    model: Mapped[str | None] = mapped_column(String(80), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)


class DecisionRow(Base):
    """Meta output + risk verdict + routing for one symbol/mode (section 12)."""

    __tablename__ = "decisions"
    __table_args__ = (Index("ix_decisions_batch_time", "batch_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    regime: Mapped[str | None] = mapped_column(_str(), nullable=True)
    confluence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    decision: Mapped[str] = mapped_column(_str())  # TradeDecision
    direction: Mapped[str | None] = mapped_column(_str(), nullable=True)
    confidence: Mapped[int] = mapped_column(Integer, default=0)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # strategies, confluence, sl candidates
    risk_approved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    risk_rule: Mapped[str | None] = mapped_column(String(48), nullable=True)
    risk_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    qty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_action: Mapped[str] = mapped_column(_str(), default="NO_TRADE")
    # final_action: NO_TRADE | RISK_REJECTED | QUEUED | EXECUTED | SIGNAL_ONLY | FILTERED
    source: Mapped[str] = mapped_column(_str(), default="SYSTEM")
    settings_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)


# =========================================================================== approvals & execution


class ApprovalItem(Base):
    __tablename__ = "approval_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(_str())
    direction: Mapped[str] = mapped_column(_str())
    entry_type: Mapped[str] = mapped_column(_str())
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target_1: Mapped[float] = mapped_column(Float)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty: Mapped[int] = mapped_column(Integer)
    max_qty: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[int] = mapped_column(Integer)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    proposal_json: Mapped[dict[str, Any]] = mapped_column(JSON)   # TradeProposal
    risk_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # top bull/bear points etc.
    status: Mapped[str] = mapped_column(_str(), index=True)        # ApprovalStatus
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    expires_at: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    decided_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_edits_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    user_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_mode: Mapped[str | None] = mapped_column(_str(), nullable=True)
    order_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_rule: Mapped[str | None] = mapped_column(String(48), nullable=True)  # risk rule if my approval was blocked


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (CheckConstraint("is_paper", name="ck_orders_paper_only"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    position_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    product: Mapped[str] = mapped_column(_str())
    side: Mapped[str] = mapped_column(_str())          # OrderSide
    order_type: Mapped[str] = mapped_column(_str())    # OrderType
    purpose: Mapped[str] = mapped_column(_str())       # OrderPurpose
    qty: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(_str(), index=True)  # OrderStatus
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    valid_until: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    charges_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(_str(), default="SYSTEM")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)   # exit reason / reject reason
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    meta_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # proposal snapshot etc.


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    product: Mapped[str] = mapped_column(_str())
    direction: Mapped[str] = mapped_column(_str())
    status: Mapped[str] = mapped_column(_str(), index=True)   # PositionStatus
    qty: Mapped[int] = mapped_column(Integer)                  # current open qty
    initial_qty: Mapped[int] = mapped_column(Integer)
    avg_entry: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    initial_stop_loss: Mapped[float] = mapped_column(Float)
    target_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_rule: Mapped[str] = mapped_column(_str(), default="NONE")
    max_holding_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    holding_days: Mapped[int] = mapped_column(Integer, default=0)
    opened_at: Mapped[datetime] = mapped_column(ISTDateTime)
    closed_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    regime: Mapped[str | None] = mapped_column(_str(), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    source: Mapped[str] = mapped_column(_str(), default="SYSTEM")
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)
    margin_blocked: Mapped[float] = mapped_column(Float)       # capital blocked for the open qty
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    entry_charges: Mapped[float] = mapped_column(Float, default=0.0)  # accrued for the open qty
    realized_gross: Mapped[float] = mapped_column(Float, default=0.0)  # from partial exits
    realized_charges: Mapped[float] = mapped_column(Float, default=0.0)
    partial_done: Mapped[bool] = mapped_column(Boolean, default=False)
    breakeven_moved: Mapped[bool] = mapped_column(Boolean, default=False)
    highest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    lowest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_price_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    pin_no_event_exit: Mapped[bool] = mapped_column(Boolean, default=False)
    overnight: Mapped[bool] = mapped_column(Boolean, default=False)
    converted_from_mode: Mapped[str | None] = mapped_column(_str(), nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settings_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settings_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # frozen mode settings
    sl_history_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)     # [{ts, old, new, by}]
    notes_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class Trade(Base):
    """One closed leg (full or partial exit) of a position."""

    __tablename__ = "trades"
    __table_args__ = (Index("ix_trades_batch_exit", "batch_id", "exit_time"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer)
    position_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    product: Mapped[str] = mapped_column(_str())
    direction: Mapped[str] = mapped_column(_str())
    qty: Mapped[int] = mapped_column(Integer)
    entry_time: Mapped[datetime] = mapped_column(ISTDateTime)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_time: Mapped[datetime] = mapped_column(ISTDateTime)
    exit_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)            # initial SL (for R multiple)
    target_1: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str] = mapped_column(_str())           # ExitReason
    is_partial: Mapped[bool] = mapped_column(Boolean, default=False)
    gross_pnl: Mapped[float] = mapped_column(Float)
    charges: Mapped[float] = mapped_column(Float)
    net_pnl: Mapped[float] = mapped_column(Float)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    charges_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(_str())
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    regime: Mapped[str | None] = mapped_column(_str(), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    holding_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    holding_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ShadowTrade(Base):
    """Rejected setups followed to their would-be outcome without capital (section 7.6)."""

    __tablename__ = "shadow_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    approval_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(_str())
    direction: Mapped[str] = mapped_column(_str())
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    target_1: Mapped[float] = mapped_column(Float)
    target_2: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty: Mapped[int] = mapped_column(Integer, default=0)       # hypothetical (risk-manager max) qty
    rejection_source: Mapped[str] = mapped_column(_str())      # RejectionSource
    rejection_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    grade: Mapped[str | None] = mapped_column(String(4), nullable=True)
    confidence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    expires_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    entered_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    status: Mapped[str] = mapped_column(_str(), default="OPEN")  # OPEN | CLOSED
    outcome: Mapped[str] = mapped_column(_str(), default="OPEN")  # ShadowOutcome
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    hypothetical_net_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)


class CorporateActionAdjustment(Base):
    __tablename__ = "corporate_action_adjustments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    position_id: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(32))
    action_type: Mapped[str] = mapped_column(_str())   # SPLIT | BONUS | EX_DIVIDEND
    ex_date: Mapped[date] = mapped_column(Date)
    ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty_before: Mapped[int] = mapped_column(Integer)
    qty_after: Mapped[int] = mapped_column(Integer)
    price_before: Mapped[float] = mapped_column(Float)
    price_after: Mapped[float] = mapped_column(Float)
    sl_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    cash_credit: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)


# =========================================================================== capital


class CapitalAccount(Base):
    """Current virtual account balances for a batch (section 6.1)."""

    __tablename__ = "capital_accounts"

    batch_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    starting_capital: Mapped[float] = mapped_column(Float)
    base_capital: Mapped[float] = mapped_column(Float)
    trading_cash: Mapped[float] = mapped_column(Float)
    blocked: Mapped[float] = mapped_column(Float, default=0.0)
    reserve: Mapped[float] = mapped_column(Float, default=0.0)
    reserve_peak: Mapped[float] = mapped_column(Float, default=0.0)
    hwm: Mapped[float] = mapped_column(Float)
    sizing_capital: Mapped[float] = mapped_column(Float)
    sizing_up_streak: Mapped[int] = mapped_column(Integer, default=0)
    cum_realized_net: Mapped[float] = mapped_column(Float, default=0.0)
    cum_manual_adjustments: Mapped[float] = mapped_column(Float, default=0.0)
    last_eod_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    halted: Mapped[bool] = mapped_column(Boolean, default=False)
    halt_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    tier: Mapped[str] = mapped_column(_str(), default="NORMAL")
    updated_at: Mapped[datetime] = mapped_column(ISTDateTime)


class Watchlist(Base):
    """A named list of stocks the owner follows (or the system's next-day plan).

    kind: MANUAL (owner's follow-up lists) | NEXT_DAY (end-of-day plan for ``for_date``) | SYSTEM.
    ``use_for_trading``: INTRADAY cycles analyse these symbols first (they still pass every risk rule).
    Owner data, not trading state: the web dashboard writes it directly (audited), not via commands.
    """

    __tablename__ = "watchlists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    kind: Mapped[str] = mapped_column(_str(), default="MANUAL", index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    use_for_trading: Mapped[bool] = mapped_column(Boolean, default=False)
    for_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)   # NEXT_DAY plans
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime)
    updated_at: Mapped[datetime] = mapped_column(ISTDateTime)
    meta_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class WatchlistItem(Base):
    """One stock in a watchlist, with an optional plan (levels) and its latest analysis."""

    __tablename__ = "watchlist_items"
    __table_args__ = (UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    watchlist_id: Mapped[int] = mapped_column(Integer, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str | None] = mapped_column(_str(), nullable=True)
    source: Mapped[str] = mapped_column(_str(), default="MANUAL")     # MANUAL | ANALYSIS | SCAN | EOD_PLAN
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    added_at: Mapped[datetime] = mapped_column(ISTDateTime)
    added_by: Mapped[str] = mapped_column(String(64), default="owner")
    added_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    plan_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # setup, trigger, stop, target, score, reasons
    status: Mapped[str] = mapped_column(_str(), default="ACTIVE")    # ACTIVE | TRIGGERED | INVALIDATED
    status_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)        # analysis it was added from
    last_analysis_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_analysis_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    last_summary: Mapped[str | None] = mapped_column(Text, nullable=True)


class FundingEvent(Base):
    """Owner money in / out when a batch starts (the "how much to add to get back to 1000" record).

    kind: INITIAL (first batch), TOP_UP (+ money added), WITHDRAW (amount < 0: profit taken out),
    NONE (carried equity already equals the new starting capital). Paper money only.
    """

    __tablename__ = "funding_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)          # the NEW batch
    kind: Mapped[str] = mapped_column(_str())
    amount: Mapped[float] = mapped_column(Float)                         # signed: + in, - out
    starting_capital: Mapped[float] = mapped_column(Float)
    prev_batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prev_equity: Mapped[float | None] = mapped_column(Float, nullable=True)  # carried in from the old batch
    actor: Mapped[str] = mapped_column(String(64), default="system")
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class CapitalLedgerEntry(Base):
    __tablename__ = "capital_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    entry_type: Mapped[str] = mapped_column(_str())       # LedgerEntryType
    amount: Mapped[float] = mapped_column(Float)          # signed movement (see entry_type)
    from_account: Mapped[str | None] = mapped_column(_str(), nullable=True)
    to_account: Mapped[str | None] = mapped_column(_str(), nullable=True)
    trading_cash_after: Mapped[float] = mapped_column(Float)
    blocked_after: Mapped[float] = mapped_column(Float)
    reserve_after: Mapped[float] = mapped_column(Float)
    sizing_capital_after: Mapped[float] = mapped_column(Float)
    position_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trade_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str | None] = mapped_column(_str(), nullable=True)
    actor: Mapped[str] = mapped_column(String(64), default="SYSTEM")
    confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class DailySummary(Base):
    __tablename__ = "daily_summary"
    __table_args__ = (UniqueConstraint("batch_id", "day", "mode", name="uq_daily_batch_day_mode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer, index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    mode: Mapped[str] = mapped_column(_str())  # Mode value or "TOTAL"
    trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    charges: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    trading_balance: Mapped[float | None] = mapped_column(Float, nullable=True)
    trading_cash: Mapped[float | None] = mapped_column(Float, nullable=True)
    reserve: Mapped[float | None] = mapped_column(Float, nullable=True)
    equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    hwm: Mapped[float | None] = mapped_column(Float, nullable=True)
    drawdown_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    tier: Mapped[str | None] = mapped_column(_str(), nullable=True)
    sizing_capital: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_locked: Mapped[float] = mapped_column(Float, default=0.0)
    refilled: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    extra_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


# =========================================================================== state


class ModeState(Base):
    """Runtime state per batch+mode: pauses, consecutive losses, intraday day-stop."""

    __tablename__ = "mode_states"
    __table_args__ = (UniqueConstraint("batch_id", "mode", name="uq_mode_state"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(_str())
    paused: Mapped[bool] = mapped_column(Boolean, default=False)          # owner pause (dashboard)
    paused_until: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)  # automatic pause
    pause_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    consecutive_losses: Mapped[int] = mapped_column(Integer, default=0)
    stopped_for_day: Mapped[date | None] = mapped_column(Date, nullable=True)  # intraday giveback stop
    day_peak_profit: Mapped[float] = mapped_column(Float, default=0.0)
    day_peak_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(ISTDateTime)


class StrategyState(Base):
    __tablename__ = "strategy_states"
    __table_args__ = (UniqueConstraint("batch_id", "strategy", "mode", name="uq_strategy_state"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(_str())
    auto_disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    disabled_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    disabled_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    expectancy_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    kelly_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(ISTDateTime)


class ControlCommand(Base):
    """Dashboard -> engine commands (section 14.2). The dashboard never calls the engine directly."""

    __tablename__ = "control_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    command_type: Mapped[str] = mapped_column(String(40))  # CommandType
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(_str(), index=True)  # CommandStatus
    requested_by: Mapped[str] = mapped_column(String(64))
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    picked_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class EngineStatus(Base):
    """Single-row engine heartbeat/status (id = 1) written by the engine, read by the dashboard."""

    __tablename__ = "engine_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state: Mapped[str] = mapped_column(_str())  # EngineState
    heartbeat_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    activity: Mapped[str | None] = mapped_column(Text, nullable=True)
    next_cycle_at: Mapped[datetime | None] = mapped_column(ISTDateTime, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settings_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_feed_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    llm_ok: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    market_open: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    regime_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


class MarketEventRow(Base):
    """Cached corporate/macro event calendar (results, ex-dates, RBI, Fed ...)."""

    __tablename__ = "market_events"
    __table_args__ = (UniqueConstraint("symbol", "event_type", "event_date", name="uq_market_event"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(_str())
    event_date: Mapped[date] = mapped_column(Date, index=True)
    title: Mapped[str] = mapped_column(Text, default="")
    ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="unknown")
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(ISTDateTime)


# =========================================================================== audit


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class LoginAttempt(Base):
    __tablename__ = "login_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    username: Mapped[str] = mapped_column(String(64))
    success: Mapped[bool] = mapped_column(Boolean)
    reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    client: Mapped[str | None] = mapped_column(String(200), nullable=True)


class ErrorLog(Base):
    __tablename__ = "errors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    component: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="ERROR")
    message: Mapped[str] = mapped_column(Text)
    traceback: Mapped[str | None] = mapped_column(Text, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class NotificationLog(Base):
    __tablename__ = "notifications_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(ISTDateTime, index=True)
    email_type: Mapped[str] = mapped_column(String(32))
    recipients: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))  # SENT | FAILED | SKIPPED
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    batch_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
