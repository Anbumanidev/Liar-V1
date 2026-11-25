"""Point-in-time inputs for the deterministic risk manager.

``RiskState`` is a plain snapshot so ``RiskManager.evaluate`` stays a pure function (easy to
test, replay-safe). ``build_risk_state`` assembles it from the database, the capital manager
and (optionally) the market data service.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.clock import IST, Clock, to_ist
from core.enums import BatchStatus, EngineState, Mode, Timeframe
from core.types import CapitalSnapshot, OpenPositionView
from paper_engine.views import open_position_views
from storage.models import Batch, EngineStatus, ModeState, Position, StrategyState, Trade


class ModeStateView(BaseModel):
    """Runtime state of one mode (pauses / day stops / loss streak)."""

    model_config = ConfigDict(extra="forbid")

    paused: bool = False                  # owner pause from the dashboard
    paused_until: datetime | None = None  # automatic pause (consecutive losses)
    pause_reason: str | None = None
    stopped_for_day: date | None = None   # intraday profit give-back stop
    consecutive_losses: int = 0


class RiskState(BaseModel):
    """Everything the risk manager needs besides the proposal itself."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    now: datetime
    batch_id: int
    batch_status: BatchStatus = BatchStatus.ACTIVE
    engine_paused: bool = False
    market_open: bool = True
    capital: CapitalSnapshot
    open_positions: list[OpenPositionView] = Field(default_factory=list)
    trades_today: dict[Mode, int] = Field(default_factory=dict)        # entries opened today
    realized_today: dict[Mode, float] = Field(default_factory=dict)    # net realized per mode
    realized_today_total: float = 0.0
    intraday_open_pnl: float = 0.0
    mode_states: dict[Mode, ModeStateView] = Field(default_factory=dict)
    disabled_strategies: set[str] = Field(default_factory=set)         # "name" or "name:MODE"
    kelly_fractions: dict[str, float] = Field(default_factory=dict)    # "name:MODE" -> fraction
    correlations: dict[str, float] = Field(default_factory=dict)       # open symbol -> corr with candidate
    quote_age_seconds: float | None = None
    atr: float | None = None                                           # fallback ATR (manual trades)
    last_sl_exit: dict[str, datetime] = Field(default_factory=dict)    # "SYMBOL:MODE" -> last stop-loss exit

    def mode_state(self, mode: Mode) -> ModeStateView:
        return self.mode_states.get(Mode(mode), ModeStateView())

    def strategy_disabled(self, strategy: str | None, mode: Mode) -> bool:
        if not strategy:
            return False
        return strategy in self.disabled_strategies or f"{strategy}:{Mode(mode).value}" in self.disabled_strategies


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time(0, 0), tzinfo=IST)
    return start, start + timedelta(days=1)


def load_mode_states(session: Session, batch_id: int) -> dict[Mode, ModeStateView]:
    """Mode state rows of a batch as views."""
    out: dict[Mode, ModeStateView] = {}
    for row in session.execute(select(ModeState).where(ModeState.batch_id == batch_id)).scalars():
        try:
            mode = Mode(row.mode)
        except ValueError:
            continue
        out[mode] = ModeStateView(
            paused=bool(row.paused),
            paused_until=row.paused_until,
            pause_reason=row.pause_reason,
            stopped_for_day=row.stopped_for_day,
            consecutive_losses=row.consecutive_losses or 0,
        )
    return out


def trades_opened_today(session: Session, batch_id: int, day: date) -> dict[Mode, int]:
    """Number of positions opened on ``day`` per mode (closed ones included)."""
    start, end = _day_bounds(day)
    rows = session.execute(
        select(Position.mode, func.count(Position.id))
        .where(Position.batch_id == batch_id, Position.opened_at >= start, Position.opened_at < end)
        .group_by(Position.mode)
    ).all()
    return {Mode(m): int(n) for m, n in rows}


def realized_by_mode(session: Session, batch_id: int, day: date) -> dict[Mode, float]:
    """Net realized P&L of trades exited on ``day`` per mode."""
    start, end = _day_bounds(day)
    rows = session.execute(
        select(Trade.mode, func.coalesce(func.sum(Trade.net_pnl), 0.0))
        .where(Trade.batch_id == batch_id, Trade.exit_time >= start, Trade.exit_time < end)
        .group_by(Trade.mode)
    ).all()
    return {Mode(m): float(v) for m, v in rows}


def last_stop_exits(session: Session, batch_id: int, since: datetime) -> dict[str, datetime]:
    """Most recent stop-loss exit per ``SYMBOL:MODE`` since ``since`` (re-entry cooldown)."""
    rows = session.execute(
        select(Trade.symbol, Trade.mode, func.max(Trade.exit_time))
        .where(Trade.batch_id == batch_id, Trade.exit_time >= since, Trade.exit_reason.in_(("SL", "GAP")))
        .group_by(Trade.symbol, Trade.mode)
    ).all()
    return {f"{sym}:{mode}": to_ist(ts) for sym, mode, ts in rows if ts is not None}


def disabled_strategy_keys(session: Session, batch_id: int) -> tuple[set[str], dict[str, float]]:
    """Auto-disabled strategies (``name:MODE``) and stored Kelly fractions."""
    disabled: set[str] = set()
    kelly: dict[str, float] = {}
    for row in session.execute(select(StrategyState).where(StrategyState.batch_id == batch_id)).scalars():
        key = f"{row.strategy}:{row.mode}"
        if row.auto_disabled:
            disabled.add(key)
        if row.kelly_fraction is not None:
            kelly[key] = float(row.kelly_fraction)
    return disabled, kelly


def engine_is_paused(session: Session) -> bool:
    """True when the engine status row says PAUSED (owner pause / kill switch)."""
    row = session.get(EngineStatus, 1)
    return bool(row is not None and row.state == EngineState.PAUSED.value)


def build_risk_state(
    session: Session,
    *,
    batch_id: int,
    capital_manager: Any,
    clock: Clock,
    calendar: Any | None = None,
    data: Any | None = None,
    symbol: str | None = None,
    correlation_lookback: int = 20,
    ltps: dict[str, float] | None = None,
    engine_paused: bool | None = None,
) -> RiskState:
    """Assemble a ``RiskState`` for ``batch_id`` as of ``clock.now()``.

    Args:
        capital_manager: ``capital.accounts.CapitalManager``.
        calendar: ``data.calendar.NSECalendar`` (market-open check). None -> assume open.
        data: ``data.service.MarketDataService`` for correlations / quote age (optional).
        symbol: candidate symbol (correlations vs open positions, quote age).
        ltps: live prices by symbol for MTM (defaults to positions.last_price).
        engine_paused: override; None -> read ``engine_status``.
    """
    now = to_ist(clock.now())
    today = now.date()
    batch = session.get(Batch, batch_id)
    status = BatchStatus(batch.status) if batch is not None else BatchStatus.CLOSED
    views = open_position_views(session, batch_id, ltps)
    realized = realized_by_mode(session, batch_id, today)
    realized_total = sum(realized.values())
    capital: CapitalSnapshot = capital_manager.snapshot(session, batch_id, views, realized_total, update_hwm=False)
    disabled, kelly = disabled_strategy_keys(session, batch_id)
    intraday_open = sum(v.unrealized_gross - v.entry_charges for v in views if v.mode == Mode.INTRADAY)

    correlations: dict[str, float] = {}
    quote_age: float | None = None
    if symbol and data is not None:
        sym = symbol.strip().upper()
        try:
            from features.correlation import return_correlation

            cand = data.candles(sym, Timeframe.D1, lookback_bars=correlation_lookback + 5)
            if not cand.empty:
                for v in views:
                    if v.symbol == sym:
                        continue
                    other = data.candles(v.symbol, Timeframe.D1, lookback_bars=correlation_lookback + 5)
                    if other.empty:
                        continue
                    corr = return_correlation(cand["close"], other["close"], correlation_lookback)
                    if corr is not None:
                        correlations[v.symbol] = corr
        except Exception as exc:  # noqa: BLE001 - correlation is advisory input; log and continue
            logger.warning(f"[risk] correlation computation failed for {sym}: {exc}")
        try:
            quote = data.quote(sym)
            if quote is not None:
                quote_age = max(0.0, (now - to_ist(quote.timestamp)).total_seconds())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[risk] quote lookup failed for {sym}: {exc}")

    market_open = True if calendar is None else bool(calendar.is_market_open(now))
    return RiskState(
        now=now,
        batch_id=batch_id,
        batch_status=status,
        engine_paused=engine_is_paused(session) if engine_paused is None else engine_paused,
        market_open=market_open,
        capital=capital,
        open_positions=views,
        trades_today=trades_opened_today(session, batch_id, today),
        realized_today=realized,
        realized_today_total=realized_total,
        intraday_open_pnl=intraday_open,
        mode_states=load_mode_states(session, batch_id),
        disabled_strategies=disabled,
        kelly_fractions=kelly,
        correlations=correlations,
        quote_age_seconds=quote_age,
        last_sl_exit=last_stop_exits(session, batch_id, now - timedelta(days=7)),
    )
