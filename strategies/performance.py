"""Strategy self-evaluation (section 7.6).

* Per (strategy, mode) metrics computed from the batch's ``trades`` table (partial-exit legs
  merged per position; MANUAL trades excluded - they are the owner's, not the strategy's).
* Auto-disable: once a strategy has ``settings.evaluation.auto_disable_min_trades`` trades
  and a negative expectancy, ``strategy_states.auto_disabled`` is set with a reason. The owner
  re-enables it from the dashboard (:func:`reenable_strategy`); after a re-enable only trades
  closed AFTER the re-enable count towards the next auto-disable decision, so the strategy is
  not immediately disabled again on its old history.
* Optional fractional Kelly (``evaluation.kelly_enabled``): once a strategy has
  ``kelly_min_trades`` trades, ``kelly = fraction x (W - (1 - W) / R)`` floored at 0, where R
  = average win / average loss (in R multiples, rupees as fallback). Stored in
  ``strategy_states.kelly_fraction``; always capped later by the risk manager.

The re-enable timestamp is stored in ``disabled_reason`` as ``"REENABLED@<iso>|<actor>"``
because ``strategy_states`` has no dedicated column (see the report's contract request).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from analytics.metrics import (
    avg_r,
    expectancy,
    kelly_fraction,
    max_drawdown,
    positions_from_trades,
    profit_factor,
    trades_to_df,
    win_rate,
)
from config.schema import Settings
from core.clock import Clock, to_ist
from core.enums import Mode, TradeSource
from core.logging import log_event
from storage.audit import audit, record_error
from storage.models import StrategyState, Trade

REENABLED_PREFIX = "REENABLED@"


@dataclass
class StrategyPerformance:
    """Metrics of one strategy in one mode (positions, not legs)."""

    strategy: str
    mode: str
    trades: int
    wins: int
    losses: int
    win_rate: float | None          # fraction 0..1
    avg_r: float | None
    expectancy_r: float | None
    expectancy_rs: float | None
    profit_factor: float | None
    max_drawdown: float
    max_drawdown_pct: float
    avg_win_r: float | None
    avg_loss_r: float | None        # positive number
    net_pnl: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# =========================================================================== metrics


def load_trades_df(session: Session, batch_id: int) -> pd.DataFrame:
    """All trade legs of ``batch_id`` as a DataFrame (see :func:`analytics.metrics.trades_to_df`)."""
    rows = session.scalars(select(Trade).where(Trade.batch_id == batch_id).order_by(Trade.exit_time, Trade.id)).all()
    return trades_to_df(rows)


def performance_of(strategy: str, mode: str, positions: pd.DataFrame) -> StrategyPerformance:
    """Metrics for one group of merged positions."""
    pnl = positions["net_pnl"] if not positions.empty else pd.Series(dtype=float)
    rs = positions["r_multiple"] if not positions.empty else pd.Series(dtype=float)
    rs_clean = rs.dropna()
    wins_r = rs_clean[rs_clean > 0]
    losses_r = rs_clean[rs_clean < 0]
    eq = pnl.fillna(0.0).cumsum()
    dd_abs, dd_pct = max_drawdown(pd.concat([pd.Series([0.0]), eq], ignore_index=True))
    return StrategyPerformance(
        strategy=strategy,
        mode=mode,
        trades=int(len(positions)),
        wins=int((pnl > 0).sum()),
        losses=int((pnl < 0).sum()),
        win_rate=win_rate(pnl),
        avg_r=avg_r(rs),
        expectancy_r=expectancy(rs),
        expectancy_rs=expectancy(pnl),
        profit_factor=profit_factor(pnl),
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        avg_win_r=float(wins_r.mean()) if len(wins_r) else None,
        avg_loss_r=float(-losses_r.mean()) if len(losses_r) else None,
        net_pnl=float(pnl.fillna(0).sum()),
    )


def strategy_positions(trades: pd.DataFrame) -> pd.DataFrame:
    """Merged positions eligible for strategy evaluation (strategy set, not MANUAL)."""
    df = trades_to_df(trades)
    if df.empty:
        return df
    df = df[df["strategy"].notna() & (df["source"] != TradeSource.MANUAL.value)]
    return positions_from_trades(df)


def compute_strategy_performance(trades: pd.DataFrame) -> list[StrategyPerformance]:
    """Per (strategy, mode) metrics, sorted by strategy then mode."""
    pos = strategy_positions(trades)
    if pos.empty:
        return []
    out: list[StrategyPerformance] = []
    for (strat, mode), g in pos.groupby(["strategy", "mode"], sort=True):
        out.append(performance_of(str(strat), str(mode), g))
    return out


def _is_negative(perf: StrategyPerformance) -> bool:
    """Negative expectancy: in R when most trades carry an R multiple, else in rupees."""
    if perf.expectancy_r is not None and perf.avg_r is not None:
        return perf.expectancy_r < 0
    return perf.expectancy_rs is not None and perf.expectancy_rs < 0


def _kelly(perf: StrategyPerformance, positions: pd.DataFrame, fraction: float) -> float:
    if perf.win_rate is None:
        return 0.0
    if perf.avg_win_r and perf.avg_loss_r:
        payoff = perf.avg_win_r / perf.avg_loss_r
        w = perf.win_rate
    else:
        pnl = positions["net_pnl"].dropna()
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        if wins.empty:
            return 0.0
        if losses.empty:
            return fraction  # no losses yet: full fraction (still capped by the risk manager)
        payoff = float(wins.mean() / -losses.mean())
        w = perf.win_rate
    return round(kelly_fraction(w, payoff, fraction), 6)


def _reenabled_at(state: StrategyState | None) -> datetime | None:
    if state is None or not state.disabled_reason or not state.disabled_reason.startswith(REENABLED_PREFIX):
        return None
    stamp = state.disabled_reason[len(REENABLED_PREFIX):].split("|", 1)[0]
    try:
        return to_ist(datetime.fromisoformat(stamp))
    except ValueError:
        logger.warning(f"unparseable re-enable marker on strategy_state {state.id}: {state.disabled_reason!r}")
        return None


# =========================================================================== state upserts


def get_state(session: Session, batch_id: int, strategy: str, mode: Mode | str) -> StrategyState | None:
    """The ``strategy_states`` row for (batch, strategy, mode), if any."""
    return session.scalars(
        select(StrategyState).where(
            StrategyState.batch_id == batch_id,
            StrategyState.strategy == strategy,
            StrategyState.mode == str(Mode(mode).value),
        )
    ).first()


def _upsert(session: Session, batch_id: int, strategy: str, mode: str, now: datetime) -> StrategyState:
    st = get_state(session, batch_id, strategy, mode)
    if st is None:
        st = StrategyState(batch_id=batch_id, strategy=strategy, mode=mode, auto_disabled=False,
                           trades_count=0, updated_at=now)
        session.add(st)
    return st


def evaluate_strategies(session: Session, batch_id: int, settings: Settings, clock: Clock) -> list[StrategyState]:
    """Recompute every strategy's metrics for ``batch_id`` and apply auto-disable / Kelly.

    Upserts one ``strategy_states`` row per (strategy, mode) that has trades (flushes, the
    caller commits). Never raises on bad data: a failure is recorded and the rows updated so
    far are returned.
    """
    now = clock.now()
    ev = settings.evaluation
    out: list[StrategyState] = []
    try:
        trades = load_trades_df(session, batch_id)
        pos_all = strategy_positions(trades)
        if pos_all.empty:
            return out
        for (strat, mode), g in pos_all.groupby(["strategy", "mode"], sort=True):
            strat, mode = str(strat), str(mode)
            perf = performance_of(strat, mode, g)
            st = _upsert(session, batch_id, strat, mode, now)
            st.trades_count = perf.trades
            st.expectancy_r = perf.expectancy_r if perf.expectancy_r is not None else None
            st.updated_at = now
            if ev.kelly_enabled and perf.trades >= ev.kelly_min_trades:
                st.kelly_fraction = _kelly(perf, g, ev.kelly_fraction)
            else:
                st.kelly_fraction = None
            if not st.auto_disabled:
                since = _reenabled_at(st)
                window = g
                if since is not None and g["exit_time"].notna().any():
                    window = g[pd.to_datetime(g["exit_time"]).map(lambda t: to_ist(t.to_pydatetime()) > since)]
                wperf = performance_of(strat, mode, window) if since is not None else perf
                if wperf.trades >= ev.auto_disable_min_trades and _is_negative(wperf):
                    exp_txt = (f"{wperf.expectancy_r:.3f}R" if wperf.expectancy_r is not None
                               else f"Rs {wperf.expectancy_rs:.2f}")
                    st.auto_disabled = True
                    st.disabled_at = now
                    st.disabled_reason = (
                        f"AUTO: expectancy {exp_txt} < 0 after {wperf.trades} trades "
                        f"(min {ev.auto_disable_min_trades})"
                    )
                    audit(session, "SYSTEM", "STRATEGY_AUTO_DISABLED", f"{strat}:{mode}",
                          {"metrics": wperf.as_dict()}, batch_id=batch_id, clock=clock)
                    logger.warning(f"strategy {strat} ({mode}) auto-disabled in batch {batch_id}: {st.disabled_reason}")
            log_event("strategy_eval", {"batch_id": batch_id, **_jsonable(perf.as_dict()),
                                        "auto_disabled": st.auto_disabled, "kelly_fraction": st.kelly_fraction},
                      clock=clock)
            out.append(st)
        session.flush()
    except Exception as exc:
        record_error(session, "strategies.performance", exc, {"batch_id": batch_id}, batch_id=batch_id, clock=clock)
    return out


def _jsonable(d: dict[str, object]) -> dict[str, object]:
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def reenable_strategy(
    session: Session, batch_id: int, strategy: str, mode: Mode | str, actor: str, clock: Clock
) -> StrategyState:
    """Owner re-enable (dashboard REENABLE_STRATEGY). Creates the row if missing.

    Only trades closed after this moment count towards the next auto-disable decision.
    """
    now = clock.now()
    m = str(Mode(mode).value)
    st = _upsert(session, batch_id, strategy, m, now)
    st.auto_disabled = False
    st.disabled_at = None
    st.disabled_reason = f"{REENABLED_PREFIX}{now.isoformat()}|{actor}"
    st.updated_at = now
    audit(session, actor, "STRATEGY_REENABLED", f"{strategy}:{m}", {}, batch_id=batch_id, clock=clock)
    session.flush()
    return st


def disabled_strategies(session: Session, batch_id: int) -> set[str]:
    """Auto-disabled strategies of the batch as ``"<name>:<MODE>"`` keys (registry format)."""
    rows = session.scalars(
        select(StrategyState).where(StrategyState.batch_id == batch_id, StrategyState.auto_disabled.is_(True))
    ).all()
    return {f"{r.strategy}:{r.mode}" for r in rows}


def kelly_fractions(session: Session, batch_id: int) -> dict[str, float]:
    """``{"<name>:<MODE>": kelly_fraction}`` for rows that have one."""
    rows = session.scalars(
        select(StrategyState).where(StrategyState.batch_id == batch_id, StrategyState.kelly_fraction.is_not(None))
    ).all()
    return {f"{r.strategy}:{r.mode}": float(r.kelly_fraction) for r in rows if r.kelly_fraction is not None}
