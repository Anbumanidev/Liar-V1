"""Read-only data layer of the analytics pages (section 14.4).

Every public function takes a SQLAlchemy ``Session`` (plus ``batch_id`` / options), or a
DataFrame produced by one, and returns a DataFrame or a dict - no Streamlit, no writes - so
each one is unit testable on an in-memory DB.

Rules:

* ``batch_id=None`` -> empty result. Every analytics page filters by the sidebar batch.
* Datetimes in returned frames are naive IST wall time (the DB stores IST; dropping the tz
  keeps tables and plotly axes in IST without any conversion).
* Capital semantics (sections 6 and 8): TRADING = trading_cash + blocked; TOTAL EQUITY =
  TRADING + unrealized (MTM of open positions from ``positions.last_price``, net of accrued
  entry charges) + RESERVE. Floors: absolute = starting_capital x (1 - hard_stop%), trailing
  = HWM x (1 - hard_stop%); the stricter (higher) one wins. Tiers come from the drawdown off
  the HWM at ``risk.drawdown.caution_pct`` / ``defensive_pct`` / ``hard_stop_pct``; HALT also
  when equity <= the stricter floor (same math as ``capital.protection``, which the dashboard
  does not import).
* Trade metrics reuse :mod:`analytics.metrics` (partial-exit legs merged per position).
* Every number is deterministic Python - no LLM ever computes one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from analytics import metrics as mx
from config.schema import Settings
from config.store import diff_settings
from core.clock import IST
from core.enums import (
    ApprovalStatus,
    BatchStatus,
    Direction,
    DrawdownTier,
    LedgerEntryType,
    Mode,
    ShadowOutcome,
    TradeSource,
)
from dashboard import services as svc
from modes.base import is_overnight
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    AuditLog,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    ControlCommand,
    DailySummary,
    DebateRow,
    DecisionRow,
    Order,
    Position,
    Run,
    SettingsVersion,
    ShadowTrade,
    StockAnalyzed,
    Trade,
)

#: Rupee tolerance for floor comparisons (same as capital.protection).
_EPS = 1e-9
#: Time stamped on daily_summary (EOD) points.
EOD_TIME = time(15, 30)
#: Label for trades without a strategy (manual trades).
NO_STRATEGY = "(none)"


# =========================================================================== small helpers


def naive_ist(value: datetime | None) -> datetime | None:
    """IST wall-clock time without tzinfo (``None`` stays ``None``)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(IST).replace(tzinfo=None)


def _day_start(day: date) -> datetime:
    """Midnight IST (aware) - comparable with ``ISTDateTime`` columns in queries."""
    return datetime.combine(day, time.min, tzinfo=IST)


def date_bounds(start: date | None, end: date | None) -> tuple[datetime | None, datetime | None]:
    """Inclusive date range -> ``[start 00:00, end + 1 day 00:00)`` aware IST bounds."""
    return (_day_start(start) if start else None, _day_start(end + timedelta(days=1)) if end else None)


def date_mask(ts: pd.Series, start: date | None, end: date | None) -> pd.Series:
    """Boolean mask of ``ts`` (naive datetimes) inside the inclusive date range."""
    if start is None and end is None:
        return pd.Series(True, index=ts.index)
    s = pd.to_datetime(ts, errors="coerce")
    mask = s.notna()
    if start is not None:
        mask &= s >= pd.Timestamp(start)
    if end is not None:
        mask &= s < pd.Timestamp(end + timedelta(days=1))
    return mask


def normalize_date_range(value: Any) -> tuple[date | None, date | None]:
    """``st.date_input`` range value -> ``(start, end)``; a half-picked range is open-ended."""
    if isinstance(value, date):
        return value, value
    if isinstance(value, (list, tuple)):
        days = [v for v in value if isinstance(v, date)]
        if len(days) >= 2:
            return min(days[0], days[1]), max(days[0], days[1])
        if days:
            return days[0], None
    return None, None


def _num(value: Any) -> float | None:
    """Float or ``None`` (``None`` / NaN / non-numeric -> ``None``)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def jsonish(value: Any, limit: int | None = None) -> str:
    """Compact one-line text of a JSON-like value (for table cells)."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 3)] + "..."
    return text


def round_frame(df: pd.DataFrame, nd: int = 2) -> pd.DataFrame:
    """Round every float column (display helper)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(nd)
    return out


def _plain(row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Row dict with naive-IST datetimes."""
    if row is None:
        return None
    return {k: naive_ist(v) if isinstance(v, datetime) else v for k, v in row.items()}


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _values(obj: Any, columns: Sequence[str]) -> dict[str, Any]:
    """``{column: value}`` of an ORM row with naive-IST datetimes."""
    out: dict[str, Any] = {}
    for c in columns:
        v = getattr(obj, c)
        out[c] = naive_ist(v) if isinstance(v, datetime) else v
    return out


def _mode_is_overnight(mode: str | None) -> bool:
    try:
        return is_overnight(Mode(str(mode)))
    except ValueError:
        return False


# =========================================================================== capital protection math


@dataclass(frozen=True)
class RiskParams:
    """The section-8 parameters the analytics need (from the active settings)."""

    hard_stop_pct: float = 9.0
    caution_pct: float = 3.0
    defensive_pct: float = 6.0
    gap_buffer_pct: float = 3.0
    loss_budget_divisor: float = 3.0

    @classmethod
    def from_settings(cls, settings: Settings | None) -> RiskParams:
        """From ``settings.risk`` (defaults when ``settings`` is None)."""
        if settings is None:
            return cls()
        risk = settings.risk
        return cls(
            hard_stop_pct=float(risk.hard_stop_pct),
            caution_pct=float(risk.drawdown.caution_pct),
            defensive_pct=float(risk.drawdown.defensive_pct),
            gap_buffer_pct=float(risk.gap_buffer_pct),
            loss_budget_divisor=float(risk.loss_budget_divisor),
        )


def floors(starting_capital: float, hwm: float, hard_stop_pct: float) -> tuple[float, float]:
    """``(absolute_floor, trailing_floor)``."""
    keep = 1.0 - hard_stop_pct / 100.0
    return starting_capital * keep, hwm * keep


def drawdown_pct(equity: float, hwm: float) -> float:
    """Drawdown off the high-water mark in percent (never negative; 0 when hwm <= 0)."""
    if hwm is None or hwm <= 0:
        return 0.0
    return max(0.0, (hwm - equity) / hwm * 100.0)


def tier_for(dd_pct: float, equity: float, effective_floor: float | None, params: RiskParams) -> str:
    """NORMAL / CAUTION / DEFENSIVE / HALT (HALT also when equity has reached the floor)."""
    if (effective_floor is not None and equity <= effective_floor + _EPS) or dd_pct >= params.hard_stop_pct - _EPS:
        return DrawdownTier.HALT.value
    if dd_pct >= params.defensive_pct - _EPS:
        return DrawdownTier.DEFENSIVE.value
    if dd_pct >= params.caution_pct - _EPS:
        return DrawdownTier.CAUTION.value
    return DrawdownTier.NORMAL.value


def position_risk(
    direction: str,
    qty: int,
    avg_entry: float,
    stop_loss: float,
    ltp: float | None = None,
    entry_charges: float = 0.0,
    *,
    overnight: bool = False,
    gap_buffer_pct: float = 0.0,
) -> float:
    """Rupees an open position can still take off TOTAL EQUITY (mirrors capital.protection).

    Worst of the stop risk measured from the entry and from the last price (an MTM gain can be
    given back), gap-adjusted for overnight positions, plus accrued entry charges. The engine
    also adds estimated exit charges, so its budget can be a few rupees smaller.
    """
    if qty <= 0:
        return 0.0
    gap = gap_buffer_pct / 100.0 if overnight else 0.0
    refs = [avg_entry] + ([ltp] if ltp is not None and ltp > 0 else [])
    worst = 0.0
    for ref in refs:
        if direction == Direction.LONG.value:
            stop_risk = max(0.0, (ref - stop_loss) * qty)
        else:
            stop_risk = max(0.0, (stop_loss - ref) * qty)
        worst = max(worst, stop_risk, qty * ref * gap)
    return worst + max(0.0, entry_charges or 0.0)


# =========================================================================== batches


def batch_info(session: Session, batch_id: int | None) -> dict[str, Any] | None:
    """Batch row as a plain dict (naive datetimes) or None."""
    if batch_id is None:
        return None
    batch = session.get(Batch, batch_id)
    return _plain(svc.row_to_dict(batch)) if batch is not None else None


# =========================================================================== live positions & capital

OPEN_POSITION_COLUMNS: tuple[str, ...] = (
    "id", "symbol", "mode", "product", "direction", "qty", "avg_entry", "last_price", "price_age_s", "stop_loss",
    "target_1", "target_2", "unrealized", "unrealized_pct", "r_now", "risk_to_sl", "opened_at", "holding_days",
    "strategy", "margin_blocked", "overnight",
)


def _open_positions(session: Session, batch_id: int | None) -> list[Position]:
    return [] if batch_id is None else svc.open_positions(session, batch_id)


def _is_overnight_position(p: Position) -> bool:
    return bool(p.overnight) or _mode_is_overnight(p.mode)


def open_positions_frame(
    session: Session, batch_id: int | None, *, now: datetime | None = None, params: RiskParams | None = None
) -> pd.DataFrame:
    """Open positions with live price, unrealized P&L (net of entry charges), R now and risk to SL."""
    params = params or RiskParams()
    rows: list[dict[str, Any]] = []
    for p in _open_positions(session, batch_id):
        upnl = svc.unrealized_pnl(p)
        sign = 1.0 if p.direction == Direction.LONG.value else -1.0
        gross = None if p.last_price is None else (p.last_price - p.avg_entry) * p.qty * sign
        init_risk = abs(p.avg_entry - (p.initial_stop_loss or p.stop_loss)) * p.qty
        cost = p.avg_entry * p.qty
        age = (now - p.last_price_at).total_seconds() if now is not None and p.last_price_at is not None else None
        rows.append({
            "id": p.id, "symbol": p.symbol, "mode": p.mode, "product": p.product, "direction": p.direction,
            "qty": p.qty, "avg_entry": p.avg_entry, "last_price": p.last_price,
            "price_age_s": None if age is None else round(age),
            "stop_loss": p.stop_loss, "target_1": p.target_1, "target_2": p.target_2, "unrealized": upnl,
            "unrealized_pct": round(upnl / cost * 100.0, 2) if upnl is not None and cost > 0 else None,
            "r_now": round(gross / init_risk, 2) if gross is not None and init_risk > 0 else None,
            "risk_to_sl": round(position_risk(p.direction, p.qty, p.avg_entry, p.stop_loss, p.last_price,
                                              p.entry_charges, overnight=_is_overnight_position(p),
                                              gap_buffer_pct=params.gap_buffer_pct), 2),
            "opened_at": naive_ist(p.opened_at), "holding_days": p.holding_days, "strategy": p.strategy,
            "margin_blocked": p.margin_blocked, "overnight": _is_overnight_position(p),
        })
    return pd.DataFrame(rows, columns=list(OPEN_POSITION_COLUMNS))


def capital_snapshot(
    session: Session, batch_id: int | None, params: RiskParams | None = None
) -> dict[str, Any] | None:
    """Live capital picture of a batch (None when it has no capital account).

    Keys: starting_capital, trading_cash, blocked, trading, reserve, reserve_peak, unrealized,
    equity, hwm, drawdown_pct, tier (computed live), engine_tier (stored), halted, abs_floor,
    trailing_floor, effective_floor, open_risk, loss_budget, max_new_trade_risk,
    budget_left_pct, sizing_capital, open_positions, unpriced_positions, net_since_start,
    return_pct, updated_at.
    """
    params = params or RiskParams()
    acct = session.get(CapitalAccount, batch_id) if batch_id is not None else None
    if acct is None:
        return None
    positions = _open_positions(session, batch_id)
    unrealized = 0.0
    unpriced = 0
    open_risk = 0.0
    for p in positions:
        upnl = svc.unrealized_pnl(p)
        if upnl is None:
            unpriced += 1
        else:
            unrealized += upnl
        open_risk += position_risk(p.direction, p.qty, p.avg_entry, p.stop_loss, p.last_price, p.entry_charges,
                                   overnight=_is_overnight_position(p), gap_buffer_pct=params.gap_buffer_pct)
    trading = float(acct.trading_cash) + float(acct.blocked)
    reserve = float(acct.reserve)
    equity = trading + unrealized + reserve
    hwm = max(float(acct.hwm), equity)
    abs_floor, trail_floor = floors(float(acct.starting_capital), hwm, params.hard_stop_pct)
    eff = max(abs_floor, trail_floor)
    dd = drawdown_pct(equity, hwm)
    tier = DrawdownTier.HALT.value if acct.halted else tier_for(dd, equity, eff, params)
    budget = max(0.0, equity - open_risk - eff)
    cushion = max(0.0, equity - eff)
    start = float(acct.starting_capital)
    return {
        "batch_id": batch_id,
        "starting_capital": start,
        "trading_cash": float(acct.trading_cash),
        "blocked": float(acct.blocked),
        "trading": trading,
        "reserve": reserve,
        "reserve_peak": float(acct.reserve_peak),
        "unrealized": round(unrealized, 2),
        "equity": round(equity, 2),
        "hwm": round(hwm, 2),
        "drawdown_pct": round(dd, 3),
        "tier": tier,
        "engine_tier": acct.tier,
        "halted": bool(acct.halted),
        "halt_reason": acct.halt_reason,
        "abs_floor": round(abs_floor, 2),
        "trailing_floor": round(trail_floor, 2),
        "effective_floor": round(eff, 2),
        "open_risk": round(open_risk, 2),
        "loss_budget": round(budget, 2),
        "max_new_trade_risk": round(budget / params.loss_budget_divisor, 2) if params.loss_budget_divisor else 0.0,
        "budget_left_pct": round(budget / cushion * 100.0, 1) if cushion > _EPS else 0.0,
        "sizing_capital": float(acct.sizing_capital),
        "open_positions": len(positions),
        "unpriced_positions": unpriced,
        "net_since_start": round(equity - start, 2),
        "return_pct": round((equity / start - 1.0) * 100.0, 3) if start > 0 else None,
        "updated_at": naive_ist(acct.updated_at),
    }


def live_point(snapshot: dict[str, Any] | None, now: datetime) -> dict[str, Any] | None:
    """Equity-series point for 'now' built from :func:`capital_snapshot`."""
    if not snapshot:
        return None
    return {"ts": naive_ist(now), "trading": snapshot["trading"], "reserve": snapshot["reserve"],
            "unrealized": snapshot["unrealized"], "equity": snapshot["equity"], "hwm": snapshot["hwm"]}


def regime_rows(regime: dict[str, Any] | None) -> pd.DataFrame:
    """Flatten ``engine_status.regime_json`` into ``scope / regime / detail`` rows."""
    cols = ["scope", "regime", "detail"]
    if not regime:
        return _empty(cols)
    rows: list[dict[str, Any]] = []
    scalars = {k: v for k, v in regime.items() if not isinstance(v, (dict, list))}
    if "regime" in scalars:
        rest = {k: v for k, v in scalars.items() if k != "regime"}
        rows.append({"scope": "market", "regime": str(scalars["regime"]), "detail": jsonish(rest, 200) if rest else ""})
        scalars = {}
    for key, value in regime.items():
        if isinstance(value, dict):
            label = value.get("regime") or value.get("label") or value.get("state") or ""
            rest = {k: v for k, v in value.items() if k not in ("regime", "label", "state")}
            rows.append({"scope": str(key), "regime": str(label), "detail": jsonish(rest, 200) if rest else ""})
        elif isinstance(value, list):
            rows.append({"scope": str(key), "regime": "", "detail": jsonish(value, 200)})
        elif key in scalars:
            rows.append({"scope": str(key), "regime": str(value), "detail": ""})
    return pd.DataFrame(rows, columns=cols)


# =========================================================================== trades

TRADE_FRAME_COLUMNS: tuple[str, ...] = (
    "id", "batch_id", "position_id", "symbol", "mode", "product", "direction", "qty", "entry_time", "entry_price",
    "exit_time", "exit_price", "stop_loss", "target_1", "target_2", "exit_reason", "is_partial", "gross_pnl",
    "charges", "net_pnl", "r_multiple", "source", "strategy", "regime", "grade", "holding_minutes",
    "holding_days", "decision_id", "approval_id",
)


def trades_frame(session: Session, batch_id: int | None) -> pd.DataFrame:
    """Every closed leg of the batch (oldest exit first). Missing regimes come from the decision."""
    cols = list(TRADE_FRAME_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    rows = session.execute(
        select(Trade).where(Trade.batch_id == batch_id).order_by(Trade.exit_time, Trade.id)
    ).scalars().all()
    data: list[dict[str, Any]] = []
    for t in rows:
        d = {c: getattr(t, c) for c in cols}
        d["entry_time"] = naive_ist(t.entry_time)
        d["exit_time"] = naive_ist(t.exit_time)
        data.append(d)
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    missing = [int(x) for x, r in zip(df["decision_id"], df["regime"], strict=True)
               if pd.notna(x) and not (isinstance(r, str) and r)]
    if missing:
        regimes = dict(session.execute(
            select(DecisionRow.id, DecisionRow.regime).where(DecisionRow.id.in_(sorted(set(missing))))
        ).all())
        df["regime"] = [r if isinstance(r, str) and r else (regimes.get(int(d)) if pd.notna(d) else None)
                        for r, d in zip(df["regime"], df["decision_id"], strict=True)]
    return df


def trade_filter_options(df: pd.DataFrame) -> dict[str, list[str]]:
    """Distinct modes / strategies / sources present in a trades frame."""
    if df.empty:
        return {"modes": [], "strategies": [], "sources": []}
    return {
        "modes": sorted({str(v) for v in df["mode"].dropna()}),
        "strategies": sorted({str(v) if isinstance(v, str) and v else NO_STRATEGY for v in df["strategy"]}),
        "sources": sorted({str(v) for v in df["source"].dropna()}),
    }


def filter_trades(
    df: pd.DataFrame,
    *,
    modes: Iterable[str] | None = None,
    strategies: Iterable[str] | None = None,
    sources: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Filter a trades frame (empty/None filters = no filter; dates on the exit time)."""
    if df.empty:
        return df
    mask = date_mask(df["exit_time"], start, end)
    if modes:
        mask &= df["mode"].isin(list(modes))
    if strategies:
        labels = df["strategy"].map(lambda v: v if isinstance(v, str) and v else NO_STRATEGY)
        mask &= labels.isin(list(strategies))
    if sources:
        mask &= df["source"].isin(list(sources))
    return df[mask].reset_index(drop=True)


def overview_metrics(trades_df: pd.DataFrame) -> dict[str, Any]:
    """Batch-level trade metrics (positions merged): net/gross/charges, win rate, PF, expectancy."""
    keys = list(mx.SUMMARY_COLUMNS)
    out: dict[str, Any] = dict.fromkeys(keys)
    out.update(trades=0, wins=0, losses=0, gross_pnl=0.0, charges=0.0, net_pnl=0.0)
    summary = mx.summarize(trades_df, by=None)
    if summary.empty:
        return out
    row = summary.iloc[0].to_dict()
    for k in keys:
        v = row.get(k)
        out[k] = int(v) if k in ("trades", "wins", "losses") else _num(v)
    return out


def performance_table(trades_df: pd.DataFrame, by: str | Sequence[str] | None) -> pd.DataFrame:
    """:func:`analytics.metrics.summarize` grouped by ``by`` (rounded for display)."""
    return round_frame(mx.summarize(trades_df, by=by))


DAILY_COLUMNS: tuple[str, ...] = ("day", "exits", "wins", "losses", "gross_pnl", "charges", "net_pnl", "cum_net_pnl")


def daily_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Realized P&L per exit day (one row per day, legs counted as exits)."""
    cols = list(DAILY_COLUMNS)
    if trades_df.empty or trades_df["exit_time"].isna().all():
        return _empty(cols)
    df = trades_df.dropna(subset=["exit_time"]).copy()
    df["day"] = pd.to_datetime(df["exit_time"]).dt.date
    for c in ("gross_pnl", "charges", "net_pnl"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["win"] = (df["net_pnl"] > 0).astype(int)
    df["loss"] = (df["net_pnl"] < 0).astype(int)
    g = df.groupby("day", sort=True)
    out = pd.DataFrame({
        "exits": g.size(), "wins": g["win"].sum(), "losses": g["loss"].sum(),
        "gross_pnl": g["gross_pnl"].sum(), "charges": g["charges"].sum(), "net_pnl": g["net_pnl"].sum(),
    }).reset_index()
    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    return round_frame(out[cols])


# =========================================================================== equity series

EQUITY_COLUMNS: tuple[str, ...] = ("ts", "trading", "reserve", "unrealized", "equity", "hwm", "source")
LEVEL_COLUMNS: tuple[str, ...] = (
    "abs_floor", "trailing_floor", "caution_level", "defensive_level", "drawdown_pct", "tier",
)


def equity_series(session: Session, batch_id: int | None, *, live: dict[str, Any] | None = None) -> pd.DataFrame:
    """TRADING / RESERVE / unrealized / TOTAL EQUITY / HWM over time.

    Source, first available: ``daily_summary`` TOTAL rows (EOD points at 15:30), else the
    capital ledger (balances after each movement, realized only), else starting capital +
    cumulative realized net P&L. A ``START`` point (batch start, starting capital) is prepended
    and ``live`` (see :func:`live_point`) appended. ``hwm`` is the running max of the recorded
    HWM and equity.
    """
    cols = list(EQUITY_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    batch = session.get(Batch, batch_id)
    acct = session.get(CapitalAccount, batch_id)
    starting = float(acct.starting_capital) if acct else (float(batch.starting_capital) if batch else None)
    start_ts = naive_ist((batch.started_at or batch.created_at) if batch else None)
    rows: list[dict[str, Any]] = []
    daily = session.execute(
        select(DailySummary).where(DailySummary.batch_id == batch_id, DailySummary.mode == "TOTAL")
        .order_by(DailySummary.day)
    ).scalars().all()
    if daily:
        for r in daily:
            reserve = float(r.reserve or 0.0)
            unreal = float(r.unrealized_pnl or 0.0)
            trading = _num(r.trading_balance)
            if trading is None:
                trading = _num(r.trading_cash)
            equity = _num(r.equity)
            if trading is None and equity is not None:
                trading = equity - reserve - unreal
            if trading is None:
                continue
            if equity is None:
                equity = trading + reserve + unreal
            rows.append({"ts": datetime.combine(r.day, EOD_TIME), "trading": trading, "reserve": reserve,
                         "unrealized": unreal, "equity": equity, "hwm": _num(r.hwm), "source": "EOD"})
    else:
        ledger = session.execute(
            select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id)
            .order_by(CapitalLedgerEntry.ts, CapitalLedgerEntry.id)
        ).scalars().all()
        if ledger:
            for e in ledger:
                trading = float(e.trading_cash_after) + float(e.blocked_after)
                reserve = float(e.reserve_after)
                rows.append({"ts": naive_ist(e.ts), "trading": trading, "reserve": reserve, "unrealized": 0.0,
                             "equity": trading + reserve, "hwm": None, "source": "LEDGER"})
        elif starting is not None:
            running = starting
            for ts, pnl in session.execute(
                select(Trade.exit_time, Trade.net_pnl).where(Trade.batch_id == batch_id)
                .order_by(Trade.exit_time, Trade.id)
            ).all():
                running += float(pnl or 0.0)
                rows.append({"ts": naive_ist(ts), "trading": running, "reserve": 0.0, "unrealized": 0.0,
                             "equity": running, "hwm": None, "source": "TRADES"})
    if starting is not None and start_ts is not None and (not rows or start_ts < min(r["ts"] for r in rows)):
        rows.insert(0, {"ts": start_ts, "trading": starting, "reserve": 0.0, "unrealized": 0.0,
                        "equity": starting, "hwm": starting, "source": "START"})
    if live:
        ts = live["ts"]
        if rows and ts <= max(r["ts"] for r in rows):
            ts = max(r["ts"] for r in rows) + timedelta(minutes=1)
        rows.append({"ts": ts, "trading": live["trading"], "reserve": live["reserve"],
                     "unrealized": live.get("unrealized", 0.0), "equity": live["equity"], "hwm": live.get("hwm"),
                     "source": "LIVE"})
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    for c in ("trading", "reserve", "unrealized", "equity"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    recorded = pd.to_numeric(df["hwm"], errors="coerce").astype(float)
    df["hwm"] = np.fmax(recorded.fillna(df["equity"]), df["equity"]).cummax()
    return df


def with_protection_levels(
    series: pd.DataFrame, starting_capital: float | None, params: RiskParams | None = None
) -> pd.DataFrame:
    """Add floors, tier thresholds, drawdown % and tier to an :func:`equity_series` frame."""
    params = params or RiskParams()
    df = series.copy()
    if df.empty:
        for c in LEVEL_COLUMNS:
            df[c] = pd.Series(dtype=object if c == "tier" else float)
        return df
    hwm = df["hwm"].astype(float)
    keep = 1.0 - params.hard_stop_pct / 100.0
    df["abs_floor"] = float(starting_capital) * keep if starting_capital is not None else np.nan
    df["trailing_floor"] = hwm * keep
    df["caution_level"] = hwm * (1.0 - params.caution_pct / 100.0)
    df["defensive_level"] = hwm * (1.0 - params.defensive_pct / 100.0)
    df["drawdown_pct"] = [round(drawdown_pct(e, h), 3) for e, h in zip(df["equity"], hwm, strict=True)]
    eff = np.fmax(df["abs_floor"].astype(float), df["trailing_floor"])
    df["tier"] = [tier_for(dd, e, None if math.isnan(f) else f, params)
                  for dd, e, f in zip(df["drawdown_pct"], df["equity"], eff, strict=True)]
    return df


# =========================================================================== capital ledger

LEDGER_COLUMNS: tuple[str, ...] = (
    "id", "ts", "entry_type", "amount", "from_account", "to_account", "trading_cash", "blocked", "trading",
    "reserve", "total", "sizing_capital", "cum_locked", "cum_refilled", "cum_manual_transfer", "cum_realized",
    "mode", "position_id", "trade_id", "actor", "confirmed", "note",
)


def ledger_frame(session: Session, batch_id: int | None) -> pd.DataFrame:
    """Every capital_ledger row (oldest first) with running balances.

    ``trading`` = cash + blocked after the movement, ``total`` = trading + reserve; the
    ``cum_*`` columns are running totals of profit locks, refills, manual transfers and
    realized P&L (computed over the whole ledger, so they stay right after filtering).
    """
    cols = list(LEDGER_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    rows = session.execute(
        select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id)
        .order_by(CapitalLedgerEntry.ts, CapitalLedgerEntry.id)
    ).scalars().all()
    data = [{
        "id": e.id, "ts": naive_ist(e.ts), "entry_type": e.entry_type, "amount": float(e.amount),
        "from_account": e.from_account, "to_account": e.to_account, "trading_cash": float(e.trading_cash_after),
        "blocked": float(e.blocked_after), "trading": float(e.trading_cash_after) + float(e.blocked_after),
        "reserve": float(e.reserve_after),
        "total": float(e.trading_cash_after) + float(e.blocked_after) + float(e.reserve_after),
        "sizing_capital": float(e.sizing_capital_after), "mode": e.mode, "position_id": e.position_id,
        "trade_id": e.trade_id, "actor": e.actor, "confirmed": bool(e.confirmed), "note": e.note,
    } for e in rows]
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    amt = df["amount"].astype(float)
    kind = df["entry_type"]
    df["cum_locked"] = amt.abs().where(kind == LedgerEntryType.PROFIT_LOCK.value, 0.0).cumsum()
    df["cum_refilled"] = amt.abs().where(kind == LedgerEntryType.REFILL.value, 0.0).cumsum()
    df["cum_manual_transfer"] = amt.abs().where(kind == LedgerEntryType.MANUAL_TRANSFER.value, 0.0).cumsum()
    df["cum_realized"] = amt.where(kind.isin([LedgerEntryType.REALIZED_PNL.value,
                                              LedgerEntryType.DIVIDEND.value]), 0.0).cumsum()
    return df


def filter_ledger(
    df: pd.DataFrame,
    *,
    entry_types: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    modes: Iterable[str] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Filter a ledger frame (type / inclusive date range / mode / free text in note+actor)."""
    if df.empty:
        return df
    mask = date_mask(df["ts"], start, end)
    if entry_types:
        mask &= df["entry_type"].isin(list(entry_types))
    if modes:
        mask &= df["mode"].isin(list(modes))
    if search:
        needle = search.strip().lower()
        text = (df["note"].fillna("").astype(str) + " " + df["actor"].fillna("").astype(str)).str.lower()
        mask &= text.str.contains(needle, regex=False)
    return df[mask].reset_index(drop=True)


def ledger_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Totals of a (filtered) ledger frame + per-type counts and the last balances."""
    out: dict[str, Any] = {"rows": int(len(df)), "locked": 0.0, "refilled": 0.0, "manual_transfers": 0.0,
                           "manual_adjustments": 0.0, "realized": 0.0, "sizing_changes": 0, "last": None,
                           "by_type": _empty(["entry_type", "count", "amount"])}
    if df.empty:
        return out
    amt = df["amount"].astype(float)
    kind = df["entry_type"]
    out["locked"] = round(float(amt.abs()[kind == LedgerEntryType.PROFIT_LOCK.value].sum()), 2)
    out["refilled"] = round(float(amt.abs()[kind == LedgerEntryType.REFILL.value].sum()), 2)
    out["manual_transfers"] = round(float(amt.abs()[kind == LedgerEntryType.MANUAL_TRANSFER.value].sum()), 2)
    out["manual_adjustments"] = round(float(amt[kind == LedgerEntryType.MANUAL_ADJUSTMENT.value].sum()), 2)
    out["realized"] = round(float(amt[kind.isin([LedgerEntryType.REALIZED_PNL.value,
                                                 LedgerEntryType.DIVIDEND.value])].sum()), 2)
    out["sizing_changes"] = int((kind == LedgerEntryType.SIZING_CHANGE.value).sum())
    last = df.iloc[-1]
    out["last"] = {k: float(last[k]) for k in ("trading_cash", "blocked", "trading", "reserve", "total",
                                               "sizing_capital")}
    by = df.groupby("entry_type").agg(count=("id", "size"), amount=("amount", "sum")).reset_index()
    out["by_type"] = round_frame(by.sort_values("count", ascending=False).reset_index(drop=True))
    return out


# =========================================================================== day-wise timeline

TIMELINE_COLUMNS: tuple[str, ...] = ("ts", "kind", "symbol", "mode", "detail", "ref")
TIMELINE_KINDS: tuple[str, ...] = (
    "RUN", "SCAN", "DECISION", "APPROVAL", "ORDER", "ENTRY", "EXIT", "CAPITAL", "SHADOW", "EOD",
)
_KIND_ORDER = {k: i for i, k in enumerate(TIMELINE_KINDS)}


def activity_days(session: Session, batch_id: int | None) -> list[date]:
    """Days with any recorded activity in the batch, newest first."""
    if batch_id is None:
        return []
    stamps: list[datetime | None] = []
    for col, owner in ((Run.started_at, Run.batch_id), (DecisionRow.created_at, DecisionRow.batch_id),
                       (ApprovalItem.created_at, ApprovalItem.batch_id), (Position.opened_at, Position.batch_id),
                       (Trade.exit_time, Trade.batch_id), (CapitalLedgerEntry.ts, CapitalLedgerEntry.batch_id)):
        stamps.extend(session.execute(select(col).where(owner == batch_id)).scalars().all())
    days = {naive_ist(s).date() for s in stamps if s is not None}
    days |= set(session.execute(select(DailySummary.day).where(DailySummary.batch_id == batch_id)).scalars())
    return sorted(days, reverse=True)


def _fmt_px(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _ev(ts: datetime | None, kind: str, symbol: str | None, mode: str | None, detail: str, ref: str) -> dict[str, Any]:
    return {"ts": naive_ist(ts), "kind": kind, "symbol": symbol or "", "mode": mode or "", "detail": detail,
            "ref": ref}


def day_timeline(
    session: Session, batch_id: int | None, day: date, *, include_scans: bool = True, scan_limit: int = 2000
) -> pd.DataFrame:
    """Everything that happened in a batch on ``day``, merged and time sorted.

    Kinds: RUN (scans/cycles), SCAN (stocks_analyzed), DECISION, APPROVAL (queued and
    decided), ORDER, ENTRY (position opened), EXIT (closed leg), CAPITAL (ledger), SHADOW
    (rejected setup tracked / resolved) and EOD (daily summary).
    """
    cols = list(TIMELINE_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    lo, hi = _day_start(day), _day_start(day + timedelta(days=1))
    ev: list[dict[str, Any]] = []

    for r in session.execute(select(Run).where(Run.batch_id == batch_id, Run.started_at >= lo, Run.started_at < hi)
                             ).scalars():
        dur = (r.finished_at - r.started_at).total_seconds() if r.finished_at else None
        detail = f"{r.run_type}{(' ' + r.mode) if r.mode else ''} - {r.status}, {r.n_symbols} symbol(s)"
        if dur is not None:
            detail += f", {dur:,.1f}s"
        if r.error:
            detail += f" - error: {jsonish(r.error, 160)}"
        if r.summary_json:
            detail += f" | {jsonish(r.summary_json, 160)}"
        ev.append(_ev(r.started_at, "EOD" if r.run_type == "EOD" else "RUN", None, r.mode, detail, f"run:{r.id}"))

    if include_scans:
        stmt = (select(StockAnalyzed).where(StockAnalyzed.batch_id == batch_id, StockAnalyzed.as_of >= lo,
                                            StockAnalyzed.as_of < hi)
                .order_by(StockAnalyzed.as_of, StockAnalyzed.id).limit(scan_limit))
        for s in session.execute(stmt).scalars():
            verdict = "included" if s.included else f"excluded: {s.reason or '-'}"
            extra = []
            if s.regime:
                extra.append(s.regime)
            if s.confluence is not None:
                extra.append(f"confluence {s.confluence:.0f}{(' ' + s.grade) if s.grade else ''}")
            detail = f"{s.stage} - {verdict}" + (f" ({', '.join(extra)})" if extra else "")
            ev.append(_ev(s.as_of, "SCAN", s.symbol, s.mode, detail, f"stock:{s.id}"))

    for d in session.execute(select(DecisionRow).where(DecisionRow.batch_id == batch_id,
                                                       DecisionRow.created_at >= lo, DecisionRow.created_at < hi)
                             ).scalars():
        detail = (f"{d.decision}{(' ' + d.direction) if d.direction else ''} conf {d.confidence}"
                  f"{(' grade ' + d.grade) if d.grade else ''} -> {d.final_action}")
        if d.strategy or d.regime:
            detail += f" ({d.strategy or '-'} / {d.regime or '-'})"
        if d.entry_price is not None:
            detail += f" | entry {_fmt_px(d.entry_price)} SL {_fmt_px(d.stop_loss)} T1 {_fmt_px(d.target_1)}"
        if d.risk_rule:
            detail += f" | risk rule {d.risk_rule}"
        ev.append(_ev(d.created_at, "DECISION", d.symbol, d.mode, detail, f"decision:{d.id}"))

    approvals = session.execute(select(ApprovalItem).where(ApprovalItem.batch_id == batch_id)).scalars().all()
    for a in approvals:
        if a.created_at is not None and lo <= a.created_at < hi:
            ev.append(_ev(a.created_at, "APPROVAL", a.symbol, a.mode,
                          f"queued: {a.direction} {a.qty} @ {_fmt_px(a.entry_price)} SL {_fmt_px(a.stop_loss)} "
                          f"T1 {_fmt_px(a.target_1)} (expires {naive_ist(a.expires_at):%H:%M})",
                          f"approval:{a.id}"))
        decided = a.decided_at or (a.expires_at if a.status == ApprovalStatus.TIMEOUT.value else None)
        if decided is not None and lo <= decided < hi and a.status != ApprovalStatus.PENDING.value:
            detail = f"{a.status}{(' by ' + a.decided_by) if a.decided_by else ''}"
            if a.user_reason:
                detail += f" - {a.user_reason}"
            if a.final_mode and a.final_mode != a.mode:
                detail += f" | as {a.final_mode}"
            if a.user_edits_json:
                detail += f" | edits {jsonish(a.user_edits_json, 120)}"
            if a.block_rule:
                detail += f" | blocked by risk rule {a.block_rule}"
            ev.append(_ev(decided, "APPROVAL", a.symbol, a.final_mode or a.mode, detail, f"approval:{a.id}"))

    for o in session.execute(select(Order).where(Order.batch_id == batch_id, Order.created_at >= lo,
                                                 Order.created_at < hi)).scalars():
        detail = f"{o.side} {o.purpose} {o.order_type} {o.qty} -> {o.status}"
        if o.fill_price is not None:
            detail += f" @ {_fmt_px(o.fill_price)}"
        elif o.limit_price is not None:
            detail += f" (limit {_fmt_px(o.limit_price)})"
        if o.reason:
            detail += f" - {jsonish(o.reason, 100)}"
        ev.append(_ev(o.filled_at or o.created_at, "ORDER", o.symbol, o.mode, detail, f"order:{o.id}"))

    for p in session.execute(select(Position).where(Position.batch_id == batch_id, Position.opened_at >= lo,
                                                    Position.opened_at < hi)).scalars():
        ev.append(_ev(p.opened_at, "ENTRY", p.symbol, p.mode,
                      f"{p.direction} {p.initial_qty} @ {_fmt_px(p.avg_entry)} SL {_fmt_px(p.initial_stop_loss)} "
                      f"T1 {_fmt_px(p.target_1)} [{p.source}{(' / ' + p.strategy) if p.strategy else ''}]",
                      f"position:{p.id}"))

    for t in session.execute(select(Trade).where(Trade.batch_id == batch_id, Trade.exit_time >= lo,
                                                 Trade.exit_time < hi)).scalars():
        r = f", R {t.r_multiple:+.2f}" if t.r_multiple is not None else ""
        ev.append(_ev(t.exit_time, "EXIT", t.symbol, t.mode,
                      f"{t.exit_reason}{' (partial)' if t.is_partial else ''} {t.qty} @ {_fmt_px(t.exit_price)} - "
                      f"net {t.net_pnl:+,.2f} (gross {t.gross_pnl:+,.2f}, charges {t.charges:,.2f}{r})",
                      f"trade:{t.id}"))

    for e in session.execute(select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id,
                                                              CapitalLedgerEntry.ts >= lo, CapitalLedgerEntry.ts < hi)
                             ).scalars():
        detail = (f"{e.entry_type} {e.amount:+,.2f} -> TRADING {e.trading_cash_after + e.blocked_after:,.2f}, "
                  f"RESERVE {e.reserve_after:,.2f}, sizing {e.sizing_capital_after:,.2f}")
        if e.note:
            detail += f" - {jsonish(e.note, 100)}"
        ev.append(_ev(e.ts, "CAPITAL", None, e.mode, detail, f"ledger:{e.id}"))

    for s in session.execute(select(ShadowTrade).where(ShadowTrade.batch_id == batch_id)).scalars():
        if lo <= s.created_at < hi:
            ev.append(_ev(s.created_at, "SHADOW", s.symbol, s.mode,
                          f"tracking {s.rejection_source} rejection: {s.direction} @ {_fmt_px(s.entry_price)} "
                          f"SL {_fmt_px(s.stop_loss)} T1 {_fmt_px(s.target_1)}"
                          f"{(' - ' + s.rejection_rule) if s.rejection_rule else ''}", f"shadow:{s.id}"))
        if s.exit_time is not None and lo <= s.exit_time < hi:
            pnl = f", hypothetical net {s.hypothetical_net_pnl:+,.2f}" if s.hypothetical_net_pnl is not None else ""
            r = f", R {s.r_multiple:+.2f}" if s.r_multiple is not None else ""
            ev.append(_ev(s.exit_time, "SHADOW", s.symbol, s.mode, f"outcome {s.outcome}{r}{pnl}", f"shadow:{s.id}"))

    for ds in session.execute(select(DailySummary).where(DailySummary.batch_id == batch_id,
                                                         DailySummary.day == day)).scalars():
        detail = (f"{ds.mode}: {ds.trades} trade(s), {ds.wins}W/{ds.losses}L, net {ds.net_pnl:+,.2f} "
                  f"(charges {ds.charges:,.2f})")
        if ds.mode == "TOTAL":
            detail += (f" | equity {_fmt_px(ds.equity)}, HWM {_fmt_px(ds.hwm)}, DD "
                       f"{(ds.drawdown_pct or 0.0):.2f}%, tier {ds.tier or '-'}, locked {ds.profit_locked:,.2f}, "
                       f"refilled {ds.refilled:,.2f}")
        ev.append(_ev(datetime.combine(day, EOD_TIME), "EOD", None, None if ds.mode == "TOTAL" else ds.mode,
                      detail, f"daily:{ds.id}"))

    df = pd.DataFrame(ev, columns=cols)
    if df.empty:
        return df
    df["_k"] = df["kind"].map(_KIND_ORDER).fillna(99)
    return df.sort_values(["ts", "_k"], kind="stable").drop(columns="_k").reset_index(drop=True)


# =========================================================================== trade drill-down


def trade_choices(df: pd.DataFrame) -> list[tuple[int, str]]:
    """``(trade_id, label)`` for a picker, newest exit first."""
    if df.empty:
        return []
    out: list[tuple[int, str]] = []
    for row in df.sort_values(["exit_time", "id"], ascending=False, na_position="last").itertuples(index=False):
        when = row.exit_time.strftime("%d-%b %H:%M") if isinstance(row.exit_time, datetime) else "-"
        net = _num(row.net_pnl) or 0.0
        out.append((int(row.id), f"#{row.id} {row.symbol} {row.mode} {row.direction} {when} net {net:+,.2f}"))
    return out


_APPROVAL_COMMANDS = ("APPROVE", "REJECT", "MODIFY_APPROVE", "BULK_REJECT")
_POSITION_COMMANDS = ("CLOSE_POSITION", "TIGHTEN_SL", "UPDATE_TARGETS", "UPDATE_TRAILING", "CONVERT_POSITION",
                      "UPDATE_HOLDING", "SET_EVENT_PIN")
HISTORY_COLUMNS: tuple[str, ...] = ("ts", "event", "by", "detail")


def approval_history(
    session: Session, approval: ApprovalItem | None, position: Position | None, batch_id: int | None
) -> pd.DataFrame:
    """Approval + edit history of a trade: queue, decision, edits, owner commands, SL changes, audit."""
    cols = list(HISTORY_COLUMNS)
    rows: list[dict[str, Any]] = []
    aid = approval.id if approval is not None else None
    pid = position.id if position is not None else None
    if approval is not None:
        a = approval
        rows.append({"ts": naive_ist(a.created_at), "event": "QUEUED", "by": "SYSTEM",
                     "detail": f"{a.mode} {a.direction} {a.entry_type} {a.qty} (max {a.max_qty}) @ "
                               f"{_fmt_px(a.entry_price)} SL {_fmt_px(a.stop_loss)} T1 {_fmt_px(a.target_1)} "
                               f"T2 {_fmt_px(a.target_2)} | conf {a.confidence} grade {a.grade or '-'} | "
                               f"expires {naive_ist(a.expires_at):%Y-%m-%d %H:%M}"})
        if a.status != ApprovalStatus.PENDING.value:
            detail = a.status
            if a.user_reason:
                detail += f" - reason: {a.user_reason}"
            if a.final_mode and a.final_mode != a.mode:
                detail += f" | approved as {a.final_mode}"
            if a.user_edits_json:
                detail += f" | edits: {jsonish(a.user_edits_json)}"
            if a.block_rule:
                detail += f" | blocked by risk rule {a.block_rule}"
            rows.append({"ts": naive_ist(a.decided_at or a.expires_at), "event": f"DECISION {a.status}",
                         "by": a.decided_by or ("TIMEOUT" if a.status == ApprovalStatus.TIMEOUT.value else ""),
                         "detail": detail})
    if aid is not None or pid is not None:
        stmt = select(ControlCommand).where(ControlCommand.command_type.in_(_APPROVAL_COMMANDS + _POSITION_COMMANDS))
        if batch_id is not None:
            stmt = stmt.where(ControlCommand.batch_id == batch_id)
        for c in session.execute(stmt.order_by(ControlCommand.id)).scalars():
            payload = c.payload_json or {}
            hit = (aid is not None and (payload.get("approval_id") == aid or aid in (payload.get("approval_ids") or [])))
            hit = hit or (pid is not None and payload.get("position_id") == pid)
            if not hit:
                continue
            res = c.result_json or {}
            info = {k: v for k, v in payload.items() if k not in ("approval_id", "approval_ids", "position_id")}
            detail = f"{c.status}" + (f" | {jsonish(info, 160)}" if info else "")
            if res.get("rule"):
                detail += f" | rule {res['rule']}"
            if res.get("message"):
                detail += f" - {jsonish(res['message'], 160)}"
            rows.append({"ts": naive_ist(c.created_at), "event": f"COMMAND {c.command_type}", "by": c.requested_by,
                         "detail": detail})
    if position is not None:
        for item in position.sl_history_json or []:
            if isinstance(item, dict):
                ts = item.get("ts")
                try:
                    ts_val = naive_ist(datetime.fromisoformat(ts)) if isinstance(ts, str) else naive_ist(ts)
                except ValueError:
                    ts_val = None
                rows.append({"ts": ts_val, "event": "SL CHANGE", "by": str(item.get("by") or ""),
                             "detail": f"{item.get('old')} -> {item.get('new')}"
                                       + (f" ({item['reason']})" if item.get("reason") else "")})
    targets = [t for t in (f"approval:{aid}" if aid else None, f"position:{pid}" if pid else None) if t]
    if targets:
        for log in session.execute(select(AuditLog).where(AuditLog.target.in_(targets)).order_by(AuditLog.id)).scalars():
            rows.append({"ts": naive_ist(log.ts), "event": f"AUDIT {log.action}", "by": log.actor,
                         "detail": jsonish(log.details_json, 200)})
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df.sort_values("ts", kind="stable", na_position="last").reset_index(drop=True)


def trade_detail(session: Session, trade_id: int) -> dict[str, Any] | None:
    """Everything behind one trade: legs, position, orders, decision (agents/debate/meta/risk),
    approval and its edit history. None when the trade does not exist."""
    t = session.get(Trade, trade_id)
    if t is None:
        return None
    pos = session.get(Position, t.position_id) if t.position_id is not None else None
    legs = session.execute(
        select(Trade).where(Trade.batch_id == t.batch_id, Trade.position_id == t.position_id)
        .order_by(Trade.exit_time, Trade.id)
    ).scalars().all()
    decision_id = t.decision_id or (pos.decision_id if pos is not None else None)
    approval_id = t.approval_id or (pos.approval_id if pos is not None else None)
    approval = session.get(ApprovalItem, approval_id) if approval_id else None
    if approval is None and decision_id is not None:
        approval = session.execute(
            select(ApprovalItem).where(ApprovalItem.decision_id == decision_id).order_by(ApprovalItem.id.desc()).limit(1)
        ).scalar_one_or_none()
    orders = session.execute(
        select(Order).where(Order.batch_id == t.batch_id, Order.position_id == t.position_id)
        .order_by(Order.created_at, Order.id)
    ).scalars().all()
    leg_cols = ["id", "exit_time", "qty", "exit_price", "exit_reason", "is_partial", "gross_pnl", "charges",
                "net_pnl", "r_multiple"]
    order_cols = ["id", "created_at", "side", "purpose", "order_type", "qty", "limit_price", "status", "filled_at",
                  "fill_price", "slippage", "reason"]
    trade = _plain(svc.row_to_dict(t)) or {}
    if not trade.get("regime") and decision_id is not None:
        dec = session.get(DecisionRow, decision_id)
        trade["regime"] = dec.regime if dec is not None else None
    trade["decision_id"] = decision_id
    trade["approval_id"] = approval.id if approval is not None else approval_id
    return {
        "trade": trade,
        "position": _plain(svc.row_to_dict(pos)) if pos is not None else None,
        "legs": pd.DataFrame([_values(x, leg_cols) for x in legs], columns=leg_cols),
        "orders": pd.DataFrame([_values(o, order_cols) for o in orders], columns=order_cols),
        "decision": svc.decision_details(session, decision_id),
        "approval": _plain(svc.row_to_dict(approval)) if approval is not None else None,
        "history": approval_history(session, approval, pos, t.batch_id),
    }


# --------------------------------------------------------------------------- candles for the trade chart

_CANDLE_KEYS = ("candles", "ohlcv", "bars", "chart_candles")
DAILY_TIMEFRAMES = ("1d", "1wk")
CANDLE_COLUMNS = ("ts", "open", "high", "low", "close")


def candles_from_records(data: Any) -> pd.DataFrame | None:
    """Normalise stored candles (list of records or dict of lists) -> ``ts/open/high/low/close``.

    ``ts`` may be called ts/time/timestamp/datetime/date; returned naive IST, sorted. None if
    the data is not usable.
    """
    if not data:
        return None
    try:
        df = pd.DataFrame(data)
        lower = {str(c).lower(): c for c in df.columns}
        ts_col = next((lower[k] for k in ("ts", "time", "timestamp", "datetime", "date") if k in lower), None)
        price_cols = [lower.get(k) for k in ("open", "high", "low", "close")]
        if df.empty or ts_col is None or any(c is None for c in price_cols):
            return None
        try:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
        except (TypeError, ValueError):  # mixed UTC offsets
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if not pd.api.types.is_datetime64_any_dtype(ts):
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(IST).dt.tz_localize(None)
        out = pd.DataFrame({"ts": ts})
        for name, col in zip(("open", "high", "low", "close"), price_cols, strict=True):
            out[name] = pd.to_numeric(df[col], errors="coerce")
        out = out.dropna().sort_values("ts", kind="stable").reset_index(drop=True)
        return out if not out.empty else None
    except Exception as exc:  # noqa: BLE001 - malformed stored data must never break the page
        logger.debug(f"unusable stored candles: {exc}")
        return None


def stored_candles(session: Session, decision_id: int | None, symbol: str | None = None) -> pd.DataFrame | None:
    """Candles stored in the DB for a decision (decision context/meta JSON or the run's
    stocks_analyzed features), if any."""
    if decision_id is None:
        return None
    d = session.get(DecisionRow, decision_id)
    if d is None:
        return None
    blobs: list[Any] = [d.context_json, d.meta_json]
    if d.run_id is not None:
        blobs.extend(session.execute(
            select(StockAnalyzed.features_json).where(StockAnalyzed.run_id == d.run_id,
                                                      StockAnalyzed.symbol == (symbol or d.symbol))
        ).scalars().all())
    for blob in blobs:
        if not isinstance(blob, dict):
            continue
        for key in _CANDLE_KEYS:
            df = candles_from_records(blob.get(key))
            if df is not None:
                return df
    return None


def cached_daily_candles(cache_dir: Path | None, symbol: str) -> pd.DataFrame | None:
    """Daily candles from the market-data disk cache (``<cache>/candles/<provider>/SYM_1d_*.parquet``).

    Read-only file access, never a network call. None when nothing usable is cached.
    """
    if cache_dir is None:
        return None
    root = Path(cache_dir) / "candles"
    if not root.is_dir():
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in symbol)
    for path in sorted(root.glob(f"*/{safe}_1d_*.parquet"), key=lambda p: p.name, reverse=True):
        try:
            raw = pd.read_parquet(path)
            if "ts" not in raw.columns:
                raw = raw.reset_index()
                raw = raw.rename(columns={raw.columns[0]: "ts"})
            df = candles_from_records(raw.to_dict(orient="list"))
        except Exception as exc:  # noqa: BLE001 - a corrupt cache file is skipped
            logger.debug(f"unreadable candle cache {path}: {exc}")
            continue
        if df is not None:
            return df
    return None


def trade_candles(
    session: Session,
    trade: dict[str, Any],
    *,
    timeframe: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Candles around a trade and where they came from (``"db"``, ``"disk cache"`` or ``"none"``).

    Order: candles stored in the DB for the trade's decision; for daily/weekly timeframes the
    market-data disk cache. The window is the trade +/- 30 days (daily) or 3 hours (intraday).
    """
    entry = trade.get("entry_time")
    exit_ = trade.get("exit_time") or entry
    df = stored_candles(session, trade.get("decision_id"), trade.get("symbol"))
    source = "db"
    if df is None and (timeframe or "") in DAILY_TIMEFRAMES and trade.get("symbol"):
        df = cached_daily_candles(cache_dir, str(trade["symbol"]))
        source = "disk cache"
    if df is None or df.empty or not isinstance(entry, datetime):
        return None, "none"
    pad = timedelta(days=30) if (timeframe or "") in DAILY_TIMEFRAMES else timedelta(hours=3)
    window = df[(df["ts"] >= entry - pad) & (df["ts"] <= exit_ + pad)]
    if window.empty:
        return None, "none"
    return window.reset_index(drop=True), source


# =========================================================================== rejected / not traded

REJECTED_COLUMNS: tuple[str, ...] = (
    "ts", "symbol", "mode", "direction", "strategy", "category", "reason", "rule", "confidence", "grade", "entry",
    "stop_loss", "target_1", "decision_id", "approval_id", "shadow_id", "shadow_closed", "shadow_outcome",
    "shadow_r", "shadow_pnl",
)
CATEGORY_LABELS: dict[str, str] = {
    "META": "Meta NO_TRADE",
    "RISK": "Risk rule",
    "USER": "Owner rejected",
    "TIMEOUT": "Approval timeout",
    "SIGNALS_ONLY": "Signals-only mode",
    "FILTER": "Filtered",
    "NOT_FILLED": "Approved, not filled",
}
_ACTION_CATEGORY = {"NO_TRADE": "META", "RISK_REJECTED": "RISK", "SIGNAL_ONLY": "SIGNALS_ONLY",
                    "SIGNALS_ONLY": "SIGNALS_ONLY", "FILTERED": "FILTER"}


def shadow_pnl(sh: ShadowTrade | None) -> float | None:
    """Hypothetical net P&L of a shadow trade (0 when it never triggered, None while open)."""
    if sh is None:
        return None
    if sh.hypothetical_net_pnl is not None:
        return float(sh.hypothetical_net_pnl)
    if sh.outcome == ShadowOutcome.NOT_TRIGGERED.value:
        return 0.0
    return None


def shadow_closed(sh: ShadowTrade | None) -> bool:
    return sh is not None and (sh.status == "CLOSED" or sh.outcome != ShadowOutcome.OPEN.value)


def _traded_approval_ids(session: Session, batch_id: int) -> set[int]:
    ids = set(session.execute(select(Trade.approval_id).where(Trade.batch_id == batch_id,
                                                              Trade.approval_id.is_not(None))).scalars())
    ids |= set(session.execute(select(Position.approval_id).where(Position.batch_id == batch_id,
                                                                  Position.approval_id.is_not(None))).scalars())
    return {int(i) for i in ids}


def _approval_reason(a: ApprovalItem, traded: set[int]) -> tuple[str, str, str | None] | None:
    """(category, reason, rule) of an approval that did not become a trade, else None."""
    if a.status == ApprovalStatus.PENDING.value:
        return None
    if a.block_rule:
        return "RISK", f"Owner approval blocked by risk rule {a.block_rule}", a.block_rule
    if a.status == ApprovalStatus.REJECTED.value:
        why = f": {a.user_reason}" if a.user_reason else ""
        return "USER", f"Rejected by {a.decided_by or 'owner'}{why}", None
    if a.status == ApprovalStatus.TIMEOUT.value:
        return "TIMEOUT", f"No decision before {naive_ist(a.expires_at):%Y-%m-%d %H:%M}", None
    if a.id in traded:
        return None
    return "NOT_FILLED", f"{a.status.title()} but never filled", None


def _decision_reason(d: DecisionRow) -> tuple[str, str, str | None]:
    """(category, reason, rule) of a decision that did not reach the approval queue / execution."""
    cat = _ACTION_CATEGORY.get(d.final_action, "META")
    if cat == "RISK":
        risk = d.risk_json or {}
        msg = risk.get("message") or risk.get("reason") or ""
        return cat, f"Risk veto: {d.risk_rule or 'rule not recorded'}{(' - ' + str(msg)) if msg else ''}", d.risk_rule
    if cat == "SIGNALS_ONLY":
        return cat, "Mode policy SIGNALS_ONLY (signal recorded, never executed)", None
    if cat == "FILTER":
        ctx = d.context_json or {}
        why = ctx.get("reason") or ctx.get("filter_reason")
        return cat, f"Filtered: {why}" if why else "Filtered before meta (strategy / confluence)", None
    meta = d.meta_json or {}
    text = meta.get("reasoning") or meta.get("reason") or meta.get("summary") or ""
    head = f"Meta {d.decision} (confidence {d.confidence})"
    return cat, f"{head}: {jsonish(text, 220)}" if text else head, None


def _rejected_row(
    *, ts: datetime | None, symbol: str, mode: str | None, direction: str | None, strategy: str | None,
    category: str, reason: str, rule: str | None, confidence: int | None, grade: str | None,
    entry: float | None, stop: float | None, target: float | None, decision_id: int | None,
    approval_id: int | None, shadow: ShadowTrade | None,
) -> dict[str, Any]:
    return {
        "ts": naive_ist(ts), "symbol": symbol, "mode": mode, "direction": direction, "strategy": strategy,
        "category": category, "reason": reason, "rule": rule, "confidence": confidence, "grade": grade,
        "entry": entry, "stop_loss": stop, "target_1": target, "decision_id": decision_id, "approval_id": approval_id,
        "shadow_id": shadow.id if shadow is not None else None, "shadow_closed": shadow_closed(shadow),
        "shadow_outcome": shadow.outcome if shadow is not None else None,
        "shadow_r": shadow.r_multiple if shadow is not None else None, "shadow_pnl": shadow_pnl(shadow),
    }


def rejected_frame(
    session: Session, batch_id: int | None, *, start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """Everything the batch did NOT trade, with the reason and the shadow outcome (newest first).

    * decisions: meta NO_TRADE (META), risk veto (RISK), signals-only (SIGNALS_ONLY), filtered
      (FILTER); a QUEUED decision takes its approval's fate:
    * approvals: owner rejection (USER), timeout (TIMEOUT), owner approval blocked by a risk rule
      (RISK), approved but never filled (NOT_FILLED); pending items are skipped;
    * shadow trades not linked to any of those rows are listed with their rejection source.
    The shadow trade is matched by approval id first, then decision id.
    """
    cols = list(REJECTED_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    lo, hi = date_bounds(start, end)

    def in_range(ts: datetime | None) -> bool:
        return ts is not None and (lo is None or ts >= lo) and (hi is None or ts < hi)

    dstmt = select(DecisionRow).where(DecisionRow.batch_id == batch_id, DecisionRow.final_action != "EXECUTED")
    sstmt = select(ShadowTrade).where(ShadowTrade.batch_id == batch_id)
    if lo is not None:
        dstmt = dstmt.where(DecisionRow.created_at >= lo)
        sstmt = sstmt.where(ShadowTrade.created_at >= lo)
    if hi is not None:
        dstmt = dstmt.where(DecisionRow.created_at < hi)
        sstmt = sstmt.where(ShadowTrade.created_at < hi)
    decisions = session.execute(dstmt.order_by(DecisionRow.created_at, DecisionRow.id)).scalars().all()
    approvals = session.execute(select(ApprovalItem).where(ApprovalItem.batch_id == batch_id)
                                .order_by(ApprovalItem.id)).scalars().all()
    by_decision: dict[int, ApprovalItem] = {}
    for a in approvals:
        if a.decision_id is not None:
            by_decision[a.decision_id] = a
    shadows = session.execute(sstmt.order_by(ShadowTrade.id)).scalars().all()
    sh_by_approval = {s.approval_id: s for s in shadows if s.approval_id is not None}
    sh_by_decision = {s.decision_id: s for s in shadows if s.decision_id is not None}
    traded = _traded_approval_ids(session, batch_id)
    seen_decisions = {d.id for d in decisions}
    used_shadows: set[int] = set()
    rows: list[dict[str, Any]] = []

    def pick_shadow(a: ApprovalItem | None, decision_id: int | None) -> ShadowTrade | None:
        sh = (sh_by_approval.get(a.id) if a is not None else None) or (
            sh_by_decision.get(decision_id) if decision_id is not None else None)
        if sh is not None:
            used_shadows.add(sh.id)
        return sh

    for d in decisions:
        a = by_decision.get(d.id)
        if d.final_action == "QUEUED" or (a is not None and d.final_action not in _ACTION_CATEGORY):
            if a is None:
                continue
            verdict = _approval_reason(a, traded)
            if verdict is None:
                continue
        else:
            verdict = _decision_reason(d)
        cat, reason, rule = verdict
        rows.append(_rejected_row(
            ts=d.created_at, symbol=d.symbol, mode=(a.final_mode or a.mode) if a is not None else d.mode,
            direction=d.direction, strategy=d.strategy or (a.strategy if a is not None else None), category=cat,
            reason=reason, rule=rule, confidence=d.confidence, grade=d.grade,
            entry=a.entry_price if a is not None else d.entry_price,
            stop=a.stop_loss if a is not None else d.stop_loss,
            target=a.target_1 if a is not None else d.target_1, decision_id=d.id,
            approval_id=a.id if a is not None else None, shadow=pick_shadow(a, d.id)))

    for a in approvals:
        if a.decision_id is not None and a.decision_id in seen_decisions:
            continue
        if not in_range(a.created_at):
            continue
        verdict = _approval_reason(a, traded)
        if verdict is None:
            continue
        cat, reason, rule = verdict
        rows.append(_rejected_row(
            ts=a.created_at, symbol=a.symbol, mode=a.final_mode or a.mode, direction=a.direction,
            strategy=a.strategy, category=cat, reason=reason, rule=rule, confidence=a.confidence, grade=a.grade,
            entry=a.entry_price, stop=a.stop_loss, target=a.target_1, decision_id=a.decision_id, approval_id=a.id,
            shadow=pick_shadow(a, a.decision_id)))

    for s in shadows:
        if s.id in used_shadows or not in_range(s.created_at) or (s.approval_id is not None and s.approval_id in traded):
            continue  # matched above, out of range, or the original of a proposal I modified and traded
        reason = s.rejection_reason or CATEGORY_LABELS.get(s.rejection_source, s.rejection_source)
        rows.append(_rejected_row(
            ts=s.created_at, symbol=s.symbol, mode=s.mode, direction=s.direction, strategy=s.strategy,
            category=s.rejection_source, reason=reason, rule=s.rejection_rule, confidence=s.confidence,
            grade=s.grade, entry=s.entry_price, stop=s.stop_loss, target=s.target_1, decision_id=s.decision_id,
            approval_id=s.approval_id, shadow=s))

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df.sort_values("ts", ascending=False, kind="stable").reset_index(drop=True)


REJECTED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "category", "label", "count", "shadow_tracked", "shadow_closed", "would_win", "would_lose", "missed_profit",
    "avoided_loss", "net_shadow_pnl", "avg_shadow_r",
)


def rejected_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per category: count, shadow coverage, would-be wins/losses, missed profit (positive
    hypothetical P&L), avoided loss (negative hypothetical P&L, as a positive number)."""
    cols = list(REJECTED_SUMMARY_COLUMNS)
    if df.empty:
        return _empty(cols)
    rows: list[dict[str, Any]] = []
    for cat, g in df.groupby("category", sort=True):
        pnl = pd.to_numeric(g["shadow_pnl"], errors="coerce")
        closed = g["shadow_closed"].astype(bool)
        known = pnl[closed & pnl.notna()]
        r = pd.to_numeric(g["shadow_r"], errors="coerce")[closed]
        rows.append({
            "category": cat, "label": CATEGORY_LABELS.get(str(cat), str(cat)), "count": int(len(g)),
            "shadow_tracked": int(g["shadow_id"].notna().sum()), "shadow_closed": int(closed.sum()),
            "would_win": int((known > 0).sum()), "would_lose": int((known < 0).sum()),
            "missed_profit": round(float(known[known > 0].sum()), 2),
            "avoided_loss": round(abs(float(known[known < 0].sum())), 2),
            "net_shadow_pnl": round(float(known.sum()), 2),
            "avg_shadow_r": round(float(r.mean()), 2) if r.notna().any() else None,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("count", ascending=False, kind="stable").reset_index(drop=True)


FILTERED_COLUMNS: tuple[str, ...] = ("ts", "symbol", "mode", "stage", "reason", "regime", "confluence", "grade")


def filtered_stocks_frame(
    session: Session, batch_id: int | None, *, start: date | None = None, end: date | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Stocks excluded by the universe / watchlist / scan filters, newest first."""
    cols = list(FILTERED_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    lo, hi = date_bounds(start, end)
    stmt = select(StockAnalyzed).where(StockAnalyzed.batch_id == batch_id, StockAnalyzed.included.is_(False))
    if lo is not None:
        stmt = stmt.where(StockAnalyzed.as_of >= lo)
    if hi is not None:
        stmt = stmt.where(StockAnalyzed.as_of < hi)
    rows = session.execute(stmt.order_by(StockAnalyzed.as_of.desc(), StockAnalyzed.id.desc()).limit(limit)).scalars()
    return pd.DataFrame([{"ts": naive_ist(s.as_of), "symbol": s.symbol, "mode": s.mode, "stage": s.stage,
                          "reason": s.reason, "regime": s.regime, "confluence": s.confluence, "grade": s.grade}
                         for s in rows], columns=cols)


# =========================================================================== me vs system

MVS_COLUMNS: tuple[str, ...] = (
    "ts", "symbol", "mode", "category", "approval_id", "trade_status", "my_pnl", "system_pnl", "edge", "comparable",
    "note",
)
MVS_CATEGORIES: tuple[str, ...] = ("APPROVED", "MODIFIED", "REJECTED", "TIMEOUT", "BLOCKED", "MANUAL")
MVS_LABELS: dict[str, str] = {
    "APPROVED": "Approved as proposed",
    "MODIFIED": "Modified by me",
    "REJECTED": "Rejected by me",
    "TIMEOUT": "Timed out (no answer)",
    "BLOCKED": "My approval blocked by risk",
    "MANUAL": "My manual trades",
}


def me_vs_system(session: Session, batch_id: int | None) -> pd.DataFrame:
    """Owner decisions vs what the system alone would have done (sections 7.6 / 14.4 #7).

    One row per decided approval plus one per manual position:

    * APPROVED as proposed: me = realized net of the trade; system = the same trade.
    * MODIFIED: me = realized net of my version; system = shadow outcome of the original proposal.
    * REJECTED / TIMEOUT / BLOCKED (my approval vetoed by a risk rule): me = 0 (no trade);
      system = shadow outcome of the proposal.
    * MANUAL (my own trades, no approval): me = realized net; system = 0 (it did not propose it).

    ``comparable`` is False while a side is unknown (position or shadow still open, proposal not
    shadow-tracked); ``edge`` = me - system for comparable rows.
    """
    cols = list(MVS_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    trades = session.execute(select(Trade).where(Trade.batch_id == batch_id)).scalars().all()
    positions = session.execute(select(Position).where(Position.batch_id == batch_id)).scalars().all()
    pos_by_id = {p.id: p for p in positions}
    realized: dict[int, float] = {}
    for t in trades:
        aid = t.approval_id
        if aid is None and t.position_id in pos_by_id:
            aid = pos_by_id[t.position_id].approval_id
        if aid is not None:
            realized[aid] = realized.get(aid, 0.0) + float(t.net_pnl or 0.0)
    open_by_approval = {p.approval_id for p in positions if p.approval_id is not None and p.status == "OPEN"}
    shadows = session.execute(select(ShadowTrade).where(ShadowTrade.batch_id == batch_id)).scalars().all()
    sh_by_approval = {s.approval_id: s for s in shadows if s.approval_id is not None}
    sh_by_decision = {s.decision_id: s for s in shadows if s.decision_id is not None}
    rows: list[dict[str, Any]] = []
    approvals = session.execute(select(ApprovalItem).where(ApprovalItem.batch_id == batch_id,
                                                           ApprovalItem.status != ApprovalStatus.PENDING.value)
                                .order_by(ApprovalItem.created_at, ApprovalItem.id)).scalars().all()
    for a in approvals:
        cat = "BLOCKED" if a.block_rule else a.status
        if cat not in MVS_CATEGORIES:
            continue
        sh = sh_by_approval.get(a.id) or (sh_by_decision.get(a.decision_id) if a.decision_id is not None else None)
        sys_known = shadow_closed(sh) and shadow_pnl(sh) is not None
        my: float | None
        system: float | None
        note = ""
        if cat in ("APPROVED", "MODIFIED"):
            if a.id in open_by_approval:
                my, status, note = None, "OPEN", "position still open"
            elif a.id in realized:
                my, status = realized[a.id], "CLOSED"
            else:
                my, status, note = 0.0, "NO FILL", "approved but never filled"
            if cat == "APPROVED":
                system = my
            else:
                system = shadow_pnl(sh) if sys_known else None
                if system is None:
                    note = note or ("original proposal shadow still open" if sh is not None
                                    else "original proposal not shadow-tracked")
        else:
            my, status = 0.0, "NOT TRADED"
            system = shadow_pnl(sh) if sys_known else None
            if system is None:
                note = "shadow still open" if sh is not None else "not shadow-tracked"
        comparable = my is not None and system is not None
        rows.append({"ts": naive_ist(a.decided_at or a.created_at), "symbol": a.symbol, "mode": a.final_mode or a.mode,
                     "category": cat, "approval_id": a.id, "trade_status": status, "my_pnl": my,
                     "system_pnl": system, "edge": (my - system) if comparable else None, "comparable": comparable,
                     "note": note})
    manual: dict[int, list[Trade]] = {}
    for t in trades:
        if t.source == TradeSource.MANUAL.value and t.approval_id is None:
            manual.setdefault(t.position_id, []).append(t)
    for pid, legs in manual.items():
        pos = pos_by_id.get(pid)
        still_open = pos is not None and pos.status == "OPEN"
        my = None if still_open else round(sum(float(t.net_pnl or 0.0) for t in legs), 2)
        first = min(legs, key=lambda t: t.entry_time)
        rows.append({"ts": naive_ist(first.entry_time), "symbol": first.symbol, "mode": first.mode,
                     "category": "MANUAL", "approval_id": None, "trade_status": "OPEN" if still_open else "CLOSED",
                     "my_pnl": my, "system_pnl": 0.0, "edge": my, "comparable": my is not None,
                     "note": "partially closed" if still_open else ""})
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


MVS_SUMMARY_COLUMNS: tuple[str, ...] = ("category", "label", "items", "compared", "my_pnl", "system_pnl", "edge")


def me_vs_system_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Per category over the comparable rows: my P&L, the system-alone P&L and my edge."""
    cols = list(MVS_SUMMARY_COLUMNS)
    if detail.empty:
        return _empty(cols)
    rows: list[dict[str, Any]] = []
    for cat in MVS_CATEGORIES:
        g = detail[detail["category"] == cat]
        if g.empty:
            continue
        c = g[g["comparable"].astype(bool)]
        my = round(float(pd.to_numeric(c["my_pnl"]).sum()), 2)
        system = round(float(pd.to_numeric(c["system_pnl"]).sum()), 2)
        rows.append({"category": cat, "label": MVS_LABELS[cat], "items": int(len(g)), "compared": int(len(c)),
                     "my_pnl": my, "system_pnl": system, "edge": round(my - system, 2)})
    return pd.DataFrame(rows, columns=cols)


def me_vs_system_totals(detail: pd.DataFrame) -> dict[str, Any]:
    """Headline numbers over the comparable rows."""
    if detail.empty:
        return {"items": 0, "compared": 0, "my_pnl": 0.0, "system_pnl": 0.0, "edge": 0.0, "pending": 0}
    c = detail[detail["comparable"].astype(bool)]
    my = round(float(pd.to_numeric(c["my_pnl"]).sum()), 2)
    system = round(float(pd.to_numeric(c["system_pnl"]).sum()), 2)
    return {"items": int(len(detail)), "compared": int(len(c)), "my_pnl": my, "system_pnl": system,
            "edge": round(my - system, 2), "pending": int(len(detail) - len(c))}


# =========================================================================== agents

AGENT_ACCURACY_COLUMNS: tuple[str, ...] = (
    "agent", "votes", "agreed", "opposed", "neutral", "hit_rate_pct", "agreed_win_rate_pct", "agreed_net_pnl",
    "agreed_expectancy_r", "avg_confidence",
)


def agent_accuracy(session: Session, batch_id: int | None, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Per agent, judged on traded decisions (positions merged; failed / skipped calls ignored).

    A vote *agrees* when BULLISH on a LONG / BEARISH on a SHORT and *opposes* the other way;
    ``hit_rate_pct`` = (agreed & won + opposed & lost) / (agreed + opposed). Metrics of the
    trades the agent agreed with: win rate, net P&L and expectancy (R).
    """
    cols = list(AGENT_ACCURACY_COLUMNS)
    if batch_id is None or trades_df.empty:
        return _empty(cols)
    pos = mx.positions_from_trades(trades_df)
    pos = pos[pos["decision_id"].notna()] if "decision_id" in pos.columns else pos.iloc[0:0]
    if pos.empty:
        return _empty(cols)
    outcome: dict[int, tuple[str, float, float | None]] = {}
    for row in pos.itertuples(index=False):
        outcome[int(row.decision_id)] = (str(row.direction), _num(row.net_pnl) or 0.0, _num(row.r_multiple))
    reports = session.execute(
        select(AgentReportRow.agent, AgentReportRow.decision_id, AgentReportRow.signal, AgentReportRow.confidence)
        .where(AgentReportRow.batch_id == batch_id, AgentReportRow.decision_id.in_(sorted(outcome)),
               AgentReportRow.status == "OK")
    ).all()
    stats: dict[str, dict[str, Any]] = {}
    for agent, dec_id, signal, conf in reports:
        st_ = stats.setdefault(agent, {"votes": 0, "agreed": 0, "opposed": 0, "neutral": 0, "hits": 0,
                                       "conf": [], "agreed_pnl": [], "agreed_r": []})
        direction, net, r = outcome[int(dec_id)]
        st_["votes"] += 1
        st_["conf"].append(conf or 0)
        bull = signal == "BULLISH"
        bear = signal == "BEARISH"
        if not (bull or bear):
            st_["neutral"] += 1
            continue
        agrees = (bull and direction == Direction.LONG.value) or (bear and direction == Direction.SHORT.value)
        if agrees:
            st_["agreed"] += 1
            st_["agreed_pnl"].append(net)
            if r is not None:
                st_["agreed_r"].append(r)
            st_["hits"] += int(net > 0)
        else:
            st_["opposed"] += 1
            st_["hits"] += int(net <= 0)
    rows = []
    for agent, s in sorted(stats.items()):
        decided = s["agreed"] + s["opposed"]
        wr = mx.win_rate(s["agreed_pnl"])
        exp_r = mx.expectancy(s["agreed_r"])
        rows.append({
            "agent": agent, "votes": s["votes"], "agreed": s["agreed"], "opposed": s["opposed"],
            "neutral": s["neutral"],
            "hit_rate_pct": round(s["hits"] / decided * 100.0, 1) if decided else None,
            "agreed_win_rate_pct": round(wr * 100.0, 1) if wr is not None else None,
            "agreed_net_pnl": round(float(sum(s["agreed_pnl"])), 2),
            "agreed_expectancy_r": round(exp_r, 2) if exp_r is not None else None,
            "avg_confidence": round(float(np.mean(s["conf"])), 1) if s["conf"] else None,
        })
    return pd.DataFrame(rows, columns=cols)


AGENT_HEALTH_COLUMNS: tuple[str, ...] = (
    "agent", "calls", "ok", "failed", "skipped", "failure_rate_pct", "avg_latency_ms", "p50_latency_ms",
    "p95_latency_ms", "max_latency_ms", "avg_tokens_in", "avg_tokens_out", "models",
)


def agent_health(session: Session, batch_id: int | None) -> pd.DataFrame:
    """Latency and failure rate per agent (agent_reports), plus the debate and the meta decision.

    ``failure_rate_pct`` = FAILED / calls; SKIPPED calls count as calls but not failures.
    """
    cols = list(AGENT_HEALTH_COLUMNS)
    if batch_id is None:
        return _empty(cols)
    data: list[tuple[str, str, Any, Any, Any, Any]] = list(session.execute(
        select(AgentReportRow.agent, AgentReportRow.status, AgentReportRow.latency_ms, AgentReportRow.tokens_in,
               AgentReportRow.tokens_out, AgentReportRow.model).where(AgentReportRow.batch_id == batch_id)
    ).all())
    data.extend(("debate", status, lat, None, None, model) for status, lat, model in session.execute(
        select(DebateRow.status, DebateRow.latency_ms, DebateRow.model).where(DebateRow.batch_id == batch_id)
    ).all())
    data.extend(("meta (decision)", "OK", lat, None, None, None) for (lat,) in session.execute(
        select(DecisionRow.latency_ms).where(DecisionRow.batch_id == batch_id, DecisionRow.latency_ms.is_not(None))
    ).all())
    if not data:
        return _empty(cols)
    df = pd.DataFrame(data, columns=["agent", "status", "latency_ms", "tokens_in", "tokens_out", "model"])
    rows = []
    for agent, g in df.groupby("agent", sort=True):
        lat = pd.to_numeric(g["latency_ms"], errors="coerce").dropna()
        calls = int(len(g))
        failed = int((g["status"] == "FAILED").sum())
        tin = pd.to_numeric(g["tokens_in"], errors="coerce").dropna()
        tout = pd.to_numeric(g["tokens_out"], errors="coerce").dropna()
        rows.append({
            "agent": agent, "calls": calls, "ok": int((g["status"] == "OK").sum()), "failed": failed,
            "skipped": int((g["status"] == "SKIPPED").sum()),
            "failure_rate_pct": round(failed / calls * 100.0, 1) if calls else None,
            "avg_latency_ms": round(float(lat.mean()), 1) if not lat.empty else None,
            "p50_latency_ms": round(float(lat.quantile(0.5)), 1) if not lat.empty else None,
            "p95_latency_ms": round(float(lat.quantile(0.95)), 1) if not lat.empty else None,
            "max_latency_ms": round(float(lat.max()), 1) if not lat.empty else None,
            "avg_tokens_in": round(float(tin.mean()), 1) if not tin.empty else None,
            "avg_tokens_out": round(float(tout.mean()), 1) if not tout.empty else None,
            "models": ", ".join(sorted({str(m) for m in g["model"].dropna()})),
        })
    return pd.DataFrame(rows, columns=cols)


# =========================================================================== batch comparison

BATCH_COMPARE_COLUMNS: tuple[str, ...] = (
    "batch_id", "batch", "status", "starting_capital", "started", "ended", "days", "trades", "win_rate_pct",
    "profit_factor", "expectancy_r", "expectancy_rs", "net_pnl", "charges", "equity", "return_pct",
    "max_drawdown_pct", "reserve", "hwm",
)
_LIVE_STATUSES = (BatchStatus.ACTIVE.value, BatchStatus.PAUSED.value, BatchStatus.HALTED.value,
                  BatchStatus.CLOSING.value)


def _batch_curve(session: Session, batch: Batch, now: datetime | None) -> pd.DataFrame:
    live = None
    if now is not None and batch.status in _LIVE_STATUSES:
        live = live_point(capital_snapshot(session, batch.id), now)
    return equity_series(session, batch.id, live=live)


def batch_comparison(session: Session, batch_ids: Sequence[int], *, now: datetime | None = None) -> pd.DataFrame:
    """Side-by-side metrics, one row per batch (in the given order)."""
    cols = list(BATCH_COMPARE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for bid in batch_ids:
        batch = session.get(Batch, bid)
        if batch is None:
            continue
        m = overview_metrics(trades_frame(session, bid))
        curve = _batch_curve(session, batch, now)
        snap = capital_snapshot(session, bid) if batch.status in _LIVE_STATUSES else None
        acct = session.get(CapitalAccount, bid)
        start = float(batch.starting_capital)
        if snap is not None:
            equity = snap["equity"]
        elif not curve.empty:
            equity = float(curve["equity"].iloc[-1])
        elif acct is not None:
            equity = float(acct.trading_cash + acct.blocked + acct.reserve)
        else:
            equity = start + float(m["net_pnl"] or 0.0)
        started = batch.started_at or batch.created_at
        finished = batch.ended_at or now
        days = (finished - started).total_seconds() / 86400.0 if finished is not None and started is not None else None
        _abs, dd_pct = mx.max_drawdown(curve["equity"]) if not curve.empty else (0.0, 0.0)
        pf = m["profit_factor"]
        rows.append({
            "batch_id": bid, "batch": batch.name, "status": batch.status, "starting_capital": start,
            "started": naive_ist(started), "ended": naive_ist(batch.ended_at),
            "days": round(days, 1) if days is not None else None, "trades": m["trades"],
            "win_rate_pct": round(m["win_rate_pct"], 1) if m["win_rate_pct"] is not None else None,
            "profit_factor": round(pf, 2) if pf is not None and math.isfinite(pf) else pf,
            "expectancy_r": round(m["expectancy_r"], 3) if m["expectancy_r"] is not None else None,
            "expectancy_rs": round(m["expectancy_rs"], 2) if m["expectancy_rs"] is not None else None,
            "net_pnl": round(float(m["net_pnl"] or 0.0), 2), "charges": round(float(m["charges"] or 0.0), 2),
            "equity": round(equity, 2), "return_pct": round((equity / start - 1.0) * 100.0, 2) if start > 0 else None,
            "max_drawdown_pct": round(dd_pct, 2),
            "reserve": round(float(acct.reserve), 2) if acct is not None else None,
            "hwm": round(float(curve["hwm"].max()), 2) if not curve.empty else None,
        })
    return pd.DataFrame(rows, columns=cols)


BATCH_CURVE_COLUMNS: tuple[str, ...] = ("batch_id", "batch", "ts", "day_n", "equity", "return_pct")


def batch_equity_curves(session: Session, batch_ids: Sequence[int], *, now: datetime | None = None) -> pd.DataFrame:
    """Long-format equity curves for overlaying batches: ``day_n`` = days since the batch's first
    point, ``return_pct`` relative to its starting capital (comparable across capitals)."""
    cols = list(BATCH_CURVE_COLUMNS)
    frames: list[pd.DataFrame] = []
    for bid in batch_ids:
        batch = session.get(Batch, bid)
        if batch is None:
            continue
        curve = _batch_curve(session, batch, now)
        if curve.empty:
            continue
        start = float(batch.starting_capital) or float(curve["equity"].iloc[0])
        t0 = curve["ts"].iloc[0]
        frames.append(pd.DataFrame({
            "batch_id": bid, "batch": f"#{bid} {batch.name}", "ts": curve["ts"],
            "day_n": [(t - t0).total_seconds() / 86400.0 for t in curve["ts"]],
            "equity": curve["equity"], "return_pct": (curve["equity"] / start - 1.0) * 100.0 if start else np.nan,
        }))
    if not frames:
        return _empty(cols)
    return round_frame(pd.concat(frames, ignore_index=True)[cols], 3)


def batch_settings(session: Session, batch_id: int) -> tuple[dict[str, Any] | None, str]:
    """``(settings dict, source)`` of a batch: its snapshot, else its settings version, else the
    last version saved for / before the batch. ``(None, "none")`` when nothing is stored."""
    batch = session.get(Batch, batch_id)
    if batch is None:
        return None, "none"
    if batch.settings_snapshot:
        return dict(batch.settings_snapshot), "batch snapshot"
    if batch.settings_version_id is not None:
        row = session.get(SettingsVersion, batch.settings_version_id)
        if row is not None:
            return dict(row.settings_json), f"settings version {row.id}"
    row = session.execute(select(SettingsVersion).where(SettingsVersion.batch_id == batch_id)
                          .order_by(SettingsVersion.id.desc()).limit(1)).scalar_one_or_none()
    if row is None and batch.created_at is not None:
        row = session.execute(select(SettingsVersion).where(SettingsVersion.created_at <= batch.created_at)
                              .order_by(SettingsVersion.id.desc()).limit(1)).scalar_one_or_none()
    if row is not None:
        return dict(row.settings_json), f"settings version {row.id}"
    return None, "none"


def settings_diff_frame(a: dict[str, Any] | None, b: dict[str, Any] | None) -> pd.DataFrame:
    """``config.store.diff_settings`` of two settings dicts -> ``path / a / b`` (values as text)."""
    cols = ["path", "a", "b"]
    if a is None or b is None:
        return _empty(cols)
    changes = diff_settings(a, b)
    return pd.DataFrame([{"path": c["path"], "a": jsonish(c["old"]), "b": jsonish(c["new"])} for c in changes],
                        columns=cols)
