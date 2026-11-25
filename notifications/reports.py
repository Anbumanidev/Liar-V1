"""End-of-day report data (section 13) + the equity chart PNG.

``build_eod_report`` only READS the database and returns a JSON-friendly ``dict``
(dates/datetimes as ISO strings) consumed by ``Notifier.eod`` and the e-mail template.
All arithmetic is deterministic Python; nothing here mutates trading state.

Report keys::

    batch_id, batch_name, day, generated_at, empty
    modes[]            {mode, trades, wins, losses, gross, charges, net, win_rate}
    totals             {trades, wins, losses, gross, charges, net, win_rate, legs}
    trades[]           closed legs of the day
    open_positions[]   {symbol, mode, product, qty, avg_entry, last_price, unrealized_net, days_held, ...}
    open_unrealized_total
    capital            {trading_cash, blocked, trading_balance, reserve, unrealized_pnl, total_equity, hwm,
                        drawdown_pct, tier, sizing_capital, mode_sizing, realized_today, profit_locked,
                        loss_absorbed, refilled, ...}
    not_traded[]       {symbol, mode, source, stage, action, reason, grade, confidence}; not_traded_total
    approvals          {total, approved, modified, rejected, timeout, pending, items[]}
    shadow             {total, open, closed, target_hits, sl_hits, hypothetical_net, by_source[],
                        cumulative_by_source[], items[]}
    agent_failures     {total, by_agent[]}
    equity_series[]    {day, equity}

"Trades" in the per-mode table are POSITIONS with at least one closed leg on ``day``
(partial exits of one position are combined); a win is a combined net P&L > 0.
"""

from __future__ import annotations

import io
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from core.clock import Clock, ist_datetime, to_ist
from core.enums import (
    AgentStatus,
    ApprovalStatus,
    Direction,
    LedgerEntryType,
    Mode,
    PositionStatus,
    ShadowOutcome,
)
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    DailySummary,
    DecisionRow,
    Position,
    ShadowTrade,
    StockAnalyzed,
    Trade,
)

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

TOTAL_MODE = "TOTAL"
_MODE_ORDER = {m.value: i for i, m in enumerate(Mode)}
_WIN_OUTCOMES = {ShadowOutcome.TARGET_1.value, ShadowOutcome.TARGET_2.value}


# =========================================================================== helpers


def _r2(value: float | None, nd: int = 2) -> float | None:
    """Round for presentation (None stays None)."""
    return None if value is None else round(float(value), nd)


def _iso(value: datetime | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_ist(value).isoformat()
    return value.isoformat()


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = ist_datetime(day, "00:00")
    return start, start + timedelta(days=1)


def _win_rate(wins: int, count: int) -> float | None:
    return round(wins / count * 100.0, 2) if count > 0 else None


def _mode_sort_key(mode: str) -> tuple[int, str]:
    return (_MODE_ORDER.get(mode, len(_MODE_ORDER)), mode)


def unrealized_pnl(direction: str, qty: int, avg_entry: float, last_price: float) -> float:
    """Gross mark-to-market P&L of ``qty`` shares (LONG: last - entry; SHORT: entry - last)."""
    diff = (last_price - avg_entry) if direction == Direction.LONG.value else (avg_entry - last_price)
    return diff * qty


# =========================================================================== sections


def _trades_section(session: Session, batch_id: int, start: datetime, end: datetime) -> dict[str, Any]:
    rows = session.scalars(
        select(Trade)
        .where(Trade.batch_id == batch_id, Trade.exit_time >= start, Trade.exit_time < end)
        .order_by(Trade.exit_time, Trade.id)
    ).all()

    legs: list[dict[str, Any]] = []
    # per position: combined net/gross/charges of today's legs
    per_position: dict[int, dict[str, Any]] = {}
    for t in rows:
        legs.append(
            {
                "id": t.id,
                "position_id": t.position_id,
                "symbol": t.symbol,
                "mode": t.mode,
                "product": t.product,
                "direction": t.direction,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": _iso(t.entry_time),
                "exit_time": _iso(t.exit_time),
                "exit_reason": t.exit_reason,
                "is_partial": bool(t.is_partial),
                "gross": _r2(t.gross_pnl),
                "charges": _r2(t.charges),
                "net": _r2(t.net_pnl),
                "r_multiple": _r2(t.r_multiple),
                "strategy": t.strategy,
                "source": t.source,
            }
        )
        agg = per_position.setdefault(
            t.position_id, {"mode": t.mode, "gross": 0.0, "charges": 0.0, "net": 0.0}
        )
        agg["gross"] += float(t.gross_pnl or 0.0)
        agg["charges"] += float(t.charges or 0.0)
        agg["net"] += float(t.net_pnl or 0.0)

    by_mode: dict[str, dict[str, Any]] = {}
    for agg in per_position.values():
        row = by_mode.setdefault(
            agg["mode"],
            {"mode": agg["mode"], "trades": 0, "wins": 0, "losses": 0, "gross": 0.0, "charges": 0.0, "net": 0.0},
        )
        row["trades"] += 1
        row["wins"] += 1 if agg["net"] > 0 else 0
        row["losses"] += 1 if agg["net"] < 0 else 0
        row["gross"] += agg["gross"]
        row["charges"] += agg["charges"]
        row["net"] += agg["net"]

    modes: list[dict[str, Any]] = []
    for mode in sorted(by_mode, key=_mode_sort_key):
        row = by_mode[mode]
        modes.append(
            {
                **row,
                "gross": _r2(row["gross"]),
                "charges": _r2(row["charges"]),
                "net": _r2(row["net"]),
                "win_rate": _win_rate(row["wins"], row["trades"]),
            }
        )
    n = sum(r["trades"] for r in by_mode.values())
    wins = sum(r["wins"] for r in by_mode.values())
    totals = {
        "trades": n,
        "legs": len(legs),
        "wins": wins,
        "losses": sum(r["losses"] for r in by_mode.values()),
        "gross": _r2(sum(r["gross"] for r in by_mode.values())) or 0.0,
        "charges": _r2(sum(r["charges"] for r in by_mode.values())) or 0.0,
        "net": _r2(sum(r["net"] for r in by_mode.values())) or 0.0,
        "win_rate": _win_rate(wins, n),
    }
    # raw (unrounded) realized net for the capital section
    realized_raw = sum(float(t.net_pnl or 0.0) for t in rows)
    return {"modes": modes, "totals": totals, "trades": legs, "realized_raw": realized_raw,
            "traded_keys": {(t.symbol, t.mode) for t in rows}}


def _open_positions_section(session: Session, batch_id: int, day: date, end: datetime) -> dict[str, Any]:
    rows = session.scalars(
        select(Position)
        .where(Position.batch_id == batch_id, Position.status == PositionStatus.OPEN.value,
               Position.opened_at < end)
        .order_by(Position.opened_at, Position.id)
    ).all()
    out: list[dict[str, Any]] = []
    total_net = 0.0
    for p in rows:
        price_missing = p.last_price is None or p.last_price <= 0
        mark = float(p.avg_entry) if price_missing else float(p.last_price)  # type: ignore[arg-type]
        gross = unrealized_pnl(p.direction, int(p.qty), float(p.avg_entry), mark)
        net = gross - float(p.entry_charges or 0.0)
        total_net += net
        cost = float(p.avg_entry) * int(p.qty)
        opened_day = to_ist(p.opened_at).date()
        out.append(
            {
                "id": p.id,
                "symbol": p.symbol,
                "mode": p.mode,
                "product": p.product,
                "direction": p.direction,
                "qty": p.qty,
                "avg_entry": p.avg_entry,
                "last_price": None if price_missing else p.last_price,
                "last_price_at": _iso(p.last_price_at),
                "price_missing": price_missing,
                "stop_loss": p.stop_loss,
                "target_1": p.target_1,
                "target_2": p.target_2,
                "entry_charges": _r2(p.entry_charges),
                "unrealized_gross": _r2(gross),
                "unrealized_net": _r2(net),
                "unrealized_pct": _r2(net / cost * 100.0) if cost > 0 else None,
                "days_held": int(p.holding_days or 0),
                "calendar_days": max(0, (day - opened_day).days),
                "opened_at": _iso(p.opened_at),
                "strategy": p.strategy,
                "margin_blocked": _r2(p.margin_blocked),
            }
        )
    return {"open_positions": out, "open_unrealized_total": _r2(total_net) or 0.0, "unrealized_raw": total_net}


def _ledger_day(session: Session, batch_id: int, start: datetime, end: datetime) -> dict[str, float]:
    rows = session.scalars(
        select(CapitalLedgerEntry)
        .where(CapitalLedgerEntry.batch_id == batch_id, CapitalLedgerEntry.ts >= start,
               CapitalLedgerEntry.ts < end)
        .order_by(CapitalLedgerEntry.ts, CapitalLedgerEntry.id)
    ).all()
    sums: dict[str, float] = defaultdict(float)
    eod_marker = False
    for e in rows:
        amount = float(e.amount or 0.0)
        et = e.entry_type
        if et == LedgerEntryType.PROFIT_LOCK.value:
            sums["profit_locked"] += abs(amount)
            eod_marker = True
        elif et == LedgerEntryType.REFILL.value:
            sums["refilled"] += abs(amount)
            eod_marker = True
        elif et == LedgerEntryType.LOSS_ABSORB.value:
            sums["loss_absorb_marker"] += abs(amount)
            eod_marker = True
        elif et == LedgerEntryType.MANUAL_TRANSFER.value:
            if e.confirmed:
                sums["manual_transfers"] += abs(amount)
        elif et == LedgerEntryType.MANUAL_ADJUSTMENT.value:
            if e.confirmed:
                sums["manual_adjustments"] += amount
        elif et == LedgerEntryType.SIZING_CHANGE.value:
            sums["sizing_change"] += amount
        elif et == LedgerEntryType.DIVIDEND.value:
            sums["dividends"] += amount
    sums["eod_marker"] = 1.0 if eod_marker else 0.0
    return sums


def _capital_section(
    session: Session,
    batch_id: int,
    day: date,
    start: datetime,
    end: datetime,
    realized_trades: float,
    unrealized_total: float,
    settings: "Settings | None",
) -> dict[str, Any]:
    acct = session.get(CapitalAccount, batch_id)
    ledger = _ledger_day(session, batch_id, start, end)
    summary_rows = session.scalars(
        select(DailySummary).where(DailySummary.batch_id == batch_id, DailySummary.day == day)
    ).all()
    total_row = next((r for r in summary_rows if r.mode == TOTAL_MODE), None)

    realized_today = realized_trades + ledger.get("dividends", 0.0)
    base: dict[str, Any] = {
        "available": acct is not None,
        "realized_today": _r2(realized_today) or 0.0,
        "dividends_today": _r2(ledger.get("dividends", 0.0)) or 0.0,
        "profit_locked": _r2(ledger.get("profit_locked", 0.0)) or 0.0,
        "refilled": _r2(ledger.get("refilled", 0.0)) or 0.0,
        "loss_absorbed": _r2(max(0.0, -realized_today)) or 0.0,
        "manual_transfers": _r2(ledger.get("manual_transfers", 0.0)) or 0.0,
        "manual_adjustments": _r2(ledger.get("manual_adjustments", 0.0)) or 0.0,
        "sizing_change": _r2(ledger.get("sizing_change", 0.0)) or 0.0,
        "mode_sizing": {},
        "source": "none",
    }
    if acct is None:
        base.update(
            {
                "starting_capital": None, "base_capital": None, "trading_cash": None, "blocked": None,
                "trading_balance": None, "reserve": None, "reserve_peak": None,
                "unrealized_pnl": _r2(unrealized_total) or 0.0, "total_equity": None, "hwm": None,
                "drawdown_pct": None, "tier": "UNKNOWN", "sizing_capital": None, "halted": False,
                "halt_reason": None, "eod_processed": False,
            }
        )
        return base

    trading_balance = float(acct.trading_cash) + float(acct.blocked)
    equity_live = trading_balance + float(acct.reserve) + unrealized_total
    if total_row is not None and total_row.equity is not None:
        equity = float(total_row.equity)
        hwm = float(total_row.hwm) if total_row.hwm is not None else max(float(acct.hwm), equity)
        tier = total_row.tier or acct.tier
        sizing = float(total_row.sizing_capital) if total_row.sizing_capital is not None else float(acct.sizing_capital)
        source = "daily_summary"
        # prefer ledger movements; fall back to the summary row if the ledger has none
        if not ledger.get("profit_locked") and total_row.profit_locked:
            base["profit_locked"] = _r2(total_row.profit_locked) or 0.0
        if not ledger.get("refilled") and total_row.refilled:
            base["refilled"] = _r2(total_row.refilled) or 0.0
        unreal = float(total_row.unrealized_pnl) if total_row.unrealized_pnl is not None else unrealized_total
    else:
        equity = equity_live
        hwm = max(float(acct.hwm), equity)
        tier = acct.tier
        sizing = float(acct.sizing_capital)
        source = "computed"
        unreal = unrealized_total
    drawdown = (hwm - equity) / hwm * 100.0 if hwm > 0 else 0.0
    if total_row is not None and total_row.drawdown_pct is not None:
        drawdown = float(total_row.drawdown_pct)

    mode_sizing: dict[str, float] = {}
    for r in sorted(summary_rows, key=lambda r: _mode_sort_key(r.mode)):
        if r.mode != TOTAL_MODE and r.sizing_capital is not None:
            mode_sizing[r.mode] = _r2(r.sizing_capital) or 0.0
    if not mode_sizing and settings is not None:
        for mode in Mode:
            cfg = settings.modes.modes.get(mode)
            if cfg is not None and cfg.enabled and cfg.allocation_pct > 0:
                mode_sizing[mode.value] = _r2(sizing * cfg.allocation_pct / 100.0) or 0.0

    base.update(
        {
            "starting_capital": _r2(acct.starting_capital),
            "base_capital": _r2(acct.base_capital),
            "trading_cash": _r2(acct.trading_cash),
            "blocked": _r2(acct.blocked),
            "trading_balance": _r2(trading_balance),
            "reserve": _r2(acct.reserve),
            "reserve_peak": _r2(acct.reserve_peak),
            "unrealized_pnl": _r2(unreal) or 0.0,
            "total_equity": _r2(equity),
            "hwm": _r2(hwm),
            "drawdown_pct": _r2(max(0.0, drawdown)),
            "tier": tier,
            "sizing_capital": _r2(sizing),
            "mode_sizing": mode_sizing,
            "halted": bool(acct.halted),
            "halt_reason": acct.halt_reason,
            "eod_processed": bool(acct.last_eod_date == day or ledger.get("eod_marker")),
            "source": source,
        }
    )
    return base


def _decision_reason(d: DecisionRow, approval: ApprovalItem | None) -> str:
    action = d.final_action
    meta = d.meta_json or {}
    if action == "RISK_REJECTED":
        return f"Risk veto: {d.risk_rule or 'rule not recorded'}"
    if action == "QUEUED":
        if approval is None:
            return "Queued for approval"
        if approval.status == ApprovalStatus.REJECTED.value:
            why = f": {approval.user_reason}" if approval.user_reason else ""
            return f"Rejected by owner{why}"
        if approval.status == ApprovalStatus.TIMEOUT.value:
            return "Approval timed out"
        if approval.status == ApprovalStatus.PENDING.value:
            return "Awaiting owner approval"
        if approval.block_rule:
            return f"Owner approved, blocked by risk: {approval.block_rule}"
        return f"Approval {approval.status.lower()}, not filled"
    if action == "SIGNAL_ONLY":
        return "Signals-only mode (never executed)"
    if action == "FILTERED":
        ctx = d.context_json or {}
        reason = ctx.get("reason") or ctx.get("filter_reason")
        return f"Filtered: {reason}" if reason else "Filtered before meta (strategy/confluence)"
    # NO_TRADE and anything else
    text = meta.get("reasoning") or meta.get("reason") or meta.get("summary")
    conf = f" (confidence {d.confidence})" if d.confidence else ""
    if text:
        text = " ".join(str(text).split())
        if len(text) > 160:
            text = text[:157] + "..."
        return f"Meta NO_TRADE{conf}: {text}"
    return f"Meta {d.decision}{conf}" if d.decision else "No trade"


def _not_traded_section(
    session: Session, batch_id: int, start: datetime, end: datetime, traded: set[tuple[str, str]]
) -> dict[str, Any]:
    decisions = session.scalars(
        select(DecisionRow)
        .where(DecisionRow.batch_id == batch_id, DecisionRow.created_at >= start, DecisionRow.created_at < end,
               DecisionRow.final_action != "EXECUTED")
        .order_by(DecisionRow.created_at, DecisionRow.id)
    ).all()
    approvals_by_decision: dict[int, ApprovalItem] = {}
    decision_ids = [d.id for d in decisions]
    if decision_ids:
        for a in session.scalars(
            select(ApprovalItem).where(ApprovalItem.decision_id.in_(decision_ids))
        ).all():
            if a.decision_id is not None:
                approvals_by_decision[a.decision_id] = a

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for d in decisions:  # ascending -> later rows overwrite earlier ones
        key = (d.symbol, d.mode)
        if key in traded:
            continue
        latest[key] = {
            "symbol": d.symbol,
            "mode": d.mode,
            "source": "DECISION",
            "stage": "ANALYZED",
            "action": d.final_action,
            "reason": _decision_reason(d, approvals_by_decision.get(d.id)),
            "grade": d.grade,
            "confidence": d.confidence,
            "decision_id": d.id,
        }

    excluded = session.scalars(
        select(StockAnalyzed)
        .where(StockAnalyzed.batch_id == batch_id, StockAnalyzed.as_of >= start, StockAnalyzed.as_of < end,
               StockAnalyzed.included.is_(False))
        .order_by(StockAnalyzed.as_of, StockAnalyzed.id)
    ).all()
    filtered: dict[tuple[str, str], dict[str, Any]] = {}
    for s in excluded:
        key = (s.symbol, s.mode)
        if key in traded or key in latest:
            continue
        filtered[key] = {
            "symbol": s.symbol,
            "mode": s.mode,
            "source": "FILTER",
            "stage": s.stage,
            "action": "EXCLUDED",
            "reason": f"{s.stage.title()} filter: {s.reason}" if s.reason else f"Excluded at {s.stage.lower()} stage",
            "grade": s.grade,
            "confidence": None,
            "decision_id": None,
        }
    items = sorted(latest.values(), key=lambda r: (_mode_sort_key(r["mode"]), r["symbol"])) + sorted(
        filtered.values(), key=lambda r: (_mode_sort_key(r["mode"]), r["symbol"])
    )
    return {"not_traded": items, "not_traded_total": len(items)}


def _approvals_section(session: Session, batch_id: int, start: datetime, end: datetime) -> dict[str, Any]:
    rows = session.scalars(
        select(ApprovalItem)
        .where(
            ApprovalItem.batch_id == batch_id,
            or_(
                and_(ApprovalItem.created_at >= start, ApprovalItem.created_at < end),
                and_(ApprovalItem.decided_at >= start, ApprovalItem.decided_at < end),
            ),
        )
        .order_by(ApprovalItem.created_at, ApprovalItem.id)
    ).all()
    counts = {s.value: 0 for s in ApprovalStatus}
    items: list[dict[str, Any]] = []
    for a in rows:
        counts[a.status] = counts.get(a.status, 0) + 1
        items.append(
            {
                "id": a.id,
                "symbol": a.symbol,
                "mode": a.mode,
                "direction": a.direction,
                "status": a.status,
                "decided_by": a.decided_by,
                "user_reason": a.user_reason,
                "block_rule": a.block_rule,
                "created_at": _iso(a.created_at),
                "decided_at": _iso(a.decided_at),
                "order_id": a.order_id,
            }
        )
    return {
        "total": len(rows),
        "approved": counts.get(ApprovalStatus.APPROVED.value, 0),
        "modified": counts.get(ApprovalStatus.MODIFIED.value, 0),
        "rejected": counts.get(ApprovalStatus.REJECTED.value, 0),
        "timeout": counts.get(ApprovalStatus.TIMEOUT.value, 0),
        "pending": counts.get(ApprovalStatus.PENDING.value, 0),
        "items": items,
    }


def _shadow_rollup(rows: Iterable[ShadowTrade], closed_only: bool = False) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for s in rows:
        is_closed = s.outcome != ShadowOutcome.OPEN.value
        if closed_only and not is_closed:
            continue
        row = by_source.setdefault(
            s.rejection_source,
            {"source": s.rejection_source, "count": 0, "closed": 0, "target_hits": 0, "sl_hits": 0,
             "not_triggered": 0, "hypothetical_net": 0.0},
        )
        row["count"] += 1
        if is_closed:
            row["closed"] += 1
        if s.outcome in _WIN_OUTCOMES:
            row["target_hits"] += 1
        elif s.outcome == ShadowOutcome.SL.value:
            row["sl_hits"] += 1
        elif s.outcome == ShadowOutcome.NOT_TRIGGERED.value:
            row["not_triggered"] += 1
        if s.hypothetical_net_pnl is not None and is_closed:
            row["hypothetical_net"] += float(s.hypothetical_net_pnl)
    out = []
    for src in sorted(by_source):
        row = by_source[src]
        row["hypothetical_net"] = _r2(row["hypothetical_net"]) or 0.0
        out.append(row)
    return out


def _shadow_section(session: Session, batch_id: int, start: datetime, end: datetime) -> dict[str, Any]:
    rows = session.scalars(
        select(ShadowTrade)
        .where(
            ShadowTrade.batch_id == batch_id,
            or_(
                and_(ShadowTrade.created_at >= start, ShadowTrade.created_at < end),
                and_(ShadowTrade.exit_time >= start, ShadowTrade.exit_time < end),
            ),
        )
        .order_by(ShadowTrade.created_at, ShadowTrade.id)
    ).all()
    cumulative = session.scalars(
        select(ShadowTrade).where(ShadowTrade.batch_id == batch_id, ShadowTrade.created_at < end)
    ).all()
    items = [
        {
            "id": s.id,
            "symbol": s.symbol,
            "mode": s.mode,
            "direction": s.direction,
            "rejection_source": s.rejection_source,
            "rejection_rule": s.rejection_rule,
            "outcome": s.outcome,
            "r_multiple": _r2(s.r_multiple),
            "hypothetical_net": _r2(s.hypothetical_net_pnl),
            "created_at": _iso(s.created_at),
            "exit_time": _iso(s.exit_time),
        }
        for s in rows
    ]
    closed = [s for s in rows if s.outcome != ShadowOutcome.OPEN.value]
    return {
        "total": len(rows),
        "open": len(rows) - len(closed),
        "closed": len(closed),
        "target_hits": sum(1 for s in rows if s.outcome in _WIN_OUTCOMES),
        "sl_hits": sum(1 for s in rows if s.outcome == ShadowOutcome.SL.value),
        "hypothetical_net": _r2(sum(float(s.hypothetical_net_pnl or 0.0) for s in closed)) or 0.0,
        "by_source": _shadow_rollup(rows),
        "cumulative_by_source": _shadow_rollup(cumulative, closed_only=True),
        "items": items,
    }


def _agent_failures_section(session: Session, batch_id: int, start: datetime, end: datetime) -> dict[str, Any]:
    rows = session.scalars(
        select(AgentReportRow)
        .where(AgentReportRow.batch_id == batch_id, AgentReportRow.status == AgentStatus.FAILED.value,
               AgentReportRow.created_at >= start, AgentReportRow.created_at < end)
        .order_by(AgentReportRow.created_at, AgentReportRow.id)
    ).all()
    by_agent: dict[str, dict[str, Any]] = {}
    for r in rows:
        row = by_agent.setdefault(r.agent, {"agent": r.agent, "count": 0, "symbols": [], "last_error": None})
        row["count"] += 1
        if r.symbol not in row["symbols"]:
            row["symbols"].append(r.symbol)
        if r.error:
            row["last_error"] = " ".join(r.error.split())
    ordered = sorted(by_agent.values(), key=lambda x: (-x["count"], x["agent"]))
    return {"total": len(rows), "by_agent": ordered}


def _equity_series(session: Session, batch_id: int, day: date, today_equity: float | None) -> list[dict[str, Any]]:
    rows = session.scalars(
        select(DailySummary)
        .where(DailySummary.batch_id == batch_id, DailySummary.mode == TOTAL_MODE, DailySummary.day <= day,
               DailySummary.equity.is_not(None))
        .order_by(DailySummary.day)
    ).all()
    series = [{"day": r.day.isoformat(), "equity": _r2(r.equity)} for r in rows]
    if today_equity is not None and (not series or series[-1]["day"] != day.isoformat()):
        series.append({"day": day.isoformat(), "equity": _r2(today_equity)})
    return series


# =========================================================================== public API


def build_eod_report(
    session: Session,
    batch_id: int,
    day: date,
    clock: Clock,
    *,
    settings: "Settings | None" = None,
) -> dict[str, Any]:
    """Collect everything the EOD e-mail shows for ``batch_id`` on ``day`` (IST).

    Args:
        session: open SQLAlchemy session (read-only use).
        batch_id: batch to report on.
        day: trading day (IST).
        clock: time source for ``generated_at``.
        settings: optional; used only to derive per-mode sizing capital when the
            ``daily_summary`` per-mode rows do not carry it.

    Returns:
        JSON-friendly dict (see module docstring). An empty day yields zeroed sections.
    """
    start, end = _day_bounds(day)
    batch = session.get(Batch, batch_id)

    trades = _trades_section(session, batch_id, start, end)
    positions = _open_positions_section(session, batch_id, day, end)

    opened_today = session.execute(
        select(Position.symbol, Position.mode).where(
            Position.batch_id == batch_id, Position.opened_at >= start, Position.opened_at < end
        )
    ).all()
    traded_keys: set[tuple[str, str]] = set(trades["traded_keys"]) | {(s, m) for s, m in opened_today}

    capital = _capital_section(
        session, batch_id, day, start, end, trades["realized_raw"], positions["unrealized_raw"], settings
    )
    not_traded = _not_traded_section(session, batch_id, start, end, traded_keys)
    approvals = _approvals_section(session, batch_id, start, end)
    shadow = _shadow_section(session, batch_id, start, end)
    failures = _agent_failures_section(session, batch_id, start, end)
    series = _equity_series(session, batch_id, day, capital.get("total_equity"))

    empty = (
        trades["totals"]["legs"] == 0
        and not positions["open_positions"]
        and approvals["total"] == 0
        and not_traded["not_traded_total"] == 0
        and shadow["total"] == 0
        and failures["total"] == 0
    )
    report: dict[str, Any] = {
        "batch_id": batch_id,
        "batch_name": batch.name if batch is not None else None,
        "day": day.isoformat(),
        "generated_at": clock.now().isoformat(),
        "empty": empty,
        "modes": trades["modes"],
        "totals": trades["totals"],
        "trades": trades["trades"],
        "open_positions": positions["open_positions"],
        "open_unrealized_total": positions["open_unrealized_total"],
        "capital": capital,
        "not_traded": not_traded["not_traded"],
        "not_traded_total": not_traded["not_traded_total"],
        "approvals": approvals,
        "shadow": shadow,
        "agent_failures": failures,
        "equity_series": series,
    }
    logger.debug(
        f"EOD report built for batch {batch_id} {day}: {trades['totals']['trades']} trades, "
        f"{len(positions['open_positions'])} open, net {trades['totals']['net']}"
    )
    return report


#: Fields of a next-day plan item shown in the EOD e-mail.
PLAN_FIELDS = ("setup", "direction", "trigger", "stop", "target", "rr", "score", "reasons", "affordable")


def next_day_plan_section(session: Session, on_or_after: date, *, opening_check: str = "09:20") -> dict[str, Any]:
    """Optional EOD keys for tomorrow's INTRADAY plan (``data.watchlists.latest_plan``).

    Returns ``{"next_day_plan": [ {symbol, setup, direction, trigger, stop, target, rr, score, reasons,
    affordable, status}, ...], "next_day_plan_for": ISO date, "next_day_plan_name": str,
    "next_day_plan_check": "HH:MM"}`` or ``{}`` when no plan exists (the template tolerates both).
    Read-only; the plan is a paper watchlist, not advice - every trade still needs the risk manager.
    """
    from data.watchlists import latest_plan

    plan = latest_plan(session, on_or_after=on_or_after)
    if plan is None:
        return {}
    rows: list[dict[str, Any]] = []
    for item in plan["items"]:
        p = item.get("plan") or {}
        row = {"symbol": item["symbol"], "status": item.get("status")}
        row.update({k: p.get(k) for k in PLAN_FIELDS})
        row["reasons"] = list(row.get("reasons") or [])
        rows.append(row)
    return {"next_day_plan": rows, "next_day_plan_for": plan.get("for_date"), "next_day_plan_name": plan.get("name"),
            "next_day_plan_check": opening_check}


# =========================================================================== chart

# Light e-mail surface; single series -> no legend (the title names it).
_CHART_SURFACE = "#fcfcfb"
_CHART_SERIES = "#2a78d6"
_CHART_GRID = "#e2e1dc"
_CHART_TEXT = "#0b0b0b"
_CHART_TEXT_2 = "#52514e"
_CHART_REF = "#8a8984"


def _coerce_points(points: Sequence[Any]) -> list[tuple[date, float]]:
    out: list[tuple[date, float]] = []
    for pt in points:
        try:
            if isinstance(pt, dict):
                raw_day, value = pt.get("day"), pt.get("equity")
            else:
                raw_day, value = pt[0], pt[1]
            if value is None:
                continue
            if isinstance(raw_day, datetime):
                d = to_ist(raw_day).date()
            elif isinstance(raw_day, date):
                d = raw_day
            else:
                d = date.fromisoformat(str(raw_day)[:10])
            out.append((d, float(value)))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            logger.warning(f"equity_chart_png: skipping bad point {pt!r}: {exc}")
    out.sort(key=lambda x: x[0])
    return out


def _fmt_axis_inr(value: float, _pos: Any = None) -> str:
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.1f}L"
    if abs(value) >= 10_000:
        return f"₹{value / 1000:.0f}K"
    return f"₹{value:,.0f}"


def equity_chart_png(
    points: Sequence[Any],
    *,
    baseline: float | None = None,
    floor: float | None = None,
    title: str = "Total equity (₹, end of day)",
    width_px: int = 1136,
    height_px: int = 520,
) -> bytes:
    """Render the equity curve as PNG bytes (matplotlib Agg, no GUI, thread-safe).

    Args:
        points: ``[{"day": date|iso, "equity": float}, ...]`` or ``[(day, equity), ...]``.
        baseline: optional starting capital reference line.
        floor: optional capital-protection floor reference line.
        title: chart title (single series: the title names it, no legend).
        width_px / height_px: output size (rendered at 2x for sharpness in e-mail).

    Returns:
        PNG bytes. With no usable points a small placeholder image is returned.
    """
    from matplotlib.figure import Figure  # local import keeps module import light
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    data = _coerce_points(points)
    dpi = 200
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=_CHART_SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(_CHART_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_CHART_GRID)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=_CHART_TEXT_2, labelsize=6, length=0, pad=3)
    ax.grid(axis="y", color=_CHART_GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8, color=_CHART_TEXT, fontweight="bold", pad=8)

    if not data:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No equity history yet", ha="center", va="center", fontsize=8,
                color=_CHART_TEXT_2, transform=ax.transAxes)
    else:
        xs = list(range(len(data)))
        ys = [v for _, v in data]
        ax.plot(xs, ys, color=_CHART_SERIES, linewidth=1.0, solid_joinstyle="round", solid_capstyle="round",
                zorder=3)
        ax.scatter([xs[-1]], [ys[-1]], s=10, color=_CHART_SERIES, edgecolors=_CHART_SURFACE, linewidths=0.8,
                   zorder=4)
        if len(xs) == 1:
            ax.scatter(xs, ys, s=10, color=_CHART_SERIES, zorder=4)
        # value label at the line end (text wears text ink, not the series colour)
        ax.annotate(_fmt_axis_inr(ys[-1]) if abs(ys[-1]) >= 10_000 else f"₹{ys[-1]:,.2f}",
                    (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=6,
                    color=_CHART_TEXT, va="center", ha="left")
        lo, hi = min(ys), max(ys)
        for ref, label in ((baseline, "start"), (floor, "floor")):
            if ref is not None:
                ax.axhline(ref, color=_CHART_REF, linewidth=0.5, linestyle="-", zorder=2)
                ax.annotate(f"{label} {_fmt_axis_inr(ref)}", (0, ref), xytext=(0, 2), textcoords="offset points",
                            fontsize=5, color=_CHART_TEXT_2, va="bottom", ha="left")
                lo, hi = min(lo, ref), max(hi, ref)
        pad = (hi - lo) * 0.12 if hi > lo else max(abs(hi) * 0.02, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        if len(xs) == 1:
            ax.set_xlim(-1.0, 1.5)
        else:
            ax.set_xlim(-0.3, len(xs) - 1 + 0.8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_axis_inr))
        n_ticks = min(len(xs), 6)
        if n_ticks > 0:
            step = max(1, (len(xs) - 1) // max(n_ticks - 1, 1)) if len(xs) > 1 else 1
            idx = list(range(0, len(xs), step))
            if idx[-1] != len(xs) - 1:
                idx.append(len(xs) - 1)
            ax.set_xticks(idx)
            ax.set_xticklabels([data[i][0].strftime("%d %b") for i in idx])

    fig.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=_CHART_SURFACE)
    return buf.getvalue()
