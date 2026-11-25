"""Read-only DB queries behind the web API. Plain functions over a ``Session`` returning dicts.

All arithmetic (P&L, R, progress, win rate, funnel counts) is deterministic Python - no LLM
ever computes a number. Money/prices are rounded to 2 dp; times are IST ISO strings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import or_, func, select
from sqlalchemy.orm import Session

from core.clock import to_ist
from core.enums import ApprovalStatus, BatchStatus, Direction, OrderStatus, PositionStatus
from storage.batches import CURRENT_STATUSES
from storage.models import (
    AgentReportRow,
    ApprovalItem,
    Batch,
    CapitalAccount,
    CapitalLedgerEntry,
    ControlCommand,
    DailySummary,
    DebateRow,
    DecisionRow,
    EngineStatus,
    ErrorLog,
    Order,
    Position,
    StockAnalyzed,
    Trade,
)
from webui.util import chart_time, iso, r2, r4, scalars_only, str_list, text

EXIT_REASON_LABELS: dict[str, str] = {
    "SL": "Stop-loss hit",
    "TARGET": "Target hit",
    "TRAIL": "Trailing stop",
    "TIME": "Time stop",
    "SQUAREOFF": "Square-off",
    "GAP": "Gap exit",
    "EVENT": "Event exit",
    "HALT": "Capital halt",
    "MANUAL": "Manual close",
    "KILL_SWITCH": "Kill switch",
    "BATCH_CLOSE": "Batch closed",
}

FINAL_ACTIONS = ("EXECUTED", "QUEUED", "RISK_REJECTED", "NO_TRADE", "FILTERED", "SIGNAL_ONLY", "ANALYZED")


# =========================================================================== helpers


def sign(direction: str | None) -> float:
    return -1.0 if direction == Direction.SHORT.value else 1.0


def current_batch(session: Session) -> Batch | None:
    return session.execute(
        select(Batch).where(Batch.status.in_(CURRENT_STATUSES), Batch.deleted.is_(False))
        .order_by(Batch.id.desc()).limit(1)
    ).scalar_one_or_none()


def batch_brief(b: Batch | None) -> dict[str, Any] | None:
    if b is None:
        return None
    return {"id": b.id, "name": b.name, "status": b.status, "starting_capital": r2(b.starting_capital),
            "created_at": iso(b.created_at), "started_at": iso(b.started_at), "ended_at": iso(b.ended_at),
            "halted_at": iso(b.halted_at), "halt_reason": text(b.halt_reason, 300)}


def exit_label(reason: str | None, *, direction: str | None = None, exit_price: float | None = None,
               target_2: float | None = None, partial: bool = False, breakeven: bool = False) -> str:
    """Plain-language exit reason ("Stop-loss hit", "Target 1", "Trailing stop", ...)."""
    code = (reason or "").upper()
    if code == "TARGET":
        if target_2 is not None and exit_price is not None and sign(direction) * (exit_price - target_2) >= -1e-9:
            return "Target 2"
        return "Partial profit (T1)" if partial else "Target 1"
    if code == "SL" and breakeven:
        return "Stop-loss hit (breakeven)"
    label = EXIT_REASON_LABELS.get(code, code.title() if code else "-")
    return f"{label} (partial)" if partial and code not in ("TARGET",) else label


# =========================================================================== engine


def engine_view(session: Session, now: datetime, stale_seconds: int) -> dict[str, Any]:
    """Engine heartbeat/status row as the status bar needs it."""
    row = session.get(EngineStatus, 1)
    if row is None:
        return {"state": "UNKNOWN", "label": "NO HEARTBEAT", "alive": False, "stale": True, "heartbeat_at": None,
                "heartbeat_age_s": None, "activity": None, "next_cycle_at": None, "market_open": None,
                "data_feed_ok": None, "llm_ok": None, "last_error": None, "batch_id": None, "started_at": None,
                "regime": {}, "status": {}, "stale_after_s": stale_seconds}
    age = (now - row.heartbeat_at).total_seconds() if row.heartbeat_at else None
    stale = age is None or age > stale_seconds
    state = row.state or "UNKNOWN"
    return {
        "state": state,
        "label": f"{state} (stale)" if stale and state != "STOPPED" else state,
        "alive": (not stale) and state != "STOPPED",
        "stale": stale,
        "heartbeat_at": iso(row.heartbeat_at),
        "heartbeat_age_s": None if age is None else round(max(age, 0.0), 1),
        "activity": text(row.activity, 300),
        "next_cycle_at": iso(row.next_cycle_at),
        "market_open": row.market_open,
        "data_feed_ok": row.data_feed_ok,
        "llm_ok": row.llm_ok,
        "last_error": text(row.last_error, 500),
        "batch_id": row.batch_id,
        "settings_version_id": row.settings_version_id,
        "started_at": iso(row.started_at),
        "regime": scalars_only(row.regime_json),
        "status": scalars_only(row.status_json),
        "stale_after_s": stale_seconds,
    }


# =========================================================================== positions


def _quote_ltp(quote: Any) -> float | None:
    if quote is None:
        return None
    ltp = getattr(quote, "ltp", None)
    if ltp is None and isinstance(quote, dict):
        ltp = quote.get("ltp")
    try:
        return float(ltp) if ltp is not None else None
    except (TypeError, ValueError):
        return None


def quote_dict(symbol: str, quote: Any) -> dict[str, Any] | None:
    """Plain quote with change % vs previous close (when the provider gives one)."""
    ltp = _quote_ltp(quote)
    if ltp is None:
        return None
    prev = getattr(quote, "prev_close", None) if not isinstance(quote, dict) else quote.get("prev_close")
    ts = getattr(quote, "timestamp", None) if not isinstance(quote, dict) else quote.get("timestamp")
    change = change_pct = None
    if prev:
        change = ltp - float(prev)
        change_pct = change / float(prev) * 100.0
    return {"symbol": symbol, "ltp": r2(ltp), "prev_close": r2(prev), "change": r2(change),
            "change_pct": r2(change_pct), "ts": iso(ts) if isinstance(ts, datetime) else ts,
            "source": getattr(quote, "source", None) if not isinstance(quote, dict) else quote.get("source")}


def progress_axis(direction: str, stop: float, entry: float, t1: float | None, t2: float | None,
                  ltp: float | None) -> dict[str, Any] | None:
    """Positions along the SL -> T1 -> T2 axis in % (0 = stop-loss, 100 = furthest target).

    ``fraction`` is the SL->T1 progress (0..1, clamped). Works for LONG and SHORT.
    """
    s = sign(direction)
    far = t2 if t2 is not None else t1
    if far is None:
        far = entry + s * 2 * abs(entry - stop)
    span = s * (far - stop)
    if span <= 0:
        return None

    def pct(x: float | None) -> float | None:
        if x is None:
            return None
        return round(max(0.0, min(100.0, s * (x - stop) / span * 100.0)), 2)

    frac = None
    if ltp is not None and t1 is not None and s * (t1 - stop) > 0:
        frac = round(max(0.0, min(1.0, s * (ltp - stop) / (s * (t1 - stop)))), 4)
    return {"sl_pct": 0.0, "entry_pct": pct(entry), "t1_pct": pct(t1), "t2_pct": pct(t2) if t2 is not None else None,
            "ltp_pct": pct(ltp), "fraction": frac, "far": r2(far)}


def position_row(p: Position, quote: Any = None) -> dict[str, Any]:
    """Open position with live LTP (quote) falling back to the engine's ``last_price``."""
    live = _quote_ltp(quote)
    ltp = live if live is not None else p.last_price
    s = sign(p.direction)
    unreal = unreal_pct = r_now = None
    if ltp is not None:
        unreal = s * (ltp - p.avg_entry) * p.qty - (p.entry_charges or 0.0)
        unreal_pct = s * (ltp - p.avg_entry) / p.avg_entry * 100.0 if p.avg_entry else None
        risk = abs(p.avg_entry - p.initial_stop_loss)
        r_now = s * (ltp - p.avg_entry) / risk if risk > 0 else None
    risk_to_sl = s * (p.avg_entry - p.stop_loss) * p.qty
    return {
        "id": p.id, "batch_id": p.batch_id, "symbol": p.symbol, "mode": p.mode, "product": p.product,
        "direction": p.direction, "qty": p.qty, "initial_qty": p.initial_qty, "avg_entry": r2(p.avg_entry),
        "ltp": r2(ltp), "ltp_source": "live" if live is not None else ("engine" if p.last_price is not None else None),
        "ltp_at": iso(p.last_price_at) if live is None else None,
        "unrealized": r2(unreal), "unrealized_pct": r2(unreal_pct), "r_now": r2(r_now),
        "stop_loss": r2(p.stop_loss), "initial_stop_loss": r2(p.initial_stop_loss),
        "target_1": r2(p.target_1), "target_2": r2(p.target_2), "trailing_rule": p.trailing_rule,
        "risk_to_sl": r2(risk_to_sl), "opened_at": iso(p.opened_at), "holding_days": p.holding_days,
        "max_holding_days": p.max_holding_days, "strategy": p.strategy, "grade": p.grade, "source": p.source,
        "partial_done": bool(p.partial_done), "breakeven_moved": bool(p.breakeven_moved),
        "overnight": bool(p.overnight), "entry_charges": r2(p.entry_charges),
        "realized_gross": r2(p.realized_gross), "realized_charges": r2(p.realized_charges),
        "decision_id": p.decision_id,
        "progress": progress_axis(p.direction, p.stop_loss, p.avg_entry, p.target_1, p.target_2, ltp),
    }


def live_batches(column: Any) -> Any:
    """SQL filter: rows of non-deleted batches (or with no batch). Used when no batch_id is given so the
    Overall scope matches capital.funding.funding_summary, which skips soft-deleted batches."""
    return or_(column.is_(None), column.not_in(select(Batch.id).where(Batch.deleted.is_(True))))


def open_positions(session: Session, batch_id: int | None = None) -> list[Position]:
    stmt = select(Position).where(Position.status == PositionStatus.OPEN.value)
    if batch_id is not None:
        stmt = stmt.where(Position.batch_id == batch_id)
    else:
        stmt = stmt.where(live_batches(Position.batch_id))
    return list(session.execute(stmt.order_by(Position.opened_at.asc(), Position.id.asc())).scalars())


def unrealized_total(rows: Iterable[dict[str, Any]], batch_id: int | None = None) -> float:
    return round(sum(r["unrealized"] or 0.0 for r in rows if batch_id is None or r["batch_id"] == batch_id), 2)


# =========================================================================== capital


def capital_view(session: Session, batch_id: int | None, unrealized: float = 0.0) -> dict[str, Any] | None:
    """Capital account balances; ``equity`` includes live unrealized P&L."""
    if batch_id is None:
        return None
    acct = session.get(CapitalAccount, batch_id)
    if acct is None:
        return None
    trading = acct.trading_cash + acct.blocked
    realized_equity = trading + acct.reserve
    equity = realized_equity + unrealized
    hwm = max(acct.hwm, equity)
    dd = (hwm - equity) / hwm * 100.0 if hwm else 0.0
    pnl = equity - acct.starting_capital
    return {
        "batch_id": batch_id, "starting_capital": r2(acct.starting_capital), "trading_cash": r2(acct.trading_cash),
        "blocked": r2(acct.blocked), "trading": r2(trading), "reserve": r2(acct.reserve),
        "realized_equity": r2(realized_equity), "unrealized": r2(unrealized), "equity": r2(equity), "hwm": r2(hwm),
        "drawdown_pct": r2(dd), "tier": acct.tier, "halted": bool(acct.halted), "halt_reason": text(acct.halt_reason),
        "sizing_capital": r2(acct.sizing_capital), "cum_realized_net": r2(acct.cum_realized_net),
        "pnl": r2(pnl), "pnl_pct": r2(pnl / acct.starting_capital * 100.0) if acct.starting_capital else None,
        "updated_at": iso(acct.updated_at),
    }


# =========================================================================== trades (position level)


def group_legs(legs: Sequence[Trade], positions: dict[int, Position]) -> list[dict[str, Any]]:
    """Closed legs -> one row per position (newest exit first)."""
    by_pos: dict[int, list[Trade]] = defaultdict(list)
    for t in legs:
        by_pos[t.position_id].append(t)
    rows = [position_trade_row(sorted(ts, key=lambda x: (x.exit_time, x.id)), positions.get(pid))
            for pid, ts in by_pos.items()]
    rows.sort(key=lambda r: (r["exit_time"] or "", r["position_id"]), reverse=True)
    return rows


def position_trade_row(legs: Sequence[Trade], pos: Position | None) -> dict[str, Any]:
    first, last = legs[0], legs[-1]
    direction = pos.direction if pos else first.direction
    qty_closed = sum(t.qty for t in legs)
    gross = sum(t.gross_pnl for t in legs)
    charges = sum(t.charges for t in legs)
    net = sum(t.net_pnl for t in legs)
    exit_px = sum(t.exit_price * t.qty for t in legs) / qty_closed if qty_closed else last.exit_price
    r_vals = [(t.r_multiple, t.qty) for t in legs if t.r_multiple is not None]
    r_mult = sum(r * q for r, q in r_vals) / sum(q for _, q in r_vals) if r_vals and sum(q for _, q in r_vals) else None
    entry_time = pos.opened_at if pos else min(t.entry_time for t in legs)
    exit_time = max(t.exit_time for t in legs)
    hold = max((t.holding_minutes or 0) for t in legs) or None
    if hold is None and entry_time and exit_time:
        hold = int((exit_time - entry_time).total_seconds() // 60)
    target_2 = pos.target_2 if pos else first.target_2
    breakeven = bool(pos.breakeven_moved) if pos else False
    reasons: list[str] = []
    labels: list[str] = []
    for t in legs:
        if t.exit_reason not in reasons:
            reasons.append(t.exit_reason)
        lab = exit_label(t.exit_reason, direction=direction, exit_price=t.exit_price, target_2=target_2,
                         partial=bool(t.is_partial), breakeven=breakeven and not t.is_partial)
        if lab not in labels:
            labels.append(lab)
    still_open = pos is not None and pos.status == PositionStatus.OPEN.value
    result = "OPEN" if still_open else ("WIN" if net > 0.005 else ("LOSS" if net < -0.005 else "FLAT"))
    return {
        "position_id": first.position_id, "batch_id": first.batch_id, "symbol": first.symbol, "mode": first.mode,
        "product": first.product, "direction": direction, "strategy": (pos.strategy if pos else None) or first.strategy,
        "grade": (pos.grade if pos else None) or first.grade, "source": first.source, "regime": first.regime,
        "qty": pos.initial_qty if pos else qty_closed, "qty_closed": qty_closed,
        "entry_time": iso(entry_time), "entry_price": r2(pos.avg_entry if pos else first.entry_price),
        "exit_time": iso(exit_time), "exit_price": r2(exit_px),
        "initial_sl": r2(pos.initial_stop_loss if pos else first.stop_loss),
        "final_sl": r2(pos.stop_loss) if pos else None,
        "target_1": r2(pos.target_1 if pos else first.target_1), "target_2": r2(target_2),
        "exit_reasons": reasons, "exit_reason_labels": labels, "exit_reason": " + ".join(labels),
        "gross": r2(gross), "charges": r2(charges), "net": r2(net), "r_multiple": r2(r_mult),
        "holding_minutes": hold, "holding_days": max((t.holding_days or 0) for t in legs),
        "legs": len(legs), "partial": any(t.is_partial for t in legs), "result": result,
        "decision_id": (pos.decision_id if pos else None) or first.decision_id,
    }


def _positions_for(session: Session, ids: Iterable[int]) -> dict[int, Position]:
    ids = list({i for i in ids if i is not None})
    if not ids:
        return {}
    return {p.id: p for p in session.execute(select(Position).where(Position.id.in_(ids))).scalars()}


def trades_between(session: Session, lo: datetime | None, hi: datetime | None, *, batch_id: int | None = None,
                   mode: str | None = None, symbol: str | None = None, legs_in_range_only: bool = False
                   ) -> list[dict[str, Any]]:
    """Position-level rows for positions with at least one leg exiting in ``[lo, hi)``."""
    stmt = select(Trade.position_id).distinct()
    if lo is not None:
        stmt = stmt.where(Trade.exit_time >= lo)
    if hi is not None:
        stmt = stmt.where(Trade.exit_time < hi)
    if batch_id is not None:
        stmt = stmt.where(Trade.batch_id == batch_id)
    else:
        stmt = stmt.where(live_batches(Trade.batch_id))
    if mode:
        stmt = stmt.where(Trade.mode == mode)
    if symbol:
        stmt = stmt.where(Trade.symbol == symbol)
    pids = [p for p in session.execute(stmt).scalars()]
    if not pids:
        return []
    legs_stmt = select(Trade).where(Trade.position_id.in_(pids))
    if legs_in_range_only:
        if lo is not None:
            legs_stmt = legs_stmt.where(Trade.exit_time >= lo)
        if hi is not None:
            legs_stmt = legs_stmt.where(Trade.exit_time < hi)
    legs = list(session.execute(legs_stmt).scalars())
    return group_legs(legs, _positions_for(session, pids))


def trade_totals(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    closed = [r for r in rows if r["result"] != "OPEN"]
    wins = [r for r in closed if r["result"] == "WIN"]
    losses = [r for r in closed if r["result"] == "LOSS"]
    r_vals = [r["r_multiple"] for r in closed if r["r_multiple"] is not None]
    gross_win = sum(r["net"] for r in wins)
    gross_loss = -sum(r["net"] for r in losses)
    best = max(rows, key=lambda r: r["net"] or 0.0) if rows else None
    worst = min(rows, key=lambda r: r["net"] or 0.0) if rows else None
    return {
        "trades": len(rows), "closed": len(closed), "wins": len(wins), "losses": len(losses),
        "flat": len(closed) - len(wins) - len(losses),
        "win_rate": r2(len(wins) / len(closed) * 100.0) if closed else None,
        "gross": r2(sum(r["gross"] or 0 for r in rows)), "charges": r2(sum(r["charges"] or 0 for r in rows)),
        "net": r2(sum(r["net"] or 0 for r in rows)),
        "avg_r": r2(sum(r_vals) / len(r_vals)) if r_vals else None,
        "avg_win": r2(gross_win / len(wins)) if wins else None,
        "avg_loss": r2(-gross_loss / len(losses)) if losses else None,
        "profit_factor": r2(gross_win / gross_loss) if gross_loss > 0 else None,
        "best": _brief(best) if best and (best["net"] or 0) > 0 else None,
        "worst": _brief(worst) if worst and (worst["net"] or 0) < 0 else None,
    }


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    return {"position_id": row["position_id"], "symbol": row["symbol"], "net": row["net"], "mode": row["mode"]}


def breakdown(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r.get(key) or "(none)"].append(r)
    out = []
    for k, items in groups.items():
        t = trade_totals(items)
        out.append({key: k, "trades": t["closed"], "wins": t["wins"], "losses": t["losses"],
                    "win_rate": t["win_rate"], "net": t["net"], "gross": t["gross"], "charges": t["charges"],
                    "avg_r": t["avg_r"]})
    out.sort(key=lambda x: x["net"] or 0.0, reverse=True)
    return out


def today_realized(session: Session, lo: datetime, hi: datetime, batch_id: int | None) -> dict[str, Any]:
    rows = trades_between(session, lo, hi, batch_id=batch_id, legs_in_range_only=True)
    t = trade_totals(rows)
    return {"realized": t["net"], "gross": t["gross"], "charges": t["charges"], "trades": t["closed"],
            "wins": t["wins"], "losses": t["losses"], "win_rate": t["win_rate"]}


# =========================================================================== decisions


def _reason(final_action: str, filtered: str | None, risk_rule: str | None, risk_msg: str | None,
            why_not: str | None, qty: int | None) -> str:
    if final_action == "FILTERED":
        return filtered or "filtered before the AI agents"
    if final_action == "RISK_REJECTED":
        return risk_msg or (f"risk rule {risk_rule}" if risk_rule else "rejected by the risk manager")
    if final_action == "NO_TRADE":
        return why_not or "AI meta agent decided not to trade"
    if final_action == "QUEUED":
        return "waiting for your approval"
    if final_action == "EXECUTED":
        return f"order placed ({qty} qty)" if qty else "order placed"
    if final_action == "SIGNAL_ONLY":
        return "signal only (not executed)"
    if final_action == "ANALYZED":
        return why_not or "analysis only (Analyze now) - not routed for execution"
    return ""


_FEED_COLUMNS = (
    DecisionRow.id, DecisionRow.created_at, DecisionRow.batch_id, DecisionRow.symbol, DecisionRow.mode,
    DecisionRow.strategy, DecisionRow.direction, DecisionRow.confluence_score, DecisionRow.grade,
    DecisionRow.decision, DecisionRow.confidence, DecisionRow.final_action, DecisionRow.risk_rule,
    DecisionRow.entry_price, DecisionRow.stop_loss, DecisionRow.target_1, DecisionRow.target_2,
    DecisionRow.latency_ms, DecisionRow.qty, DecisionRow.regime,
    DecisionRow.context_json["filtered_reason"].as_string().label("filtered_reason"),
    DecisionRow.risk_json["message"].as_string().label("risk_message"),
    DecisionRow.meta_json["why_not"].as_string().label("why_not"),
)


def decision_feed(session: Session, *, since_id: int | None = None, before_id: int | None = None, limit: int = 100,
                  action: str | None = None, symbol: str | None = None, hide_filtered: bool = False,
                  lo: datetime | None = None, hi: datetime | None = None, batch_id: int | None = None
                  ) -> list[dict[str, Any]]:
    stmt = select(*_FEED_COLUMNS)
    if batch_id is not None:
        stmt = stmt.where(DecisionRow.batch_id == batch_id)
    else:
        stmt = stmt.where(live_batches(DecisionRow.batch_id))
    if since_id is not None:
        stmt = stmt.where(DecisionRow.id > since_id)
    if before_id is not None:
        stmt = stmt.where(DecisionRow.id < before_id)
    if action:
        actions = [a.strip().upper() for a in action.split(",") if a.strip()]
        stmt = stmt.where(DecisionRow.final_action.in_(actions))
    if hide_filtered:
        stmt = stmt.where(DecisionRow.final_action != "FILTERED")
    if symbol:
        stmt = stmt.where(DecisionRow.symbol == symbol)
    if lo is not None:
        stmt = stmt.where(DecisionRow.created_at >= lo)
    if hi is not None:
        stmt = stmt.where(DecisionRow.created_at < hi)
    stmt = stmt.order_by(DecisionRow.id.desc()).limit(limit)
    out = []
    for r in session.execute(stmt).all():
        out.append({
            "id": r.id, "time": iso(r.created_at), "batch_id": r.batch_id, "symbol": r.symbol, "mode": r.mode,
            "strategy": r.strategy, "direction": r.direction, "confluence": r2(r.confluence_score), "grade": r.grade,
            "decision": r.decision, "confidence": r.confidence, "final_action": r.final_action,
            "risk_rule": r.risk_rule, "regime": r.regime,
            "reason": text(_reason(r.final_action, r.filtered_reason, r.risk_rule, r.risk_message, r.why_not, r.qty),
                           400),
            "entry": r2(r.entry_price), "stop_loss": r2(r.stop_loss), "target_1": r2(r.target_1),
            "target_2": r2(r.target_2), "latency_ms": r.latency_ms, "qty": r.qty,
        })
    return out


def _report_view(r: AgentReportRow) -> dict[str, Any]:
    rep = r.report_json or {}
    extra = {k: v for k, v in rep.items() if k not in ("agent", "symbol", "mode", "timestamp", "signal", "confidence",
                                                         "key_findings", "data_used") and isinstance(
        v, (str, int, float, bool))}
    return {"agent": r.agent, "signal": r.signal, "confidence": r.confidence, "status": r.status, "model": r.model,
            "latency_ms": r.latency_ms, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
            "error": text(r.error, 500), "key_findings": str_list(rep.get("key_findings"), 10),
            "extra": {k: text(v, 400) if isinstance(v, str) else v for k, v in list(extra.items())[:8]}}


def _side(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if not d:
        return None
    return {"stance": text(d.get("stance"), 600), "arguments": str_list(d.get("top_arguments") or d.get("arguments"), 8),
            "rebuttals": str_list(d.get("rebuttals"), 6), "conviction": d.get("conviction")}


def debate_view(row: DebateRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    turns = []
    for t in (row.transcript_json or [])[:20]:
        if isinstance(t, dict):
            turns.append({"round": t.get("round"), "speaker": t.get("speaker") or t.get("side") or t.get("role"),
                          "phase": t.get("phase"), "stance": text(t.get("stance"), 500),
                          "arguments": str_list(t.get("arguments"), 8), "rebuttals": str_list(t.get("rebuttals"), 6),
                          "conviction": t.get("conviction"),
                          "text": None if t.get("arguments") else text(t.get("content") or t.get("text"), 1500)})
        else:
            turns.append({"text": text(t, 1500)})
    return {"id": row.id, "rounds": row.rounds, "status": row.status, "model": row.model, "latency_ms": row.latency_ms,
            "bull": _side(row.bull_json), "bear": _side(row.bear_json), "transcript": turns}


def risk_view(risk: dict[str, Any] | None) -> dict[str, Any] | None:
    if not risk:
        return None
    checks = []
    for c in risk.get("checks") or []:
        if isinstance(c, dict):
            checks.append({"rule": c.get("rule"), "passed": bool(c.get("passed")), "detail": text(c.get("detail"), 300)})
    charges = risk.get("charges_estimate")
    charges_total = None
    if isinstance(charges, dict):
        vals = [v for v in charges.values() if isinstance(v, (int, float))]
        charges_total = r2(sum(vals)) if vals else None
    return {"approved": risk.get("approved"), "rule": risk.get("rule"), "message": text(risk.get("message"), 500),
            "qty": risk.get("qty"), "max_qty": risk.get("max_qty"), "risk_amount": r2(risk.get("risk_amount")),
            "effective_risk": r2(risk.get("effective_risk")), "risk_budget": r2(risk.get("risk_budget")),
            "position_value": r2(risk.get("position_value")), "charges_estimate_total": charges_total,
            "expected_net_at_t1": r2(risk.get("expected_net_at_t1")), "tier": risk.get("tier"),
            "size_multiplier": risk.get("size_multiplier"), "checks": checks,
            "failed": [c for c in checks if not c["passed"]]}


_BEHAVIOUR_KEYS = ("range_pos", "range_vs_adr", "vwap", "vwap_ext_atr", "day_high", "day_low", "day_open", "adr",
                   "atr", "pdh", "pdl", "pdc", "gap_pct", "continuation_rate")
_BEHAVIOUR_TOP = {"range_pos": "range_pos", "range_vs_adr": "range_vs_adr", "vwap_ext_atr": "vwap_ext_atr",
                  "continuation_rate": "continuation_rate", "behaviour_samples": "samples"}
BEHAVIOUR_VERDICTS = ("OK", "CHASING", "FADES")


def behaviour_view(context: dict[str, Any] | None, entry: float | None = None) -> dict[str, Any] | None:
    """Deterministic price-behaviour snapshot of a decision (``context_json["proposal"]["behaviour"]``).

    None for older decisions without it. Numbers are rounded; notes are plain, truncated strings.
    """
    if not isinstance(context, dict):
        return None
    prop = context.get("proposal")
    if not isinstance(prop, dict):
        return None
    beh = prop.get("behaviour") if isinstance(prop.get("behaviour"), dict) else {}
    out: dict[str, Any] = {k: r4(beh.get(k)) for k in _BEHAVIOUR_KEYS}
    for src, dst in _BEHAVIOUR_TOP.items():
        if out.get(dst) is None and prop.get(src) is not None:
            out[dst] = r4(prop.get(src))
    samples = beh.get("samples", out.get("samples"))
    try:
        out["samples"] = int(samples) if samples is not None else None
    except (TypeError, ValueError):
        out["samples"] = None
    if not any(v is not None for v in out.values()):
        return None
    verdict = str(beh.get("verdict") or "").upper()
    out["verdict"] = verdict if verdict in BEHAVIOUR_VERDICTS else None
    direction = str(beh.get("direction") or prop.get("direction") or "").upper()
    out["direction"] = direction if direction in ("LONG", "SHORT") else None
    out["notes"] = str_list(beh.get("notes"), 8, 300)
    out["entry"] = r2(entry if entry is not None else prop.get("entry_price"))
    out["ltp"] = r2(prop.get("ltp"))
    return out


def decision_detail(session: Session, decision_id: int) -> dict[str, Any] | None:
    d = session.get(DecisionRow, decision_id)
    if d is None:
        return None
    ctxj = d.context_json or {}
    meta = d.meta_json or {}
    conf = ctxj.get("confluence") or {}
    vote = ctxj.get("vote") or None
    signals = []
    for s in (ctxj.get("signals") or [])[:10]:
        if isinstance(s, dict):
            signals.append({k: s.get(k) for k in ("strategy", "mode", "direction", "entry_price", "entry_type",
                                                    "stop_loss", "target_1", "target_2", "strength", "regime_fit",
                                                    "trailing_rule", "max_holding_days")}
                           | {"notes": str_list(s.get("notes"), 6)})
    sl_candidates = [{k: c.get(k) for k in ("method", "price", "distance", "distance_atr", "note")}
                     for c in (ctxj.get("sl_candidates") or [])[:12] if isinstance(c, dict)]
    components = {k: r2(v) for k, v in (conf.get("components") or {}).items() if isinstance(v, (int, float))}
    market = ctxj.get("market") or {}
    reports = [_report_view(r) for r in session.execute(
        select(AgentReportRow).where(AgentReportRow.decision_id == decision_id).order_by(AgentReportRow.agent)
    ).scalars()]
    debate = session.execute(select(DebateRow).where(DebateRow.decision_id == decision_id).order_by(DebateRow.id.desc())
                             .limit(1)).scalar_one_or_none()
    orders = list(session.execute(select(Order).where(Order.decision_id == decision_id).order_by(Order.id)).scalars())
    positions = list(session.execute(select(Position.id, Position.status, Position.symbol)
                                     .where(Position.decision_id == decision_id)).all())
    approval = session.execute(select(ApprovalItem).where(ApprovalItem.decision_id == decision_id)
                               .order_by(ApprovalItem.id.desc()).limit(1)).scalar_one_or_none()
    vote_view = None
    if isinstance(vote, dict):
        vote_view = {k: vote.get(k) for k in ("score", "bullish", "bearish", "neutral", "failed", "skipped",
                                              "tradeable", "total_weight", "event_risk")}
        vote_view["contributions"] = {k: r4(v) for k, v in (vote.get("contributions") or {}).items()}
        vote_view["failed_agents"] = str_list(vote.get("failed_agents"), 12)
        vote_view["tradeable_reasons"] = str_list(vote.get("tradeable_reasons"), 6)
    return {
        "id": d.id, "time": iso(d.created_at), "batch_id": d.batch_id, "symbol": d.symbol, "mode": d.mode,
        "strategy": d.strategy, "regime": d.regime, "direction": d.direction, "decision": d.decision,
        "confidence": d.confidence, "final_action": d.final_action, "confluence": r2(d.confluence_score),
        "grade": d.grade, "entry": r2(d.entry_price), "stop_loss": r2(d.stop_loss), "target_1": r2(d.target_1),
        "target_2": r2(d.target_2), "qty": d.qty, "latency_ms": d.latency_ms, "source": d.source,
        "filtered_reason": text(ctxj.get("filtered_reason"), 400),
        "reason": text(_reason(d.final_action, ctxj.get("filtered_reason"), d.risk_rule,
                               (d.risk_json or {}).get("message"), meta.get("why_not"), d.qty), 500),
        "signals": signals,
        "confluence_detail": {"score": r2(conf.get("score")), "grade": conf.get("grade"), "components": components,
                              "notes": str_list(conf.get("notes"), 10)} if conf else None,
        "sl_candidates": sl_candidates,
        "vote": vote_view,
        "market": scalars_only(market),
        "events": str_list([e.get("title") if isinstance(e, dict) else e for e in (ctxj.get("events") or [])], 8),
        "agents": reports,
        "debate": debate_view(debate),
        "meta": {"decision": meta.get("decision"), "confidence": meta.get("confidence"),
                 "reasons_for": str_list(meta.get("reasons_for"), 10),
                 "reasons_against": str_list(meta.get("reasons_against_considered"), 10),
                 "why_not": text(meta.get("why_not"), 800), "status": meta.get("status"), "model": meta.get("model"),
                 "latency_ms": meta.get("latency_ms"), "error": text(meta.get("error"), 400),
                 "time_validity_minutes": meta.get("time_validity_minutes")} if meta else None,
        "risk": risk_view(d.risk_json),
        "behaviour": behaviour_view(ctxj, d.entry_price),
        "orders": [order_row(o) for o in orders],
        "positions": [{"id": p.id, "status": p.status, "symbol": p.symbol} for p in positions],
        "approval": {"id": approval.id, "status": approval.status, "decided_by": approval.decided_by,
                     "user_reason": text(approval.user_reason), "block_rule": approval.block_rule}
        if approval else None,
    }


def decision_funnel(session: Session, lo: datetime | None, hi: datetime | None, batch_id: int | None = None
                    ) -> dict[str, Any]:
    stmt = select(DecisionRow.final_action, DecisionRow.risk_rule,
                  DecisionRow.context_json["filtered_reason"].as_string().label("filtered_reason"))
    if lo is not None:
        stmt = stmt.where(DecisionRow.created_at >= lo)
    if hi is not None:
        stmt = stmt.where(DecisionRow.created_at < hi)
    if batch_id is not None:
        stmt = stmt.where(DecisionRow.batch_id == batch_id)
    else:
        stmt = stmt.where(live_batches(DecisionRow.batch_id))
    counts: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    filters: Counter[str] = Counter()
    for action, rule, filtered in session.execute(stmt).all():
        counts[action] += 1
        if action == "RISK_REJECTED":
            rules[rule or "(unknown)"] += 1
        elif action == "FILTERED":
            filters[_filter_bucket(filtered)] += 1
    total = sum(counts.values())
    filtered_n = counts.get("FILTERED", 0)
    reached_ai = total - filtered_n
    trade_signal = reached_ai - counts.get("NO_TRADE", 0)
    scan_stmt = select(func.count(StockAnalyzed.id))
    if lo is not None:
        scan_stmt = scan_stmt.where(StockAnalyzed.as_of >= lo)
    if hi is not None:
        scan_stmt = scan_stmt.where(StockAnalyzed.as_of < hi)
    if batch_id is not None:
        scan_stmt = scan_stmt.where(StockAnalyzed.batch_id == batch_id)
    scanned = int(session.execute(scan_stmt).scalar_one() or 0)
    return {
        "scanned": scanned,
        "counts": {a: counts.get(a, 0) for a in FINAL_ACTIONS} | {"TOTAL": total},
        "stages": [
            {"key": "analysed", "label": "Analysed", "count": total,
             "help": "Stocks that went through strategies + confluence scoring"},
            {"key": "passed_filter", "label": "Passed strategy filter", "count": reached_ai,
             "help": "Had a valid strategy setup with enough confluence; sent to the AI agents"},
            {"key": "trade_signal", "label": "AI said trade", "count": trade_signal,
             "help": "Meta agent decided BUY / SELL SHORT (the rest were NO_TRADE)"},
            {"key": "risk_rejected", "label": "Risk-rejected", "count": counts.get("RISK_REJECTED", 0),
             "help": "Vetoed by the deterministic risk manager"},
            {"key": "queued", "label": "Queued for approval", "count": counts.get("QUEUED", 0),
             "help": "Waited for your Approve / Reject"},
            {"key": "executed", "label": "Executed", "count": counts.get("EXECUTED", 0),
             "help": "Paper order placed"},
        ],
        "risk_rules_top": [{"rule": k, "count": v} for k, v in rules.most_common(5)],
        "filter_reasons_top": [{"reason": k, "count": v} for k, v in filters.most_common(5)],
    }


def _filter_bucket(reason: str | None) -> str:
    """Group filter reasons ("confluence 53.5 < 60.0 (grade C)" -> "confluence below minimum")."""
    if not reason:
        return "(unspecified)"
    r = reason.lower()
    if r.startswith("confluence"):
        return "confluence below minimum"
    return reason if len(reason) <= 80 else reason[:79] + "…"


# =========================================================================== orders / timeline


def order_row(o: Order) -> dict[str, Any]:
    charges = (o.charges_json or {}).get("total") if isinstance(o.charges_json, dict) else None
    return {"id": o.id, "batch_id": o.batch_id, "position_id": o.position_id, "decision_id": o.decision_id,
            "approval_id": o.approval_id, "symbol": o.symbol, "mode": o.mode, "product": o.product, "side": o.side,
            "type": o.order_type, "purpose": o.purpose, "qty": o.qty, "limit_price": r2(o.limit_price),
            "status": o.status, "created_at": iso(o.created_at), "filled_at": iso(o.filled_at),
            "fill_price": r2(o.fill_price), "slippage": r2(o.slippage), "charges": r2(charges), "source": o.source,
            "reason": text(o.reason, 300), "time": iso(o.filled_at or o.created_at)}


def leg_row(t: Trade) -> dict[str, Any]:
    return {"id": t.id, "position_id": t.position_id, "qty": t.qty, "entry_time": iso(t.entry_time),
            "entry_price": r2(t.entry_price), "exit_time": iso(t.exit_time), "exit_price": r2(t.exit_price),
            "exit_reason": t.exit_reason, "is_partial": bool(t.is_partial), "gross": r2(t.gross_pnl),
            "charges": r2(t.charges), "net": r2(t.net_pnl), "r_multiple": r2(t.r_multiple),
            "holding_minutes": t.holding_minutes}


def _as_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return to_ist(value)
    if isinstance(value, str):
        try:
            return to_ist(datetime.fromisoformat(value))
        except ValueError:
            return None
    return None


def sl_moves(p: Position) -> list[dict[str, Any]]:
    out = []
    for h in p.sl_history_json or []:
        if not isinstance(h, dict):
            continue
        out.append({"ts": iso(_as_dt(h.get("ts"))), "old": r2(h.get("old")), "new": r2(h.get("new")),
                    "by": h.get("by")})
    return out


def day_timeline(session: Session, lo: datetime, hi: datetime, batch_id: int | None = None,
                 limit: int = 400) -> list[dict[str, Any]]:
    """Orders filled/rejected, exits, SL moves, risk rejections and approvals in ``[lo, hi)``."""
    ev: list[dict[str, Any]] = []

    def add(ts: datetime | None, kind: str, symbol: str | None, mode: str | None, title: str, detail: str,
            ref: dict[str, Any] | None = None, tone: str = "neutral") -> None:
        if ts is None:
            return
        ev.append({"ts": iso(ts), "kind": kind, "symbol": symbol, "mode": mode, "title": title, "detail": detail,
                   "ref": ref or {}, "tone": tone})

    o_stmt = select(Order).where(func.coalesce(Order.filled_at, Order.created_at) >= lo,
                                 func.coalesce(Order.filled_at, Order.created_at) < hi)
    if batch_id is not None:
        o_stmt = o_stmt.where(Order.batch_id == batch_id)
    for o in session.execute(o_stmt).scalars():
        if o.status == OrderStatus.FILLED.value and o.purpose == "ENTRY":
            add(o.filled_at, "ENTRY", o.symbol, o.mode, f"Entered {o.side} {o.qty} @ {o.fill_price:,.2f}",
                f"{o.order_type} order filled" + (f", slippage {o.slippage:,.2f}" if o.slippage else ""),
                {"type": "position", "id": o.position_id}, "info")
        elif o.status in (OrderStatus.REJECTED.value, OrderStatus.CANCELLED.value, OrderStatus.EXPIRED.value):
            add(o.created_at, "ORDER", o.symbol, o.mode, f"{o.purpose.title()} order {o.status.lower()}",
                text(o.reason, 200) or "", {"type": "decision", "id": o.decision_id} if o.decision_id else None,
                "warn")
        elif o.status == OrderStatus.PENDING.value:
            add(o.created_at, "ORDER", o.symbol, o.mode, f"{o.side} {o.purpose.lower()} order pending",
                f"{o.order_type} {o.qty}" + (f" limit {o.limit_price:,.2f}" if o.limit_price else ""), None, "info")

    t_stmt = select(Trade).where(Trade.exit_time >= lo, Trade.exit_time < hi)
    if batch_id is not None:
        t_stmt = t_stmt.where(Trade.batch_id == batch_id)
    for t in session.execute(t_stmt).scalars():
        label = exit_label(t.exit_reason, direction=t.direction, exit_price=t.exit_price, target_2=t.target_2,
                           partial=bool(t.is_partial))
        add(t.exit_time, "EXIT", t.symbol, t.mode, f"{label}: {t.qty} @ {t.exit_price:,.2f}",
            f"net {t.net_pnl:+,.2f} (gross {t.gross_pnl:+,.2f}, charges {t.charges:,.2f})"
            + (f", R {t.r_multiple:+.2f}" if t.r_multiple is not None else ""),
            {"type": "position", "id": t.position_id}, "good" if t.net_pnl > 0 else "bad")

    p_stmt = select(Position).where((Position.closed_at.is_(None)) | (Position.closed_at >= lo),
                                    Position.opened_at < hi)
    if batch_id is not None:
        p_stmt = p_stmt.where(Position.batch_id == batch_id)
    for p in session.execute(p_stmt).scalars():
        for h in p.sl_history_json or []:
            if not isinstance(h, dict) or (h.get("by") or "").upper() == "ENTRY":
                continue
            ts = _as_dt(h.get("ts"))
            if ts is None or not (lo <= ts < hi):
                continue
            add(ts, "SL_MOVE", p.symbol, p.mode, f"Stop-loss moved {h.get('old')} → {h.get('new')}",
                f"by {h.get('by') or 'system'}", {"type": "position", "id": p.id}, "info")

    d_stmt = select(DecisionRow.id, DecisionRow.created_at, DecisionRow.symbol, DecisionRow.mode,
                    DecisionRow.risk_rule, DecisionRow.risk_json["message"].as_string().label("msg")
                    ).where(DecisionRow.final_action == "RISK_REJECTED", DecisionRow.created_at >= lo,
                            DecisionRow.created_at < hi)
    if batch_id is not None:
        d_stmt = d_stmt.where(DecisionRow.batch_id == batch_id)
    for d in session.execute(d_stmt).all():
        add(d.created_at, "RISK", d.symbol, d.mode, f"Risk rejected: {d.risk_rule or '-'}", text(d.msg, 250) or "",
            {"type": "decision", "id": d.id}, "warn")

    a_stmt = select(ApprovalItem).where(ApprovalItem.created_at >= lo - timedelta(days=3))
    if batch_id is not None:
        a_stmt = a_stmt.where(ApprovalItem.batch_id == batch_id)
    for a in session.execute(a_stmt).scalars():
        if lo <= a.created_at < hi:
            add(a.created_at, "APPROVAL", a.symbol, a.mode, f"Queued for approval: {a.direction} {a.qty}",
                f"entry {a.entry_price:,.2f} SL {a.stop_loss:,.2f} T1 {a.target_1:,.2f}",
                {"type": "decision", "id": a.decision_id} if a.decision_id else None, "info")
        decided = a.decided_at or (a.expires_at if a.status == ApprovalStatus.TIMEOUT.value else None)
        if decided is not None and lo <= decided < hi and a.status != ApprovalStatus.PENDING.value:
            add(decided, "APPROVAL", a.symbol, a.mode, f"Approval {a.status.lower()}",
                (f"by {a.decided_by}" if a.decided_by else "") + (f" - {text(a.user_reason, 120)}" if a.user_reason
                                                                  else "")
                + (f" | blocked by {a.block_rule}" if a.block_rule else ""),
                {"type": "decision", "id": a.decision_id} if a.decision_id else None,
                "good" if a.status == ApprovalStatus.APPROVED.value else "warn")
    ev.sort(key=lambda e: e["ts"] or "")
    return ev[-limit:]


def day_equity(session: Session, lo: datetime, hi: datetime, batch_id: int | None) -> list[dict[str, Any]]:
    """Realized equity points through the day from the capital ledger (start-of-day point first)."""
    if batch_id is None:
        return []
    before = session.execute(
        select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id, CapitalLedgerEntry.ts < lo)
        .order_by(CapitalLedgerEntry.ts.desc(), CapitalLedgerEntry.id.desc()).limit(1)
    ).scalar_one_or_none()
    rows = list(session.execute(
        select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id, CapitalLedgerEntry.ts >= lo,
                                         CapitalLedgerEntry.ts < hi)
        .order_by(CapitalLedgerEntry.ts, CapitalLedgerEntry.id)
    ).scalars())
    pts: list[dict[str, Any]] = []
    if before is not None:
        pts.append({"ts": iso(lo), "time": chart_time(lo), "equity": r2(_ledger_equity(before))})
    for e in rows:
        pts.append({"ts": iso(e.ts), "time": chart_time(e.ts), "equity": r2(_ledger_equity(e))})
    return _dedupe_times(pts)


def _ledger_equity(e: CapitalLedgerEntry) -> float:
    return float(e.trading_cash_after) + float(e.blocked_after) + float(e.reserve_after)


def _dedupe_times(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lightweight Charts needs strictly increasing times: keep the last value per second."""
    out: dict[int, dict[str, Any]] = {}
    for p in points:
        out[p["time"]] = p
    return [out[k] for k in sorted(out)]


# =========================================================================== batches


def batch_trade_stats(session: Session, batch_id: int) -> dict[str, Any]:
    rows = trades_between(session, None, None, batch_id=batch_id)
    return trade_totals(rows)


def batch_equity_curve(session: Session, batch_id: int, live_equity: float | None, now: datetime
                       ) -> list[dict[str, Any]]:
    """EOD daily summaries if present, else the capital ledger, else cumulative trade net."""
    b = session.get(Batch, batch_id)
    acct = session.get(CapitalAccount, batch_id)
    starting = float(acct.starting_capital) if acct else (float(b.starting_capital) if b else None)
    pts: list[dict[str, Any]] = []
    start_ts = (b.started_at or b.created_at) if b else None
    if start_ts is not None and starting is not None:
        pts.append({"ts": iso(start_ts), "time": chart_time(start_ts), "equity": r2(starting), "source": "START"})
    daily = list(session.execute(select(DailySummary).where(DailySummary.batch_id == batch_id,
                                                            DailySummary.mode == "TOTAL")
                                 .order_by(DailySummary.day)).scalars())
    if daily:
        for r in daily:
            if r.equity is None:
                continue
            ts = datetime.combine(r.day, datetime.min.time()).replace(hour=15, minute=30, tzinfo=now.tzinfo)
            pts.append({"ts": iso(ts), "time": chart_time(ts), "equity": r2(r.equity), "source": "EOD"})
    else:
        ledger = list(session.execute(select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id)
                                      .order_by(CapitalLedgerEntry.ts, CapitalLedgerEntry.id)).scalars())
        if ledger:
            for e in ledger:
                pts.append({"ts": iso(e.ts), "time": chart_time(e.ts), "equity": r2(_ledger_equity(e)),
                            "source": "LEDGER"})
        elif starting is not None:
            running = starting
            for ts, pnl in session.execute(select(Trade.exit_time, Trade.net_pnl).where(Trade.batch_id == batch_id)
                                           .order_by(Trade.exit_time, Trade.id)).all():
                running += float(pnl or 0.0)
                pts.append({"ts": iso(ts), "time": chart_time(ts), "equity": r2(running), "source": "TRADES"})
    if live_equity is not None and b is not None and b.status in CURRENT_STATUSES:
        last = max((p["time"] for p in pts), default=0)
        t = max(chart_time(now), last + 1)
        pts.append({"ts": iso(now), "time": t, "equity": r2(live_equity), "source": "LIVE"})
    return _dedupe_times(pts)


def batches_fallback(session: Session, unrealized_by_batch: dict[int, float]) -> dict[str, Any]:
    """Approximation of ``capital.funding.funding_summary`` from Batch + CapitalAccount + trades.

    First batch: ``top_up`` = its starting capital (the initial deposit). Later batches:
    ``top_up = starting_capital - carried_in`` where ``carried_in`` is the previous batch's
    equity (+ = money added, - = withdrawn).
    """
    batches = list(session.execute(select(Batch).where(Batch.deleted.is_(False)).order_by(Batch.id)).scalars())
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    prev_equity: float | None = None
    deposited = withdrawn = 0.0
    for b in batches:
        acct = session.get(CapitalAccount, b.id)
        unreal = unrealized_by_batch.get(b.id, 0.0)
        equity = (acct.trading_cash + acct.blocked + acct.reserve + unreal) if acct else b.starting_capital
        stats = batch_trade_stats(session, b.id)
        if prev_equity is None:
            top_up, carried = b.starting_capital, None
            events.append({"ts": iso(b.started_at or b.created_at), "batch_id": b.id, "kind": "INITIAL",
                           "amount": r2(b.starting_capital), "note": f"{b.name} initial capital"})
        else:
            carried = prev_equity
            top_up = b.starting_capital - prev_equity
            if abs(top_up) >= 0.005:
                events.append({"ts": iso(b.started_at or b.created_at), "batch_id": b.id,
                               "kind": "TOP_UP" if top_up > 0 else "WITHDRAW", "amount": r2(abs(top_up)),
                               "note": f"{b.name}: start {b.starting_capital:,.2f} vs carried {prev_equity:,.2f}"})
        if top_up > 0:
            deposited += top_up
        else:
            withdrawn += -top_up
        pnl = equity - b.starting_capital
        rows.append({
            "batch_id": b.id, "name": b.name, "status": b.status, "starting_capital": r2(b.starting_capital),
            "top_up": r2(top_up), "carried_in": r2(carried), "equity": r2(equity),
            "realized_net": r2(stats["net"]), "unrealized": r2(unreal), "pnl": r2(pnl),
            "pnl_pct": r2(pnl / b.starting_capital * 100.0) if b.starting_capital else None,
            "trades": stats["closed"], "wins": stats["wins"], "losses": stats["losses"],
            "win_rate": stats["win_rate"], "started_at": iso(b.started_at or b.created_at), "ended_at": iso(b.ended_at),
        })
        prev_equity = equity
    current = rows[-1]["equity"] if rows else 0.0
    net_invested = deposited - withdrawn
    overall = (current or 0.0) - net_invested
    return {
        "total_deposited": r2(deposited), "total_withdrawn": r2(withdrawn), "net_invested": r2(net_invested),
        "current_capital": r2(current), "overall_pnl": r2(overall),
        "overall_pnl_pct": r2(overall / net_invested * 100.0) if net_invested else None,
        "batches": rows, "events": events,
    }


# =========================================================================== approvals / commands / errors


def approval_view(a: ApprovalItem, now: datetime, decision_risk: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = a.summary_json or {}
    tax = summary.get("tax") if isinstance(summary.get("tax"), dict) else None
    risk = risk_view(a.risk_json or decision_risk)
    per_share = abs(a.entry_price - a.stop_loss)
    left = (a.expires_at - now).total_seconds()
    tax_view = None
    if tax:
        tax_view = {"head": tax.get("head"), "rate_pct": tax.get("rate_pct"),
                    "charges_pct_of_value": tax.get("charges_pct_of_value"),
                    "breakeven_move_pct": tax.get("breakeven_move_pct"), "post_tax_rr": tax.get("post_tax_rr"),
                    "scenarios": [{"label": sc.get("label"), "exit_price": r2(sc.get("exit_price")),
                                   "gross": r2(sc.get("gross")), "charges": r2(sc.get("charges")),
                                   "tax": r2(sc.get("tax")), "net_after_tax": r2(sc.get("net_after_tax"))}
                                  for sc in (tax.get("scenarios") or []) if isinstance(sc, dict)][:6],
                    "notes": str_list(tax.get("notes"), 4)}
    return {
        "id": a.id, "batch_id": a.batch_id, "decision_id": a.decision_id, "symbol": a.symbol, "mode": a.mode,
        "direction": a.direction, "entry_type": a.entry_type, "entry_price": r2(a.entry_price),
        "stop_loss": r2(a.stop_loss), "target_1": r2(a.target_1), "target_2": r2(a.target_2), "qty": a.qty,
        "max_qty": a.max_qty, "confidence": a.confidence, "grade": a.grade, "strategy": a.strategy,
        "status": a.status, "created_at": iso(a.created_at), "expires_at": iso(a.expires_at),
        "seconds_left": max(0, int(left)), "expired": left <= 0,
        "risk_amount": r2(per_share * a.qty), "position_value": r2(a.entry_price * a.qty),
        "rr": r2(abs(a.target_1 - a.entry_price) / per_share) if per_share > 0 else None,
        "bull_points": str_list(summary.get("bull_points") or summary.get("top_bull"), 5),
        "bear_points": str_list(summary.get("bear_points") or summary.get("top_bear"), 5),
        "charges_estimate": risk["charges_estimate_total"] if risk else None,
        "expected_net_at_t1": risk["expected_net_at_t1"] if risk else None,
        "tax": tax_view,
    }


def pending_approvals(session: Session, now: datetime) -> list[dict[str, Any]]:
    items = list(session.execute(select(ApprovalItem).where(ApprovalItem.status == ApprovalStatus.PENDING.value)
                                 .order_by(ApprovalItem.expires_at.asc(), ApprovalItem.id.asc())).scalars())
    risk_by_decision: dict[int, Any] = {}
    ids = [a.decision_id for a in items if a.decision_id and not a.risk_json]
    if ids:
        for did, rj in session.execute(select(DecisionRow.id, DecisionRow.risk_json).where(DecisionRow.id.in_(ids))):
            risk_by_decision[did] = rj
    return [approval_view(a, now, risk_by_decision.get(a.decision_id)) for a in items]


def pending_count(session: Session) -> int:
    return int(session.execute(select(func.count(ApprovalItem.id))
                               .where(ApprovalItem.status == ApprovalStatus.PENDING.value)).scalar_one() or 0)


def command_view(row: ControlCommand) -> dict[str, Any]:
    result = row.result_json or {}
    finished = row.status in ("DONE", "FAILED", "REJECTED")
    ok = None
    if finished:
        ok = bool(result.get("ok")) if result else row.status == "DONE"
    return {"id": row.id, "type": row.command_type, "status": row.status, "finished": finished, "ok": ok,
            "message": text(result.get("message") or row.error or "", 600), "rule": result.get("rule"),
            "data": result.get("data") if isinstance(result.get("data"), dict) else {},
            "created_at": iso(row.created_at), "completed_at": iso(row.completed_at),
            "requested_by": row.requested_by}


def errors_grouped(session: Session, hi: datetime | None, limit: int = 50) -> list[dict[str, Any]]:
    stmt = select(ErrorLog).order_by(ErrorLog.id.desc()).limit(limit)
    if hi is not None:
        stmt = select(ErrorLog).where(ErrorLog.ts < hi).order_by(ErrorLog.id.desc()).limit(limit)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for e in session.execute(stmt).scalars():
        key = (e.component, e.message[:300])
        g = groups.get(key)
        if g is None:
            groups[key] = {"component": e.component, "message": text(e.message, 300), "severity": e.severity,
                           "count": 1, "last_ts": iso(e.ts), "first_ts": iso(e.ts)}
        else:
            g["count"] += 1
            g["first_ts"] = iso(e.ts)
    return sorted(groups.values(), key=lambda g: (-g["count"], g["last_ts"] or ""))


def recent_symbols(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    """Most recently analysed distinct symbols with their latest decision."""
    rows = session.execute(select(DecisionRow.id, DecisionRow.symbol, DecisionRow.created_at, DecisionRow.final_action,
                                  DecisionRow.confluence_score, DecisionRow.grade, DecisionRow.mode)
                           .order_by(DecisionRow.id.desc()).limit(600)).all()
    seen: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.symbol in seen:
            continue
        seen[r.symbol] = {"symbol": r.symbol, "last_decision_id": r.id, "time": iso(r.created_at),
                          "final_action": r.final_action, "confluence": r2(r.confluence_score), "grade": r.grade,
                          "mode": r.mode}
        if len(seen) >= limit:
            break
    return list(seen.values())


def latest_ids(session: Session) -> dict[str, int]:
    return {
        "decision": int(session.execute(select(func.max(DecisionRow.id))).scalar_one() or 0),
        "order": int(session.execute(select(func.max(Order.id))).scalar_one() or 0),
        "trade": int(session.execute(select(func.max(Trade.id))).scalar_one() or 0),
    }


def batch_detail_rows(session: Session, batch_id: int) -> list[dict[str, Any]]:
    return trades_between(session, None, None, batch_id=batch_id)


def day_of(dt: datetime | None) -> date | None:
    return to_ist(dt).date() if dt is not None else None


def batch_status_closed(status: str) -> bool:
    return status in (BatchStatus.CLOSED.value, BatchStatus.CLOSING.value)


def closed_position_ids(session: Session, *, lo: datetime | None = None, hi: datetime | None = None,
                        batch_id: int | None = None, limit: int = 50) -> list[int]:
    """Latest closed positions (newest exit first) in a scope - for aggregated post-mortems."""
    stmt = select(Position.id).where(Position.status == PositionStatus.CLOSED.value, Position.closed_at.is_not(None))
    if lo is not None:
        stmt = stmt.where(Position.closed_at >= lo)
    if hi is not None:
        stmt = stmt.where(Position.closed_at < hi)
    if batch_id is not None:
        stmt = stmt.where(Position.batch_id == batch_id)
    else:
        stmt = stmt.where(live_batches(Position.batch_id))
    return list(session.execute(stmt.order_by(Position.closed_at.desc(), Position.id.desc()).limit(limit)).scalars())


def batches_brief(session: Session) -> list[dict[str, Any]]:
    return [{"id": b.id, "name": b.name, "status": b.status, "starting_capital": r2(b.starting_capital),
             "started_at": iso(b.started_at or b.created_at), "ended_at": iso(b.ended_at)}
            for b in session.execute(select(Batch).where(Batch.deleted.is_(False)).order_by(Batch.id.desc())).scalars()]


def overall_capital_curve(curves: dict[int, list[dict[str, Any]]], batches: list[dict[str, Any]]
                          ) -> list[dict[str, Any]]:
    """One line for ALL batches: money invested so far + every started batch's P&L at that moment.

    value(t) = sum(top-ups of batches started by t) + sum(equity_b(t) - start_b). It ends at
    funding_summary's current_capital and stays continuous across batch switches (the old batch's
    equity is carried into the new one), unlike interleaving per-batch equity points.
    """
    info = {int(b["batch_id"]): b for b in batches if b.get("batch_id") is not None}
    series = {bid: sorted(pts, key=lambda p: p["time"]) for bid, pts in curves.items() if pts and bid in info}
    times = sorted({p["time"] for pts in series.values() for p in pts})
    out: list[dict[str, Any]] = []
    for t in times:
        total, any_started = 0.0, False
        for bid, pts in series.items():
            if pts[0]["time"] > t:
                continue
            last = next(p for p in reversed(pts) if p["time"] <= t)
            b = info[bid]
            start = float(b.get("starting_capital") or 0.0)
            total += float(b.get("top_up") or 0.0) + float(last["equity"] or 0.0) - start
            any_started = True
        if any_started:
            src = next((p for pts in series.values() for p in pts if p["time"] == t), {})
            out.append({"ts": src.get("ts"), "time": t, "equity": r2(total), "source": "OVERALL"})
    return out


def concat_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge several equity curves into one strictly time-ordered series (overall view)."""
    pts = [p for c in curves for p in c]
    pts.sort(key=lambda p: p["time"])
    return _dedupe_times(pts)
