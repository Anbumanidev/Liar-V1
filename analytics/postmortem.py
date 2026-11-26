"""Trade post-mortem: what happened, why it lost, and which system rule now prevents it.

``analyze_position`` rebuilds one paper trade end to end from the database (decision, agent
reports, debate, entry order, stop-loss history, exit legs) plus the price candles around it,
then runs deterministic diagnosis rules. Every number is computed here; nothing comes from an
LLM. ``analyze_day`` aggregates the issues of one day ("what went wrong today").

Diagnosis codes (severity): STALE_SIGNAL, CHASED_ENTRY, TIGHT_STOP, AGENTS_FAILED (high);
DEBATE_LOST, NO_ROOM, REVENGE_REENTRY, NOISE_STOP (medium); MIDDAY_ENTRY, CHARGES_HEAVY,
NEVER_WORKED (low). ``fixed_by`` names the rule/setting that now blocks the pattern (None =
advice only).
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.clock import IST, to_ist
from core.enums import Timeframe
from storage.models import AgentReportRow, DebateRow, DecisionRow, Order, Position, Trade

#: Diagnosis catalogue: code -> (severity, title, fix, fixed_by)
CATALOG: dict[str, tuple[str, str, str, str | None]] = {
    "STALE_SIGNAL": ("high", "Entered on stale analysis",
                     "Never execute a decision older than ~1 bar: decisions now expire (max age per mode, "
                     "signal validity) and only the best 2 setups per cycle go to the LLM so analysis finishes "
                     "inside the 5-minute cycle.",
                     "risk rule SIGNAL_EXPIRED (INTRADAY max 6 min) + two-phase cycle (top 2 setups, 240 s budget)"),
    "CHASED_ENTRY": ("high", "Chased the price - reward:risk collapsed at the fill",
                     "Re-price the plan at the live price before ordering and again at the fill; skip if R:R "
                     "to target 1 is no longer acceptable.",
                     "risk rule PRICE_MOVED (live R:R >= 0.8 x min R:R, drift <= 1 ATR) + fill check fill_rr_min 1.0"),
    "TIGHT_STOP": ("high", "Stop-loss inside normal noise",
                   "Stops must sit at least ~0.8-1 ATR from the actual entry, beyond structure.",
                   "risk.sl_min_atr 0.3 -> 0.8 + fill check (skip fills too close to the stop)"),
    "BOUGHT_HIGH": ("high", "Bought near the day's high in an already exhausted range",
                    "Do not chase: no longs in the upper part of a day range that has already moved its normal "
                    "daily distance (or far above VWAP), and skip stocks whose recent highs usually fade. "
                    "Wait for a pullback toward VWAP / support, or the next session.",
                    "risk rules CHASING_HIGH (no longs above 80% of the range when stretched, above 60% on an "
                    "exhausted day) + HISTORY_FADES (the stock's own last 20 sessions)"),
    "AGENTS_FAILED": ("high", "Most LLM agents failed - an uninformed decision",
                      "Refuse trades when the agent panel did not report; fix the cause (one 4B model on a 6 GB "
                      "GPU, 2 concurrent calls, 1 timeout retry).",
                      "risk rule AGENT_QUORUM (max 34% failed) + models low profile = qwen3:4b only"),
    "DEBATE_LOST": ("medium", "The bear case won the debate, the trade was taken anyway",
                    "A clearly lost debate is a veto, whatever the meta agent says.",
                    "risk rule DEBATE_AGAINST (bear - bull >= 10)"),
    "NO_ROOM": ("medium", "Resistance / support right in front of the entry",
                "Require at least 1R of room to the nearest opposing level before entering.",
                "risk rule NO_ROOM (min_room_r 0.5, noise levels ignored)"),
    "REVENGE_REENTRY": ("medium", "Re-entered a symbol right after being stopped out",
                        "Cool down after a stop-loss in the same symbol.",
                        "risk rule REENTRY_COOLDOWN (INTRADAY 60 min)"),
    "NOISE_STOP": ("medium", "Stopped out, then the price reached the target",
                   "The idea was right but the stop was too close or the entry too early; prefer structural "
                   "stops and wait for the bar to confirm.",
                   "risk.sl_min_atr 0.8 + fill check"),
    "MIDDAY_ENTRY": ("low", "Entered in the midday lull (11:45-13:15)",
                     "Indian large caps drift with low volume around lunch; intraday breakouts often fail.",
                     "INTRADAY entry windows 09:30-11:45 and 13:15-14:45"),
    "CHARGES_HEAVY": ("low", "Charges ate a large part of the move",
                      "With Rs 1,000 and 1-share positions, charges are ~20-30% of a 1R move. Prefer setups "
                      "with larger targets, or let capital grow before trading INTRADAY often.",
                      None),
    "NEVER_WORKED": ("low", "The move never started",
                     "Price never went 0.3R in favour: direction or timing was wrong from the start.",
                     None),
}

_ROOM_RE = re.compile(r"=\s*([0-9.]+)R")


def _f(x: Any) -> float | None:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _r2(x: Any) -> float | None:
    v = _f(x)
    return None if v is None else round(v, 2)


def _dt(x: Any) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, datetime):
        return to_ist(x)
    try:
        return to_ist(datetime.fromisoformat(str(x)))
    except ValueError:
        return None


def _iso(x: datetime | None) -> str | None:
    return x.isoformat() if x is not None else None


def _hm(x: datetime | None) -> str:
    return f"{x:%H:%M}" if x is not None else "--:--"


def _issue(code: str, detail: str) -> dict[str, Any]:
    sev, title, fix, fixed_by = CATALOG[code]
    return {"code": code, "severity": sev, "title": title, "detail": detail, "fix": fix, "fixed_by": fixed_by}


def _candles(data: Any, symbol: str, start: datetime, end: datetime) -> tuple[pd.DataFrame, str]:
    """1m candles for the window (5m fallback). Empty frame if unavailable."""
    if data is None:
        return pd.DataFrame(), ""
    for tf in (Timeframe.M1, Timeframe.M5):
        try:
            df = data.candles(symbol, tf, start=start)
        except Exception:  # noqa: BLE001 - post-mortem must work without data
            continue
        if df is not None and not df.empty:
            idx = df.index
            mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
            out = df.loc[mask]
            if not out.empty:
                return out, tf.value
    return pd.DataFrame(), ""


def _decision_block(session: Session, d: DecisionRow | None) -> dict[str, Any] | None:
    if d is None:
        return None
    meta = d.meta_json or {}
    agents = [
        {"agent": a.agent, "signal": a.signal, "confidence": a.confidence, "status": a.status,
         "latency_ms": a.latency_ms, "error": (a.error or "")[:160] or None}
        for a in session.execute(select(AgentReportRow).where(AgentReportRow.decision_id == d.id)
                                 .order_by(AgentReportRow.id)).scalars()
    ]
    deb = session.execute(select(DebateRow).where(DebateRow.decision_id == d.id)).scalars().first()
    debate = None
    if deb is not None:
        bull, bear = deb.bull_json or {}, deb.bear_json or {}
        debate = {"bull": bull.get("conviction"), "bear": bear.get("conviction"),
                  "bull_stance": bull.get("stance"), "bear_stance": bear.get("stance"),
                  "bull_points": (bull.get("top_arguments") or [])[:3],
                  "bear_points": (bear.get("top_arguments") or [])[:3]}
    return {"id": d.id, "created_at": _iso(_dt(d.created_at)), "latency_ms": d.latency_ms,
            "decision": d.decision, "confidence": d.confidence, "confluence": _r2(d.confluence_score),
            "grade": d.grade, "strategy": d.strategy, "final_action": d.final_action,
            "reasons_for": meta.get("reasons_for") or [], "reasons_against": meta.get("reasons_against_considered") or [],
            "agents": agents, "debate": debate}


def analyze_position(session: Session, position_id: int, data: Any = None, *,
                     settings: Any = None) -> dict[str, Any]:
    """Full post-mortem of one position (open or closed). Raises ``KeyError`` if unknown."""
    pos = session.get(Position, position_id)
    if pos is None:
        raise KeyError(f"position {position_id} not found")
    long_ = pos.direction == "LONG"
    sign = 1.0 if long_ else -1.0
    legs = list(session.execute(select(Trade).where(Trade.position_id == pos.id).order_by(Trade.exit_time)).scalars())
    entry_order = session.execute(select(Order).where(Order.position_id == pos.id, Order.purpose == "ENTRY")
                                  .order_by(Order.id)).scalars().first()
    decision = session.get(DecisionRow, pos.decision_id) if pos.decision_id else None
    prop = ((entry_order.meta_json or {}).get("proposal") or {}) if entry_order is not None else {}
    ctx = (decision.context_json or {}) if decision is not None else {}

    entry = float(pos.avg_entry)
    sl0 = float(pos.initial_stop_loss)
    risk = abs(entry - sl0) or None
    t1, t2 = _f(pos.target_1), _f(pos.target_2)
    opened = _dt(pos.opened_at)
    closed = _dt(pos.closed_at)
    qty0 = pos.initial_qty
    gross = sum(t.gross_pnl for t in legs)
    charges = sum(t.charges for t in legs)
    net = sum(t.net_pnl for t in legs)
    exit_qty = sum(t.qty for t in legs)
    exit_px = (sum(t.exit_price * t.qty for t in legs) / exit_qty) if exit_qty else None
    status_open = pos.status == "OPEN"
    result = "OPEN" if status_open else ("WIN" if net > 0.005 else "LOSS" if net < -0.005 else "FLAT")
    r_mult = (net / (risk * qty0)) if risk and qty0 else None

    # ---- planned numbers (proposal / signal) -------------------------------------------------
    planned_entry = _f(prop.get("entry_price")) or _f(decision.entry_price if decision else None)
    atr = _f(prop.get("atr"))
    data_at = _dt(prop.get("created_at"))           # when the analysed data was read
    signal = None
    for sg in ctx.get("signals") or []:
        if decision is not None and sg.get("strategy") == (decision.strategy or pos.strategy):
            signal = sg
            break
    if signal is None and ctx.get("signals"):
        signal = ctx["signals"][0]
    valid_until = _dt(prop.get("valid_until")) or _dt((signal or {}).get("valid_until"))
    decided_at = _dt(decision.created_at) if decision is not None else None
    vote = ctx.get("vote") or {}
    voting = (vote.get("bullish", 0) + vote.get("bearish", 0) + vote.get("neutral", 0)) - vote.get("skipped", 0)
    failed = vote.get("failed")
    failed_pct = _f(prop.get("failed_agents_pct"))
    if failed_pct is None and failed is not None and voting > 0:
        failed_pct = 100.0 * failed / voting
    room_r = _f(prop.get("room_r"))
    if room_r is None:
        for note in (ctx.get("confluence") or {}).get("notes") or []:
            if note.startswith("room:"):
                m = _ROOM_RE.search(note)
                room_r = float(m.group(1)) if m else None
    dec = _decision_block(session, decision)
    deb = (dec or {}).get("debate") or {}

    # ---- price behaviour (MFE / MAE, what happened after the exit) ---------------------------
    win_start = (opened or datetime.now(IST)) - timedelta(minutes=5)
    day_end = datetime.combine((opened or datetime.now(IST)).date(), time(15, 30), tzinfo=IST)
    in_trade, tf_used = _candles(data, pos.symbol, win_start, closed or day_end)
    mfe = mae = None
    if not in_trade.empty and opened is not None:
        seg = in_trade.loc[in_trade.index >= pd.Timestamp(opened.replace(second=0, microsecond=0))]
        if closed is not None:
            seg = seg.loc[seg.index <= pd.Timestamp(closed)]
        if not seg.empty:
            hi, lo = float(seg["high"].max()), float(seg["low"].min())
            mfe = (hi - entry) if long_ else (entry - lo)
            mae = (entry - lo) if long_ else (hi - entry)
    if mfe is None and pos.highest_price is not None and pos.lowest_price is not None:
        mfe = (pos.highest_price - entry) if long_ else (entry - pos.lowest_price)
        mae = (entry - pos.lowest_price) if long_ else (pos.highest_price - entry)
    after_hit_t1: datetime | None = None
    if closed is not None and t1 is not None and pos.mode == "INTRADAY":
        after, _ = _candles(data, pos.symbol, closed, day_end)
        if not after.empty:
            hit = after[(after["high"] >= t1)] if long_ else after[(after["low"] <= t1)]
            if not hit.empty:
                after_hit_t1 = to_ist(hit.index[0].to_pydatetime())

    # ---- story ------------------------------------------------------------------------------
    story: list[str] = []
    if data_at is not None:
        conf = (ctx.get("confluence") or {}).get("score")
        story.append(f"{_hm(data_at)} market data read for analysis: {pos.strategy or 'setup'} "
                     f"{pos.direction} plan entry {planned_entry if planned_entry is not None else '?'}, "
                     f"stop {sl0}, target 1 {t1}" + (f"; confluence {conf:.1f} ({decision.grade})" if conf and decision else ""))
    if decided_at is not None:
        lag = (decided_at - data_at).total_seconds() / 60 if data_at else None
        fails = f"; {failed}/{voting} LLM agents failed" if failed else ""
        debate_txt = f"; debate bull {deb.get('bull')} vs bear {deb.get('bear')}" if deb.get("bull") is not None else ""
        story.append(f"{_hm(decided_at)} decision {decision.decision} (confidence {decision.confidence})"
                     + (f" {lag:.1f} min after the data" if lag is not None else "") + fails + debate_txt)
    if entry_order is not None:
        filled = _dt(entry_order.filled_at)
        diff = (entry - planned_entry) if planned_entry else None
        story.append(f"{_hm(filled)} filled {qty0} @ {entry:.2f}"
                     + (f" ({'+' if diff >= 0 else ''}{diff:.2f} vs plan)" if diff is not None else ""))
    for h in pos.sl_history_json or []:
        if h.get("by") != "ENTRY":
            story.append(f"{_hm(_dt(h.get('ts')))} stop moved {h.get('old')} -> {h.get('new')} ({h.get('by')})")
    for t in legs:
        story.append(f"{_hm(_dt(t.exit_time))} exit {t.qty} @ {t.exit_price:.2f} ({t.exit_reason}"
                     f"{', partial' if t.is_partial else ''}): net {t.net_pnl:+.2f}")
    if closed is not None and t1 is not None and pos.mode == "INTRADAY":
        story.append(f"after the exit: price {'reached' if after_hit_t1 else 'did not reach'} target 1 {t1}"
                     + (f" at {_hm(after_hit_t1)}" if after_hit_t1 else " that day"))

    # ---- diagnosis --------------------------------------------------------------------------
    issues: list[dict[str, Any]] = []
    max_age = 6.0
    if settings is not None:
        try:
            max_age = float(settings.mode(pos.mode).max_signal_age_minutes or max_age)
        except Exception:  # noqa: BLE001
            pass
    if entry_order is not None and data_at is not None:
        filled = _dt(entry_order.filled_at) or decided_at
        age = (filled - data_at).total_seconds() / 60 if filled else None
        expired = valid_until is not None and filled is not None and filled > valid_until
        if (age is not None and age > max_age) or expired:
            parts = []
            if age is not None:
                parts.append(f"filled {age:.1f} min after the data was read (max {max_age:g})")
            if expired:
                parts.append(f"the signal expired at {_hm(valid_until)}")
            if dec and dec.get("latency_ms"):
                parts.append(f"the analysis itself took {dec['latency_ms'] / 60000:.1f} min")
            issues.append(_issue("STALE_SIGNAL", "; ".join(parts)))
    if t1 is not None and risk:
        rr_fill = ((t1 - entry) * sign) / risk
        rr_plan = None
        if planned_entry is not None and abs(planned_entry - sl0) > 0:
            rr_plan = ((t1 - planned_entry) * sign) / abs(planned_entry - sl0)
        if rr_fill < 1.0:
            issues.append(_issue("CHASED_ENTRY", f"R:R to target 1 was {rr_fill:.2f} at the fill {entry:.2f}"
                                 + (f" (planned {rr_plan:.2f} at {planned_entry:.2f})" if rr_plan else "")))
    if atr and risk and risk < 1.0 * atr:
        issues.append(_issue("TIGHT_STOP", f"stop {risk:.2f} below the fill = {risk / atr:.2f} ATR "
                             f"(5m ATR {atr:.2f}); normal 5-minute noise reaches it"))
    if failed_pct is not None and failed_pct > 34:
        issues.append(_issue("AGENTS_FAILED", f"{failed_pct:.0f}% of the LLM agents failed"
                             + (f" ({failed}/{voting}, mostly timeouts)" if failed else "")))
    if deb.get("bull") is not None and deb.get("bear") is not None and deb["bear"] - deb["bull"] >= 10:
        issues.append(_issue("DEBATE_LOST", f"bear conviction {deb['bear']} vs bull {deb['bull']}"))
    if room_r is not None and room_r < 0.5:
        issues.append(_issue("NO_ROOM", f"nearest opposing level only {room_r:.2f}R from the planned entry"))
    if opened is not None:
        prev = session.execute(
            select(Trade).where(Trade.batch_id == pos.batch_id, Trade.symbol == pos.symbol, Trade.mode == pos.mode,
                                Trade.exit_reason.in_(("SL", "GAP")), Trade.exit_time < pos.opened_at,
                                Trade.exit_time >= pos.opened_at - timedelta(minutes=60))
            .order_by(Trade.exit_time.desc())).scalars().first()
        if prev is not None:
            mins = (opened - _dt(prev.exit_time)).total_seconds() / 60
            issues.append(_issue("REVENGE_REENTRY", f"{pos.symbol} was stopped out {mins:.0f} min earlier"))
        if pos.mode == "INTRADAY" and time(11, 45) <= opened.time() < time(13, 15):
            issues.append(_issue("MIDDAY_ENTRY", f"entry at {_hm(opened)}"))
    beh = prop.get("behaviour") if isinstance(prop.get("behaviour"), dict) else None
    if beh is None and data is not None and opened is not None and pos.mode in ("INTRADAY", "BTST"):
        try:
            from features.price_behaviour import compute

            if settings is None:
                from config.loader import load_default_settings

                settings = load_default_settings()
            beh = compute(data, pos.symbol, opened, direction=pos.direction, live_price=entry,
                          thresholds=settings.risk)
        except Exception:  # noqa: BLE001 - advisory
            beh = None
    if beh and beh.get("range_pos") is not None and beh.get("range_vs_adr") is not None:
        side_pos = beh["range_pos"] if long_ else 1.0 - beh["range_pos"]
        if beh.get("verdict") in ("CHASING", "FADES") or (side_pos >= 0.7 and beh["range_vs_adr"] >= 1.0):
            parts = [f"entry at {side_pos * 100:.0f}% toward the {'high' if long_ else 'low'} of the day's range "
                     f"{beh.get('day_low')}-{beh.get('day_high')}",
                     f"the day had already moved {beh['range_vs_adr'] * 100:.0f}% of its normal range"]
            if beh.get("vwap_ext_atr") is not None:
                parts.append(f"{abs(beh['vwap_ext_atr']):.1f} ATR {'above' if beh['vwap_ext_atr'] >= 0 else 'below'} "
                             f"VWAP {beh.get('vwap')}")
            if beh.get("samples"):
                parts.append(f"in its last {beh['samples']} comparable sessions only "
                             f"{(beh.get('continuation_rate') or 0) * 100:.0f}% of such moves continued")
            issues.append(_issue("BOUGHT_HIGH", "; ".join(parts)))
    stopped = any(t.exit_reason in ("SL", "GAP") for t in legs)
    if stopped and after_hit_t1 is not None:
        issues.append(_issue("NOISE_STOP", f"stopped at {_hm(closed)}, target 1 {t1} reached at {_hm(after_hit_t1)}"))
    if not status_open and risk and mfe is not None and mfe < 0.3 * risk and result == "LOSS":
        issues.append(_issue("NEVER_WORKED", f"best move in favour {mfe:.2f} = {mfe / risk:.2f}R"))
    if not status_open and charges > 0 and risk and qty0 and charges > 0.15 * risk * qty0:
        issues.append(_issue("CHARGES_HEAVY", f"charges {charges:.2f} = {100 * charges / (risk * qty0):.0f}% of the 1R risk "
                             f"({risk * qty0:.2f})"))
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])

    chart_from = min(x for x in (data_at, opened) if x is not None) - timedelta(minutes=45) if (data_at or opened) else None
    chart_to = (closed or datetime.now(IST)) + timedelta(minutes=45)
    if chart_to.time() > time(15, 30):
        chart_to = datetime.combine(chart_to.date(), time(15, 30), tzinfo=IST)
    return {
        "position_id": pos.id, "batch_id": pos.batch_id, "symbol": pos.symbol, "mode": pos.mode,
        "direction": pos.direction, "strategy": pos.strategy, "grade": pos.grade, "qty": qty0,
        "entry_time": _iso(opened), "entry_price": _r2(entry), "planned_entry": _r2(planned_entry),
        "exit_time": _iso(closed), "exit_price": _r2(exit_px),
        "initial_sl": _r2(sl0), "final_sl": _r2(pos.stop_loss), "target_1": _r2(t1), "target_2": _r2(t2),
        "exit_reasons": sorted({t.exit_reason for t in legs}),
        "gross_pnl": _r2(gross), "charges": _r2(charges), "net_pnl": _r2(net),
        "r_multiple": _r2(r_mult), "holding_minutes": int((closed - opened).total_seconds() // 60) if closed and opened else None,
        "result": result,
        "legs": [{"qty": t.qty, "exit_time": _iso(_dt(t.exit_time)), "exit_price": _r2(t.exit_price),
                  "reason": t.exit_reason, "partial": bool(t.is_partial), "net_pnl": _r2(t.net_pnl)} for t in legs],
        "sl_history": pos.sl_history_json or [],
        "atr": _r2(atr), "risk_per_share": _r2(risk),
        "mfe": _r2(mfe), "mae": _r2(mae),
        "mfe_r": _r2(mfe / risk) if mfe is not None and risk else None,
        "mae_r": _r2(mae / risk) if mae is not None and risk else None,
        "data_read_at": _iso(data_at), "decided_at": _iso(decided_at), "signal_valid_until": _iso(valid_until),
        "failed_agents_pct": _r2(failed_pct), "room_r": _r2(room_r), "behaviour": beh,
        "story": story, "diagnosis": issues, "decision": dec,
        "chart": {"timeframe": "5m" if not tf_used or tf_used == "5m" else "1m",
                  "from": _iso(chart_from), "to": _iso(chart_to)},
    }


def analyze_day(session: Session, day: date, batch_id: int | None = None, data: Any = None, *,
                settings: Any = None) -> dict[str, Any]:
    """Post-mortem of every position opened on ``day`` (optionally one batch)."""
    start = datetime.combine(day, time(0, 0), tzinfo=IST)
    stmt = select(Position.id).where(Position.opened_at >= start, Position.opened_at < start + timedelta(days=1))
    if batch_id is not None:
        stmt = stmt.where(Position.batch_id == batch_id)
    trades = [analyze_position(session, pid, data, settings=settings)
              for pid in session.execute(stmt.order_by(Position.opened_at)).scalars()]
    closed = [t for t in trades if t["result"] != "OPEN"]
    counts: dict[str, int] = {}
    for t in trades:
        for i in t["diagnosis"]:
            counts[i["code"]] = counts.get(i["code"], 0) + 1
    sev = {"high": 0, "medium": 1, "low": 2}
    top = sorted(counts.items(), key=lambda kv: (sev[CATALOG[kv[0]][0]], -kv[1]))
    return {
        "day": day.isoformat(),
        "trades": trades,
        "summary": {"trades": len(closed), "open": len(trades) - len(closed),
                    "wins": sum(1 for t in closed if t["result"] == "WIN"),
                    "losses": sum(1 for t in closed if t["result"] == "LOSS"),
                    "net_pnl": round(sum(t["net_pnl"] or 0 for t in closed), 2),
                    "charges": round(sum(t["charges"] or 0 for t in closed), 2)},
        "top_issues": [{"code": c, "count": n, "severity": CATALOG[c][0], "title": CATALOG[c][1],
                        "fix": CATALOG[c][2], "fixed_by": CATALOG[c][3]} for c, n in top],
    }


__all__ = ["CATALOG", "analyze_day", "analyze_position"]
