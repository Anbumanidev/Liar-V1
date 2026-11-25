"""Scheduled jobs (section 10). Each job is ``fn(services) -> dict`` and runs only on NSE
trading days. Jobs that open NEW trades also require the engine to be RUNNING and the batch
ACTIVE; position monitoring, exits and EOD always run.

The live engine fires them from APScheduler cron triggers; REPLAY fires the very same
functions from ``scheduler.plan.DailyPlan`` while advancing a simulated clock.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import func, select

from agents.graph import close_run, open_run
from core.clock import parse_hhmm, to_ist
from core.enums import BatchStatus, EngineState, Mode, PositionStatus, Product, RunType
from core.logging import log_event
from paper_engine.views import open_position_views
from storage.audit import record_error
from storage.batches import closing_batches, get_current_batch
from storage.models import DailySummary, DecisionRow, Position, StockAnalyzed, Trade

JobFn = Callable[[Any], dict[str, Any]]


def _now(svc: Any) -> datetime:
    return to_ist(svc.clock.now())


def _trading_day(svc: Any, d: date) -> bool:
    return bool(svc.calendar.is_trading_day(d)) if svc.calendar is not None else d.weekday() < 5


def _engine_running(svc: Any) -> bool:
    from control.status import get_engine_state

    if svc.ctx.is_replay:
        return svc.extras.get("engine_state", EngineState.RUNNING) == EngineState.RUNNING
    with svc.session() as s:
        return get_engine_state(s) == EngineState.RUNNING


def _active_batch_id(svc: Any) -> int | None:
    with svc.session() as s:
        b = get_current_batch(s)
        return b.id if b is not None and b.status == BatchStatus.ACTIVE.value else None


def _mode_tradeable(svc: Any, mode: Mode, batch_id: int) -> bool:
    cfg = svc.settings.mode(mode)
    if not cfg.enabled:
        return False
    from risk.state import load_mode_states

    with svc.session() as s:
        ms = load_mode_states(s, batch_id).get(mode)
    return not (ms is not None and ms.paused)


def _risk_state(svc: Any, batch_id: int, symbol: str | None = None) -> Any:
    from risk.state import build_risk_state

    with svc.session() as s:
        st = build_risk_state(s, batch_id=batch_id, capital_manager=svc.capital, clock=svc.clock,
                              calendar=svc.calendar, data=None, symbol=symbol)
        s.rollback()
    return st


# =========================================================================== universe / watchlist


def build_watchlist(svc: Any, mode: Mode, batch_id: int, run_id: int | None = None) -> list[str]:
    """Universe -> filters (incl. 'risk manager can size >= 1 share') -> logged watchlist."""
    settings = svc.settings
    symbols = svc.universe.build_universe(mode, settings) if svc.universe is not None else []
    max_price = svc.risk.max_affordable_price(_risk_state(svc, batch_id), mode)
    included, excluded = svc.universe.filter_universe(symbols, svc.data, settings, mode, max_affordable_price=max_price)
    now = _now(svc)
    with svc.session() as s:
        for sym in included:
            s.add(StockAnalyzed(run_id=run_id, batch_id=batch_id, symbol=sym, mode=mode.value, as_of=now,
                                stage="WATCHLIST", included=True, reason="passed universe filters"))
        for sym, reason in excluded:
            s.add(StockAnalyzed(run_id=run_id, batch_id=batch_id, symbol=sym, mode=mode.value, as_of=now,
                                stage="UNIVERSE", included=False, reason=reason))
        s.commit()
    log_event("scans", {"event": "watchlist", "mode": mode.value, "included": len(included),
                        "excluded": len(excluded), "max_affordable_price": max_price}, clock=svc.clock)
    svc.extras.setdefault("watchlist", {})[mode.value] = {"day": now.date().isoformat(), "symbols": included}
    return included


def _watchlist(svc: Any, mode: Mode, batch_id: int) -> list[str]:
    wl = svc.extras.get("watchlist", {}).get(mode.value)
    if wl and wl.get("day") == _now(svc).date().isoformat():
        return list(wl["symbols"])
    return build_watchlist(svc, mode, batch_id)


def _ranked(svc: Any, mode: Mode, symbols: list[str], cap: int) -> list[str]:
    """Top ``cap`` 'stocks in play' (cached for ``engine.ranking_refresh_minutes``)."""
    from strategies.ranking import rank_in_play

    now = _now(svc)
    bucket = (now.hour * 60 + now.minute) // svc.settings.engine.ranking_refresh_minutes
    key = f"ranking:{mode.value}"
    cached = svc.extras.get(key)
    if cached and cached[0] == (now.date(), bucket):
        return cached[1]
    ranked = rank_in_play(symbols, svc.data, now, top_n=cap)
    chunk = [r.symbol for r in ranked]
    svc.extras[key] = ((now.date(), bucket), chunk)
    log_event("scans", {"event": "in_play_ranking", "mode": mode.value,
                        "top": [{"symbol": r.symbol, "score": r.score, "rvol": r.rvol, "move_pct": r.move_pct}
                                for r in ranked]}, clock=svc.clock)
    return chunk


def _scan(svc: Any, mode: Mode, run_type: RunType, *, rotate: bool, check_timing: bool) -> dict[str, Any]:
    batch_id = _active_batch_id(svc)
    if batch_id is None or not _engine_running(svc) or not _mode_tradeable(svc, mode, batch_id):
        return {"skipped": "engine not running / batch not active / mode disabled or paused"}
    symbols = _watchlist(svc, mode, batch_id)
    cap = svc.settings.mode(mode).max_symbols_per_cycle
    if rotate and len(symbols) > cap and svc.settings.engine.watchlist_ranking:
        chunk = _ranked(svc, mode, symbols, cap)
    elif rotate and len(symbols) > cap:
        key = f"rotation:{mode.value}"
        start = svc.extras.get(key, 0) % len(symbols)
        chunk = (symbols[start:] + symbols[:start])[:cap]
        svc.extras[key] = start + cap
    else:
        chunk = symbols[:cap]
    with svc.session() as s:
        run = open_run(s, run_type, batch_id=batch_id, mode=mode, clock=svc.clock,
                       settings_version_id=svc.ctx.settings_version_id, is_replay=svc.ctx.is_replay)
        s.commit()
        run_id = run.id
    actions = _analyse_chunk(svc, mode, chunk, batch_id=batch_id, run_id=run_id, check_timing=check_timing)
    with svc.session() as s:
        run = s.get(type(run), run_id)
        close_run(s, run, clock=svc.clock, n_symbols=len(chunk), summary={"actions": actions})
        s.commit()
    return {"run_id": run_id, "symbols": len(chunk), "actions": actions}


def _analyse_chunk(svc: Any, mode: Mode, chunk: list[str], *, batch_id: int, run_id: int,
                   check_timing: bool) -> dict[str, int]:
    """Two phases: deterministic screen of every symbol, then the LLM for the best few only.

    Phase 1 (seconds): features + strategies + confluence for the whole chunk. Phase 2: the
    ``engine.max_llm_candidates_per_cycle`` highest-confluence setups go through the agents,
    debate, meta and risk, strongest first, until ``engine.cycle_llm_budget_seconds`` is used
    (LLM backends only). Every other setup is recorded as FILTERED with the reason and shadow-
    tracked, so nothing disappears from the audit trail.
    """
    import time as _time

    eng = svc.settings.engine
    actions: dict[str, int] = {}

    def _count(res: Any) -> None:
        actions[res.final_action] = actions.get(res.final_action, 0) + 1

    pipeline = svc.pipeline
    if not hasattr(pipeline, "screen"):   # test doubles with only analyze()
        for sym in chunk:
            _count(pipeline.analyze(sym, mode, batch_id=batch_id, run_id=run_id, check_timing=check_timing))
        return actions
    pending = []
    for sym in chunk:
        h = pipeline.screen(sym, mode, batch_id=batch_id, run_id=run_id, check_timing=check_timing)
        if h.pending:
            pending.append(h)
        else:
            _count(h.result)
    pending.sort(key=lambda h: -h.confluence)
    top_k = max(1, int(eng.max_llm_candidates_per_cycle))
    budgeted = eng.llm_backend != "rules"
    window_ok, window_why = True, ""
    if check_timing and getattr(svc, "risk", None) is not None and hasattr(svc.risk, "in_entry_window"):
        window_ok, window_why = svc.risk.in_entry_window(mode, _now(svc))
    t0 = _time.monotonic()
    sent = 0
    for i, h in enumerate(pending):
        reason = None
        if not window_ok:
            reason = f"outside the entry window ({window_why}); no LLM time spent"
        elif i >= top_k:
            reason = (f"setup passed the filter (confluence {h.confluence:.1f}) but was not in this cycle's "
                      f"top {top_k} by confluence")
        elif budgeted and i > 0 and _time.monotonic() - t0 > eng.cycle_llm_budget_seconds:
            reason = (f"cycle LLM time budget ({eng.cycle_llm_budget_seconds}s) used up before this setup "
                      f"(confluence {h.confluence:.1f})")
        sent += reason is None
        _count(pipeline.resume(h, skip_reason=reason))
    if pending:
        log_event("scans", {"event": "cycle_llm", "mode": mode.value, "run_id": run_id,
                            "screened": len(chunk), "passed_filter": len(pending), "sent_to_llm": sent,
                            "candidates": [{"symbol": h.symbol, "confluence": round(h.confluence, 2)} for h in pending],
                            "llm_seconds": round(_time.monotonic() - t0, 1)}, clock=svc.clock)
    return actions


# =========================================================================== jobs


def job_premarket(svc: Any) -> dict[str, Any]:
    """08:30: event calendar refresh, corporate actions on held CNC positions, caches."""
    today = _now(svc).date()
    out: dict[str, Any] = {}
    if svc.events is not None:
        try:
            out["events"] = svc.events.refresh(today, today + timedelta(days=45))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[premarket] event refresh failed: {exc}")
        from paper_engine.corporate_actions import apply_corporate_actions

        with svc.session() as s:
            rows = apply_corporate_actions(s, svc.events.corporate_actions_on(today), today, svc.capital, svc.clock)
            s.commit()
        out["corporate_actions"] = len(rows)
    svc.extras.pop("watchlist", None)
    return out


def job_watchlist(svc: Any) -> dict[str, Any]:
    """09:00: intraday watchlist (every stock and reason logged)."""
    batch_id = _active_batch_id(svc)
    if batch_id is None or not svc.settings.mode(Mode.INTRADAY).enabled:
        return {"skipped": "no active batch / INTRADAY disabled"}
    return {"symbols": len(build_watchlist(svc, Mode.INTRADAY, batch_id))}


def job_preopen_check(svc: Any) -> dict[str, Any]:
    """09:08: overnight gap check on open CNC positions (pre-open price vs stop)."""
    alerts = 0
    with svc.session() as s:
        rows = s.execute(select(Position).where(Position.status == PositionStatus.OPEN.value,
                                                Position.product == Product.CNC.value)).scalars().all()
        for p in rows:
            q = None
            try:
                q = svc.data.quote(p.symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[preopen] quote {p.symbol}: {exc}")
            if q is None:
                continue
            if (p.direction == "LONG" and q.ltp <= p.stop_loss) or (p.direction == "SHORT" and q.ltp >= p.stop_loss):
                alerts += 1
                if svc.notifier is not None:
                    svc.notifier.urgent(f"Pre-open gap through stop-loss: {p.symbol}",
                                        f"{p.symbol} pre-open {q.ltp} vs stop {p.stop_loss}; it will exit at the open.",
                                        {"position_id": p.id, "mode": p.mode})
    return {"gap_alerts": alerts}


def job_intraday_cycle(svc: Any) -> dict[str, Any]:
    """09:30-14:45 every N minutes: INTRADAY analysis cycle over the (rotating) watchlist."""
    t = _now(svc).time()
    sc = svc.settings.schedule
    if not (parse_hhmm(sc.intraday_cycle_start) <= t <= parse_hhmm(sc.intraday_cycle_end)):
        return {"skipped": "outside the intraday cycle window"}
    return _scan(svc, Mode.INTRADAY, RunType.INTRADAY_CYCLE, rotate=True, check_timing=True)


def job_btst_scan(svc: Any) -> dict[str, Any]:
    """14:30-15:15: BTST scan and entries (if enabled)."""
    if not svc.settings.mode(Mode.BTST).enabled:
        return {"skipped": "BTST disabled"}
    return _scan(svc, Mode.BTST, RunType.BTST_SCAN, rotate=False, check_timing=True)


def job_monitor(svc: Any) -> dict[str, Any]:
    """Fills, exits, trailing, time stops, square-off, HALT (also fired at 15:15 / 15:20)."""
    with svc.session() as s:
        rep = svc.monitor.check(s)
        s.commit()
    return {"opened": rep.opened, "exits": rep.exits, "halted": rep.halted_batches, "errors": rep.errors}


def job_squareoff(svc: Any) -> dict[str, Any]:
    return job_monitor(svc)


def job_cnc_time_stop(svc: Any) -> dict[str, Any]:
    return job_monitor(svc)


def _daily_rows(svc: Any, batch_id: int, day: date) -> None:
    """Per-mode daily_summary rows (TOTAL is written by CapitalManager.run_eod)."""
    from datetime import time as dtime

    from core.clock import IST

    start = datetime.combine(day, dtime(0, 0), tzinfo=IST)
    end = start + timedelta(days=1)
    with svc.session() as s:
        for mode in Mode:
            q = select(func.count(Trade.id), func.coalesce(func.sum(Trade.gross_pnl), 0.0),
                       func.coalesce(func.sum(Trade.charges), 0.0), func.coalesce(func.sum(Trade.net_pnl), 0.0)) \
                .where(Trade.batch_id == batch_id, Trade.mode == mode.value, Trade.exit_time >= start,
                       Trade.exit_time < end)
            n, gross, charges, net = s.execute(q).one()
            wins = s.scalar(select(func.count(Trade.id)).where(
                Trade.batch_id == batch_id, Trade.mode == mode.value, Trade.exit_time >= start, Trade.exit_time < end,
                Trade.net_pnl > 0)) or 0
            row = s.execute(select(DailySummary).where(DailySummary.batch_id == batch_id, DailySummary.day == day,
                                                       DailySummary.mode == mode.value)).scalar_one_or_none()
            if row is None:
                row = DailySummary(batch_id=batch_id, day=day, mode=mode.value)
                s.add(row)
            row.trades, row.wins, row.losses = int(n), int(wins), int(n) - int(wins)
            row.gross_pnl, row.charges, row.net_pnl = float(gross), float(charges), float(net)
        s.commit()


def job_eod(svc: Any) -> dict[str, Any]:
    """15:30: capital EOD (profit lock / loss absorb / capped refill / sizing / HWM), daily summary,
    strategy self-evaluation, finalisation of CLOSING batches."""
    day = _now(svc).date()
    out: dict[str, Any] = {}
    with svc.session() as s:
        batch_ids = [b.id for b in ([get_current_batch(s)] + closing_batches(s)) if b is not None]
    for bid in batch_ids:
        with svc.session() as s:
            views = open_position_views(s, bid)
            snap = svc.capital.snapshot(s, bid, views, 0.0, update_hwm=True)
            res = svc.capital.run_eod(s, bid, day, None, snap.total_equity, snap.tier)
            s.commit()
        _daily_rows(svc, bid, day)
        with svc.session() as s:
            from strategies.performance import evaluate_strategies

            evaluate_strategies(s, bid, svc.settings, svc.clock)
            ok, details = svc.capital.check_ledger_balance(s, bid)
            if not ok:
                record_error(s, "scheduler.eod", f"ledger imbalance for batch {bid}: {details.get('problems')}",
                             details, batch_id=bid, clock=svc.clock)
            s.commit()
        out[str(bid)] = res.as_dict()
    with svc.session() as s:
        from control.batches import finalize_closing_batches

        out["closed_batches"] = finalize_closing_batches(s, svc)
        s.commit()
    svc.extras.pop("watchlist", None)
    return out


def job_eod_report(svc: Any) -> dict[str, Any]:
    """15:45: EOD report e-mail."""
    if svc.notifier is None:
        return {"skipped": "notifier disabled"}
    from notifications.reports import build_eod_report

    day = _now(svc).date()
    with svc.session() as s:
        b = get_current_batch(s)
        if b is None:
            return {"skipped": "no batch"}
        report = build_eod_report(s, b.id, day, svc.clock, settings=svc.settings)
        try:
            from storage.models import Position as _Pos
            from storage.models import Trade as _Trade
            from tax.engine import summarize, tax_tips, trade_tax_lines

            trades = list(s.execute(select(_Trade).where(_Trade.batch_id == b.id)).scalars())
            opens = list(s.execute(select(_Pos).where(_Pos.batch_id == b.id,
                                                      _Pos.status == PositionStatus.OPEN.value)).scalars())
            summary = summarize(trade_tax_lines(trades, svc.settings.charges, svc.settings.tax), svc.settings.tax, day)
            report = dict(report)
            report["tax"] = summary.as_dict()
            report["tax_tips"] = [t.as_dict() for t in tax_tips(summary, opens, day, svc.settings.tax,
                                                                svc.settings.charges)
                                  if t.severity in ("saving", "warning")][:5]
        except Exception as exc:  # noqa: BLE001 - tax section is informational
            logger.warning(f"[eod_report] tax section failed: {exc}")
    svc.notifier.eod(report)
    return {"sent": True}


def job_swing_scan(svc: Any) -> dict[str, Any]:
    """16:00-18:00: SWING / POSITIONAL scan on daily data -> next-session candidates."""
    out: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for mode, rt in ((Mode.SWING, RunType.SWING_SCAN), (Mode.POSITIONAL, RunType.POSITIONAL_SCAN)):
        if not svc.settings.mode(mode).enabled:
            continue
        res = _scan(svc, mode, rt, rotate=False, check_timing=False)
        out[mode.value] = res
        run_id = res.get("run_id")
        if run_id is None:
            continue
        with svc.session() as s:
            for d in s.execute(select(DecisionRow).where(DecisionRow.run_id == run_id,
                                                         DecisionRow.final_action.in_(["QUEUED", "EXECUTED",
                                                                                       "SIGNAL_ONLY"]))).scalars():
                candidates.append({"symbol": d.symbol, "mode": d.mode, "direction": d.direction, "entry": d.entry_price,
                                   "stop_loss": d.stop_loss, "target_1": d.target_1, "target_2": d.target_2,
                                   "qty": d.qty, "confidence": d.confidence, "grade": d.grade,
                                   "status": d.final_action})
    if svc.notifier is not None and out:
        nxt = svc.calendar.next_trading_day(_now(svc).date()) if svc.calendar is not None else None
        svc.notifier.swing_scan(candidates, for_day=nxt)
    out["candidates"] = len(candidates)
    return out


def job_shadow_update(svc: Any) -> dict[str, Any]:
    with svc.session() as s:
        n = svc.shadow.update(s)
        s.commit()
    return {"closed": n}


def job_approval_expiry(svc: Any) -> dict[str, Any]:
    with svc.session() as s:
        expired = svc.queue.expire(s)
        s.commit()
    return {"expired": [i.id for i in expired]}


JOB_FUNCS: dict[str, JobFn] = {
    "premarket": job_premarket,
    "watchlist": job_watchlist,
    "preopen_check": job_preopen_check,
    "intraday_cycle": job_intraday_cycle,
    "btst_scan": job_btst_scan,
    "squareoff": job_squareoff,
    "cnc_time_stop": job_cnc_time_stop,
    "eod": job_eod,
    "eod_report": job_eod_report,
    "swing_scan": job_swing_scan,
    "monitor": job_monitor,
    "shadow_update": job_shadow_update,
    "approval_expiry": job_approval_expiry,
}

#: Jobs that also run on non-trading days.
ALWAYS_JOBS = frozenset({"approval_expiry"})


def run_job(svc: Any, name: str, *, force: bool = False) -> dict[str, Any]:
    """Run job ``name`` safely (trading-day check, error capture, JSONL log). Never raises."""
    fn = JOB_FUNCS.get(name)
    if fn is None:
        return {"error": f"unknown job {name}"}
    today = _now(svc).date()
    if not force and name not in ALWAYS_JOBS and not _trading_day(svc, today):
        return {"skipped": "not a trading day"}
    try:
        result = fn(svc)
    except Exception as exc:  # noqa: BLE001 - a failing job never kills the engine
        with svc.session() as s:
            record_error(s, f"scheduler.{name}", exc, {}, batch_id=svc.ctx.batch_id, clock=svc.clock)
            s.commit()
        return {"error": str(exc)}
    if name not in ("monitor", "approval_expiry", "shadow_update"):
        log_event("scans", {"job": name, "result": result}, clock=svc.clock)
    return result
