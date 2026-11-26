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
from core.activity import activity
from core.logging import log_event
from paper_engine.views import open_position_views
from storage.audit import record_error
from storage.batches import closing_batches, get_current_batch
from storage.models import DailySummary, DecisionRow, Position, Run, StockAnalyzed, Trade

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
    # the pool / cap change when follow-up stocks take the first places, so they are part of the key
    sig = (now.date(), bucket, cap, len(symbols), hash(tuple(symbols)))
    cached = svc.extras.get(key)
    if cached and cached[0] == sig:
        return cached[1]
    ranked = rank_in_play(symbols, svc.data, now, top_n=cap)
    chunk = [r.symbol for r in ranked]
    svc.extras[key] = (sig, chunk)
    log_event("scans", {"event": "in_play_ranking", "mode": mode.value,
                        "top": [{"symbol": r.symbol, "score": r.score, "rvol": r.rvol, "move_pct": r.move_pct}
                                for r in ranked]}, clock=svc.clock)
    return chunk


def _chunk_for(svc: Any, mode: Mode, symbols: list[str], *, rotate: bool, advance: bool = True,
               priority: list[str] | None = None) -> list[str]:
    """The symbols to analyse now. ``advance=False`` (Scan now) reads the rotation without moving it,
    so an analysis-only scan never changes which stocks the next live cycle trades.

    ``priority`` (follow-up stocks: today's plan + "use for trading" lists) takes the first places;
    the remaining ``max_symbols_per_cycle`` places are filled by the normal ranking / rotation."""
    cap = svc.settings.mode(mode).max_symbols_per_cycle
    first = list(dict.fromkeys(priority or []))[:cap]
    taken = set(first)
    pool = [s for s in symbols if s not in taken] if taken else symbols
    room = cap - len(first)
    if room <= 0:
        return first
    if rotate and len(pool) > room and svc.settings.engine.watchlist_ranking:
        return first + [s for s in _ranked(svc, mode, pool, room) if s not in taken]
    if rotate and len(pool) > room:
        key = f"rotation:{mode.value}"
        start = svc.extras.get(key, 0) % len(pool)
        if advance:
            svc.extras[key] = start + room
        return first + (pool[start:] + pool[:start])[:room]
    return first + pool[:room]


def _followup_priority(svc: Any, mode: Mode, symbols: list[str], batch_id: int | None) -> list[str]:
    """INTRADAY + ``engine.followup_priority``: the stocks the cycle analyses FIRST.

    ``data.watchlists.trading_symbols`` (today's next-day plan, best score first, then the owner's
    "use for trading" lists), restricted to the mode's tradeable watchlist ``symbols``. A plan stock the
    universe filter left out (other index / liquidity) is still added when it is affordable now (price
    <= the risk manager's max affordable price) and not blacklisted. Never raises (returns [])."""
    if mode != Mode.INTRADAY or not svc.settings.engine.followup_priority:
        return []
    try:
        from data.watchlists import LIVE_STATUSES, plan_items, trading_symbols

        today = _now(svc).date()
        with svc.session() as s:
            wanted = trading_symbols(s, today=today)
            plan = {i.symbol: dict(i.plan_json or {}) for i in plan_items(s, for_date=today, statuses=LIVE_STATUSES)}
        if not wanted:
            return []
        universe = set(symbols)
        black = {b.strip().upper() for b in svc.settings.universe.blacklist}
        out: list[str] = []
        dropped: list[str] = []
        max_price: float | None = None
        for sym in wanted:
            if sym in universe:
                out.append(sym)
                continue
            p = plan.get(sym)
            if p is None or sym in black or not p.get("affordable", False):
                dropped.append(sym)
                continue
            if max_price is None and batch_id is not None:
                max_price = float(svc.risk.max_affordable_price(_risk_state(svc, batch_id), mode))
            price = None
            try:
                q = svc.data.quote(sym)
                price = float(q.ltp) if q is not None and q.ltp else None
            except Exception:  # noqa: BLE001
                price = None
            price = price if price is not None else p.get("close")
            if max_price is not None and price is not None and float(price) > max_price:
                dropped.append(sym)
                continue
            out.append(sym)
        if out or dropped:
            activity("cycle", f"{mode.value}: follow-up stocks first: {', '.join(out) or 'none'}"
                     + (f" (not tradeable now, skipped: {', '.join(dropped)})" if dropped else ""),
                     clock=svc.clock, mode=mode.value, symbols=len(out))
        return out
    except Exception as exc:  # noqa: BLE001 - priority is a nicety; never break a cycle
        logger.warning(f"[followup] priority symbols unavailable: {exc}")
        return []


def _record_followups(svc: Any, results: list[Any], *, routed: bool) -> int:
    """Store the latest analysis (one-line summary) on every follow-up list item of the analysed
    symbols. Returns rows updated. Never raises - a cycle must never fail because of this."""
    if not results:
        return 0
    try:
        from control.processor import describe_result
        from data.watchlists import followed_symbols, record_analysis

        now = _now(svc)
        n = 0
        with svc.session() as s:
            followed = followed_symbols(s)
            if not followed:
                return 0
            for r in results:
                sym = str(getattr(r, "symbol", "") or "").upper()
                if sym not in followed:
                    continue
                try:
                    summary = describe_result(r, routed=routed)
                except Exception:  # noqa: BLE001 - test doubles / partial results
                    summary = f"{sym}: {getattr(r, 'final_action', 'analysed')}"
                n += record_analysis(s, sym, decision_id=getattr(r, "decision_id", None), summary=summary, now=now)
            s.commit()
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[followup] could not record analyses: {exc}")
        return 0


def _scan(svc: Any, mode: Mode, run_type: RunType, *, rotate: bool, check_timing: bool) -> dict[str, Any]:
    batch_id = _active_batch_id(svc)
    if batch_id is None or not _engine_running(svc) or not _mode_tradeable(svc, mode, batch_id):
        why = ("no active batch" if batch_id is None else "engine paused" if not _engine_running(svc)
               else f"{mode.value} disabled or paused")
        activity("cycle", f"{run_type.value} skipped: {why}", clock=svc.clock, level="warning", job=run_type.value,
                 mode=mode.value)
        return {"skipped": "engine not running / batch not active / mode disabled or paused"}
    symbols = _watchlist(svc, mode, batch_id)
    chunk = _chunk_for(svc, mode, symbols, rotate=rotate, priority=_followup_priority(svc, mode, symbols, batch_id))
    with svc.session() as s:
        run = open_run(s, run_type, batch_id=batch_id, mode=mode, clock=svc.clock,
                       settings_version_id=svc.ctx.settings_version_id, is_replay=svc.ctx.is_replay)
        s.commit()
        run_id = run.id
    actions = _analyse_chunk(svc, mode, chunk, batch_id=batch_id, run_id=run_id, check_timing=check_timing,
                             **_llm_limits(svc, mode, check_timing))
    with svc.session() as s:
        run = s.get(type(run), run_id)
        close_run(s, run, clock=svc.clock, n_symbols=len(chunk), summary={"actions": actions})
        s.commit()
    return {"run_id": run_id, "symbols": len(chunk), "actions": actions}


def _llm_limits(svc: Any, mode: Mode, check_timing: bool) -> dict[str, Any]:
    """Live 5-minute cycles: top-K + time budget. Evening SWING/POSITIONAL scan: more, no budget."""
    eng = svc.settings.engine
    if check_timing:
        return {"top_k": eng.max_llm_candidates_per_cycle, "budget_s": eng.cycle_llm_budget_seconds}
    return {"top_k": eng.max_llm_candidates_evening, "budget_s": None}


_WEAK_WORDS = {"trend": "weak trend", "volume": "low volume", "regime": "regime does not fit",
               "market": "market not aligned", "candles": "no confirming candle", "room": "resistance close",
               "sentiment": "news against", "fundamentals": "weak fundamentals"}


def _near_miss(low: list[Any]) -> str:
    """'ITC ema_9_21_pullback 57.6 (weak: low volume, resistance close)' for the best sub-threshold setup."""
    from strategies.confluence import COMPONENT_MAX

    best = max(low, key=lambda r: r.confluence.score if r.confluence is not None else 0.0)
    conf = best.confluence
    if conf is None:
        return best.symbol
    strat = next((n.split(" ", 2)[1] for n in conf.notes if n.startswith("strategy ")), "")
    fracs = sorted(((v / COMPONENT_MAX[k], k) for k, v in conf.components.items()
                    if k in COMPONENT_MAX and COMPONENT_MAX[k] > 0), key=lambda x: x[0])
    weak = [_WEAK_WORDS.get(k, k) for f, k in fracs if f < 0.35][:3]
    return f"{best.symbol} {strat} {conf.score:.1f}" + (f" (weak: {', '.join(weak)})" if weak else "")


def _analyse_chunk(svc: Any, mode: Mode, chunk: list[str], *, batch_id: int, run_id: int,
                   check_timing: bool, top_k: int | None = None, budget_s: float | None = -1.0,
                   route: bool = True, use_llm: bool = True, results: list[Any] | None = None) -> dict[str, int]:
    """Two phases: deterministic screen of every symbol, then the LLM for the best few only.

    Phase 1 (seconds): features + strategies + confluence for the whole chunk. Phase 2: the
    ``top_k`` highest-confluence setups (default ``engine.max_llm_candidates_per_cycle``) go
    through the agents, debate, meta and risk, strongest first, until ``budget_s`` seconds are
    used (default ``engine.cycle_llm_budget_seconds``; None = no budget; LLM backends only).
    Every other setup is recorded as FILTERED with the reason and shadow-tracked, so nothing
    disappears from the audit trail. ``results`` (optional) collects every PipelineResult.
    Symbols that are in a follow-up list get their latest analysis recorded on the list item.
    """
    import time as _time

    eng = svc.settings.engine
    actions: dict[str, int] = {}
    done: list[Any] = []

    def _count(res: Any) -> None:
        actions[res.final_action] = actions.get(res.final_action, 0) + 1
        done.append(res)
        if results is not None:
            results.append(res)

    pipeline = svc.pipeline
    if not hasattr(pipeline, "screen"):   # test doubles with only analyze()
        for sym in chunk:
            _count(pipeline.analyze(sym, mode, batch_id=batch_id, run_id=run_id, check_timing=check_timing,
                                    route=route))
        _record_followups(svc, done, routed=route)
        return actions
    t_screen = _time.monotonic()
    activity("cycle", f"{mode.value}: screening {len(chunk)} stock(s): {', '.join(chunk[:12])}"
             + (" ..." if len(chunk) > 12 else ""), clock=svc.clock, mode=mode.value, symbols=len(chunk))
    pending = []
    for sym in chunk:
        h = pipeline.screen(sym, mode, batch_id=batch_id, run_id=run_id, check_timing=check_timing, route=route)
        if h.pending:
            pending.append(h)
        else:
            _count(h.result)
    pending.sort(key=lambda h: -h.confluence)
    top_k = max(1, int(top_k if top_k is not None else eng.max_llm_candidates_per_cycle))
    if budget_s is not None and budget_s < 0:
        budget_s = eng.cycle_llm_budget_seconds
    budgeted = eng.llm_backend != "rules" and budget_s is not None
    window_ok, window_why = True, ""
    if check_timing and getattr(svc, "risk", None) is not None and hasattr(svc.risk, "in_entry_window"):
        window_ok, window_why = svc.risk.in_entry_window(mode, _now(svc))
    screen_s = _time.monotonic() - t_screen
    screened = [r for r in done if r is not None]
    no_setup = sum(1 for r in screened if r.filtered_reason == "no strategy setup")
    low = [r for r in screened if r.filtered_reason and r.filtered_reason.startswith("confluence")]
    other = len(screened) - no_setup - len(low)
    parts = [f"{no_setup} without a setup"]
    if low:
        parts.append(f"{len(low)} setup(s) below the confluence bar {svc.settings.mode(mode).min_confluence:g} - "
                     f"closest: {_near_miss(low)}")
    if other:
        parts.append(f"{other} skipped (no data / errors)")
    tail = "; ".join(parts) + f" ({len(chunk)} screened in {screen_s:.0f}s)"
    if pending:
        cands = ", ".join(f"{h.symbol} ({h.confluence:.0f})" for h in pending)
        nxt = ("none - outside the entry window" if not window_ok else "none - screening only" if not use_llm
               else ", ".join(h.symbol for h in pending[:top_k]))
        activity("cycle", f"{mode.value}: {len(pending)} setup(s) passed the filter: {cands}; to the AI: {nxt}; "
                          + tail, clock=svc.clock, mode=mode.value)
    else:
        activity("cycle", f"{mode.value}: no setup good enough for the AI this cycle - " + tail, clock=svc.clock,
                 mode=mode.value)
    t0 = _time.monotonic()
    sent = 0
    for i, h in enumerate(pending):
        reason = None
        if not window_ok:
            reason = f"outside the entry window ({window_why}); no LLM time spent"
        elif not use_llm:
            reason = f"screening only (no AI requested); confluence {h.confluence:.1f}"
        elif i >= top_k:
            reason = (f"setup passed the filter (confluence {h.confluence:.1f}) but was not in this cycle's "
                      f"top {top_k} by confluence")
        elif budgeted and i > 0 and _time.monotonic() - t0 > float(budget_s):
            reason = (f"cycle LLM time budget ({budget_s:.0f}s) used up before this setup "
                      f"(confluence {h.confluence:.1f})")
        sent += reason is None
        _count(pipeline.resume(h, skip_reason=reason))
    if pending:
        log_event("scans", {"event": "cycle_llm", "mode": mode.value, "run_id": run_id,
                            "screened": len(chunk), "passed_filter": len(pending), "sent_to_llm": sent,
                            "candidates": [{"symbol": h.symbol, "confluence": round(h.confluence, 2)} for h in pending],
                            "llm_seconds": round(_time.monotonic() - t0, 1)}, clock=svc.clock)
    _record_followups(svc, done, routed=route)
    return actions


#: Most stocks one explicit "Scan these" request analyses.
MAX_SCAN_SYMBOLS = 100


def _explicit_symbols(svc: Any, symbols: list[str] | None, watchlist_id: int | None) -> tuple[list[str], str, list[str]]:
    """``(symbols, label, rejected)`` for a Scan now on chosen stocks and/or one follow-up list.

    ``ValueError`` for an unknown list. Upper-cased, de-duplicated, at most ``MAX_SCAN_SYMBOLS``."""
    from data.watchlists import get_watchlist, normalize_symbols

    wanted: list[str] = []
    label: list[str] = []
    if watchlist_id is not None:
        with svc.session() as s:
            wl = get_watchlist(s, watchlist_id)
        if wl is None:
            raise ValueError(f"watchlist #{watchlist_id} not found")
        wanted += [i["symbol"] for i in sorted(wl["items"], key=lambda i: (i["added_at"] or "", i["id"]))]
        label.append(f"list '{wl['name']}'")
    if symbols:
        wanted += [str(x) for x in symbols]
        label.append("chosen stocks")
    ok, bad = normalize_symbols(wanted, limit=MAX_SCAN_SYMBOLS)
    return ok, " + ".join(label), bad


def scan_now(svc: Any, mode: Mode, *, use_llm: bool = True, symbols: list[str] | None = None,
             watchlist_id: int | None = None) -> dict[str, Any]:
    """Owner "Scan now": screen the mode's watchlist now and run the AI on the best setups.

    ``symbols`` and/or ``watchlist_id`` (a follow-up list): scan exactly those stocks (upper-cased,
    de-duplicated, max 100) instead of the mode's watchlist; the live rotation is never advanced.
    Analysis only - nothing is traded or queued, at any time of day (decisions are stored as
    ANALYZED / FILTERED so they show up in the dashboard). Never raises.
    """
    import time as _time

    from control.processor import describe_result

    t0 = _time.monotonic()
    batch_id = _active_batch_id(svc) or svc.ctx.batch_id
    if batch_id is None:
        return {"error": "no batch", "message": "No active batch - start one first."}
    explicit = bool(symbols) or watchlist_id is not None
    rejected: list[str] = []
    if explicit:
        try:
            chunk, label, rejected = _explicit_symbols(svc, symbols, watchlist_id)
        except Exception as exc:  # noqa: BLE001 - unknown list / DB error
            return {"error": str(exc), "message": f"{mode.value} scan failed: {exc}"}
        if not chunk:
            return {"symbols": 0, "message": f"{mode.value} scan: no valid stock to scan in {label or 'the request'}"
                                             + (f" (invalid: {', '.join(rejected)})" if rejected else "") + "."}
        activity("cycle", f"Scan now {mode.value}: {len(chunk)} stock(s) from {label}: {', '.join(chunk[:12])}"
                 + (" ..." if len(chunk) > 12 else "") + " (analysis only)", clock=svc.clock, job="scan_now",
                 mode=mode.value, symbols=len(chunk))
    else:
        try:
            symbols_wl = _watchlist(svc, mode, batch_id)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "message": f"{mode.value} scan failed: could not build the watchlist ({exc})"}
        if not symbols_wl:
            return {"symbols": 0, "message": f"{mode.value} scan: the watchlist is empty (the universe filters "
                                             "excluded every stock - see the Live log)."}
        chunk = _chunk_for(svc, mode, symbols_wl, rotate=mode == Mode.INTRADAY, advance=False,
                           priority=_followup_priority(svc, mode, symbols_wl, batch_id))
        activity("cycle", f"Scan now {mode.value}: {len(chunk)} of {len(symbols_wl)} watchlist stocks (analysis only)",
                 clock=svc.clock, job="scan_now", mode=mode.value, symbols=len(chunk))
    with svc.session() as s:
        run = open_run(s, RunType.ANALYZE_NOW, batch_id=batch_id, mode=mode, clock=svc.clock,
                       settings_version_id=svc.ctx.settings_version_id, is_replay=svc.ctx.is_replay)
        s.commit()
        run_id = run.id
    results: list[Any] = []
    limits = _llm_limits(svc, mode, mode in (Mode.INTRADAY, Mode.BTST))
    if explicit:   # the owner chose these stocks: let more of them reach the AI (still capped)
        limits["top_k"] = max(int(limits["top_k"]),
                              min(len(chunk), int(svc.settings.engine.max_llm_candidates_evening)))
    actions = _analyse_chunk(svc, mode, chunk, batch_id=batch_id, run_id=run_id, check_timing=False,
                             route=False, use_llm=use_llm, results=results, **limits)
    with svc.session() as s:
        run = s.get(Run, run_id)
        close_run(s, run, clock=svc.clock, n_symbols=len(chunk), summary={"actions": actions, "scan_now": True})
        s.commit()
    analysed = [r for r in results if r.meta is not None]
    failed = [r for r in results if r.error and r.meta is None]
    no_setup = [r for r in results if r.filtered_reason == "no strategy setup"]
    low = [r for r in results if r.filtered_reason and r.filtered_reason.startswith("confluence")]
    other = [r for r in results if r.meta is None and not r.error and r.filtered_reason
             and r not in no_setup and r not in low]   # had a setup but was not sent to the AI (any reason)
    secs = _time.monotonic() - t0
    parts = [f"{mode.value} scan of {len(chunk)} stock(s) in {secs:.0f}s: {len(no_setup)} without a setup, "
             f"{len(low)} with low confluence, {len(other)} setup(s) not sent to the AI, "
             f"{len(analysed)} analysed by the AI" + (f", {len(failed)} failed" if failed else "")]
    parts += [describe_result(r, routed=False) for r in analysed]
    if other:
        parts.append("setups not sent to the AI: " + "; ".join(f"{r.symbol} ({r.filtered_reason})" for r in other))
    if failed:
        parts.append("failed: " + "; ".join(f"{r.symbol} ({r.error})" for r in failed))
    if rejected:
        parts.append("ignored invalid symbol(s): " + ", ".join(rejected[:10]))
    parts.append("Analysis only - nothing was traded.")
    message = " | ".join(parts)
    activity("cycle", message, clock=svc.clock, level="success", job="scan_now", mode=mode.value,
             seconds=round(secs, 1), actions=actions)
    return {"run_id": run_id, "symbols": len(chunk), "actions": actions, "message": message,
            "scanned": list(chunk), "rejected": rejected,
            "analysed": [{"symbol": r.symbol, "decision_id": r.decision_id,
                          "summary": describe_result(r, routed=False)} for r in analysed]}


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
        try:
            from notifications.reports import next_day_plan_section

            report = dict(report)
            report.update(next_day_plan_section(s, _next_session(svc, day),
                                                opening_check=svc.settings.schedule.opening_check))
        except Exception as exc:  # noqa: BLE001 - the plan section is informational
            logger.warning(f"[eod_report] next-day plan section failed: {exc}")
    svc.notifier.eod(report)
    return {"sent": True}


# =========================================================================== next-day plan


def _next_session(svc: Any, d: date) -> date:
    if svc.calendar is not None:
        return svc.calendar.next_trading_day(d)
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def _next_day_universe(svc: Any) -> list[str]:
    """``engine.next_day_universe`` constituents (+ custom / whitelist) - blacklist - excluded sectors."""
    st = svc.settings
    u = st.universe
    name = st.engine.next_day_universe.strip().upper()
    norm = [s.strip().upper() for s in u.whitelist]
    if u.whitelist_only:
        base = norm
    else:
        base = (svc.universe.symbols_for(name) if svc.universe is not None and name != "CUSTOM" else [])
        base = base + [s.strip().upper() for s in u.custom_symbols] + norm
    black = {s.strip().upper() for s in u.blacklist}
    excluded = {s.strip().lower() for s in u.excluded_sectors if s.strip()}
    out: list[str] = []
    for sym in dict.fromkeys(base):
        if sym in black:
            continue
        sector = svc.universe.sector_of(sym) if svc.universe is not None else None
        if sector and sector.lower() in excluded:
            continue
        out.append(sym)
    if svc.ctx.is_replay and svc.universe is not None:
        # REPLAY only has bars for the replayed symbols: rank those (the universe is restricted there)
        have: set[str] = set()
        for m in Mode:
            try:
                have |= set(svc.universe.build_universe(m, st))
            except Exception:  # noqa: BLE001
                continue
        out = [s for s in out if s in have] or sorted(have - black)
    return out


def job_next_day_plan(svc: Any) -> dict[str, Any]:
    """15:40: rank tomorrow's INTRADAY candidates from today's CLOSED daily bars (deterministic,
    ``strategies.next_day``) and save them as the NEXT_DAY follow-up list for the next session.

    Paper plan only: it decides which stocks the INTRADAY cycles analyse first tomorrow; every
    trade still needs the full pipeline and the risk manager. Nothing is ordered here.
    """
    eng = svc.settings.engine
    if not eng.next_day_plan_enabled:
        return {"skipped": "next-day plan disabled (engine.next_day_plan_enabled)"}
    from data.watchlists import save_plan
    from strategies.next_day import build_next_day_plan

    now = _now(svc)
    nxt = _next_session(svc, now.date())
    symbols = _next_day_universe(svc)
    if not symbols:
        activity("plan", f"next-day plan: the {eng.next_day_universe} universe is empty - nothing to rank",
                 clock=svc.clock, level="warning", job="next_day_plan")
        return {"skipped": f"universe {eng.next_day_universe} is empty"}
    batch_id = _active_batch_id(svc) or svc.ctx.batch_id
    max_price: float | None = None
    if batch_id is not None and getattr(svc, "risk", None) is not None:
        try:
            max_price = float(svc.risk.max_affordable_price(_risk_state(svc, batch_id), Mode.INTRADAY))
        except Exception as exc:  # noqa: BLE001 - affordability is a flag, not a blocker
            logger.warning(f"[next_day_plan] max affordable price unavailable: {exc}")
    activity("plan", f"next-day plan: ranking {len(symbols)} {eng.next_day_universe} stock(s) for {nxt:%a %d %b} "
                      f"on today's daily bars" + (f" (INTRADAY max affordable price Rs {max_price:,.2f})"
                                                   if max_price is not None else ""),
             clock=svc.clock, job="next_day_plan", symbols=len(symbols))
    ranked = build_next_day_plan(svc.data, symbols, now, svc.settings, max_price=max_price, for_date=nxt)
    qualified = [c for c in ranked if (c.get("score") or 0.0) >= eng.next_day_min_score]
    # affordable candidates take the places first; unaffordable ones only fill what is left
    chosen = sorted(qualified, key=lambda c: (not c["affordable"], -c["score"], c["symbol"]))
    picked = sorted(chosen[:eng.next_day_max_candidates], key=lambda c: (-c["score"], not c["affordable"], c["symbol"]))
    with svc.session() as s:
        saved = save_plan(s, nxt, picked, now=now, actor="system")
        s.commit()
    if picked:
        head = "; ".join(f"{c['symbol']} {c['setup'].lower()} trigger {c['trigger']:.2f}" for c in picked[:5])
        activity("plan", f"next-day plan: {len(picked)} candidate(s) for {nxt:%a %d %b}: {head}"
                          + (" ..." if len(picked) > 5 else "") + f" (from {len(ranked)} setup(s), "
                          f"{len(qualified)} with score >= {eng.next_day_min_score:.0f})",
                 clock=svc.clock, level="success", job="next_day_plan", symbols=len(picked))
        for c in picked:
            activity("plan", f"plan {c['symbol']}: {c['setup']} {c['direction']} trigger {c['trigger']:.2f}, "
                              f"stop {c['stop']:.2f}, target {c['target']:.2f} (R:R {c['rr']:.1f}, score "
                              f"{c['score']:.0f}){'' if c['affordable'] else ' - NOT affordable at current capital'}"
                              f" - {'; '.join(c['reasons'][:2])}",
                     clock=svc.clock, job="next_day_plan", symbol=c["symbol"])
    else:
        activity("plan", f"next-day plan: no stock qualified for {nxt:%a %d %b} ({len(ranked)} setup(s), none with "
                          f"score >= {eng.next_day_min_score:.0f})", clock=svc.clock, job="next_day_plan")
    return {"for_date": nxt.isoformat(), "watchlist_id": saved["id"], "name": saved["name"],
            "universe": len(symbols), "setups": len(ranked), "qualified": len(qualified), "selected": len(picked),
            "max_affordable_price": max_price,
            "candidates": [{"symbol": c["symbol"], "setup": c["setup"], "direction": c["direction"],
                            "trigger": c["trigger"], "score": c["score"], "affordable": c["affordable"]}
                           for c in picked]}


def _opening_prices(svc: Any, symbol: str, today: date) -> tuple[float | None, float | None]:
    """``(last price, today's open)`` from the live quote, else today's closed 5m / 1m bars."""
    price: float | None = None
    day_open: float | None = None
    try:
        q = svc.data.quote(symbol)
        if q is not None and q.ltp and to_ist(q.timestamp).date() == today:
            price = float(q.ltp)
            day_open = float(q.open) if q.open else None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"[opening_check] quote {symbol}: {exc}")
    if price is None or day_open is None:
        from core.enums import Timeframe

        for tf in (Timeframe.M5, Timeframe.M1):
            try:
                bars = svc.data.candles(symbol, tf, lookback_bars=80)
            except Exception:  # noqa: BLE001
                continue
            if bars is None or bars.empty:
                continue
            tod = bars[bars.index.date == today]
            if tod.empty:
                continue
            price = price if price is not None else float(tod["close"].iloc[-1])
            day_open = day_open if day_open is not None else float(tod["open"].iloc[0])
            break
    return price, day_open


def opening_verdict(plan: dict[str, Any], price: float, day_open: float | None = None) -> tuple[str, str]:
    """``(status, note)`` of one plan item at the opening (pure, deterministic).

    LONG (SHORT mirrored): price below the stop -> INVALIDATED; more than 1 ATR past the trigger
    (gapped past - don't chase) -> INVALIDATED; at/above the trigger -> TRIGGERED; else ACTIVE.
    """
    trig, stop = float(plan["trigger"]), float(plan["stop"])
    a = float(plan.get("atr") or 0.0) or abs(trig - stop) / 1.2 or max(abs(trig) * 0.01, 0.05)
    short = str(plan.get("direction", "LONG")).upper() == "SHORT"
    d = -1.0 if short else 1.0
    opened = f" (open {day_open:.2f})" if day_open is not None else ""
    past = (price - trig) * d
    if (price - stop) * d < 0:
        return "INVALIDATED", (f"price {price:.2f}{opened} is {'above' if short else 'below'} the stop {stop:.2f} "
                               "- setup broken, not traded")
    if past > a:
        return "INVALIDATED", (f"price {price:.2f}{opened} is already {past / a:.1f} ATR past the trigger "
                               f"{trig:.2f} (gapped past) - don't chase")
    if past >= 0:
        return "TRIGGERED", (f"price {price:.2f}{opened} is past the trigger {trig:.2f} - analysed first by the "
                             "INTRADAY cycles (risk manager decides)")
    return "ACTIVE", (f"waiting for trigger {trig:.2f}: price {price:.2f}{opened}, {abs(past) / a:.2f} ATR away "
                      f"(stop {stop:.2f})")


def job_opening_check(svc: Any) -> dict[str, Any]:
    """09:20: re-check today's NEXT_DAY plan against the opening prices.

    Each live item becomes TRIGGERED / INVALIDATED / ACTIVE ("waiting for trigger") with a note.
    INVALIDATED items are dropped from the INTRADAY priority list; nothing is ordered here.
    """
    from data.watchlists import plan_items, set_item_status

    now = _now(svc)
    today = now.date()
    with svc.session() as s:
        items = [(i.id, i.symbol, dict(i.plan_json or {}), i.status) for i in plan_items(s, for_date=today)]
    if not items:
        return {"skipped": f"no next-day plan for {today:%a %d %b}"}
    counts = {"TRIGGERED": 0, "INVALIDATED": 0, "ACTIVE": 0, "no_price": 0}
    updates: list[tuple[int, str, str]] = []
    for item_id, sym, plan, status in items:
        if status == "INVALIDATED":
            counts["INVALIDATED"] += 1
            continue
        if plan.get("trigger") is None or plan.get("stop") is None:
            continue
        price, day_open = _opening_prices(svc, sym, today)
        if price is None:
            counts["no_price"] += 1
            activity("plan", f"opening check {sym}: no price yet - kept {status}", clock=svc.clock,
                     level="warning", job="opening_check", symbol=sym)
            continue
        new_status, note = opening_verdict(plan, price, day_open)
        counts[new_status] += 1
        updates.append((item_id, new_status, note))
        activity("plan", f"opening check {sym} {plan.get('setup', '')}: {new_status} - {note}", clock=svc.clock,
                 level={"TRIGGERED": "success", "INVALIDATED": "warning"}.get(new_status, "info"),
                 job="opening_check", symbol=sym)
    with svc.session() as s:
        for item_id, new_status, note in updates:
            set_item_status(s, item_id, new_status, note, now=now, actor="system")
        s.commit()
    activity("plan", f"opening check: {len(items)} plan stock(s) - {counts['TRIGGERED']} triggered, "
                      f"{counts['ACTIVE']} waiting, {counts['INVALIDATED']} invalidated"
                      + (f", {counts['no_price']} without a price" if counts["no_price"] else ""),
             clock=svc.clock, job="opening_check")
    return {"checked": len(items), "triggered": counts["TRIGGERED"], "waiting": counts["ACTIVE"],
            "invalidated": counts["INVALIDATED"], "no_price": counts["no_price"]}


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
    "next_day_plan": job_next_day_plan,
    "opening_check": job_opening_check,
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
