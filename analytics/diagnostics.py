"""Daily health diagnosis from the logs: what failed, how often, and what to do about it.

Reads ``logs/<day>/`` (JSONL categories + ``engine.log``) and the ``errors`` / ``decisions``
tables, and turns the counts into prioritized findings with a concrete action. Pure
read-only; safe to call from the dashboard at any time.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.clock import IST
from storage.models import DecisionRow, ErrorLog

_NUM = re.compile(r"\d+(\.\d+)?")
_TIMEOUT = re.compile(r"timeout|timed out", re.I)


def _jsonl(folder: Path, category: str) -> list[dict[str, Any]]:
    path = folder / f"{category}.jsonl"
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _norm(msg: str) -> str:
    """Group similar messages: digits -> #, first line, max 140 chars."""
    first = (msg or "").strip().splitlines()[0] if msg else ""
    return _NUM.sub("#", first)[:140]


def _pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


def _p90(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]


def diagnose(session: Session | None, day: date, log_dir: Path) -> dict[str, Any]:
    """Health report for ``day``. ``session`` may be None (log files only)."""
    folder = Path(log_dir) / day.strftime("%Y-%m-%d")
    agents = [r for r in _jsonl(folder, "agents") if r.get("agent") and r.get("agent") != "_summary"]
    llm_rows = [r for r in agents if r.get("status") != "SKIPPED" and r.get("model") not in (None, "rules")]
    failed = [r for r in llm_rows if r.get("status") == "FAILED"]
    timeouts = [r for r in failed if _TIMEOUT.search(str(r.get("error") or ""))]
    invalid = [r for r in failed if "invalid output" in str(r.get("error") or "")]
    lat = [float(r["latency_ms"]) / 1000 for r in llm_rows if r.get("status") == "OK" and r.get("latency_ms")]
    llm = {"calls": len(llm_rows), "failed": len(failed), "timeouts": len(timeouts), "invalid": len(invalid),
           "fail_pct": _pct(len(failed), len(llm_rows)),
           "avg_latency_s": round(mean(lat), 1) if lat else None,
           "p90_latency_s": round(_p90(lat), 1) if lat else None,
           "by_agent": dict(Counter(r["agent"] for r in failed).most_common())}

    decisions = _jsonl(folder, "decisions")
    analysed = [d for d in decisions if not d.get("filtered")]
    dl = [float(d["latency_ms"]) / 1000 for d in analysed if d.get("latency_ms")]
    pipeline = {"analyses": len(analysed), "filtered": len(decisions) - len(analysed),
                "avg_s": round(mean(dl), 1) if dl else None, "max_s": round(max(dl), 1) if dl else None,
                "slow": sum(1 for x in dl if x > 120)}

    errors = _jsonl(folder, "errors")
    # "replay: ..." errors are test / replay runs that used to leak into the live folder - not real gaps
    data_err = [e for e in errors if e.get("component") == "data" and "replay:" not in str(e.get("message") or "")]
    by_symbol = Counter((e.get("context") or {}).get("symbol") or "?" for e in data_err)
    by_tf = Counter((e.get("context") or {}).get("timeframe") or "?" for e in data_err)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for e in errors:
        key = (str(e.get("component") or "?"), _norm(str(e.get("message") or "")))
        g = grouped.setdefault(key, {"component": key[0], "message": key[1], "count": 0, "last_ts": None,
                                     "severity": e.get("severity")})
        g["count"] += 1
        g["last_ts"] = e.get("ts") or g["last_ts"]

    engine_log = folder / "engine.log"
    log_text = engine_log.read_text(encoding="utf-8", errors="replace") if engine_log.exists() else ""
    db_locked = log_text.count("database is locked")
    cycles_skipped = sum(1 for s in _jsonl(folder, "scans") if s.get("event") == "cycle_skipped")
    overlap_hint = len(re.findall(r"job intraday_cycle start", log_text))
    finbert_offline = "couldn't connect to 'https://huggingface.co'" in log_text or any(
        "huggingface" in str(e.get("message") or "") for e in errors)
    replay_mixed = (folder / "replay.jsonl").exists() or (folder / "replay.log").exists()

    actions: dict[str, int] = {}
    risk_rules: dict[str, int] = {}
    if session is not None:
        start = datetime.combine(day, time(0, 0), tzinfo=IST)
        for act, n in session.execute(select(DecisionRow.final_action, func.count(DecisionRow.id))
                                      .where(DecisionRow.created_at >= start,
                                             DecisionRow.created_at < start + timedelta(days=1))
                                      .group_by(DecisionRow.final_action)).all():
            actions[str(act)] = int(n)
        for rule, n in session.execute(select(DecisionRow.risk_rule, func.count(DecisionRow.id))
                                       .where(DecisionRow.created_at >= start,
                                              DecisionRow.created_at < start + timedelta(days=1),
                                              DecisionRow.final_action == "RISK_REJECTED")
                                       .group_by(DecisionRow.risk_rule)).all():
            risk_rules[str(rule)] = int(n)
        for row in session.execute(select(ErrorLog).where(ErrorLog.ts >= start,
                                                          ErrorLog.ts < start + timedelta(days=1))).scalars():
            key = (row.component, _norm(row.message))
            if key not in grouped:
                grouped[key] = {"component": row.component, "message": key[1], "count": 1,
                                "last_ts": row.ts.isoformat() if row.ts else None, "severity": row.severity}

    # ---- findings ---------------------------------------------------------------------------
    findings: list[dict[str, Any]] = []

    def add(sev: str, title: str, detail: str, action: str) -> None:
        findings.append({"severity": sev, "title": title, "detail": detail, "action": action})

    if llm["calls"] and llm["fail_pct"] >= 20:
        add("high", f"LLM overloaded: {llm['fail_pct']:.0f}% of agent calls failed",
            f"{llm['failed']}/{llm['calls']} failed ({llm['timeouts']} timeouts, {llm['invalid']} invalid answers). "
            "Failed agents vote NEUTRAL, so decisions were made on almost no information.",
            "Use the 'low' model profile (qwen3:4b for every role - one model fits a 6 GB GPU, no swapping), "
            "keep models.max_concurrent_requests = 2, FinBERT on CPU, and check `ollama ps` shows '100% GPU'. "
            "Trades are now refused when > 34% of agents fail (AGENT_QUORUM).")
    if pipeline["avg_s"] and pipeline["avg_s"] > 120:
        add("high", f"Analysis too slow: {pipeline['avg_s']:.0f} s per stock (max {pipeline['max_s']:.0f} s)",
            "A 5-minute cycle cannot finish; decisions were executed on prices that were 5-25 minutes old.",
            "Only the best 2 setups per cycle now reach the LLM (engine.max_llm_candidates_per_cycle) with a "
            "240 s budget; decisions older than 6 min are refused (SIGNAL_EXPIRED).")
    if overlap_hint and cycles_skipped:
        add("info", f"{cycles_skipped} cycle(s) skipped because the previous one was still running",
            "Overlapping cycles are no longer allowed (they multiplied the LLM queue and DB locks).",
            "If this happens often, lower engine.max_llm_candidates_per_cycle to 1.")
    if data_err:
        worst = ", ".join(f"{s} ({n})" for s, n in by_symbol.most_common(5))
        tfs = ", ".join(f"{t}: {n}" for t, n in by_tf.most_common())
        add("medium" if len(data_err) >= 10 else "low", f"Market data gaps: {len(data_err)} failed candle requests",
            f"Symbols: {worst}. Timeframes: {tfs}. yfinance's 1m/5m feed for NSE is delayed and sometimes "
            "returns nothing for the latest minutes; fills then fall back to 5m/daily bars.",
            "Add Angel One SmartAPI keys to .env (free, market data only) for reliable intraday candles; "
            "yfinance stays as the fallback.")
    if db_locked:
        add("medium", f"Database was locked {db_locked} time(s)",
            "Two writers waited on SQLite longer than the busy timeout (overlapping cycles + dashboard).",
            "busy_timeout is now 30 s and cycles no longer overlap. Never run replays against runtime/trading.db.")
    if finbert_offline:
        add("low", "FinBERT could not download its model",
            "News sentiment fell back to neutral until the model was cached.",
            "Run the engine once with internet access so ProsusAI/finbert is cached in ~/.cache/huggingface.")
    if replay_mixed:
        add("low", "Replay / test output found in the live log folder",
            "Replays used to write into logs/<date>/ (simulated dates even created future folders).",
            "Fixed: replays now log to logs/replay/ and tests to a temp folder. Old mixed files can be ignored.")
    if llm["invalid"]:
        add("low", f"{llm['invalid']} invalid LLM answers", "Answers that failed schema validation after retries.",
            "Usually fixed by the smaller, consistent model; check the prompt of the agents listed in by_agent.")
    if actions and not actions.get("EXECUTED") and not actions.get("QUEUED"):
        add("info", "No trade was taken", f"Decisions by outcome: {actions}.",
            "Normal on quiet days. If 'no strategy setup' dominates for weeks, review strategy parameters.")
    if risk_rules:
        top = ", ".join(f"{k} ({v})" for k, v in sorted(risk_rules.items(), key=lambda kv: -kv[1])[:5])
        add("info", "Risk manager rejections", f"Top rules: {top}.",
            "Each rule is shown on the decision; rejected setups are shadow-tracked so you can see what they would have done.")
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    findings.sort(key=lambda f: order[f["severity"]])
    return {
        "day": day.isoformat(), "log_folder": str(folder), "llm": llm, "pipeline": pipeline,
        "data": {"errors": len(data_err), "by_symbol": dict(by_symbol.most_common(10)), "by_timeframe": dict(by_tf)},
        "db": {"locked": db_locked}, "cycles_skipped": cycles_skipped,
        "decisions": actions, "risk_rejections": risk_rules,
        "errors_top": sorted(grouped.values(), key=lambda g: -g["count"])[:25],
        "findings": findings,
    }


__all__ = ["diagnose"]
