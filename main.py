"""Liar-V1 command line - PAPER TRADING ONLY.

    python main.py engine                      # run the trading engine (scheduler + control loop)
    python main.py engine --autonomy FULL_AUTO # ... fully automatic (or HUMAN_IN_THE_LOOP / HYBRID / ...)
    python main.py init-db                     # create tables, seed settings + prompts, first batch
    python main.py analyze RELIANCE --mode INTRADAY [--route]
    python main.py replay --start 2026-10-01 --end 2026-10-31 --modes INTRADAY,SWING [--symbols A,B] [--llm rules]
    python main.py walk-forward --start 2026-06-01 --end 2026-10-31 --modes SWING [--train-days 20 --test-days 5]

The dashboard is a separate process:  streamlit run dashboard/app.py --server.address 127.0.0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from core.enums import Mode


def _modes(text: str) -> list[Mode]:
    return [Mode(m.strip().upper()) for m in text.split(",") if m.strip()]


def _symbols(text: str | None) -> list[str] | None:
    return [s.strip().upper() for s in text.split(",") if s.strip()] if text else None


def cmd_engine(args: argparse.Namespace) -> int:
    from control.bootstrap import bootstrap
    from control.engine import TradingEngine

    services = bootstrap(process="engine", llm_backend=args.llm)
    if args.autonomy:
        set_autonomy(services, args.autonomy)
    TradingEngine(services).run()
    return 0


def set_autonomy(services: object, autonomy: str) -> None:
    """Persist the autonomy switch as a new (audited) settings version and apply it."""
    from config.store import apply_changes

    svc = services  # control.bootstrap.Services
    with svc.session() as s:  # type: ignore[attr-defined]
        apply_changes(s, {"modes.autonomy": autonomy.upper()}, svc.caps, author="cli",  # type: ignore[attr-defined]
                      note=f"autonomy set to {autonomy.upper()} from the command line", clock=svc.clock)  # type: ignore[attr-defined]
        s.commit()
    svc.reload_settings()  # type: ignore[attr-defined]
    print(f"Autonomy: {autonomy.upper()} (PAPER trades only)")


def cmd_init_db(args: argparse.Namespace) -> int:
    from control.bootstrap import bootstrap

    services = bootstrap(process="init", with_notifier=False, offline=True)
    print(f"Database ready. Active batch id: {services.ctx.batch_id}; settings version "
          f"{services.ctx.settings_version_id}. PAPER TRADING ONLY.")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from control.bootstrap import bootstrap

    services = bootstrap(process="analyze", llm_backend=args.llm, with_notifier=args.route)
    res = services.pipeline.analyze(args.symbol, Mode(args.mode.upper()), batch_id=services.ctx.batch_id,
                                    force=True, route=args.route)
    out = {
        "symbol": res.symbol, "mode": res.mode.value, "final_action": res.final_action,
        "decision_id": res.decision_id, "filtered_reason": res.filtered_reason, "error": res.error,
        "latency_ms": res.latency_ms,
        "meta": res.meta.model_dump(mode="json") if res.meta else None,
        "risk": res.verdict.model_dump(mode="json", exclude={"checks"}) if res.verdict else None,
        "confluence": res.confluence.model_dump(mode="json") if res.confluence else None,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from scheduler.replay import ReplayConfig, ReplayRunner

    cfg = ReplayConfig(start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
                       modes=_modes(args.modes), symbols=_symbols(args.symbols), name=args.name,
                       llm_backend=args.llm, max_symbols=args.max_symbols)
    report = ReplayRunner(cfg).run()
    print(json.dumps(report.as_dict(), indent=2, default=str))
    return 0 if report.ledger_ok else 1


def cmd_walk_forward(args: argparse.Namespace) -> int:
    from scheduler.walk_forward import walk_forward

    grid = json.loads(open(args.grid, encoding="utf-8").read()) if args.grid else None
    rep = walk_forward(date.fromisoformat(args.start), date.fromisoformat(args.end), _modes(args.modes),
                       symbols=_symbols(args.symbols), train_days=args.train_days, test_days=args.test_days,
                       grid=grid, llm_backend=args.llm, name=args.name)
    print(json.dumps(rep.as_dict(), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="liar-v1", description="Multi-agent PAPER trading system for NSE (never real orders)")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("engine", help="run the trading engine")
    e.add_argument("--llm", choices=["ollama", "rules"], default=None, help="override the LLM backend")
    e.add_argument("--autonomy", default=None,
                   choices=["PER_MODE", "FULL_AUTO", "HYBRID", "HUMAN_IN_THE_LOOP", "SIGNALS_ONLY"],
                   help="who decides: fully automatic or human in the loop (saved as a settings version)")
    e.set_defaults(func=cmd_engine)

    i = sub.add_parser("init-db", help="create tables, seed settings/prompts, create the first batch")
    i.set_defaults(func=cmd_init_db)

    a = sub.add_parser("analyze", help="run the full agent pipeline for one symbol now")
    a.add_argument("symbol")
    a.add_argument("--mode", default="INTRADAY")
    a.add_argument("--route", action="store_true", help="apply the mode's execution policy (default: analysis only)")
    a.add_argument("--llm", choices=["ollama", "rules"], default=None)
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("replay", help="historical replay (minute-by-minute, separate DB)")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--modes", default="INTRADAY,SWING")
    r.add_argument("--symbols", default=None)
    r.add_argument("--name", default="replay")
    r.add_argument("--llm", choices=["ollama", "rules"], default="rules")
    r.add_argument("--max-symbols", type=int, default=15)
    r.set_defaults(func=cmd_replay)

    w = sub.add_parser("walk-forward", help="walk-forward tuning over replays")
    w.add_argument("--start", required=True)
    w.add_argument("--end", required=True)
    w.add_argument("--modes", default="SWING")
    w.add_argument("--symbols", default=None)
    w.add_argument("--train-days", type=int, default=20)
    w.add_argument("--test-days", type=int, default=5)
    w.add_argument("--grid", default=None, help='JSON file: {"modes.modes.SWING.min_confluence": [55, 60, 70]}')
    w.add_argument("--llm", choices=["ollama", "rules"], default="rules")
    w.add_argument("--name", default="wf")
    w.set_defaults(func=cmd_walk_forward)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
