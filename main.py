"""Liar-V1 command line - PAPER TRADING ONLY.

    python main.py engine                      # run the trading engine (scheduler + control loop)
    python main.py engine --autonomy FULL_AUTO # ... fully automatic (or HUMAN_IN_THE_LOOP / HYBRID / ...)
    python main.py init-db                     # create tables, seed settings + prompts, first batch
    python main.py analyze RELIANCE --mode INTRADAY [--route]
    python main.py replay --start 2026-10-01 --end 2026-10-31 --modes INTRADAY,SWING [--symbols A,B] [--llm rules]
    python main.py walk-forward --start 2026-06-01 --end 2026-10-31 --modes SWING [--train-days 20 --test-days 5]
    python main.py web [--port 8600]           # the live web dashboard (HTML/JS) on http://127.0.0.1:8600
    python main.py new-batch [--capital 1000]  # start a fresh batch; shows the top-up needed to reach the capital
    python main.py upgrade-settings            # apply the recommended settings as an audited version
    python main.py postmortem [--date D]       # why each trade of a day won / lost, and what now prevents it
    python main.py diagnose [--date D]         # what failed in the logs of a day and what to do next

Dashboards are separate processes:  python main.py web   (main)
                                    streamlit run dashboard/app.py --server.address 127.0.0.1   (advanced console)
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


def cmd_web(args: argparse.Namespace) -> int:
    from webui.server import run

    run(host=args.host, port=args.port)
    return 0


def _light_context() -> tuple[object, object, object]:
    """(session_factory, caps, secrets) without building the full service graph."""
    from config.loader import load_hard_caps
    from config.secrets import load_secrets
    from storage.db import create_all_for_url

    secrets = load_secrets()
    return create_all_for_url(secrets.database_url), load_hard_caps(), secrets


def _engine_alive(session_factory: object) -> bool:
    from datetime import datetime as _dt

    from core.clock import IST, to_ist
    from storage.models import EngineStatus

    with session_factory() as s:  # type: ignore[operator]
        row = s.get(EngineStatus, 1)
        if row is None or row.state not in ("RUNNING", "PAUSED") or row.heartbeat_at is None:
            return False
        return (_dt.now(IST) - to_ist(row.heartbeat_at)).total_seconds() < 60


def cmd_upgrade_settings(args: argparse.Namespace) -> int:
    from control.upgrades import apply_recommended, pending_changes

    sf, caps, _ = _light_context()
    with sf() as s:  # type: ignore[operator]
        todo = pending_changes(s, caps)
        if not todo:
            print("Settings already up to date - nothing to change.")
            return 0
        for k, v in todo.items():
            print(f"  {k} = {v}")
        if args.dry_run:
            print(f"{len(todo)} change(s) would be applied (dry run).")
            return 0
        vid, _ = apply_recommended(s, caps, author="cli-upgrade")
        s.commit()
    print(f"Applied {len(todo)} change(s) as settings version v{vid}. A running engine picks it up on RELOAD_SETTINGS "
          "(or restart it). Revert from the dashboard's settings history if needed.")
    if _engine_alive(sf):
        from control.commands import enqueue_command
        from core.enums import CommandType

        with sf() as s:  # type: ignore[operator]
            enqueue_command(s, CommandType.RELOAD_SETTINGS, {}, actor="cli-upgrade")
            s.commit()
        print("Engine is running: RELOAD_SETTINGS sent.")
    return 0


def cmd_new_batch(args: argparse.Namespace) -> int:
    import time as _time

    from capital.funding import topup_preview
    from control.commands import enqueue_command, get_command
    from core.enums import CommandType

    sf, caps, _ = _light_context()
    with sf() as s:  # type: ignore[operator]
        preview = topup_preview(s, args.capital)
    print(preview["message"])
    payload = {"starting_capital": args.capital, "name": args.name, "open_cnc_handling": args.cnc,
               "confirm_text": "CONFIRM"}
    if _engine_alive(sf):
        with sf() as s:  # type: ignore[operator]
            row = enqueue_command(s, CommandType.NEW_BATCH, payload, actor="cli")
            s.commit()
            cid = row.id
        print(f"Engine is running: NEW_BATCH command #{cid} sent, waiting for the result ...")
        for _ in range(60):
            _time.sleep(1)
            with sf() as s:  # type: ignore[operator]
                row = get_command(s, cid)
                if row is not None and row.status not in ("PENDING", "RUNNING"):
                    print(f"{row.status}: {(row.result_json or {}).get('message')}")
                    return 0 if row.status == "DONE" else 1
        print("No answer from the engine within 60 s - check the dashboard.")
        return 1
    from control.bootstrap import bootstrap
    from control.processor import CommandProcessor

    services = bootstrap(process="cli", with_notifier=False, llm_backend="rules")
    with services.session() as s:
        row = enqueue_command(s, CommandType.NEW_BATCH, payload, actor="cli", batch_id=services.ctx.batch_id,
                              clock=services.clock)
        s.commit()
        cid = row.id
    CommandProcessor(services).process_pending()
    with services.session() as s:
        row = get_command(s, cid)
        print(f"{row.status}: {(row.result_json or {}).get('message')}")
        return 0 if row.status == "DONE" else 1


def _parse_day(text: str | None) -> date:
    from core.clock import IST

    from datetime import datetime as _dt

    return date.fromisoformat(text) if text else _dt.now(IST).date()


def cmd_postmortem(args: argparse.Namespace) -> int:
    from pathlib import Path

    from analytics.postmortem import analyze_day
    from config.store import get_active_settings
    from core.clock import SystemClock
    from data.calendar import load_calendar
    from data.service import build_market_data_service

    sf, caps, secrets = _light_context()
    with sf() as s:  # type: ignore[operator]
        _, settings = get_active_settings(s, caps)
        s.rollback()
    data = None
    if not args.offline:
        from core.net import configure_ssl

        configure_ssl(Path(secrets.runtime_dir), use_os_store=True, ca_bundle=secrets.ca_bundle_path)  # type: ignore[attr-defined]
        cache = Path(secrets.runtime_dir) / "cache"  # type: ignore[attr-defined]
        data = build_market_data_service(settings, secrets, SystemClock(),
                                         load_calendar(settings.schedule, cache, fetch=False), cache)
    with sf() as s:  # type: ignore[operator]
        rep = analyze_day(s, _parse_day(args.date), data=data, settings=settings)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    print(f"Post-mortem {rep['day']}: {rep['summary']}")
    for t in rep["trades"]:
        print(f"\n{t['symbol']} {t['mode']} {t['strategy']} -> {t['result']} net {t['net_pnl']} "
              f"(R {t['r_multiple']}, best move {t['mfe_r']}R, worst {t['mae_r']}R)")
        for line in t["story"]:
            print(f"   {line}")
        for i in t["diagnosis"]:
            print(f"   [{i['severity']}] {i['title']}: {i['detail']}")
            print(f"        fix: {i['fixed_by'] or i['fix']}")
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from analytics.diagnostics import diagnose

    sf, _, secrets = _light_context()
    with sf() as s:  # type: ignore[operator]
        rep = diagnose(s, _parse_day(args.date), secrets.log_dir)  # type: ignore[attr-defined]
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
        return 0
    llm, pipe = rep["llm"], rep["pipeline"]
    print(f"Diagnosis {rep['day']}  ({rep['log_folder']})")
    print(f"  LLM calls {llm['calls']}, failed {llm['failed']} ({llm['fail_pct']}%), timeouts {llm['timeouts']}, "
          f"avg {llm['avg_latency_s']} s")
    print(f"  analyses {pipe['analyses']}, avg {pipe['avg_s']} s, max {pipe['max_s']} s; data errors "
          f"{rep['data']['errors']}; DB locked {rep['db']['locked']}")
    for f in rep["findings"]:
        print(f"\n[{f['severity']}] {f['title']}\n    {f['detail']}\n    -> {f['action']}")
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

    web = sub.add_parser("web", help="live web dashboard (HTML/JS), owner login required")
    web.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 (owner-only); use an SSH tunnel for remote")
    web.add_argument("--port", type=int, default=8600)
    web.set_defaults(func=cmd_web)

    nb = sub.add_parser("new-batch", help="start a new batch (closes the current one) and record the top-up")
    nb.add_argument("--capital", type=float, default=1000.0, help="starting capital of the new batch")
    nb.add_argument("--name", default=None, help='default: "Batch N"')
    nb.add_argument("--cnc", default="EXIT", choices=["EXIT", "RUN_TO_COMPLETION"],
                    help="what to do with open CNC positions of the current batch")
    nb.set_defaults(func=cmd_new_batch)

    up = sub.add_parser("upgrade-settings", help="apply the recommended settings (audited, revertible)")
    up.add_argument("--dry-run", action="store_true")
    up.set_defaults(func=cmd_upgrade_settings)

    pm = sub.add_parser("postmortem", help="per-trade analysis of a day: what happened, why, what is fixed")
    pm.add_argument("--date", default=None)
    pm.add_argument("--offline", action="store_true", help="no candle download (no MFE/MAE from prices)")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=cmd_postmortem)

    dg = sub.add_parser("diagnose", help="what failed in a day's logs and what to do next")
    dg.add_argument("--date", default=None)
    dg.add_argument("--json", action="store_true")
    dg.set_defaults(func=cmd_diagnose)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # Windows consoles default to cp1252 (no ₹ sign)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
