"""Engine-side control command processor (section 14.2).

The engine polls ``control_commands`` every 2-5 s; each PENDING command is executed in its
own transaction and its ``CommandResult`` written back (DONE / REJECTED / FAILED). Every
trade-affecting command goes through the risk manager or the hard-coded guards - the owner is
blocked by exactly the same rules as the system, and the result carries the exact rule.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import ValidationError
from sqlalchemy.orm import Session

from control.commands import pending_commands
from control.payloads import CommandResult, validate_payload
from core.enums import (
    CommandStatus,
    CommandType,
    EngineState,
    ExitReason,
    Mode,
    OrderPurpose,
    OrderStatus,
    PositionStatus,
    RiskRule,
)
from core.activity import activity
from core.errors import StopLossLooseningError, TradingSystemError
from core.logging import log_event, to_jsonable
from paper_engine.broker import OrderRejected, get_mode_state
from paper_engine.views import position_to_view
from risk.guards import InvalidStopLossError
from storage.audit import audit, record_error
from storage.models import ControlCommand, Order, Position

Handler = Callable[[Session, Any, str], CommandResult]


def describe_result(res: Any, *, routed: bool) -> str:
    """Plain-language one-liner for an analysis result (dashboard toast / activity log)."""
    head = f"{res.symbol} {res.mode.value}"
    secs = f" [{res.latency_ms / 1000:.0f}s]" if getattr(res, "latency_ms", None) else ""
    tail = "" if routed else " (analysis only - not traded)"
    if res.error:
        return f"{head}: analysis failed - {res.error}{secs}"
    if res.filtered_reason:
        return f"{head}: no trade - {res.filtered_reason}{tail}{secs}"
    meta, verdict = res.meta, res.verdict
    if meta is not None and meta.decision.value == "NO_TRADE":
        why = meta.why_not or "the AI decided not to trade"
        if why == "no strategy signal":
            why = "no strategy setup right now (none of the mode's strategies matched the chart)"
        return f"{head}: no trade - {why}{tail}{secs}"
    if meta is not None and verdict is not None:
        side = meta.decision.value
        conf = res.confluence.score if res.confluence is not None else None
        plan = (f"{side} {meta.strategy or ''} entry {meta.entry_price}, SL {meta.stop_loss}, T1 {meta.target_1} "
                f"(confidence {meta.confidence}" + (f", confluence {conf:.0f}" if conf is not None else "") + ")")
        if verdict.approved:
            return f"{head}: {plan} - risk manager APPROVED qty {verdict.qty} -> {res.final_action}{tail}{secs}"
        return f"{head}: {plan} - risk manager REJECTED: {verdict.message}{tail}{secs}"
    return f"{head}: {res.final_action}{tail}{secs}"


class CommandProcessor:
    """Dispatches control commands to services."""

    def __init__(self, services: Any, engine: Any | None = None) -> None:
        self.svc = services
        self.engine = engine
        self.handlers: dict[CommandType, Handler] = {
            CommandType.ENGINE_START: self._engine_start,
            CommandType.ENGINE_RESUME: self._engine_start,
            CommandType.ENGINE_PAUSE: self._engine_pause,
            CommandType.ENGINE_STOP: self._engine_stop,
            CommandType.MODE_PAUSE: self._mode_pause,
            CommandType.MODE_RESUME: self._mode_resume,
            CommandType.KILL_SWITCH: self._kill_switch,
            CommandType.RELOAD_SETTINGS: self._reload,
            CommandType.APPROVE: lambda s, p, a: self.svc.queue.approve(s, p.approval_id, actor=a),
            CommandType.REJECT: lambda s, p, a: self.svc.queue.reject(s, p.approval_id, actor=a, reason=p.reason),
            CommandType.MODIFY_APPROVE: lambda s, p, a: self.svc.queue.modify_and_approve(s, p, actor=a),
            CommandType.BULK_REJECT: lambda s, p, a: self.svc.queue.bulk_reject(s, p.approval_ids, actor=a,
                                                                                reason=p.reason),
            CommandType.MANUAL_TRADE: self._manual_trade,
            CommandType.ANALYZE_NOW: self._analyze_now,
            CommandType.CLOSE_POSITION: self._close_position,
            CommandType.TIGHTEN_SL: self._tighten_sl,
            CommandType.UPDATE_TARGETS: self._update_targets,
            CommandType.UPDATE_TRAILING: self._update_trailing,
            CommandType.CONVERT_POSITION: self._convert,
            CommandType.UPDATE_HOLDING: self._update_holding,
            CommandType.SET_EVENT_PIN: self._event_pin,
            CommandType.RESUME_HALT: self._resume_halt,
            CommandType.RESERVE_TRANSFER: self._reserve_transfer,
            CommandType.NEW_BATCH: self._new_batch,
            CommandType.CLOSE_BATCH: self._close_batch,
            CommandType.REENABLE_STRATEGY: self._reenable_strategy,
            CommandType.SEND_TEST_EMAIL: self._test_email,
            CommandType.RUN_JOB: self._run_job,
            CommandType.SCAN_NOW: self._scan_now,
        }
        self._worker: Any = None   # single background thread for long commands (live engine only)
        self._queued: dict[Any, int] = {}       # future -> command id (claimed, waiting for the worker)
        self.active: dict[int, str] = {}        # command id -> label, while a long command runs

    #: Commands that can take minutes (LLM analysis). In the live engine they run on a background
    #: worker so position monitoring / stop-losses are never paused behind them (2026-11-25: one
    #: "Analyze now" held the engine loop for 78 s).
    LONG_COMMANDS = frozenset({CommandType.ANALYZE_NOW, CommandType.SCAN_NOW, CommandType.RUN_JOB})

    # ------------------------------------------------------------------ loop

    def process_pending(self, limit: int = 20) -> int:
        """Execute up to ``limit`` pending commands (oldest first). Returns how many ran / started."""
        with self.svc.session() as s:
            rows = [(c.id, c.command_type) for c in pending_commands(s, limit)]
        for cid, ctype in rows:
            if self.engine is not None and ctype in self.LONG_COMMANDS:
                if self._claim(cid):
                    fut = self._background().submit(self._run_claimed, cid, str(ctype))
                    self._queued[fut] = cid
                    fut.add_done_callback(lambda f: self._queued.pop(f, None))
            else:
                self.execute(cid)
        return len(rows)

    def _background(self) -> Any:
        if self._worker is None:
            from concurrent.futures import ThreadPoolExecutor

            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="long-command")
        return self._worker

    def _run_claimed(self, command_id: int, label: str = "") -> None:
        self.active[command_id] = label or f"command #{command_id}"
        try:
            self._run(command_id)
        except Exception as exc:  # noqa: BLE001 - never kill the worker
            logger.exception(f"[control] background command {command_id} crashed: {exc}")
        finally:
            self.active.pop(command_id, None)

    def shutdown(self) -> int:
        """Engine stop: cancel queued long commands and close them as FAILED (never left RUNNING).

        The one in flight (if any) finishes on its own; returns how many were cancelled.
        """
        if self._worker is None:
            return 0
        cancelled = [cid for fut, cid in list(self._queued.items()) if fut.cancel()]
        self._worker.shutdown(wait=False, cancel_futures=True)
        svc = self.svc
        msg = "engine stopped before this command ran - please retry"
        with svc.session() as s:
            for cid in cancelled:
                row = s.get(ControlCommand, cid)
                if row is not None and row.status == CommandStatus.RUNNING.value:
                    row.status = CommandStatus.FAILED.value
                    row.completed_at = svc.clock.now()
                    row.error = msg
                    row.result_json = {"ok": False, "message": msg, "rule": None, "data": {}}
            s.commit()
        for cid in cancelled:
            activity("command", f"command #{cid}: FAILED - {msg}", clock=svc.clock, level="warning",
                     command_id=cid, status="FAILED")
        return len(cancelled)

    def _claim(self, command_id: int) -> bool:
        """PENDING -> RUNNING (False if someone else took it)."""
        svc = self.svc
        with svc.session() as s:
            row = s.get(ControlCommand, command_id)
            if row is None or row.status != CommandStatus.PENDING.value:
                return False
            row.status = CommandStatus.RUNNING.value
            row.picked_at = svc.clock.now()
            s.commit()
            ctype, actor = row.command_type, row.requested_by
        activity("command", f"{ctype} #{command_id} from {actor}: started", clock=svc.clock,
                 command_id=command_id, type=ctype, status="RUNNING")
        return True

    def execute(self, command_id: int) -> CommandResult:
        """Run one command in its own transaction. Never raises."""
        if not self._claim(command_id):
            return CommandResult(ok=False, message="command not pending")
        return self._run(command_id)

    def _run(self, command_id: int) -> CommandResult:
        svc = self.svc
        with svc.session() as s:
            row = s.get(ControlCommand, command_id)
            ctype = CommandType(row.command_type)
            actor = row.requested_by
            try:
                payload = validate_payload(ctype, row.payload_json or {})
                result = self.handlers[ctype](s, payload, actor)
                s.flush()
                status = CommandStatus.DONE if result.ok else CommandStatus.REJECTED
            except ValidationError as exc:
                s.rollback()
                result, status = CommandResult(ok=False, message=f"invalid payload: {exc.errors()}"), CommandStatus.REJECTED
            except Exception as exc:  # noqa: BLE001 - report the failure to the dashboard
                s.rollback()
                record_error(s, "control.processor", exc, {"command_id": command_id, "type": row.command_type},
                             batch_id=svc.ctx.batch_id, clock=svc.clock)
                result = CommandResult(ok=False, message=f"{type(exc).__name__}: {exc}")
                status = CommandStatus.FAILED
                logger.error(f"[control] command {command_id} {row.command_type} failed:\n{traceback.format_exc()}")
            s.commit()   # end the handler's transaction so the re-read below sees other writers
            row = s.get(ControlCommand, command_id)
            s.refresh(row)
            if row.status != CommandStatus.RUNNING.value:
                # closed meanwhile (e.g. a restarted engine failed it as orphaned): keep that outcome
                logger.warning(f"[control] command {command_id} finished as {status.value} but was already "
                               f"{row.status}; not overwriting")
            else:
                row.status = status.value
                row.completed_at = svc.clock.now()
                row.result_json = to_jsonable(result.model_dump(mode="json"))
                row.error = None if result.ok else result.message[:2000]
            s.commit()
        log_event("control", {"event": "executed", "id": command_id, "type": ctype.value, "status": status.value,
                              "ok": result.ok, "rule": result.rule.value if result.rule else None,
                              "message": result.message}, clock=svc.clock)
        activity("command", f"{ctype.value} #{command_id}: {status.value} - {result.message}", clock=svc.clock,
                 level="success" if result.ok else "warning", command_id=command_id, type=ctype.value,
                 status=status.value)
        return result

    # ------------------------------------------------------------------ engine

    def _set_state(self, s: Session, state: EngineState, actor: str, activity: str) -> None:
        from control.status import set_engine_state

        set_engine_state(s, state, now=self.svc.clock.now(), activity=activity)
        if self.engine is not None:
            self.engine.state = state
        self.svc.extras["engine_state"] = state
        audit(s, actor, f"ENGINE_{state.value}", "engine", {}, batch_id=self.svc.ctx.batch_id, clock=self.svc.clock)

    def _engine_start(self, s: Session, p: Any, actor: str) -> CommandResult:
        self._set_state(s, EngineState.RUNNING, actor, "running")
        return CommandResult(ok=True, message="engine running")

    def _engine_pause(self, s: Session, p: Any, actor: str) -> CommandResult:
        self._set_state(s, EngineState.PAUSED, actor, "paused by owner (positions still monitored)")
        return CommandResult(ok=True, message="engine paused: no new entries; open positions still monitored")

    def _engine_stop(self, s: Session, p: Any, actor: str) -> CommandResult:
        self._set_state(s, EngineState.STOPPED, actor, "stopped by owner")
        if self.engine is not None:
            self.engine.request_stop()
        return CommandResult(ok=True, message="engine stopping")

    def _mode_pause(self, s: Session, p: Any, actor: str) -> CommandResult:
        row = get_mode_state(s, self.svc.ctx.batch_id, Mode(p.mode), self.svc.clock.now())
        row.paused = True
        row.pause_reason = p.reason or "owner pause"
        audit(s, actor, "MODE_PAUSE", Mode(p.mode).value, {"reason": p.reason}, batch_id=self.svc.ctx.batch_id,
              clock=self.svc.clock)
        return CommandResult(ok=True, message=f"{Mode(p.mode).value} paused")

    def _mode_resume(self, s: Session, p: Any, actor: str) -> CommandResult:
        row = get_mode_state(s, self.svc.ctx.batch_id, Mode(p.mode), self.svc.clock.now())
        row.paused = False
        row.pause_reason = None
        audit(s, actor, "MODE_RESUME", Mode(p.mode).value, {}, batch_id=self.svc.ctx.batch_id, clock=self.svc.clock)
        return CommandResult(ok=True, message=f"{Mode(p.mode).value} resumed")

    def _market_open(self) -> bool:
        cal = self.svc.calendar
        return bool(cal.is_market_open(self.svc.clock.now())) if cal is not None else True

    def _kill_switch(self, s: Session, p: Any, actor: str) -> CommandResult:
        """EMERGENCY: pause everything, square off intraday, optionally exit every CNC position."""
        self._set_state(s, EngineState.PAUSED, actor, f"KILL SWITCH: {p.reason}")
        open_orders = s.query(Order).filter(Order.status == OrderStatus.PENDING.value,
                                            Order.purpose == OrderPurpose.ENTRY.value).all()
        for o in open_orders:
            self.svc.broker.cancel_order(s, o, "kill switch")
        trades = self.svc.broker.close_positions(s, reason=ExitReason.KILL_SWITCH, include_mis=True,
                                                 include_cnc=p.exit_cnc, actor=actor, market_open=self._market_open())
        if self.svc.notifier is not None:
            self.svc.notifier.urgent("KILL SWITCH activated", f"{p.reason}. Engine paused; {len(trades)} position(s) "
                                     f"exited now; CNC exit={p.exit_cnc}.", {"cancelled_orders": len(open_orders)})
        return CommandResult(ok=True, message=f"kill switch: engine paused, {len(trades)} exits, "
                                              f"{len(open_orders)} orders cancelled",
                             data={"exits": len(trades), "cancelled_orders": len(open_orders)})

    def _reload(self, s: Session, p: Any, actor: str) -> CommandResult:
        vid = self.svc.reload_settings()
        if self.engine is not None and hasattr(self.engine, "reschedule"):
            self.engine.reschedule()   # schedule times may have changed
        return CommandResult(ok=True, message=f"settings version {vid} loaded", data={"settings_version_id": vid})

    # ------------------------------------------------------------------ trades

    def _manual_trade(self, s: Session, p: Any, actor: str) -> CommandResult:
        batch_id = self.svc.ctx.batch_id
        analysis: dict[str, Any] = {}
        if p.analyze_first:
            res = self.svc.pipeline.analyze(p.symbol, Mode(p.mode), batch_id=batch_id, force=True, route=False)
            analysis = {"decision_id": res.decision_id, "final_action": res.final_action,
                        "meta": res.meta.model_dump(mode="json") if res.meta else None}
            s.commit()   # new snapshot: the analysis was committed by the pipeline's own session
        result = self.svc.manual.submit(s, p, actor=actor, batch_id=batch_id)
        if analysis:
            result.data["analysis"] = analysis
        return result

    def _analyze_now(self, s: Session, p: Any, actor: str) -> CommandResult:
        res = self.svc.pipeline.analyze(p.symbol, Mode(p.mode), batch_id=self.svc.ctx.batch_id, force=True,
                                        route=p.route)
        message = describe_result(res, routed=bool(p.route))
        try:   # follow-up lists show the latest analysis of their stocks (never fails the command)
            from data.watchlists import record_analysis

            with self.svc.session() as ws:
                if record_analysis(ws, res.symbol, decision_id=res.decision_id, summary=message,
                                   now=self.svc.clock.now()):
                    ws.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[control] could not record the analysis on follow-up lists: {exc}")
        return CommandResult(ok=res.error is None, message=message,
                             data={"decision_id": res.decision_id, "final_action": res.final_action,
                                   "latency_ms": res.latency_ms, "error": res.error})

    def _scan_now(self, s: Session, p: Any, actor: str) -> CommandResult:
        from scheduler.jobs import scan_now

        out = scan_now(self.svc, Mode(p.mode), use_llm=bool(p.use_llm), symbols=getattr(p, "symbols", None),
                       watchlist_id=getattr(p, "watchlist_id", None))
        return CommandResult(ok="error" not in out, message=out.get("message", str(out)), data=to_jsonable(out))

    def _position(self, s: Session, position_id: int) -> Position:
        pos = s.get(Position, position_id)
        if pos is None or pos.status != PositionStatus.OPEN.value:
            raise OrderRejected(f"position {position_id} is not open")
        return pos

    def _guarded(self, fn: Callable[[], Any], ok_msg: str) -> CommandResult:
        try:
            fn()
        except StopLossLooseningError as exc:
            return CommandResult(ok=False, message=str(exc), rule=RiskRule.SL_LOOSENING_FORBIDDEN)
        except (OrderRejected, InvalidStopLossError, TradingSystemError, ValueError) as exc:
            return CommandResult(ok=False, message=str(exc))
        return CommandResult(ok=True, message=ok_msg)

    def _close_position(self, s: Session, p: Any, actor: str) -> CommandResult:
        def go() -> None:
            pos = self._position(s, p.position_id)
            if self._market_open():
                self.svc.broker.exit_position(s, pos, p.qty, ExitReason.MANUAL, actor=actor)
            else:
                if p.qty is not None and p.qty < pos.qty:
                    raise OrderRejected("partial exits need an open market; full exits are queued for the next open")
                self.svc.broker.request_exit(s, pos, ExitReason.MANUAL, actor=actor)
        return self._guarded(go, "exit done (or queued for the next market open)")

    def _tighten_sl(self, s: Session, p: Any, actor: str) -> CommandResult:
        return self._guarded(lambda: self.svc.broker.tighten_sl(s, self._position(s, p.position_id), p.new_stop_loss,
                                                                actor=actor), "stop-loss tightened")

    def _update_targets(self, s: Session, p: Any, actor: str) -> CommandResult:
        return self._guarded(lambda: self.svc.broker.update_targets(s, self._position(s, p.position_id), p.target_1,
                                                                    p.target_2, actor=actor), "targets updated")

    def _update_trailing(self, s: Session, p: Any, actor: str) -> CommandResult:
        return self._guarded(lambda: self.svc.broker.update_trailing(s, self._position(s, p.position_id),
                                                                     p.trailing_rule, p.trail_atr_mult, actor=actor),
                             "trailing rule updated")

    def _update_holding(self, s: Session, p: Any, actor: str) -> CommandResult:
        return self._guarded(lambda: self.svc.broker.update_holding(s, self._position(s, p.position_id),
                                                                    p.max_holding_days, actor=actor),
                             "holding period updated")

    def _event_pin(self, s: Session, p: Any, actor: str) -> CommandResult:
        return self._guarded(lambda: self.svc.broker.set_event_pin(s, self._position(s, p.position_id), p.pinned,
                                                                   actor=actor), "event pin updated")

    def _convert(self, s: Session, p: Any, actor: str) -> CommandResult:
        from risk.state import build_risk_state

        try:
            pos = self._position(s, p.position_id)
        except OrderRejected as exc:
            return CommandResult(ok=False, message=str(exc))
        mp = self.svc.broker.market_price(pos.symbol)
        view = position_to_view(pos, mp[0] if mp else None)
        state = build_risk_state(s, batch_id=pos.batch_id, capital_manager=self.svc.capital, clock=self.svc.clock,
                                 calendar=self.svc.calendar, data=self.svc.data, symbol=pos.symbol)
        events = self.svc.events_for_window(pos.symbol, Mode(p.to_mode), self.svc.clock.now())
        verdict = self.svc.risk.evaluate_conversion(view, Mode(p.to_mode), state, events=events)
        if not verdict.approved:
            return CommandResult(ok=False, message=verdict.message, rule=verdict.rule)
        return self._guarded(lambda: self.svc.broker.convert_position(s, pos, Mode(p.to_mode), verdict, actor=actor),
                             f"converted to {Mode(p.to_mode).value}")

    # ------------------------------------------------------------------ capital / batches

    def _resume_halt(self, s: Session, p: Any, actor: str) -> CommandResult:
        from control.batches import resume_halted

        return resume_halted(s, p, self.svc, actor=actor)

    def _reserve_transfer(self, s: Session, p: Any, actor: str) -> CommandResult:
        if p.confirm_text.strip() != "CONFIRM":
            return CommandResult(ok=False, message="type CONFIRM to move money from RESERVE to TRADING")
        try:
            self.svc.capital.manual_reserve_transfer(s, self.svc.ctx.batch_id, p.amount, actor, True, note=p.note)
        except TradingSystemError as exc:
            return CommandResult(ok=False, message=str(exc))
        return CommandResult(ok=True, message=f"Rs {p.amount:.2f} moved RESERVE -> TRADING")

    def _new_batch(self, s: Session, p: Any, actor: str) -> CommandResult:
        from control.batches import start_new_batch

        return start_new_batch(s, p, self.svc, actor=actor)

    def _close_batch(self, s: Session, p: Any, actor: str) -> CommandResult:
        from control.batches import close_batch

        return close_batch(s, p, self.svc, actor=actor)

    def _reenable_strategy(self, s: Session, p: Any, actor: str) -> CommandResult:
        from strategies.performance import reenable_strategy

        reenable_strategy(s, self.svc.ctx.batch_id, p.strategy, Mode(p.mode), actor, self.svc.clock)
        return CommandResult(ok=True, message=f"{p.strategy} re-enabled for {Mode(p.mode).value}")

    def _test_email(self, s: Session, p: Any, actor: str) -> CommandResult:
        if self.svc.notifier is None:
            return CommandResult(ok=False, message="notifier disabled")
        out = self.svc.notifier.test()
        return CommandResult(ok=getattr(out, "status", "") != "FAILED",
                             message=f"test email: {getattr(out, 'status', out)}")

    def _run_job(self, s: Session, p: Any, actor: str) -> CommandResult:
        from scheduler.jobs import run_job

        engine = self.engine
        if engine is not None and p.job in getattr(engine, "LOCKED_JOBS", ()) | {"monitor"}:
            with engine._lock:   # state-mutating jobs never race the monitor (runs on the long-command worker)
                res = run_job(self.svc, p.job, force=True)
        else:
            res = run_job(self.svc, p.job, force=True)
        return CommandResult(ok="error" not in res, message=f"job {p.job}: {res}", data={"result": to_jsonable(res)})
