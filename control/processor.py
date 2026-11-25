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
from core.errors import StopLossLooseningError, TradingSystemError
from core.logging import log_event, to_jsonable
from paper_engine.broker import OrderRejected, get_mode_state
from paper_engine.views import position_to_view
from risk.guards import InvalidStopLossError
from storage.audit import audit, record_error
from storage.models import ControlCommand, Order, Position

Handler = Callable[[Session, Any, str], CommandResult]


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
        }

    # ------------------------------------------------------------------ loop

    def process_pending(self, limit: int = 20) -> int:
        """Execute up to ``limit`` pending commands (oldest first). Returns how many ran."""
        with self.svc.session() as s:
            ids = [c.id for c in pending_commands(s, limit)]
        for cid in ids:
            self.execute(cid)
        return len(ids)

    def execute(self, command_id: int) -> CommandResult:
        """Run one command in its own transaction. Never raises."""
        svc = self.svc
        with svc.session() as s:
            row = s.get(ControlCommand, command_id)
            if row is None or row.status != CommandStatus.PENDING.value:
                return CommandResult(ok=False, message="command not pending")
            row.status = CommandStatus.RUNNING.value
            row.picked_at = svc.clock.now()
            s.commit()
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
            row = s.get(ControlCommand, command_id)
            row.status = status.value
            row.completed_at = svc.clock.now()
            row.result_json = to_jsonable(result.model_dump(mode="json"))
            row.error = None if result.ok else result.message[:2000]
            s.commit()
        log_event("control", {"event": "executed", "id": command_id, "type": ctype.value, "status": status.value,
                              "ok": result.ok, "rule": result.rule.value if result.rule else None,
                              "message": result.message}, clock=svc.clock)
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
        return CommandResult(ok=res.error is None,
                             message=f"{res.symbol} {res.mode.value}: {res.final_action}"
                                     + (f" ({res.meta.decision.value}, confidence {res.meta.confidence})" if res.meta else "")
                                     + (f" - {res.filtered_reason}" if res.filtered_reason else ""),
                             data={"decision_id": res.decision_id, "final_action": res.final_action,
                                   "latency_ms": res.latency_ms, "error": res.error})

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

        res = run_job(self.svc, p.job, force=True)
        return CommandResult(ok="error" not in res, message=f"job {p.job}: {res}", data={"result": to_jsonable(res)})
