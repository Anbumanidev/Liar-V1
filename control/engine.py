"""The trading engine process (``python main.py engine``).

* APScheduler (IST cron triggers, Mon-Fri) fires the daily plan's jobs; every job checks the
  NSE trading calendar itself.
* The main loop polls ``control_commands`` every ``engine.poll_interval_seconds`` (2-5 s),
  writes the heartbeat, monitors positions every ``engine.monitor_interval_seconds``, expires
  approvals and advances shadow trades.
* The dashboard is a separate process: it only writes commands / settings and reads the DB.
"""

from __future__ import annotations

import re
import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from control.processor import CommandProcessor
from control.status import heartbeat, set_engine_state
from core.activity import activity
from core.clock import IST, parse_hhmm, to_ist
from core.enums import EngineState, Mode
from core.logging import log_event
from scheduler.jobs import run_job
from scheduler.plan import DailyPlan


class TradingEngine:
    """Long-running engine loop."""

    def __init__(self, services: Any, *, use_scheduler: bool = True) -> None:
        self.svc = services
        self.state = EngineState.RUNNING
        self.processor = CommandProcessor(services, engine=self)
        self.use_scheduler = use_scheduler
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._scheduler: Any = None
        self._last_monitor = 0.0
        self._last_expiry = 0.0
        self._last_shadow = 0.0
        self._last_health = 0.0
        self._health: dict[str, bool | None] = {"data": None, "llm": None}
        # One running instance per job NAME. Each cron slot is its own APScheduler job id
        # ("intraday_cycle@1110", "@1115", ...), so APScheduler's max_instances cannot stop a
        # slow 11:10 cycle from overlapping the 11:15 one - that overlap multiplied the LLM queue
        # and SQLite lock contention on 2026-11-25. A busy job's next slot is skipped and logged.
        self._job_locks: dict[str, threading.Lock] = {}
        self._job_locks_guard = threading.Lock()
        self.skipped_jobs: dict[str, int] = {}
        self._running: set[str] = set()
        self._last_status_text = ""
        self._last_status_log = 0.0

    # ------------------------------------------------------------------ lifecycle

    def request_stop(self) -> None:
        self._stop.set()

    def _install_signals(self) -> None:
        def _handler(signum: int, _frame: Any) -> None:
            logger.warning(f"[engine] signal {signum}: stopping")
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # not in main thread / unsupported on this OS
                pass

    def _schedule(self) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        sched = BackgroundScheduler(timezone=IST, job_defaults={"coalesce": True, "max_instances": 1,
                                                                 "misfire_grace_time": 120})
        plan = DailyPlan(self.svc.settings)
        for slot in plan.slots():
            sched.add_job(self._fire, CronTrigger(day_of_week="mon-fri", hour=slot.at.hour, minute=slot.at.minute,
                                                  timezone=IST),
                          args=[slot.job], id=f"{slot.job}@{slot.at:%H%M}", replace_existing=True)
        sched.start()
        self._scheduler = sched
        logger.info(f"[engine] scheduled {len(plan.slots())} job slots")

    def reschedule(self) -> None:
        """Rebuild the cron jobs after a schedule settings change."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._schedule()

    #: Short, state-mutating jobs serialized with the monitor / command processing. The long
    #: LLM scans run unlocked (their own sessions; orders they create are filled by the monitor).
    LOCKED_JOBS = frozenset({"premarket", "watchlist", "preopen_check", "squareoff", "cnc_time_stop", "eod",
                             "eod_report"})

    def _job_lock(self, job: str) -> threading.Lock:
        with self._job_locks_guard:
            lock = self._job_locks.get(job)
            if lock is None:
                lock = self._job_locks[job] = threading.Lock()
            return lock

    def _fire(self, job: str) -> None:
        lock = self._job_lock(job)
        if not lock.acquire(blocking=False):
            self.skipped_jobs[job] = self.skipped_jobs.get(job, 0) + 1
            activity("cycle", f"{job} skipped: the previous {job} is still running "
                              f"({self.skipped_jobs[job]} skipped today)", clock=self.svc.clock, level="warning",
                     job=job)
            log_event("scans", {"event": "cycle_skipped", "job": job, "reason": "previous run still in progress",
                                "skipped_total": self.skipped_jobs[job]}, clock=self.svc.clock)
            return
        self._running.add(job)
        t0 = time.monotonic()
        try:
            activity("cycle", f"{job} started", clock=self.svc.clock, job=job)
            with self.svc.session() as s:
                heartbeat(s, now=self.svc.clock.now(), batch_id=self.svc.ctx.batch_id,
                          settings_version_id=self.svc.ctx.settings_version_id, activity=f"running {job}")
                s.commit()
            if job in self.LOCKED_JOBS:
                with self._lock:
                    res = run_job(self.svc, job)
            else:
                res = run_job(self.svc, job)
            secs = time.monotonic() - t0
            if isinstance(res, dict) and res.get("skipped"):
                activity("cycle", f"{job} skipped: {res['skipped']}", clock=self.svc.clock, job=job)
            else:
                activity("cycle", f"{job} done in {secs:.0f}s: {_summarise(res)}", clock=self.svc.clock, job=job,
                         seconds=round(secs, 1), actions=res.get("actions") if isinstance(res, dict) else None)
        finally:
            self._running.discard(job)
            lock.release()

    def _next_cycle(self, now: datetime) -> datetime | None:
        nxt = self._next_slot(now)
        return nxt[0] if nxt else None

    def _job_relevant(self, job: str) -> bool:
        """False for slots that will only say "skipped" (disabled modes)."""
        s = self.svc.settings
        if job == "btst_scan":
            return s.mode(Mode.BTST).enabled
        if job == "intraday_cycle":
            return s.mode(Mode.INTRADAY).enabled
        if job == "swing_scan":
            return s.mode(Mode.SWING).enabled or s.mode(Mode.POSITIONAL).enabled
        if job in ("next_day_plan", "opening_check"):
            return s.engine.next_day_plan_enabled
        return True

    SCAN_JOBS = frozenset({"intraday_cycle", "btst_scan", "swing_scan"})

    def _next_slot(self, now: datetime, *, scans_only: bool = False) -> tuple[datetime, str] | None:
        """Next relevant scheduled job (or scan) today (trading days only)."""
        cal = self.svc.calendar
        if cal is not None and not cal.is_trading_day(now.date()):
            return None
        for slot in DailyPlan(self.svc.settings).slots():
            ts = datetime.combine(now.date(), slot.at, tzinfo=IST)
            if ts > now and self._job_relevant(slot.job) and (not scans_only or slot.job in self.SCAN_JOBS):
                return ts, slot.job
        return None

    def status_text(self, now: datetime) -> str:
        """One line for the terminal / dashboard: what the engine is doing and what runs next."""
        if self.state != EngineState.RUNNING:
            return f"{self.state.value.lower()} - no new scans until resumed (positions are still monitored)"
        running = sorted(set(self._running)) + sorted(set(self.processor.active.values()))
        if running:
            return f"running {', '.join(running)}"
        sc = self.svc.settings.schedule
        cal = self.svc.calendar
        if cal is not None and not cal.is_trading_day(now.date()):
            nxt_day = cal.next_trading_day(now.date())
            return (f"idle - {now:%a %d %b} is not an NSE trading day; next session {nxt_day:%a %d %b} "
                    "(pre-market 08:30)")
        start, end = parse_hhmm(sc.intraday_cycle_start), parse_hhmm(sc.intraday_cycle_end)
        t = now.time()
        if not self._job_relevant("intraday_cycle"):
            why = "INTRADAY trading is disabled"
        elif t < start:
            why = f"intraday scans start at {start:%H:%M}"
        elif t <= end:
            why = f"between intraday scans (every {sc.intraday_cycle_minutes} min until {end:%H:%M})"
        else:
            why = f"intraday scans run {start:%H:%M}-{end:%H:%M} (over for today)"
        nxt = self._next_slot(now)
        scan = self._next_slot(now, scans_only=True)

        def _when(slot: tuple[datetime, str]) -> str:
            return f"{slot[1]} at {slot[0]:%H:%M} (in {max(0, int((slot[0] - now).total_seconds() // 60))} min)"

        if nxt is None:
            nxt_day = cal.next_trading_day(now.date()) if cal is not None else None
            tail = f"no more jobs today; next session {nxt_day:%a %d %b}" if nxt_day else "no more jobs today"
        else:
            tail = f"next: {_when(nxt)}"
            if scan is not None and scan != nxt:
                tail += f"; next scan: {_when(scan)}"
            elif scan is None:
                tail += "; no more scans today"
        return f"idle - {why}; {tail}. Use Scan now on the dashboard to analyse at any time."

    def _fail_orphaned_commands(self) -> None:
        """Commands left RUNNING by a previous engine process can never finish: mark them FAILED."""
        from storage.models import ControlCommand

        svc = self.svc
        with svc.session() as s:
            rows = s.query(ControlCommand).filter(ControlCommand.status == "RUNNING").all()
            for row in rows:
                row.status = "FAILED"
                row.completed_at = svc.clock.now()
                row.error = "engine restarted while this command was running - please retry"
                row.result_json = {"ok": False, "message": row.error, "rule": None, "data": {}}
            s.commit()
        if rows:
            activity("engine", f"{len(rows)} command(s) interrupted by the last stop marked FAILED", clock=svc.clock,
                     level="warning")

    def _check_health(self) -> None:
        try:
            self._health["data"] = bool(self.svc.data.healthy())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[engine] data health check failed: {exc}")
            self._health["data"] = False
        try:
            self._health["llm"] = bool(self.svc.llm.healthcheck())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[engine] LLM health check failed: {exc}")
            self._health["llm"] = False
        now = to_ist(self.svc.clock.now())
        if (self._health["data"] is False and self.svc.calendar is not None and self.svc.calendar.is_market_open(now)
                and self.svc.notifier is not None):
            self.svc.notifier.urgent("Data feed down during market hours",
                                     f"Market data provider unhealthy at {now:%H:%M}: "
                                     f"{getattr(self.svc.data, 'last_error', None)}")

    def tick(self) -> None:
        """One iteration of the main loop (also used by tests)."""
        svc = self.svc
        now_mono = time.monotonic()
        with self._lock:
            self.processor.process_pending()
        now = to_ist(svc.clock.now())
        market_open = bool(svc.calendar.is_market_open(now)) if svc.calendar is not None else True
        if now_mono - self._last_monitor >= svc.settings.engine.monitor_interval_seconds:
            self._last_monitor = now_mono
            with self._lock:
                run_job(svc, "monitor")
        if now_mono - self._last_expiry >= svc.settings.schedule.approval_timeout_check_seconds:
            self._last_expiry = now_mono
            run_job(svc, "approval_expiry")
        if now_mono - self._last_shadow >= 300:
            self._last_shadow = now_mono
            run_job(svc, "shadow_update")
        if now_mono - self._last_health >= 120:
            self._last_health = now_mono
            self._check_health()
        status = self.status_text(now)
        key = _COUNTDOWN.sub("", status)   # the "(in N min)" countdown alone is not a change worth logging
        if key != self._last_status_text or now_mono - self._last_status_log >= 600:
            if not (key.startswith("running") and self._last_status_text.startswith("running")):
                nxt = self._next_slot(now)
                activity("engine", status, clock=svc.clock, state=self.state.value,
                         next_job=nxt[1] if nxt else None, next_at=nxt[0].isoformat() if nxt else None)
            self._last_status_text, self._last_status_log = key, now_mono
        next_cycle = self._next_cycle(now)
        with svc.session() as s:
            heartbeat(s, now=now, batch_id=svc.ctx.batch_id, settings_version_id=svc.ctx.settings_version_id,
                      next_cycle_at=next_cycle, market_open=market_open,
                      data_feed_ok=self._health["data"], llm_ok=self._health["llm"],
                      status={"state": self.state.value, "text": status}, activity=status[:500],
                      clear_next_cycle=next_cycle is None)
            s.commit()

    def run(self) -> None:
        """Blocking main loop until ENGINE_STOP / Ctrl+C."""
        svc = self.svc
        self._install_signals()
        with svc.session() as s:
            set_engine_state(s, EngineState.RUNNING, now=svc.clock.now(), activity="started")
            s.commit()
        svc.extras["engine_state"] = EngineState.RUNNING
        self._fail_orphaned_commands()
        if svc.settings.engine.keep_awake and not svc.ctx.is_replay:
            from core.power import keep_awake

            self._awake = keep_awake(display=svc.settings.engine.keep_display_on)
        if (svc.settings.engine.llm_backend == "ollama" and svc.settings.engine.auto_start_ollama
                and not svc.ctx.is_replay):
            from agents.ollama_service import ensure_running

            ok = ensure_running(svc.llm, svc.settings.models.ollama_host, svc.ctx.runtime_dir)
            self._health["llm"] = ok
            if not ok:
                activity("engine", "Ollama is not running and could not be started - AI analysis will fail until "
                                   "it is up (start the Ollama app)", clock=svc.clock, level="error")
        if self.use_scheduler:
            self._schedule()
        logger.info("[engine] running (PAPER TRADING ONLY). Ctrl+C to stop.")
        now = to_ist(svc.clock.now())
        st = svc.settings
        enabled = ", ".join(m.value for m in Mode if st.mode(m).enabled) or "none"
        activity("engine", f"started: batch {svc.ctx.batch_id}, settings v{svc.ctx.settings_version_id}, "
                           f"autonomy {st.modes.autonomy.value}, modes {enabled}, LLM {st.engine.llm_backend} "
                           f"({st.models.model_for('technical')}). {self.status_text(now)}", clock=svc.clock,
                 state="RUNNING")
        try:
            while not self._stop.is_set():
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001 - the loop must survive anything
                    logger.exception(f"[engine] tick failed: {exc}")
                self._stop.wait(svc.settings.engine.poll_interval_seconds)
        finally:
            if self._scheduler is not None:
                self._scheduler.shutdown(wait=False)
            try:
                self.processor.shutdown()   # queued Analyze/Scan commands -> FAILED, never left RUNNING
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[engine] command worker shutdown: {exc}")
            activity("engine", "stopped", clock=svc.clock, state="STOPPED")
            if getattr(self, "_awake", False):
                from core.power import release

                release()
            with svc.session() as s:
                set_engine_state(s, EngineState.STOPPED, now=svc.clock.now(), activity="stopped")
                s.commit()
            if svc.notifier is not None:
                svc.notifier.close(timeout=10)
            logger.info("[engine] stopped")


_COUNTDOWN = re.compile(r" \(in \d+ min\)")


def _summarise(res: Any) -> str:
    """Short text for a job result dict."""
    if not isinstance(res, dict):
        return str(res)[:200]
    if "actions" in res:
        acts = ", ".join(f"{k} {v}" for k, v in sorted(res["actions"].items())) or "nothing analysed"
        return f"{res.get('symbols', '?')} stock(s): {acts}"
    keep = {k: v for k, v in res.items() if v not in (None, [], {}, 0)}
    return (str(keep) if keep else "nothing to do")[:300]


def seconds_until(now: datetime, target: datetime) -> float:
    """Helper for status displays."""
    return max(0.0, (target - now).total_seconds())


__all__ = ["TradingEngine", "seconds_until", "timedelta"]
