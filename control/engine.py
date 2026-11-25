"""The trading engine process (``python main.py engine``).

* APScheduler (IST cron triggers, Mon-Fri) fires the daily plan's jobs; every job checks the
  NSE trading calendar itself.
* The main loop polls ``control_commands`` every ``engine.poll_interval_seconds`` (2-5 s),
  writes the heartbeat, monitors positions every ``engine.monitor_interval_seconds``, expires
  approvals and advances shadow trades.
* The dashboard is a separate process: it only writes commands / settings and reads the DB.
"""

from __future__ import annotations

import signal
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from control.processor import CommandProcessor
from control.status import heartbeat, set_engine_state
from core.clock import IST, to_ist
from core.enums import EngineState
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
            logger.warning(f"[engine] job {job} skipped: the previous {job} is still running "
                           f"({self.skipped_jobs[job]} skipped today)")
            log_event("scans", {"event": "cycle_skipped", "job": job, "reason": "previous run still in progress",
                                "skipped_total": self.skipped_jobs[job]}, clock=self.svc.clock)
            return
        try:
            logger.info(f"[engine] job {job} start")
            with self.svc.session() as s:
                heartbeat(s, now=self.svc.clock.now(), batch_id=self.svc.ctx.batch_id,
                          settings_version_id=self.svc.ctx.settings_version_id, activity=f"job {job}")
                s.commit()
            if job in self.LOCKED_JOBS:
                with self._lock:
                    res = run_job(self.svc, job)
            else:
                res = run_job(self.svc, job)
            logger.info(f"[engine] job {job} done: {res}")
        finally:
            lock.release()

    def _next_cycle(self, now: datetime) -> datetime | None:
        plan = DailyPlan(self.svc.settings)
        for slot in plan.slots():
            ts = datetime.combine(now.date(), slot.at, tzinfo=IST)
            if ts > now:
                return ts
        return None

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
        with svc.session() as s:
            heartbeat(s, now=now, batch_id=svc.ctx.batch_id, settings_version_id=svc.ctx.settings_version_id,
                      next_cycle_at=self._next_cycle(now), market_open=market_open,
                      data_feed_ok=self._health["data"], llm_ok=self._health["llm"],
                      status={"state": self.state.value})
            s.commit()

    def run(self) -> None:
        """Blocking main loop until ENGINE_STOP / Ctrl+C."""
        svc = self.svc
        self._install_signals()
        with svc.session() as s:
            set_engine_state(s, EngineState.RUNNING, now=svc.clock.now(), activity="started")
            s.commit()
        svc.extras["engine_state"] = EngineState.RUNNING
        if self.use_scheduler:
            self._schedule()
        logger.info("[engine] running (PAPER TRADING ONLY). Ctrl+C to stop.")
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
            with svc.session() as s:
                set_engine_state(s, EngineState.STOPPED, now=svc.clock.now(), activity="stopped")
                s.commit()
            if svc.notifier is not None:
                svc.notifier.close(timeout=10)
            logger.info("[engine] stopped")


def seconds_until(now: datetime, target: datetime) -> float:
    """Helper for status displays."""
    return max(0.0, (target - now).total_seconds())


__all__ = ["TradingEngine", "seconds_until", "timedelta"]
