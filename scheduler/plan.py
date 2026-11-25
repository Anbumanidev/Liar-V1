"""The daily schedule (section 10) as data, shared by the live engine and REPLAY.

``DailyPlan(settings).slots()`` lists every scheduled job of a trading day with its IST time.
``jobs_due(prev, now)`` returns the jobs whose slot falls in ``(prev, now]`` - the replay
driver calls it every simulated minute; the live engine turns the same slots into
APScheduler cron triggers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from config.schema import Settings
from core.clock import IST, parse_hhmm


@dataclass(frozen=True, order=True)
class Slot:
    at: time
    job: str


#: Job names (also accepted by the RUN_JOB command).
JOBS: tuple[str, ...] = (
    "premarket", "watchlist", "preopen_check", "intraday_cycle", "btst_scan", "squareoff",
    "cnc_time_stop", "eod", "eod_report", "swing_scan",
)


def _range(start: str, end: str, step_min: int) -> list[time]:
    t0 = datetime.combine(date(2000, 1, 1), parse_hhmm(start))
    t1 = datetime.combine(date(2000, 1, 1), parse_hhmm(end))
    out: list[time] = []
    t = t0
    while t <= t1:
        out.append(t.time())
        t += timedelta(minutes=max(1, step_min))
    return out


class DailyPlan:
    """Scheduled job slots for one trading day."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def slots(self) -> list[Slot]:
        sc = self.settings.schedule
        out = [
            Slot(parse_hhmm(sc.premarket), "premarket"),
            Slot(parse_hhmm(sc.watchlist_build), "watchlist"),
            Slot(parse_hhmm(sc.preopen_check), "preopen_check"),
            Slot(parse_hhmm(sc.intraday_squareoff), "squareoff"),
            Slot(parse_hhmm(sc.cnc_time_stop), "cnc_time_stop"),
            Slot(parse_hhmm(sc.eod_processing), "eod"),
            Slot(parse_hhmm(sc.eod_report), "eod_report"),
            Slot(parse_hhmm(sc.swing_scan), "swing_scan"),
        ]
        out += [Slot(t, "intraday_cycle") for t in _range(sc.intraday_cycle_start, sc.intraday_cycle_end,
                                                           sc.intraday_cycle_minutes)]
        out += [Slot(t, "btst_scan") for t in _range(sc.btst_scan_window.start, sc.btst_scan_window.end,
                                                      sc.btst_scan_minutes)]
        return sorted(out)

    def jobs_due(self, prev: datetime, now: datetime) -> list[str]:
        """Jobs whose slot time on ``now``'s date lies in ``(prev, now]`` (in slot order)."""
        due: list[str] = []
        for slot in self.slots():
            ts = datetime.combine(now.date(), slot.at, tzinfo=IST)
            if prev < ts <= now:
                due.append(slot.job)
        return due

    def first_slot(self) -> time:
        return min(s.at for s in self.slots())

    def last_slot(self) -> time:
        return max(s.at for s in self.slots())
