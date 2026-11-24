"""Event & Overnight Risk Agent (section 4, agent 7) - BTST / SWING / POSITIONAL only.

Uses ``ctx.events`` (events inside the holding window, prepared by the data layer) plus
weekends / holidays inside the window. Code owns the verdict: ``event_risk`` and the list
of ``blocking_events`` (event types in ``settings.risk.avoid_event_types`` for this symbol)
cannot be cleared or invented by the LLM.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import SubAgent, mode_config
from agents.registry import register_agent
from agents.schemas import AgentOutput, truncate_list
from core.enums import EventType, Mode, Signal
from core.types import AgentReport, MarketEvent, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

DEFAULT_AVOID_TYPES: tuple[str, ...] = ("RESULTS", "BOARD_MEETING", "EX_DIVIDEND", "EX_DATE", "SPLIT", "BONUS", "RIGHTS")
#: Event types that are always informative risk (never blocking by themselves).
SOFT_RISK_TYPES: frozenset[EventType] = frozenset({EventType.MACRO, EventType.INDEX_REBALANCE, EventType.AGM})
BLOCKING_CONFIDENCE = 85
SOFT_RISK_CONFIDENCE = 50


class EventRiskOutput(AgentOutput):
    """LLM answer of the event-risk agent."""

    event_risk: bool = Field(default=False, description="true if any listed event/gap risk falls inside the holding window")
    blocking_events: list[str] = Field(default_factory=list,
                                       description="Copy the ids of blocking events from the facts (e.g. 'E1'); never invent")


def holding_window(ctx: StockContext, settings: "Settings | None") -> tuple[date, date]:
    """(entry day, last calendar day of the holding window).

    Holding days are trading days; they are converted to calendar days with 7/5 plus a
    3-day weekend buffer, so the window is a conservative superset.
    """
    start = ctx.as_of.date()
    cfg = mode_config(ctx, settings)
    if cfg is not None:
        days = cfg.max_holding_days
    else:
        days = int(ctx.mode_settings.get("max_holding_days", {Mode.BTST: 1, Mode.SWING: 10, Mode.POSITIONAL: 63}.get(ctx.mode, 1)))
    days = max(1, days)
    # BTST (1 day): next trading day may be up to 4 calendar days away (Fri -> Mon + holiday).
    calendar_days = 4 if days == 1 else int(math.ceil(days * 7 / 5)) + 3
    return start, start + timedelta(days=calendar_days)


def _event_id(i: int) -> str:
    return f"E{i + 1}"


@register_agent
class EventRiskAgent(SubAgent):
    """Events, ex-dates, macro and weekend/holiday gaps inside the holding window."""

    name = "event_risk"
    description = "Results, ex-dates, board meetings, index rebalancing, macro events, weekends/holidays"
    modes = frozenset({Mode.BTST, Mode.SWING, Mode.POSITIONAL})
    OutputModel = EventRiskOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> dict[str, Any]:
        start, end = holding_window(ctx, settings)
        avoid = {str(t) for t in (settings.risk.avoid_event_types if settings else DEFAULT_AVOID_TYPES)}
        avoid_enabled = settings.risk.avoid_events_in_window if settings else True
        events: list[MarketEvent] = sorted(
            (e for e in ctx.events if start <= e.event_date <= end
             and (e.symbol is None or e.symbol.upper() == ctx.symbol.upper())),
            key=lambda e: (e.event_date, e.event_type.value),
        )
        listed: list[dict[str, Any]] = []
        blocking: list[str] = []
        soft: list[str] = []
        for i, e in enumerate(events):
            eid = _event_id(i)
            is_block = (avoid_enabled and e.symbol is not None and e.event_type.value in avoid
                        and e.event_type not in SOFT_RISK_TYPES)
            desc = f"{eid}: {e.event_type.value} {e.event_date.isoformat()} {e.title}".strip()
            listed.append({"id": eid, "type": e.event_type.value, "date": e.event_date.isoformat(),
                           "days_from_entry": (e.event_date - start).days, "title": e.title[:160],
                           "stock_specific": e.symbol is not None, "blocking": is_block})
            (blocking if is_block else soft).append(desc)

        # Weekends / holidays inside the overnight gap(s).
        gap_notes: list[str] = []
        if ctx.mode == Mode.BTST:
            nxt = start + timedelta(days=1)
            if nxt.weekday() >= 5:
                gap_notes.append(f"BTST over a weekend: {start.isoformat()} is a {start.strftime('%A')}")
        holidays = [str(h) for h in (ctx.extra.get("holidays_in_window") or [])]
        if holidays:
            gap_notes.append(f"exchange holidays inside the window: {', '.join(holidays[:5])}")
        weekend_days = sum(1 for k in range((end - start).days + 1) if (start + timedelta(days=k)).weekday() >= 5)

        return {
            "symbol": ctx.symbol,
            "mode": ctx.mode.value,
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "avoid_event_types": sorted(avoid),
            "avoid_events_in_window": avoid_enabled,
            "events": listed,
            "blocking_event_ids": [b.split(":")[0] for b in blocking],
            "blocking_events": blocking,
            "soft_risks": soft + gap_notes,
            "weekend_days_in_window": weekend_days,
            "event_risk": bool(blocking or soft or gap_notes),
        }

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return self._analyse(ctx, settings)

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        a = self._analyse(ctx, settings)
        if not a["event_risk"]:
            return self.make_report(
                ctx, signal=Signal.NEUTRAL, confidence=60,
                key_findings=[f"no events or gap risks between {a['window_start']} and {a['window_end']}"],
                extra={"event_risk": False, "blocking_events": []},
            )
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        a = self._analyse(ctx, settings)
        if a["blocking_events"]:
            signal, conf = Signal.BEARISH, BLOCKING_CONFIDENCE
        elif a["event_risk"]:
            signal, conf = Signal.NEUTRAL, SOFT_RISK_CONFIDENCE
        else:
            signal, conf = Signal.NEUTRAL, 60
        findings = [f"BLOCKING {b}" for b in a["blocking_events"]] + a["soft_risks"]
        return self.make_report(
            ctx, signal=signal, confidence=conf, key_findings=findings or ["no event risk in window"],
            reasoning=f"{len(a['blocking_events'])} blocking event(s), {len(a['soft_risks'])} soft risk(s) "
                      f"between {a['window_start']} and {a['window_end']}.",
            extra={"event_risk": a["event_risk"], "blocking_events": a["blocking_events"],
                   "soft_risks": a["soft_risks"]},
        )

    def finalize(self, ctx: StockContext, parsed: EventRiskOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        code_blocking: list[str] = list(det.extra.get("blocking_events") or [])
        notes: list[str] = []
        known_ids = {b.split(":")[0] for b in code_blocking}
        invented = [b for b in parsed.blocking_events if b.split(":")[0].strip() not in known_ids]
        if invented:
            notes.append(f"ignored LLM blocking events not in code list: {truncate_list(invented, 3, 60)}")
        event_risk = bool(det.extra.get("event_risk")) or parsed.event_risk
        signal, conf = parsed.signal, parsed.confidence
        if signal == Signal.BULLISH:
            notes.append("event-risk agent cannot vote BULLISH; set to NEUTRAL")
            signal = Signal.NEUTRAL
        if code_blocking:
            if signal != Signal.BEARISH:
                notes.append("blocking event in window: signal forced to BEARISH")
            signal = Signal.BEARISH
            conf = max(conf, det.confidence)
        findings = [f"BLOCKING {b}" for b in code_blocking] + list(parsed.key_findings)
        rep = self.report_from_output(ctx, parsed, signal=signal, confidence=conf, key_findings=findings, extra={
            "event_risk": event_risk, "blocking_events": code_blocking,
            "soft_risks": det.extra.get("soft_risks"), "validation_notes": notes,
        })
        return self.crosscheck(rep, det)
