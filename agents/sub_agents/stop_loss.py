"""Stop-Loss Agent (section 4, agent 9).

Deterministic candidates come from ``ctx.sl_candidates`` (``features.stops``: ATR
multiples, swing points, VWAP, opening range, daily structure) plus the strategy's own stop
when it is not already listed. The LLM picks one candidate **by index** and justifies it;
code validates it (exists, protective side of the setup entry, distance between
``settings.risk.sl_min_atr`` and ``sl_max_atr`` x ATR). An invalid pick - or an invented
price - falls back to the valid candidate nearest to the LLM's pick (or to the rule choice)
and the substitution is noted. The result is in ``report.extra``:
``{'stop_loss', 'method', 'candidate_index', 'candidates', 'validation_notes', ...}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import DIRECTION_SIGNAL, SetupAgent, current_price, r2, reference_setup
from agents.registry import register_agent
from agents.schemas import AgentOutput
from core.enums import Direction, Signal, SLMethod
from core.types import AgentReport, SLCandidate, StockContext, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Prices closer than this (rupees) are considered the same level.
PRICE_TOL = 0.01


class StopLossOutput(AgentOutput):
    """LLM answer of the stop-loss agent."""

    candidate_index: int = Field(description="index of the chosen stop-loss candidate from the facts")
    price: float | None = Field(default=None, description="optional: copy the chosen candidate's price exactly")


def _atr(ctx: StockContext) -> float | None:
    f = ctx.features
    if f is None:
        return None
    if f.atr:
        return float(f.atr)
    tf = f.tf(f.primary_timeframe)
    return tf.v("atr_14") if tf else None


def sl_bounds(settings: "Settings | None") -> tuple[float, float]:
    """(min, max) stop distance in ATR multiples."""
    if settings is None:
        return 0.3, 3.0
    return float(settings.risk.sl_min_atr), float(settings.risk.sl_max_atr)


def build_candidates(ctx: StockContext, setup: StrategySignal) -> list[SLCandidate]:
    """``ctx.sl_candidates`` + the strategy stop (if new), distances recomputed vs the setup entry."""
    atr = _atr(ctx)
    entry = setup.entry_price
    raw = list(ctx.sl_candidates)
    if not any(abs(c.price - setup.stop_loss) <= PRICE_TOL for c in raw):
        raw.append(SLCandidate(method=SLMethod.STRATEGY, price=setup.stop_loss, distance=0.0,
                               note=f"{setup.strategy} strategy stop"))
    out: list[SLCandidate] = []
    for c in raw:
        dist = abs(entry - c.price)
        out.append(c.model_copy(update={"distance": round(dist, 4),
                                        "distance_atr": round(dist / atr, 3) if atr else c.distance_atr}))
    return out


def validate_candidate(c: SLCandidate, setup: StrategySignal, atr: float | None,
                       bounds: tuple[float, float]) -> str | None:
    """None if ``c`` is a valid protective stop for ``setup``, else the reason."""
    entry = setup.entry_price
    if c.price <= 0:
        return "non-positive price"
    if setup.direction == Direction.LONG and not c.price < entry:
        return f"{c.price:.2f} is not below the LONG entry {entry:.2f}"
    if setup.direction == Direction.SHORT and not c.price > entry:
        return f"{c.price:.2f} is not above the SHORT entry {entry:.2f}"
    dist = abs(entry - c.price)
    lo, hi = bounds
    d_atr = dist / atr if atr else c.distance_atr
    if d_atr is None:
        # No ATR at all: only reject absurd distances (> 10 % of entry).
        return None if dist / entry <= 0.10 else f"distance {dist:.2f} > 10% of entry without ATR"
    if d_atr < lo:
        return f"distance {d_atr:.2f} ATR < minimum {lo:.2f} ATR (too tight)"
    if d_atr > hi:
        return f"distance {d_atr:.2f} ATR > maximum {hi:.2f} ATR (too wide)"
    return None


def rule_choice(cands: list[SLCandidate], valid: list[int]) -> int | None:
    """Rule pick: the strategy's stop if valid, else the tightest valid candidate."""
    if not valid:
        return None
    for i in valid:
        if cands[i].method == SLMethod.STRATEGY:
            return i
    return min(valid, key=lambda i: cands[i].distance)


@register_agent
class StopLossAgent(SetupAgent):
    """Picks and justifies one deterministic stop-loss candidate."""

    name = "stop_loss"
    description = "Chooses one code-computed stop-loss candidate; code validates side and ATR distance"
    OutputModel = StopLossOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> dict[str, Any] | None:
        setup = reference_setup(ctx)
        if setup is None:
            return None
        atr = _atr(ctx)
        bounds = sl_bounds(settings)
        cands = build_candidates(ctx, setup)
        reasons = [validate_candidate(c, setup, atr, bounds) for c in cands]
        valid = [i for i, r in enumerate(reasons) if r is None]
        return {"setup": setup, "atr": atr, "bounds": bounds, "cands": cands, "reasons": reasons,
                "valid": valid, "rule": rule_choice(cands, valid)}

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        a = self._analyse(ctx, settings)
        if a is None:
            return {"symbol": ctx.symbol, "setup": None}
        setup: StrategySignal = a["setup"]
        return {
            "symbol": ctx.symbol, "ltp": r2(current_price(ctx)), "atr": r2(a["atr"]),
            "setup": {"strategy": setup.strategy, "direction": setup.direction.value,
                      "entry_price": r2(setup.entry_price), "strategy_stop": r2(setup.stop_loss),
                      "target_1": r2(setup.target_1)},
            "allowed_distance_atr": {"min": a["bounds"][0], "max": a["bounds"][1]},
            "candidates": [
                {"index": i, "method": c.method.value, "price": r2(c.price), "distance": r2(c.distance),
                 "distance_atr": r2(c.distance_atr), "note": c.note[:120],
                 "valid": a["reasons"][i] is None, "invalid_reason": a["reasons"][i]}
                for i, c in enumerate(a["cands"])
            ],
            "rule_suggestion_index": a["rule"],
        }

    def _extra(self, a: dict[str, Any], idx: int | None, notes: list[str]) -> dict[str, Any]:
        cands: list[SLCandidate] = a["cands"]
        setup: StrategySignal = a["setup"]
        c = cands[idx] if idx is not None else None
        return {
            "stop_loss": r2(c.price) if c else None,
            "method": c.method.value if c else None,
            "candidate_index": idx,
            "distance": r2(c.distance) if c else None,
            "distance_atr": r2(c.distance_atr) if c else None,
            "setup_strategy": setup.strategy,
            "setup_entry": r2(setup.entry_price),
            "setup_direction": setup.direction.value,
            "candidates": [{"index": i, "method": x.method.value, "price": r2(x.price),
                            "distance_atr": r2(x.distance_atr), "valid": a["reasons"][i] is None}
                           for i, x in enumerate(cands)],
            "validation_notes": notes,
        }

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        a = self._analyse(ctx, settings)
        if a is None:
            return self.make_report(ctx, key_findings=["no setup"], extra={"no_setup": True})
        idx = a["rule"]
        if idx is None:
            return self.make_report(ctx, key_findings=["no valid stop-loss candidate"],
                                    extra=self._extra(a, None, ["no candidate passed validation"]))
        c: SLCandidate = a["cands"][idx]
        return self.make_report(
            ctx, signal=DIRECTION_SIGNAL[a["setup"].direction], confidence=55,
            key_findings=[f"stop {c.price:.2f} ({c.method.value}, {c.distance_atr or 0:.2f} ATR from entry)"],
            reasoning="Rule: strategy stop if valid, else the tightest valid candidate.",
            extra=self._extra(a, idx, []),
        )

    def finalize(self, ctx: StockContext, parsed: StopLossOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        a = self._analyse(ctx, settings)
        if a is None:
            return det
        cands: list[SLCandidate] = a["cands"]
        valid: list[int] = a["valid"]
        setup: StrategySignal = a["setup"]
        notes: list[str] = []
        idx: int | None = parsed.candidate_index
        exists = idx is not None and 0 <= idx < len(cands)
        if not exists:
            notes.append(f"candidate_index {parsed.candidate_index} does not exist")
        if parsed.price is not None:
            if exists and abs(parsed.price - cands[idx].price) > PRICE_TOL:  # type: ignore[index]
                notes.append(f"LLM price {parsed.price} differs from candidate price; candidate price used")
            if not any(abs(parsed.price - c.price) <= PRICE_TOL for c in cands):
                notes.append(f"LLM price {parsed.price} is not a deterministic candidate (rejected)")

        if exists and idx in valid:
            chosen: int | None = idx
        else:
            if exists:
                notes.append(f"candidate {idx} invalid: {a['reasons'][idx]}")  # type: ignore[index]
            if not valid:
                chosen = None
            else:
                ref = cands[idx].price if exists else None  # type: ignore[index]
                if ref is None and parsed.price is not None:
                    ref = parsed.price
                if ref is None:
                    chosen = a["rule"]
                else:
                    chosen = min(valid, key=lambda i: (abs(cands[i].price - ref), cands[i].distance))
                notes.append(f"fell back to nearest valid candidate {chosen}")

        if chosen is None:
            rep = self.report_from_output(ctx, parsed, signal=Signal.NEUTRAL, confidence=0,
                                          key_findings=["no valid stop-loss candidate"] + list(parsed.key_findings),
                                          extra=self._extra(a, None, notes + ["no valid candidate"]))
            return self.crosscheck(rep, det)
        signal = self.restrict_signal(parsed.signal, setup)
        conf = parsed.confidence if chosen == idx else min(parsed.confidence, 50)
        c = cands[chosen]
        findings = [f"stop {c.price:.2f} ({c.method.value}, {c.distance_atr or 0:.2f} ATR)"] + list(parsed.key_findings)
        rep = self.report_from_output(ctx, parsed, signal=signal, confidence=conf, key_findings=findings,
                                      extra=self._extra(a, chosen, notes))
        return self.crosscheck(rep, det)
