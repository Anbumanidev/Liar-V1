"""Entry Timing Agent (section 4, agent 8): entry type, price zone, or WAIT.

Code builds the candidate entry prices for the reference setup (strongest
``StrategySignal``): MARKET at the LTP, the strategy entry, VWAP (intraday), EMA9 (and
EMA20 for longer modes) of the primary timeframe and the breakout level (opening range /
nearest resistance or support). The LLM only picks ``entry_type`` (MARKET | LIMIT | WAIT)
and one candidate index; ``finalize`` verifies the pick is a real, consistent candidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from agents.base import (
    DIRECTION_SIGNAL,
    INTRADAY_TIMEFRAMES,
    SetupAgent,
    current_price,
    primary_timeframe,
    r2,
    reference_setup,
)
from agents.registry import register_agent
from agents.schemas import AgentOutput
from core.enums import Direction, EntryType, Mode, Signal
from core.types import AgentReport, StockContext, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Candidates farther than this many ATR from the LTP are dropped.
MAX_DISTANCE_ATR = 3.0
#: Without an ATR, candidates farther than this % from the LTP are dropped.
MAX_DISTANCE_PCT = 5.0
#: Strategy entry within this many ATR of the LTP -> enter at market.
AT_MARKET_ATR = 0.2
#: Entry zone half-width in ATR.
ZONE_HALF_ATR = 0.1
#: Two candidates closer than this % are merged.
DEDUPE_PCT = 0.05

EntryChoice = Literal["MARKET", "LIMIT", "WAIT"]


class EntryTimingOutput(AgentOutput):
    """LLM answer of the entry-timing agent."""

    entry_type: EntryChoice = Field(description="MARKET (enter now), LIMIT (resting order at a candidate) or WAIT")
    candidate_index: int | None = Field(default=None, description="index of the chosen entry candidate from the facts")

    @field_validator("entry_type", mode="before")
    @classmethod
    def _v_type(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


def _atr(ctx: StockContext) -> float | None:
    f = ctx.features
    if f is None:
        return None
    if f.atr:
        return float(f.atr)
    tf = f.tf(f.primary_timeframe)
    return tf.v("atr_14") if tf else None


def entry_candidates(ctx: StockContext, setup: StrategySignal, settings: "Settings | None") -> list[dict[str, Any]]:
    """Deterministic entry candidates; index 0 is always MARKET at the LTP."""
    price = current_price(ctx)
    assert price is not None  # guarded by precheck
    atr = _atr(ctx)
    long = setup.direction == Direction.LONG
    raw: list[tuple[str, float]] = [("MARKET at LTP", price), (f"strategy {setup.strategy} entry", setup.entry_price)]
    f = ctx.features
    prim = primary_timeframe(ctx, settings)
    tf = f.tf(prim) if (f and prim) else None
    if tf is not None:
        if prim in INTRADAY_TIMEFRAMES and tf.v("vwap"):
            raw.append((f"VWAP ({prim})", tf.v("vwap")))  # type: ignore[arg-type]
        if tf.v("ema_9"):
            raw.append((f"EMA9 ({prim})", tf.v("ema_9")))  # type: ignore[arg-type]
        if ctx.mode != Mode.INTRADAY and tf.v("ema_20"):
            raw.append((f"EMA20 ({prim})", tf.v("ema_20")))  # type: ignore[arg-type]
        lv = tf.levels
        if lv is not None:
            if long and lv.resistances:
                above = [x for x in lv.resistances if x > price]
                if above:
                    raw.append(("breakout above nearest resistance", min(above)))
            if not long and lv.supports:
                below = [x for x in lv.supports if x < price]
                if below:
                    raw.append(("breakdown below nearest support", max(below)))
    if f is not None and ctx.mode == Mode.INTRADAY:
        if long and f.opening_range_high and f.opening_range_high > price:
            raw.append(("opening range high breakout", f.opening_range_high))
        if not long and f.opening_range_low and f.opening_range_low < price:
            raw.append(("opening range low breakdown", f.opening_range_low))

    out: list[dict[str, Any]] = []
    kept: list[float] = []
    for label, p in raw:
        if p is None or p <= 0:
            continue
        p = float(p)
        dist = p - price
        dist_atr = abs(dist) / atr if atr else None
        is_market = not out
        if not is_market:  # MARKET (index 0) is never dropped
            # The strategy entry is kept whatever its distance (it defines the setup).
            if not label.startswith("strategy "):
                if dist_atr is not None and dist_atr > MAX_DISTANCE_ATR:
                    continue
                if dist_atr is None and abs(dist) / price * 100 > MAX_DISTANCE_PCT:
                    continue
            # Merge near-duplicates (a strategy entry at the LTP collapses into MARKET).
            if any(abs(p - k) / price * 100 < DEDUPE_PCT for k in kept):
                continue
        # Can a resting LIMIT order sit here? (LONG: at/below LTP, SHORT: at/above LTP)
        limit_side = (p <= price) if long else (p >= price)
        out.append({
            "index": len(out), "label": label, "price": r2(p), "distance_pct": r2(dist / price * 100),
            "distance_atr": r2(dist_atr), "limit_ok": bool(limit_side and not is_market),
            "needs_trigger": bool(not limit_side),
        })
        kept.append(p)
    return out


def _decide(ctx: StockContext, setup: StrategySignal, cands: list[dict[str, Any]]) -> tuple[str, int, str]:
    """Rule-based (entry_type, candidate index, reason)."""
    price = current_price(ctx) or setup.entry_price
    atr = _atr(ctx)
    strat_idx = next((c["index"] for c in cands if c["label"].startswith("strategy ")), 0)
    gap = abs(setup.entry_price - price)
    near = gap <= (AT_MARKET_ATR * atr if atr else price * 0.002)
    if setup.entry_type == EntryType.MARKET or near or strat_idx == 0:
        return "MARKET", 0, "strategy entry is at the market price"
    c = cands[strat_idx]
    if c["limit_ok"]:
        return "LIMIT", strat_idx, "strategy entry is a pullback level below/above the market"
    return "WAIT", strat_idx, "strategy entry not triggered yet (price has not reached the trigger)"


@register_agent
class EntryTimingAgent(SetupAgent):
    """Chooses MARKET / LIMIT / WAIT among deterministic entry candidates."""

    name = "entry_timing"
    description = "Entry type and price zone among code-computed candidates, or WAIT"
    OutputModel = EntryTimingOutput

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        setup = reference_setup(ctx)
        price = current_price(ctx)
        if setup is None or price is None:
            return {"symbol": ctx.symbol, "setup": None, "ltp": r2(price)}
        cands = entry_candidates(ctx, setup, settings)
        etype, idx, why = _decide(ctx, setup, cands)
        return {
            "symbol": ctx.symbol, "ltp": r2(price), "atr": r2(_atr(ctx)),
            "setup": {"strategy": setup.strategy, "direction": setup.direction.value,
                      "entry_price": r2(setup.entry_price), "entry_type": setup.entry_type.value,
                      "stop_loss": r2(setup.stop_loss), "target_1": r2(setup.target_1),
                      "strength": setup.strength, "valid_until": setup.valid_until.isoformat() if setup.valid_until else None},
            "candidates": cands,
            "rule_suggestion": {"entry_type": etype, "candidate_index": idx, "reason": why},
            "rules": ["MARKET uses candidate 0", "LIMIT must use a candidate with limit_ok=true",
                      "WAIT means the trigger price (needs_trigger=true) has not been reached"],
        }

    def _result(self, ctx: StockContext, setup: StrategySignal, cands: list[dict[str, Any]],
                etype: str, idx: int, notes: list[str]) -> dict[str, Any]:
        atr = _atr(ctx)
        c = cands[idx]
        price = float(c["price"])
        half = ZONE_HALF_ATR * atr if atr else price * 0.001
        return {
            "entry_type": etype, "entry_price": r2(price), "candidate_index": idx, "candidate_label": c["label"],
            "zone_low": r2(price - half), "zone_high": r2(price + half),
            "setup_strategy": setup.strategy, "setup_direction": setup.direction.value,
            "candidates": cands, "validation_notes": notes,
        }

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        setup = reference_setup(ctx)
        if setup is None or current_price(ctx) is None:
            return self.make_report(ctx, key_findings=["no setup"], extra={"no_setup": True})
        cands = entry_candidates(ctx, setup, settings)
        etype, idx, why = _decide(ctx, setup, cands)
        signal = Signal.NEUTRAL if etype == "WAIT" else DIRECTION_SIGNAL[setup.direction]
        conf = 50 if etype == "WAIT" else 60
        return self.make_report(
            ctx, signal=signal, confidence=conf,
            key_findings=[f"{etype} at {cands[idx]['label']} {cands[idx]['price']}: {why}"],
            reasoning=why, extra=self._result(ctx, setup, cands, etype, idx, []),
        )

    def finalize(self, ctx: StockContext, parsed: EntryTimingOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        setup = reference_setup(ctx)
        if setup is None or current_price(ctx) is None:
            return det
        cands = entry_candidates(ctx, setup, settings)
        det_type, det_idx, _ = _decide(ctx, setup, cands)
        notes: list[str] = []
        etype: str = parsed.entry_type
        idx = parsed.candidate_index
        valid_idx = idx is not None and 0 <= idx < len(cands)
        if idx is not None and not valid_idx:
            notes.append(f"candidate_index {idx} does not exist")

        if etype == "MARKET":
            if idx not in (None, 0):
                notes.append("MARKET entry uses candidate 0 (LTP)")
            idx = 0
        elif etype == "LIMIT":
            if not valid_idx or not cands[idx]["limit_ok"]:  # type: ignore[index]
                limit_ok = [c for c in cands if c["limit_ok"]]
                if not limit_ok:
                    notes.append("no candidate can take a resting LIMIT order; using rule suggestion")
                    etype, idx = det_type, det_idx
                else:
                    target = float(cands[idx]["price"]) if valid_idx else float(cands[det_idx]["price"])  # type: ignore[index]
                    best = min(limit_ok, key=lambda c: abs(float(c["price"]) - target))
                    notes.append(f"LIMIT candidate invalid; using nearest valid LIMIT candidate {best['index']}")
                    idx = best["index"]
        else:  # WAIT
            if not valid_idx or idx == 0:
                idx = det_idx if det_idx != 0 else (1 if len(cands) > 1 else 0)
                notes.append(f"WAIT zone set to candidate {idx}")
        assert idx is not None
        signal = Signal.NEUTRAL if etype == "WAIT" else self.restrict_signal(parsed.signal, setup)
        if signal != parsed.signal:
            notes.append(f"signal {parsed.signal.value} not allowed for {etype} on a {setup.direction.value} setup")
        rep = self.report_from_output(ctx, parsed, signal=signal,
                                      extra=self._result(ctx, setup, cands, etype, idx, notes))
        return self.crosscheck(rep, det)
