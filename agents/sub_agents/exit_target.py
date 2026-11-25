"""Exit / Target Agent (section 4, agent 10).

Code computes target candidates for the reference setup from R:R multiples (the mode's
minimum R:R, 2R, 3R) and price levels (strategy targets, pivots, resistances / supports,
previous high/low, 52-week high/low), each with its exact R:R. Levels below the mode's
minimum R:R are listed as *obstacles*, not candidates. The LLM picks ``target_1_index`` /
``target_2_index``, a trailing rule and max holding days; ``finalize`` enforces min R:R,
target ordering, the trailing methods allowed for the mode and the mode's holding limits.
Result in ``report.extra``: ``target_1, target_2, rr_1, rr_2, trailing_rule,
max_holding_days, target_1_index, target_2_index, candidates, obstacles``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from agents.base import (
    DIRECTION_SIGNAL,
    SetupAgent,
    current_price,
    min_rr_for,
    mode_config,
    mode_timeframes,
    r2,
    reference_setup,
)
from agents.registry import register_agent
from agents.schemas import AgentOutput, clamp_int
from core.enums import Direction, Mode, Signal, TrailingMethod
from core.types import AgentReport, StockContext, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Trailing methods allowed per mode (section 7.4).
ALLOWED_TRAILING: dict[Mode, tuple[TrailingMethod, ...]] = {
    Mode.INTRADAY: (TrailingMethod.NONE, TrailingMethod.ATR, TrailingMethod.EMA9_5M),
    Mode.BTST: (TrailingMethod.NONE,),
    Mode.SWING: (TrailingMethod.NONE, TrailingMethod.EMA20_DAILY, TrailingMethod.ATR_DAILY_CLOSE),
    Mode.POSITIONAL: (TrailingMethod.NONE, TrailingMethod.EMA20_DAILY, TrailingMethod.ATR_DAILY_CLOSE),
}
DEFAULT_HOLDING: dict[Mode, tuple[int, int]] = {Mode.INTRADAY: (0, 0), Mode.BTST: (1, 1), Mode.SWING: (2, 10),
                                               Mode.POSITIONAL: (5, 63)}
#: Two targets closer than this % of entry are merged.
DEDUPE_PCT = 0.1
#: target_2 must be at least this many R beyond target_1.
MIN_T2_GAP_R = 0.5


class ExitTargetOutput(AgentOutput):
    """LLM answer of the exit/target agent."""

    target_1_index: int = Field(description="index of the first target candidate (must meet min R:R)")
    target_2_index: int | None = Field(default=None, description="index of a farther second target, or null")
    trailing_rule: TrailingMethod = Field(description="one of the allowed trailing methods listed in the facts")
    max_holding_days: int = Field(ge=0, le=180, description="max holding days within the mode limits (0 intraday)")

    @field_validator("trailing_rule", mode="before")
    @classmethod
    def _v_trail(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("max_holding_days", mode="before")
    @classmethod
    def _v_days(cls, v: Any) -> Any:
        return clamp_int(v, 0, 180)


def holding_limits(ctx: StockContext, settings: "Settings | None") -> tuple[int, int]:
    """(min, max) holding days of the mode."""
    cfg = mode_config(ctx, settings)
    if cfg is not None:
        return int(cfg.min_holding_days), int(cfg.max_holding_days)
    return DEFAULT_HOLDING[ctx.mode]


def default_trailing(ctx: StockContext, setup: StrategySignal, settings: "Settings | None") -> TrailingMethod:
    """Mode-config trailing (if allowed), else the setup's rule (if allowed), else NONE."""
    allowed = ALLOWED_TRAILING[ctx.mode]
    cfg = mode_config(ctx, settings)
    if cfg is not None and cfg.trailing in allowed:
        return cfg.trailing
    if setup.trailing_rule in allowed:
        return setup.trailing_rule
    return TrailingMethod.NONE


def target_candidates(ctx: StockContext, setup: StrategySignal, settings: "Settings | None",
                      min_rr: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(candidates meeting min R:R sorted by distance, obstacles below min R:R)."""
    entry, stop = setup.entry_price, setup.stop_loss
    risk = abs(entry - stop)
    if risk <= 0:
        return [], []
    long = setup.direction == Direction.LONG
    sgn = 1.0 if long else -1.0
    raw: list[tuple[str, float]] = []
    for m in sorted({round(min_rr, 2), 2.0, 3.0}):
        if m >= min_rr - 1e-9:
            raw.append((f"{m:g}R", entry + sgn * m * risk))
    raw.append((f"strategy {setup.strategy} target_1", setup.target_1))
    if setup.target_2 is not None:
        raw.append((f"strategy {setup.strategy} target_2", setup.target_2))
    f = ctx.features
    if f is not None:
        for name in mode_timeframes(ctx, settings):
            tf = f.tf(name)
            lv = tf.levels if tf else None
            if lv is None:
                continue
            names = ("r1", "r2", "r3", "prev_high") if long else ("s1", "s2", "s3", "prev_low")
            for k in names:
                v = getattr(lv, k)
                if v is not None:
                    raw.append((f"{name} {k.upper()}", v))
            for v in (lv.resistances if long else lv.supports)[:3]:
                raw.append((f"{name} {'resistance' if long else 'support'}", v))
        if ctx.mode != Mode.INTRADAY:
            if long and f.week52_high:
                raw.append(("52-week high", f.week52_high))
            if not long and f.week52_low:
                raw.append(("52-week low", f.week52_low))

    cands: list[dict[str, Any]] = []
    obstacles: list[dict[str, Any]] = []
    seen: list[float] = []
    for label, price in sorted(raw, key=lambda x: sgn * (x[1] - entry)):
        if price is None or price <= 0 or sgn * (price - entry) <= 0:
            continue  # not in the profit direction (or a non-positive R multiple of a wide SHORT)
        if any(abs(price - s) / entry * 100 < DEDUPE_PCT for s in seen):
            continue
        seen.append(price)
        rr = abs(price - entry) / risk
        item = {"label": label, "price": r2(price), "rr": r2(rr), "distance_pct": r2((price - entry) / entry * 100)}
        if rr + 1e-9 >= min_rr:
            cands.append(item)
        else:
            obstacles.append(item)
    for i, c in enumerate(cands):
        c["index"] = i
    return cands, obstacles


def rule_targets(cands: list[dict[str, Any]], setup: StrategySignal) -> tuple[int | None, int | None]:
    """Rule pick: strategy target_1 if it meets min R:R else nearest candidate; t2 farther by >= 0.5R."""
    if not cands:
        return None, None
    t1 = next((c["index"] for c in cands if c["label"].endswith("target_1")), 0)
    rr1 = float(cands[t1]["rr"])
    t2 = next((c["index"] for c in cands if c["label"].endswith("target_2") and float(c["rr"]) >= rr1 + MIN_T2_GAP_R), None)
    if t2 is None:
        t2 = next((c["index"] for c in cands if float(c["rr"]) >= rr1 + MIN_T2_GAP_R), None)
    return t1, t2


@register_agent
class ExitTargetAgent(SetupAgent):
    """Targets, trailing rule and max holding period among code-computed candidates."""

    name = "exit_target"
    description = "Targets from R:R and levels, trailing rule, max holding days; code enforces min R:R"
    OutputModel = ExitTargetOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> dict[str, Any] | None:
        setup = reference_setup(ctx)
        if setup is None:
            return None
        min_rr = min_rr_for(ctx, settings)
        cands, obstacles = target_candidates(ctx, setup, settings, min_rr)
        t1, t2 = rule_targets(cands, setup)
        lo, hi = holding_limits(ctx, settings)
        setup_days = setup.max_holding_days if setup.max_holding_days is not None else hi
        return {"setup": setup, "min_rr": min_rr, "cands": cands, "obstacles": obstacles, "t1": t1, "t2": t2,
                "trailing": default_trailing(ctx, setup, settings), "hold": (lo, hi),
                "days": max(lo, min(hi, setup_days))}

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        a = self._analyse(ctx, settings)
        if a is None:
            return {"symbol": ctx.symbol, "setup": None}
        s: StrategySignal = a["setup"]
        first = a["cands"][a["t1"]] if a["t1"] is not None else None
        blocking = [o for o in a["obstacles"]]
        if first is not None:
            blocking = [o for o in a["obstacles"] if float(o["rr"]) < float(first["rr"])]
        return {
            "symbol": ctx.symbol, "ltp": r2(current_price(ctx)),
            "setup": {"strategy": s.strategy, "direction": s.direction.value, "entry_price": r2(s.entry_price),
                      "stop_loss": r2(s.stop_loss), "risk_per_share": r2(abs(s.entry_price - s.stop_loss)),
                      "strategy_target_1": r2(s.target_1), "strategy_target_2": r2(s.target_2)},
            "min_rr": a["min_rr"],
            "target_candidates": a["cands"],
            "obstacles_below_min_rr": blocking,
            "allowed_trailing": [t.value for t in ALLOWED_TRAILING[ctx.mode]],
            "default_trailing": a["trailing"].value,
            "holding_days_limits": {"min": a["hold"][0], "max": a["hold"][1]},
            "rule_suggestion": {"target_1_index": a["t1"], "target_2_index": a["t2"],
                                "trailing_rule": a["trailing"].value, "max_holding_days": a["days"]},
        }

    def _extra(self, a: dict[str, Any], t1: int | None, t2: int | None, trailing: TrailingMethod,
               days: int, notes: list[str]) -> dict[str, Any]:
        c = a["cands"]
        return {
            "target_1": c[t1]["price"] if t1 is not None else None,
            "target_2": c[t2]["price"] if t2 is not None else None,
            "rr_1": c[t1]["rr"] if t1 is not None else None,
            "rr_2": c[t2]["rr"] if t2 is not None else None,
            "target_1_index": t1, "target_2_index": t2,
            "trailing_rule": trailing.value, "max_holding_days": days, "min_rr": a["min_rr"],
            "setup_strategy": a["setup"].strategy, "setup_direction": a["setup"].direction.value,
            "candidates": c, "obstacles": a["obstacles"], "validation_notes": notes,
        }

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        a = self._analyse(ctx, settings)
        if a is None:
            return self.make_report(ctx, key_findings=["no setup"], extra={"no_setup": True})
        if a["t1"] is None:
            return self.make_report(ctx, key_findings=[f"no target meets min R:R {a['min_rr']}"],
                                    extra=self._extra(a, None, None, a["trailing"], a["days"], ["no valid target"]))
        t1 = a["cands"][a["t1"]]
        near_obstacles = [o for o in a["obstacles"]]
        conf = 45 if near_obstacles else 60
        findings = [f"target_1 {t1['price']} ({t1['label']}, {t1['rr']}R)"]
        if a["t2"] is not None:
            t2 = a["cands"][a["t2"]]
            findings.append(f"target_2 {t2['price']} ({t2['label']}, {t2['rr']}R)")
        findings += [f"obstacle {o['label']} {o['price']} at {o['rr']}R" for o in near_obstacles[:2]]
        return self.make_report(
            ctx, signal=DIRECTION_SIGNAL[a["setup"].direction], confidence=conf, key_findings=findings,
            reasoning="Rule: strategy target_1 if it meets min R:R, else nearest qualifying level.",
            extra=self._extra(a, a["t1"], a["t2"], a["trailing"], a["days"], []),
        )

    def finalize(self, ctx: StockContext, parsed: ExitTargetOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        a = self._analyse(ctx, settings)
        if a is None:
            return det
        cands = a["cands"]
        setup: StrategySignal = a["setup"]
        notes: list[str] = []
        if not cands:
            rep = self.report_from_output(ctx, parsed, signal=Signal.NEUTRAL, confidence=0,
                                          key_findings=[f"no target meets min R:R {a['min_rr']}"],
                                          extra=self._extra(a, None, None, a["trailing"], a["days"],
                                                            ["no valid target candidate"]))
            return self.crosscheck(rep, det)

        # target_1: must exist (candidates all satisfy min R:R by construction).
        t1: int = parsed.target_1_index
        if not (0 <= t1 < len(cands)) or float(cands[t1]["rr"]) + 1e-9 < a["min_rr"]:
            notes.append(f"target_1_index {parsed.target_1_index} invalid or below min R:R {a['min_rr']}; "
                         f"using nearest valid target")
            t1 = 0
        rr1 = float(cands[t1]["rr"])

        # target_2: optional, must be beyond target_1 by >= MIN_T2_GAP_R.
        t2 = parsed.target_2_index
        if t2 is not None and not (0 <= t2 < len(cands) and float(cands[t2]["rr"]) >= rr1 + MIN_T2_GAP_R):
            notes.append(f"target_2_index {t2} invalid (must be >= {MIN_T2_GAP_R}R beyond target_1)")
            t2 = next((c["index"] for c in cands if float(c["rr"]) >= rr1 + MIN_T2_GAP_R), None)

        trailing = parsed.trailing_rule
        if trailing not in ALLOWED_TRAILING[ctx.mode]:
            notes.append(f"trailing {trailing.value} not allowed in {ctx.mode.value}; using {a['trailing'].value}")
            trailing = a["trailing"]

        lo, hi = a["hold"]
        days = parsed.max_holding_days
        if not lo <= days <= hi:
            clamped = max(lo, min(hi, days))
            notes.append(f"max_holding_days {days} outside [{lo}, {hi}]; clamped to {clamped}")
            days = clamped

        signal = self.restrict_signal(parsed.signal, setup)
        rep = self.report_from_output(ctx, parsed, signal=signal,
                                      extra=self._extra(a, t1, t2, trailing, days, notes))
        return self.crosscheck(rep, det)
