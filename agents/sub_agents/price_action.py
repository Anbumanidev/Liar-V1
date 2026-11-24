"""Candlestick & Price Action Agent (section 4, agent 3).

TA-Lib / fallback candlestick patterns on the mode's timeframes, opening range (intraday),
gaps, inside bars, trend structure (HH/HL vs LH/LL), position in the day range and, for
longer modes, distance to the 52-week high/low.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import SubAgent, current_price, mode_timeframes, pct_diff, primary_timeframe, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput, truncate_text
from agents.sub_agents._common import (
    Vote,
    confidence_from_score,
    levels_summary,
    score_votes,
    signal_from_score,
    strongest_facts,
)
from core.enums import Mode, Signal
from core.types import AgentReport, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Max patterns per timeframe shown to the LLM (most recent first).
MAX_PATTERNS = 5


class PriceActionOutput(AgentOutput):
    """LLM answer of the price-action agent."""

    key_pattern: str | None = Field(default=None, description="Name of the most important pattern listed in the facts, or null")


@register_agent
class PriceActionAgent(SubAgent):
    """Candlestick patterns and price structure."""

    name = "price_action"
    description = "Candlestick patterns, opening range, gaps, inside bars, trend structure"
    OutputModel = PriceActionOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> tuple[dict[str, Any], list[Vote], list[str]]:
        f = ctx.features
        price = current_price(ctx)
        prim = primary_timeframe(ctx, settings)
        votes: list[Vote] = []
        per_tf: dict[str, Any] = {}
        pattern_names: list[str] = []
        for name in mode_timeframes(ctx, settings):
            tf = f.tf(name) if f else None
            if tf is None:
                continue
            pats = sorted(tf.patterns, key=lambda p: p.bar_time, reverse=True)[:MAX_PATTERNS]
            w = 1.5 if name == prim else 1.0
            for p in pats:
                sign = 1 if p.direction == Signal.BULLISH else -1 if p.direction == Signal.BEARISH else 0
                votes.append(Vote(f"{name}: {p.name} ({p.direction.value}, strength {p.strength}) on bar "
                                  f"{p.bar_time.isoformat(timespec='minutes')}", sign, w))
                pattern_names.append(p.name)
            ts = tf.v("trend_structure")
            if ts is not None and ts != 0:
                votes.append(Vote(f"{name}: trend structure {'higher highs / higher lows' if ts > 0 else 'lower highs / lower lows'}",
                                  1 if ts > 0 else -1, w))
            if tf.v("inside_bar") == 1:
                votes.append(Vote(f"{name}: last closed bar is an inside bar (compression)", 0, 0))
            per_tf[name] = {
                "patterns": [{"name": p.name, "direction": p.direction.value, "strength": p.strength,
                              "bar_time": p.bar_time.isoformat(timespec="minutes")} for p in pats],
                "trend_structure": tf.v("trend_structure"),
                "inside_bar": tf.v("inside_bar"),
                "last_bar": {k: r2(tf.v(k)) for k in ("open", "high", "low", "close") if tf.v(k) is not None},
                "levels": levels_summary(tf.levels, price),
            }

        extra_facts: dict[str, Any] = {}
        if f is not None and price is not None:
            if f.gap_pct is not None:
                extra_facts["gap_pct"] = r2(f.gap_pct)
                if abs(f.gap_pct) >= 1.0:
                    votes.append(Vote(f"gap {'up' if f.gap_pct > 0 else 'down'} {f.gap_pct:+.2f}% at the open", 0, 0))
            if ctx.mode == Mode.INTRADAY and f.opening_range_high is not None and f.opening_range_low is not None:
                orh, orl = f.opening_range_high, f.opening_range_low
                pos = "ABOVE" if price > orh else "BELOW" if price < orl else "INSIDE"
                extra_facts["opening_range"] = {"high": r2(orh), "low": r2(orl), "ltp_position": pos,
                                                "distance_to_high_pct": pct_diff(price, orh),
                                                "distance_to_low_pct": pct_diff(price, orl)}
                if pos != "INSIDE":
                    votes.append(Vote(f"ltp {price:.2f} is {pos} the opening range {orl:.2f}-{orh:.2f}",
                                      1 if pos == "ABOVE" else -1, 1.5))
                else:
                    votes.append(Vote(f"ltp {price:.2f} is INSIDE the opening range {orl:.2f}-{orh:.2f}", 0, 0))
            if f.day_high is not None and f.day_low is not None and f.day_high > f.day_low:
                loc = (price - f.day_low) / (f.day_high - f.day_low) * 100
                extra_facts["day_range"] = {"high": r2(f.day_high), "low": r2(f.day_low),
                                            "ltp_position_pct_of_range": r2(loc, 1)}
                if loc >= 80:
                    votes.append(Vote(f"ltp in the top {100 - loc:.0f}% of the day's range (close near high)", 1))
                elif loc <= 20:
                    votes.append(Vote(f"ltp in the bottom {loc:.0f}% of the day's range (close near low)", -1))
            if ctx.mode in (Mode.SWING, Mode.POSITIONAL, Mode.BTST):
                if f.week52_high:
                    d = pct_diff(price, f.week52_high)
                    extra_facts["distance_to_52w_high_pct"] = d
                    if d is not None and d >= -3:
                        votes.append(Vote(f"ltp within {abs(d):.2f}% of the 52-week high {f.week52_high:.2f}", 1))
                if f.week52_low:
                    d = pct_diff(price, f.week52_low)
                    extra_facts["distance_to_52w_low_pct"] = d
                    if d is not None and d <= 3:
                        votes.append(Vote(f"ltp within {abs(d):.2f}% of the 52-week low {f.week52_low:.2f}", -1))

        payload = {
            "symbol": ctx.symbol, "ltp": r2(price), "primary_timeframe": prim,
            "timeframes": per_tf, **extra_facts,
            "facts": [v.fact for v in votes],
            "bullish_points": sum(1 for v in votes if v.vote > 0),
            "bearish_points": sum(1 for v in votes if v.vote < 0),
        }
        return payload, votes, pattern_names

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return self._analyse(ctx, settings)[0]

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if not mode_timeframes(ctx, settings):
            return self.make_report(ctx, key_findings=["no price data for the mode's timeframes"],
                                    extra={"no_data": True})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        _, votes, pattern_names = self._analyse(ctx, settings)
        score = score_votes(votes)
        signal = signal_from_score(score)
        conf = confidence_from_score(score, signal, base=30, span=50, cap=85)
        return self.make_report(
            ctx, signal=signal, confidence=conf, key_findings=strongest_facts(votes, signal),
            reasoning=f"Price-action rule score {score:+.2f} from {len(votes)} observations.",
            extra={"rule_score": round(score, 3), "key_pattern": pattern_names[0] if pattern_names else None},
        )

    def finalize(self, ctx: StockContext, parsed: PriceActionOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        _, _, pattern_names = self._analyse(ctx, settings)
        key_pattern = parsed.key_pattern
        notes: list[str] = []
        if key_pattern and key_pattern not in pattern_names:
            notes.append(f"LLM key_pattern {truncate_text(key_pattern, 60)!r} not among detected patterns; dropped")
            key_pattern = None
        rep = self.report_from_output(ctx, parsed, extra={"key_pattern": key_pattern,
                                                          "rule_score": det.extra.get("rule_score"),
                                                          "validation_notes": notes})
        return self.crosscheck(rep, det)
