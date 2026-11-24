"""Volume & Liquidity Agent (section 4, agent 4).

Relative volume and volume spikes on the mode's timeframes, VWAP deviation (intraday),
delivery % vs its 20-day average (longer modes), bid/ask spread and average daily turnover.
``liquidity_ok`` is a code verdict (thresholds from ``settings.universe``): the LLM cannot
override it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import INTRADAY_TIMEFRAMES, SubAgent, current_price, mode_timeframes, primary_timeframe, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput
from agents.sub_agents._common import (
    Vote,
    confidence_from_score,
    score_votes,
    signal_from_score,
    strongest_facts,
)
from core.enums import Mode
from core.types import AgentReport, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: RVOL at/above which a bar counts as a volume spike.
SPIKE_RVOL = 1.5
#: Delivery % above this multiple of its 20-day average counts as accumulation/distribution.
DELIVERY_MULT = 1.15
#: Confidence cap when liquidity is insufficient.
ILLIQUID_CONF_CAP = 30
_CRORE = 1e7


class VolumeOutput(AgentOutput):
    """LLM answer of the volume & liquidity agent."""

    volume_confirms: bool = Field(default=False, description="true if volume confirms the price direction")
    liquidity_ok: bool = Field(default=True, description="true if liquidity is adequate (echo the provided verdict)")


@register_agent
class VolumeLiquidityAgent(SubAgent):
    """Volume confirmation and liquidity checks."""

    name = "volume_liquidity"
    description = "Relative volume, spikes, VWAP deviation, delivery %, spread, turnover"
    OutputModel = VolumeOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> tuple[dict[str, Any], list[Vote], bool, list[str]]:
        f = ctx.features
        price = current_price(ctx)
        prim = primary_timeframe(ctx, settings)
        votes: list[Vote] = []
        per_tf: dict[str, Any] = {}
        for name in mode_timeframes(ctx, settings):
            tf = f.tf(name) if f else None
            if tf is None:
                continue
            vol, avg, rvol = tf.v("volume"), tf.v("avg_volume_20"), tf.v("rvol")
            if rvol is None and vol is not None and avg:
                rvol = vol / avg
            o, c = tf.v("open"), tf.v("close")
            bar_dir = (1 if c > o else -1 if c < o else 0) if (o is not None and c is not None) else 0
            entry: dict[str, Any] = {"volume": r2(vol, 0), "avg_volume_20": r2(avg, 0), "rvol": r2(rvol),
                                     "last_bar_direction": {1: "UP", -1: "DOWN", 0: "FLAT"}[bar_dir]}
            w = 1.5 if name == prim else 1.0
            if rvol is not None:
                if rvol >= SPIKE_RVOL and bar_dir != 0:
                    votes.append(Vote(f"{name}: volume spike RVOL {rvol:.2f}x on an {'UP' if bar_dir > 0 else 'DOWN'} bar",
                                      bar_dir, w))
                elif rvol < 0.7:
                    votes.append(Vote(f"{name}: thin volume RVOL {rvol:.2f}x (weak participation)", 0, 0))
                else:
                    votes.append(Vote(f"{name}: normal volume RVOL {rvol:.2f}x", 0, 0))
            if name in INTRADAY_TIMEFRAMES and tf.v("vwap_dev_pct") is not None:
                dev = tf.v("vwap_dev_pct")
                entry["vwap_dev_pct"] = r2(dev)
                if abs(dev) >= 0.2:
                    votes.append(Vote(f"{name}: price {dev:+.2f}% from VWAP", 1 if dev > 0 else -1, 0.5))
            per_tf[name] = entry

        facts: dict[str, Any] = {}
        if f is not None:
            if f.rvol is not None:
                facts["rvol_stock"] = r2(f.rvol)
            if f.delivery_pct is not None:
                facts["delivery_pct"] = r2(f.delivery_pct)
                facts["delivery_pct_avg_20"] = r2(f.delivery_pct_avg_20)
                if ctx.mode != Mode.INTRADAY and f.delivery_pct_avg_20:
                    ratio = f.delivery_pct / f.delivery_pct_avg_20
                    facts["delivery_vs_avg_ratio"] = r2(ratio)
                    up = price is not None and f.prev_close is not None and price > f.prev_close
                    down = price is not None and f.prev_close is not None and price < f.prev_close
                    if ratio >= DELIVERY_MULT and up:
                        votes.append(Vote(f"delivery {f.delivery_pct:.1f}% is {ratio:.2f}x its 20d average on an up day (accumulation)", 1, 1.5))
                    elif ratio >= DELIVERY_MULT and down:
                        votes.append(Vote(f"delivery {f.delivery_pct:.1f}% is {ratio:.2f}x its 20d average on a down day (distribution)", -1, 1.5))
            if f.avg_daily_turnover_20 is not None:
                facts["avg_daily_turnover_cr"] = r2(f.avg_daily_turnover_20 / _CRORE)
            if f.avg_daily_volume_20 is not None:
                facts["avg_daily_volume_20"] = r2(f.avg_daily_volume_20, 0)
        spread = ctx.quote.spread_pct if ctx.quote is not None else None
        if spread is not None:
            facts["spread_pct"] = r2(spread, 3)

        # --- liquidity verdict (code-owned)
        min_turnover_cr = settings.universe.min_avg_daily_turnover_cr if settings else 5.0
        max_spread = settings.universe.max_spread_pct if settings else 0.3
        problems: list[str] = []
        turnover_cr = facts.get("avg_daily_turnover_cr")
        if turnover_cr is not None and turnover_cr < min_turnover_cr:
            problems.append(f"avg daily turnover {turnover_cr:.2f} cr < minimum {min_turnover_cr:.2f} cr")
        if spread is not None and spread > max_spread:
            problems.append(f"spread {spread:.3f}% > maximum {max_spread:.3f}%")
        liquidity_ok = not problems
        for p in problems:
            votes.append(Vote(f"LIQUIDITY: {p}", 0, 0))

        payload = {
            "symbol": ctx.symbol, "ltp": r2(price), "primary_timeframe": prim, "timeframes": per_tf, **facts,
            "liquidity_thresholds": {"min_avg_daily_turnover_cr": min_turnover_cr, "max_spread_pct": max_spread},
            "liquidity_ok": liquidity_ok, "liquidity_problems": problems,
            "facts": [v.fact for v in votes],
        }
        return payload, votes, liquidity_ok, problems

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return self._analyse(ctx, settings)[0]

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if not mode_timeframes(ctx, settings):
            return self.make_report(ctx, key_findings=["no volume data for the mode's timeframes"],
                                    extra={"no_data": True, "liquidity_ok": None})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        _, votes, liquidity_ok, problems = self._analyse(ctx, settings)
        score = score_votes(votes)
        signal = signal_from_score(score, threshold=0.25)
        conf = confidence_from_score(score, signal, base=30, span=50, cap=80)
        if not liquidity_ok:
            conf = min(conf, ILLIQUID_CONF_CAP)
        findings = problems + strongest_facts(votes, signal)
        return self.make_report(
            ctx, signal=signal, confidence=conf, key_findings=findings,
            reasoning=f"Volume rule score {score:+.2f}; liquidity {'OK' if liquidity_ok else 'INSUFFICIENT'}.",
            extra={"rule_score": round(score, 3), "liquidity_ok": liquidity_ok,
                   "volume_confirms": signal.value != "NEUTRAL", "liquidity_problems": problems},
        )

    def finalize(self, ctx: StockContext, parsed: VolumeOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        liquidity_ok = bool(det.extra.get("liquidity_ok", True))
        notes: list[str] = []
        if parsed.liquidity_ok != liquidity_ok:
            notes.append(f"LLM liquidity_ok={parsed.liquidity_ok} overridden by code verdict {liquidity_ok}")
        conf = parsed.confidence if liquidity_ok else min(parsed.confidence, ILLIQUID_CONF_CAP)
        findings = list(det.extra.get("liquidity_problems") or []) + list(parsed.key_findings)
        rep = self.report_from_output(ctx, parsed, confidence=conf, key_findings=findings, extra={
            "liquidity_ok": liquidity_ok, "volume_confirms": parsed.volume_confirms,
            "liquidity_problems": det.extra.get("liquidity_problems"), "rule_score": det.extra.get("rule_score"),
            "validation_notes": notes,
        })
        return self.crosscheck(rep, det)
