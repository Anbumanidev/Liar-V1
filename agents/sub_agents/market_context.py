"""Market Context Agent (section 4, agent 5).

NIFTY 50 / BANK NIFTY trend, India VIX vs its 20-day average, sector strength,
advance/decline, FII/DII flows, global cues and the stock's relative strength vs NIFTY.
``risk_off`` (VIX elevated) is a code verdict the LLM cannot clear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import SubAgent, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput
from agents.sub_agents._common import (
    Vote,
    confidence_from_score,
    score_votes,
    signal_from_score,
    strongest_facts,
)
from core.enums import Signal
from core.types import AgentReport, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Confidence multiplier when VIX is elevated (risk-off).
RISK_OFF_CONF_MULT = 0.75

_TREND_VOTE = {Signal.BULLISH: 1, Signal.BEARISH: -1, Signal.NEUTRAL: 0}


class MarketOutput(AgentOutput):
    """LLM answer of the market-context agent."""

    risk_off: bool = Field(default=False, description="true if the broad market is risk-off")


@register_agent
class MarketContextAgent(SubAgent):
    """Broad market and sector backdrop."""

    name = "market_context"
    description = "NIFTY/BANKNIFTY trend, India VIX, sector strength, breadth, FII/DII, global cues"
    OutputModel = MarketOutput

    def _analyse(self, ctx: StockContext) -> tuple[dict[str, Any], list[Vote], bool]:
        m = ctx.market
        votes: list[Vote] = []
        if m is None:
            return {"available": False}, votes, False
        votes.append(Vote(f"NIFTY 50 trend {m.nifty_trend.value}"
                          + (f" ({m.nifty_change_pct:+.2f}% today)" if m.nifty_change_pct is not None else ""),
                          _TREND_VOTE[m.nifty_trend], 2.0))
        votes.append(Vote(f"BANK NIFTY trend {m.banknifty_trend.value}"
                          + (f" ({m.banknifty_change_pct:+.2f}% today)" if m.banknifty_change_pct is not None else ""),
                          _TREND_VOTE[m.banknifty_trend], 0.75))
        if m.advance_decline_ratio is not None:
            adr = m.advance_decline_ratio
            vote = 1 if adr >= 1.2 else -1 if adr <= 0.8 else 0
            votes.append(Vote(f"advance/decline ratio {adr:.2f} ({'broad buying' if vote > 0 else 'broad selling' if vote < 0 else 'mixed'})", vote))
        if m.fii_net_cr is not None:
            votes.append(Vote(f"FII net {m.fii_net_cr:+.0f} cr", 1 if m.fii_net_cr > 0 else -1 if m.fii_net_cr < 0 else 0, 0.5))
        if m.dii_net_cr is not None:
            votes.append(Vote(f"DII net {m.dii_net_cr:+.0f} cr", 0, 0))
        sector_val: float | None = None
        if ctx.sector and ctx.sector in m.sector_strength:
            sector_val = m.sector_strength[ctx.sector]
            vote = 1 if sector_val >= 0.5 else -1 if sector_val <= -0.5 else 0
            votes.append(Vote(f"sector {ctx.sector} strength {sector_val:+.2f}", vote, 1.5))
        if m.global_cues:
            avg = sum(m.global_cues.values()) / len(m.global_cues)
            vote = 1 if avg >= 0.3 else -1 if avg <= -0.3 else 0
            votes.append(Vote(f"global cues average {avg:+.2f}% over {len(m.global_cues)} markets", vote, 0.5))
        rs = ctx.features.rs_vs_nifty if ctx.features else None
        if rs is not None:
            vote = 1 if rs >= 1.0 else -1 if rs <= -1.0 else 0
            votes.append(Vote(f"stock relative strength vs NIFTY {rs:+.2f}%", vote, 1.0))
        risk_off = bool(m.vix_elevated)
        if m.india_vix is not None:
            vix_txt = f"India VIX {m.india_vix:.2f}"
            if m.vix_20d_avg:
                vix_txt += f" vs 20d avg {m.vix_20d_avg:.2f} ({(m.india_vix - m.vix_20d_avg) / m.vix_20d_avg * 100:+.1f}%)"
            votes.append(Vote(vix_txt + (" - ELEVATED (risk-off)" if risk_off else ""), 0, 0))

        ranked = sorted(m.sector_strength.items(), key=lambda kv: kv[1], reverse=True)
        payload = {
            "available": True,
            "as_of": m.as_of.isoformat(timespec="minutes"),
            "nifty_trend": m.nifty_trend.value, "nifty_change_pct": r2(m.nifty_change_pct),
            "banknifty_trend": m.banknifty_trend.value, "banknifty_change_pct": r2(m.banknifty_change_pct),
            "india_vix": r2(m.india_vix), "vix_20d_avg": r2(m.vix_20d_avg), "vix_elevated": m.vix_elevated,
            "advance_decline_ratio": r2(m.advance_decline_ratio),
            "fii_net_cr": r2(m.fii_net_cr), "dii_net_cr": r2(m.dii_net_cr),
            "stock_sector": ctx.sector, "stock_sector_strength": r2(sector_val),
            "strongest_sectors": [{"sector": k, "strength": r2(v)} for k, v in ranked[:3]],
            "weakest_sectors": [{"sector": k, "strength": r2(v)} for k, v in ranked[-3:][::-1]],
            "global_cues_pct": {k: r2(v) for k, v in m.global_cues.items()},
            "rs_vs_nifty_pct": r2(rs),
            "market_regime": m.regime.value,
            "notes": list(m.notes)[:5],
            "risk_off": risk_off,
            "facts": [v.fact for v in votes],
        }
        return payload, votes, risk_off

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return self._analyse(ctx)[0]

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if ctx.market is None:
            return self.make_report(ctx, key_findings=["no market context available"],
                                    extra={"no_data": True, "risk_off": None})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        _, votes, risk_off = self._analyse(ctx)
        score = score_votes(votes)
        signal = signal_from_score(score)
        conf = confidence_from_score(score, signal, base=30, span=55, cap=85)
        if risk_off:
            conf = int(conf * RISK_OFF_CONF_MULT)
        return self.make_report(
            ctx, signal=signal, confidence=conf, key_findings=strongest_facts(votes, signal),
            reasoning=f"Market rule score {score:+.2f}" + ("; VIX elevated (risk-off)." if risk_off else "."),
            extra={"rule_score": round(score, 3), "risk_off": risk_off},
        )

    def finalize(self, ctx: StockContext, parsed: MarketOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        code_risk_off = bool(det.extra.get("risk_off"))
        risk_off = code_risk_off or parsed.risk_off
        conf = parsed.confidence
        if code_risk_off and parsed.signal == Signal.BULLISH:
            conf = int(conf * RISK_OFF_CONF_MULT)
        rep = self.report_from_output(ctx, parsed, confidence=conf, extra={
            "risk_off": risk_off, "rule_score": det.extra.get("rule_score"),
        })
        return self.crosscheck(rep, det)
