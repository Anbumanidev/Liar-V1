"""Fundamental Agent (section 4, agent 1). Always outputs TRADEABLE true/false.

* INTRADAY / BTST: quality filter only (ASM/GSM, promoter pledge, results today, corporate
  action today, high debt). Signal stays NEUTRAL unless the quality picture is clear.
* SWING / POSITIONAL: deterministic ``fundamental_score`` 0-100 from growth, profitability,
  valuation vs sector PE, debt and promoter-holding trend, interpreted by the LLM.

TRADEABLE is a **code verdict**: the LLM's opinion is recorded but can never flip it
(``finalize`` always uses the deterministic value, and FAILED reports keep it too).

Units follow ``core.types`` conventions: growth / margins / ROE / holdings in percent units,
``debt_to_equity`` as a plain ratio (0.5 == 50 %).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agents.base import SubAgent, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput, truncate_list
from core.enums import Mode, Signal
from core.types import AgentReport, FundamentalSnapshot, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

PLEDGE_LIMIT_PCT = 25.0
HIGH_DEBT_RATIO = 2.0
#: Sectors where a high debt/equity ratio is structural (lenders) and not a red flag.
_FINANCIAL_HINTS = ("bank", "financ", "nbfc", "insurance", "capital markets")
#: Minimum share of score weight that must be computable for a directional signal.
MIN_COVERAGE = 0.4
QUALITY_MODES = frozenset({Mode.INTRADAY, Mode.BTST})
#: Confidence cap for a directional signal in quality-filter modes.
QUALITY_MODE_CONF_CAP = 60


class FundamentalOutput(AgentOutput):
    """LLM answer of the fundamental agent."""

    tradeable: bool = Field(description="Echo the provided code verdict 'tradeable' (true/false)")
    concerns: list[str] = Field(default_factory=list, description="Up to 4 fundamental concerns from the facts")


def is_financial(snap: FundamentalSnapshot, sector: str | None) -> bool:
    """True for banks / NBFCs / insurers (debt ratio not meaningful)."""
    txt = " ".join(x for x in (sector, snap.sector, snap.industry) if x).lower()
    return any(h in txt for h in _FINANCIAL_HINTS)


def quality_flags(snap: FundamentalSnapshot, sector: str | None) -> tuple[list[str], list[str]]:
    """(blocking flags -> TRADEABLE=false, warnings)."""
    blocking: list[str] = []
    warnings: list[str] = []
    if snap.in_asm:
        blocking.append("stock is in the ASM (additional surveillance) list")
    if snap.in_gsm:
        blocking.append("stock is in the GSM (graded surveillance) list")
    if snap.promoter_pledge_pct is not None and snap.promoter_pledge_pct > PLEDGE_LIMIT_PCT:
        blocking.append(f"promoter pledge {snap.promoter_pledge_pct:.1f}% > {PLEDGE_LIMIT_PCT:.0f}%")
    if snap.results_today:
        blocking.append("results announced today")
    if snap.corporate_action_today:
        blocking.append("corporate action today")
    if (snap.debt_to_equity is not None and snap.debt_to_equity > HIGH_DEBT_RATIO
            and not is_financial(snap, sector)):
        warnings.append(f"high debt: debt/equity {snap.debt_to_equity:.2f} > {HIGH_DEBT_RATIO:.1f}")
    return blocking, warnings


def _band(value: float | None, bands: list[tuple[float, float]], floor: float) -> float | None:
    """Points for the first band whose threshold ``value`` reaches (bands descending)."""
    if value is None:
        return None
    for threshold, pts in bands:
        if value >= threshold:
            return pts
    return floor


def fundamental_score(snap: FundamentalSnapshot, sector: str | None) -> tuple[float | None, float, dict[str, Any]]:
    """Deterministic 0-100 score; returns (score or None, coverage 0-1, components).

    Components (max points): earnings growth 15, revenue growth 10, ROE 15, profit margin 10,
    valuation vs sector PE 20, debt 15, promoter holding 15. Missing inputs are excluded and
    the score is normalised over the available maximum; ``coverage`` = available / 100.
    """
    comps: dict[str, tuple[float | None, float]] = {
        "earnings_growth": (_band(snap.earnings_growth, [(20, 15), (10, 11), (0, 6)], 0), 15),
        "revenue_growth": (_band(snap.revenue_growth, [(15, 10), (8, 7), (0, 4)], 0), 10),
        "roe": (_band(snap.roe, [(20, 15), (15, 12), (10, 7)], 2), 15),
        "profit_margin": (_band(snap.profit_margin, [(15, 10), (8, 7), (0, 3)], 0), 10),
    }
    if snap.pe is not None and snap.sector_pe:
        if snap.pe <= 0:
            val_pts: float | None = 0.0
        else:
            ratio = snap.pe / snap.sector_pe
            val_pts = 20 if ratio <= 0.8 else 15 if ratio <= 1.0 else 10 if ratio <= 1.3 else 5 if ratio <= 1.8 else 0
    else:
        val_pts = None
    comps["valuation_vs_sector"] = (val_pts, 20)
    if snap.debt_to_equity is not None and not is_financial(snap, sector):
        de = snap.debt_to_equity
        debt_pts: float | None = 15 if de <= 0.3 else 10 if de <= 1.0 else 5 if de <= 2.0 else 0
    else:
        debt_pts = None
    comps["debt"] = (debt_pts, 15)
    if snap.promoter_holding_change is not None or snap.promoter_holding is not None:
        pts = 0.0
        ch = snap.promoter_holding_change
        if ch is not None:
            pts += 10 if ch > 0 else 6 if ch == 0 else 0
        else:
            pts += 5
        if snap.promoter_holding is not None:
            pts += 5 if snap.promoter_holding >= 50 else 3 if snap.promoter_holding >= 30 else 0
        else:
            pts += 2
        comps["promoter"] = (pts, 15)
    else:
        comps["promoter"] = (None, 15)

    earned = sum(p for p, _ in comps.values() if p is not None)
    available = sum(m for p, m in comps.values() if p is not None)
    coverage = available / 100.0
    score = round(earned / available * 100, 1) if available > 0 else None
    detail = {k: {"points": r2(p, 1), "max": m} for k, (p, m) in comps.items()}
    return score, round(coverage, 2), detail


@register_agent
class FundamentalAgent(SubAgent):
    """Quality filter (INTRADAY/BTST) or fundamental score (SWING/POSITIONAL)."""

    name = "fundamental"
    description = "Quality filter / fundamental score; always outputs TRADEABLE"
    OutputModel = FundamentalOutput

    def _analyse(self, ctx: StockContext) -> dict[str, Any]:
        snap = ctx.fundamentals
        if snap is None:
            return {"available": False, "tradeable": True, "blocking_flags": [], "warnings": ["no fundamentals data"],
                    "fundamental_score": None, "coverage": 0.0, "mode_role": self._role(ctx.mode)}
        blocking, warnings = quality_flags(snap, ctx.sector)
        out: dict[str, Any] = {
            "available": True,
            "mode_role": self._role(ctx.mode),
            "tradeable": not blocking,
            "blocking_flags": blocking,
            "warnings": warnings,
            "quality": {
                "in_asm": snap.in_asm, "in_gsm": snap.in_gsm,
                "promoter_pledge_pct": r2(snap.promoter_pledge_pct), "pledge_limit_pct": PLEDGE_LIMIT_PCT,
                "results_today": snap.results_today, "corporate_action_today": snap.corporate_action_today,
                "debt_to_equity": r2(snap.debt_to_equity), "financial_sector": is_financial(snap, ctx.sector),
            },
            "sector": ctx.sector or snap.sector,
            "fundamental_score": None,
            "coverage": None,
        }
        if ctx.mode not in QUALITY_MODES:
            score, coverage, comps = fundamental_score(snap, ctx.sector)
            pe_vs = r2(snap.pe / snap.sector_pe, 2) if (snap.pe and snap.sector_pe) else None
            out.update({
                "fundamental_score": score,
                "coverage": coverage,
                "score_components": comps,
                "metrics": {
                    "market_cap_cr": r2(snap.market_cap / 1e7) if snap.market_cap else None,
                    "pe": r2(snap.pe), "sector_pe": r2(snap.sector_pe), "pe_to_sector_pe": pe_vs,
                    "pb": r2(snap.pb), "roe_pct": r2(snap.roe), "roce_pct": r2(snap.roce),
                    "revenue_growth_pct": r2(snap.revenue_growth), "earnings_growth_pct": r2(snap.earnings_growth),
                    "profit_margin_pct": r2(snap.profit_margin), "promoter_holding_pct": r2(snap.promoter_holding),
                    "promoter_holding_change_pct": r2(snap.promoter_holding_change),
                },
            })
        return out

    @staticmethod
    def _role(mode: Mode) -> str:
        return "QUALITY_FILTER" if mode in QUALITY_MODES else "SCORED_SIGNAL"

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return {"symbol": ctx.symbol, **self._analyse(ctx)}

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if ctx.fundamentals is None:
            return self.make_report(ctx, tradeable=True, key_findings=["no fundamentals data: quality filter not applied"],
                                    extra={"no_data": True, "blocking_flags": [], "fundamental_score": None})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        a = self._analyse(ctx)
        tradeable = bool(a["tradeable"])
        signal, conf = Signal.NEUTRAL, 50
        score = a.get("fundamental_score")
        coverage = a.get("coverage") or 0.0
        if ctx.mode in QUALITY_MODES:
            if not tradeable:
                signal, conf = Signal.BEARISH, QUALITY_MODE_CONF_CAP
            reasoning = "Quality filter: " + ("FAILED - " + "; ".join(a["blocking_flags"]) if not tradeable else "passed")
        else:
            if score is not None and coverage >= MIN_COVERAGE:
                if score >= 65:
                    signal = Signal.BULLISH
                elif score <= 35:
                    signal = Signal.BEARISH
                conf = int(min(85, 35 + abs(score - 50) * 1.2) * min(1.0, coverage / 0.8))
            else:
                conf = 20
            if not tradeable:
                signal, conf = Signal.BEARISH, max(conf, QUALITY_MODE_CONF_CAP)
            reasoning = (f"Fundamental score {score} (coverage {coverage:.0%})" if score is not None
                         else "Fundamental score not computable") + (
                "; blocked: " + "; ".join(a["blocking_flags"]) if not tradeable else "")
        findings = list(a["blocking_flags"]) + list(a["warnings"])
        if score is not None:
            findings.append(f"fundamental score {score:.1f}/100 (coverage {coverage:.0%})")
        return self.make_report(
            ctx, signal=signal, confidence=conf, tradeable=tradeable, key_findings=findings or ["no quality red flags"],
            reasoning=reasoning,
            extra={"blocking_flags": a["blocking_flags"], "warnings": a["warnings"],
                   "fundamental_score": score, "coverage": a.get("coverage"), "role": a["mode_role"]},
        )

    def finalize(self, ctx: StockContext, parsed: FundamentalOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        tradeable = bool(det.tradeable) if det.tradeable is not None else True
        notes: list[str] = []
        if parsed.tradeable != tradeable:
            notes.append(f"LLM tradeable={parsed.tradeable} ignored; code verdict tradeable={tradeable}")
        signal, conf = parsed.signal, parsed.confidence
        if ctx.mode in QUALITY_MODES and signal != Signal.NEUTRAL:
            conf = min(conf, QUALITY_MODE_CONF_CAP)
        if not tradeable and signal == Signal.BULLISH:
            notes.append("not tradeable: BULLISH vote replaced by NEUTRAL")
            signal = Signal.NEUTRAL
        findings = list(det.extra.get("blocking_flags") or []) + list(parsed.key_findings)
        rep = self.report_from_output(ctx, parsed, signal=signal, confidence=conf, tradeable=tradeable,
                                      key_findings=findings, extra={
                                          "blocking_flags": det.extra.get("blocking_flags"),
                                          "warnings": det.extra.get("warnings"),
                                          "fundamental_score": det.extra.get("fundamental_score"),
                                          "coverage": det.extra.get("coverage"),
                                          "role": det.extra.get("role"),
                                          "concerns": truncate_list(parsed.concerns, 4, 160),
                                          "llm_tradeable": parsed.tradeable,
                                          "validation_notes": notes,
                                      })
        return self.crosscheck(rep, det)
