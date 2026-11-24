"""Technical Analysis Agent (section 4, agent 2).

EMA 9/21/50/200, VWAP (intraday timeframes), RSI, MACD, Supertrend, ADX/DI, Bollinger,
ATR, pivots/CPR and support/resistance on every timeframe of the mode. All comparisons
are pre-computed as fact sentences; the LLM only interprets them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from agents.base import INTRADAY_TIMEFRAMES, SubAgent, current_price, mode_timeframes, primary_timeframe, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput
from agents.sub_agents._common import (
    Vote,
    compare_fact,
    confidence_from_score,
    levels_summary,
    score_votes,
    signal_from_score,
    strongest_facts,
    tf_values,
)
from core.enums import Signal
from core.types import AgentReport, StockContext, TimeframeFeatures

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

TECH_KEYS: tuple[str, ...] = (
    "close", "ema_9", "ema_20", "ema_21", "ema_50", "ema_200", "vwap", "vwap_dev_pct", "rsi_14", "macd",
    "macd_signal", "macd_hist", "supertrend", "supertrend_dir", "adx_14", "plus_di", "minus_di", "bb_upper",
    "bb_mid", "bb_lower", "bb_width_pct", "bb_squeeze", "atr_14", "atr_pct", "atr_expansion",
    "rsi_bull_div", "rsi_bear_div",
)


class TechnicalOutput(AgentOutput):
    """LLM answer of the technical agent."""

    trend_alignment: Literal["ALIGNED_UP", "ALIGNED_DOWN", "MIXED"] = Field(
        default="MIXED", description="Do the mode's timeframes agree? ALIGNED_UP, ALIGNED_DOWN or MIXED")

    @field_validator("trend_alignment", mode="before")
    @classmethod
    def _v_align(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


def timeframe_votes(tf: TimeframeFeatures) -> list[Vote]:
    """Rule-based votes for one timeframe (pure)."""
    v = tf.v
    label = tf.timeframe
    votes: list[Vote] = []
    close = v("close")

    for a_name, a, b_name, b in (
        ("close", close, "EMA21", v("ema_21")),
        ("EMA9", v("ema_9"), "EMA21", v("ema_21")),
        ("EMA50", v("ema_50"), "EMA200", v("ema_200")),
        ("close", close, "EMA200", v("ema_200")),
    ):
        cmp = compare_fact(a_name, a, b_name, b)
        if cmp:
            votes.append(Vote(f"{label}: {cmp[0]}", cmp[1]))
    if label in INTRADAY_TIMEFRAMES:
        cmp = compare_fact("close", close, "VWAP", v("vwap"))
        if cmp:
            votes.append(Vote(f"{label}: {cmp[0]}", cmp[1]))

    rsi = v("rsi_14")
    if rsi is not None:
        if rsi >= 60:
            votes.append(Vote(f"{label}: RSI14 {rsi:.1f} >= 60 (bullish momentum)", 1))
        elif rsi <= 40:
            votes.append(Vote(f"{label}: RSI14 {rsi:.1f} <= 40 (bearish momentum)", -1))
        else:
            votes.append(Vote(f"{label}: RSI14 {rsi:.1f} neutral (40-60)", 0))
        if rsi >= 75:
            votes.append(Vote(f"{label}: RSI14 {rsi:.1f} overbought (>= 75)", 0, 0))
        elif rsi <= 25:
            votes.append(Vote(f"{label}: RSI14 {rsi:.1f} oversold (<= 25)", 0, 0))

    hist = v("macd_hist")
    if hist is not None:
        votes.append(Vote(f"{label}: MACD histogram {hist:+.3f} ({'positive' if hist > 0 else 'negative' if hist < 0 else 'flat'})",
                          1 if hist > 0 else -1 if hist < 0 else 0))
    st = v("supertrend_dir")
    if st is not None and st != 0:
        votes.append(Vote(f"{label}: Supertrend {'UP' if st > 0 else 'DOWN'}", 1 if st > 0 else -1))

    adx, pdi, mdi = v("adx_14"), v("plus_di"), v("minus_di")
    if adx is not None:
        if adx > 25 and pdi is not None and mdi is not None:
            up = pdi > mdi
            votes.append(Vote(f"{label}: ADX {adx:.1f} > 25 strong trend, +DI {pdi:.1f} {'>' if up else '<'} -DI {mdi:.1f}",
                              1 if up else -1, 1.5))
        elif adx < 20:
            votes.append(Vote(f"{label}: ADX {adx:.1f} < 20 (weak / no trend)", 0, 0))
        else:
            votes.append(Vote(f"{label}: ADX {adx:.1f} (developing trend)", 0, 0))

    if v("bb_squeeze") == 1:
        votes.append(Vote(f"{label}: Bollinger squeeze (bandwidth {v('bb_width_pct', 0):.2f}%)", 0, 0))
    if close is not None and v("bb_upper") is not None and close > v("bb_upper"):
        votes.append(Vote(f"{label}: close above upper Bollinger band", 0, 0))
    if close is not None and v("bb_lower") is not None and close < v("bb_lower"):
        votes.append(Vote(f"{label}: close below lower Bollinger band", 0, 0))
    if v("rsi_bull_div") == 1:
        votes.append(Vote(f"{label}: bullish RSI divergence", 1))
    if v("rsi_bear_div") == 1:
        votes.append(Vote(f"{label}: bearish RSI divergence", -1))
    exp = v("atr_expansion")
    if exp is not None and exp >= 1.5:
        votes.append(Vote(f"{label}: ATR expanding ({exp:.2f}x 20-bar average)", 0, 0))
    return votes


@register_agent
class TechnicalAgent(SubAgent):
    """Multi-timeframe indicator interpretation."""

    name = "technical"
    description = "EMA/VWAP/RSI/MACD/Supertrend/ADX/Bollinger/ATR/pivots across the mode's timeframes"
    OutputModel = TechnicalOutput

    def _analyse(self, ctx: StockContext, settings: "Settings | None") -> tuple[dict[str, Any], float, list[Vote], dict[str, float]]:
        tfs = mode_timeframes(ctx, settings)
        prim = primary_timeframe(ctx, settings)
        price = current_price(ctx)
        per_tf: dict[str, Any] = {}
        all_votes: list[Vote] = []
        tf_scores: dict[str, float] = {}
        num = den = 0.0
        for name in tfs:
            tf = ctx.features.tf(name) if ctx.features else None
            if tf is None:
                continue
            votes = timeframe_votes(tf)
            s = score_votes(votes)
            tf_scores[name] = round(s, 3)
            w = 2.0 if name == prim else 1.0
            if any(vv.weight > 0 for vv in votes):
                num += s * w
                den += w
            all_votes.extend(votes)
            per_tf[name] = {
                "bars": tf.bars,
                "regime": tf.regime.value,
                "values": tf_values(tf, TECH_KEYS),
                "levels": levels_summary(tf.levels, price),
                "facts": [vv.fact for vv in votes],
                "bullish_points": sum(1 for vv in votes if vv.vote > 0),
                "bearish_points": sum(1 for vv in votes if vv.vote < 0),
            }
        score = num / den if den > 0 else 0.0
        payload = {
            "symbol": ctx.symbol,
            "ltp": r2(price),
            "primary_timeframe": prim,
            "timeframes_analysed": list(per_tf),
            "regime_primary": (ctx.features.regime.value if ctx.features else ctx.regime.value),
            "atr_primary": r2(ctx.features.atr) if ctx.features else None,
            "timeframes": per_tf,
        }
        return payload, score, all_votes, tf_scores

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        return self._analyse(ctx, settings)[0]

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if not mode_timeframes(ctx, settings):
            return self.make_report(ctx, key_findings=["no indicator data for the mode's timeframes"],
                                    extra={"no_data": True})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        payload, score, votes, tf_scores = self._analyse(ctx, settings)
        signal = signal_from_score(score)
        conf = confidence_from_score(score, signal)
        signs = {1 if s > 0.15 else -1 if s < -0.15 else 0 for s in tf_scores.values()}
        alignment = "ALIGNED_UP" if signs == {1} else "ALIGNED_DOWN" if signs == {-1} else "MIXED"
        if alignment != "MIXED" and signal != Signal.NEUTRAL:
            conf = min(95, conf + 5)
        return self.make_report(
            ctx, signal=signal, confidence=conf, key_findings=strongest_facts(votes, signal),
            reasoning=f"Rule score {score:+.2f} across {', '.join(tf_scores) or 'no timeframes'} ({alignment}).",
            extra={"rule_score": round(score, 3), "tf_scores": tf_scores, "trend_alignment": alignment},
        )

    def finalize(self, ctx: StockContext, parsed: TechnicalOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        rep = self.report_from_output(ctx, parsed, extra={
            "trend_alignment": parsed.trend_alignment,
            "rule_score": det.extra.get("rule_score"),
            "tf_scores": det.extra.get("tf_scores"),
        })
        return self.crosscheck(rep, det)
