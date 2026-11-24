"""News & Sentiment Agent (section 4, agent 6).

Headlines are already FinBERT-scored upstream (``Headline.sentiment_label/score``,
``ctx.sentiment_score``). The window is the last 24 h for INTRADAY and 7 days for the
longer modes; only headlines published at or before ``ctx.as_of`` are used (no look-ahead).
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from agents.base import SubAgent, r2
from agents.registry import register_agent
from agents.schemas import AgentOutput
from core.enums import Mode, Signal
from core.types import AgentReport, Headline, StockContext

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Max headlines shown to the LLM (newest first).
MAX_HEADLINES = 15
#: |aggregate score| needed for a directional rule-based signal.
SENTIMENT_THRESHOLD = 0.25


def news_window(mode: Mode) -> timedelta:
    """24 h intraday, 7 days for BTST / SWING / POSITIONAL."""
    return timedelta(hours=24) if mode == Mode.INTRADAY else timedelta(days=7)


def headlines_in_window(ctx: StockContext) -> list[Headline]:
    """Headlines for ``ctx.symbol`` (or market-wide) inside the mode's window, newest first."""
    start = ctx.as_of - news_window(ctx.mode)
    out = [h for h in ctx.headlines
           if start <= h.published_at <= ctx.as_of and (h.symbol is None or h.symbol.upper() == ctx.symbol.upper())]
    return sorted(out, key=lambda h: h.published_at, reverse=True)


def aggregate_sentiment(headlines: list[Headline], fallback: float | None) -> tuple[float | None, int]:
    """Mean FinBERT score of scored headlines (fallback to ``ctx.sentiment_score``)."""
    scores = [h.sentiment_score for h in headlines if h.sentiment_score is not None]
    if scores:
        return sum(scores) / len(scores), len(scores)
    return fallback, 0


@register_agent
class NewsSentimentAgent(SubAgent):
    """FinBERT-scored news interpretation."""

    name = "news_sentiment"
    description = "FinBERT sentiment of headlines (24h intraday, 7 days for longer modes)"
    OutputModel = AgentOutput

    def build_payload(self, ctx: StockContext, settings: "Settings | None" = None) -> dict[str, Any]:
        hs = headlines_in_window(ctx)
        agg, n_scored = aggregate_sentiment(hs, ctx.sentiment_score)
        labels = {s.value: sum(1 for h in hs if h.sentiment_label == s) for s in Signal}
        return {
            "symbol": ctx.symbol,
            "window_hours": int(news_window(ctx.mode).total_seconds() // 3600),
            "headline_count": len(hs),
            "scored_count": n_scored,
            "aggregate_sentiment": r2(agg, 3),
            "label_counts": labels,
            "headlines": [
                {
                    "title": h.title[:240],
                    "source": h.source,
                    "age_hours": r2((ctx.as_of - h.published_at).total_seconds() / 3600, 1),
                    "finbert_label": h.sentiment_label.value if h.sentiment_label else None,
                    "finbert_score": r2(h.sentiment_score, 3),
                    "stock_specific": h.symbol is not None,
                }
                for h in hs[:MAX_HEADLINES]
            ],
        }

    def precheck(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport | None:
        if not headlines_in_window(ctx):
            hours = int(news_window(ctx.mode).total_seconds() // 3600)
            return self.make_report(ctx, key_findings=[f"no headlines in the last {hours}h"],
                                    extra={"no_data": True, "aggregate_sentiment": None})
        return None

    def deterministic_report(self, ctx: StockContext, settings: "Settings | None" = None) -> AgentReport:
        hs = headlines_in_window(ctx)
        agg, n = aggregate_sentiment(hs, ctx.sentiment_score)
        signal = Signal.NEUTRAL
        conf = 30
        if agg is not None:
            if agg >= SENTIMENT_THRESHOLD:
                signal = Signal.BULLISH
            elif agg <= -SENTIMENT_THRESHOLD:
                signal = Signal.BEARISH
            if signal != Signal.NEUTRAL:
                conf = int(min(85, 30 + abs(agg) * 60 + min(n, 5) * 3))
        top = sorted((h for h in hs if h.sentiment_score is not None),
                     key=lambda h: -abs(h.sentiment_score or 0.0))[:4]
        findings = [f"{len(hs)} headline(s), aggregate FinBERT {agg:+.2f}" if agg is not None
                    else f"{len(hs)} headline(s), no sentiment scores"]
        findings += [f"{h.sentiment_label.value if h.sentiment_label else '?'} {h.sentiment_score:+.2f}: {h.title[:120]}"
                     for h in top]
        return self.make_report(ctx, signal=signal, confidence=conf, key_findings=findings,
                                reasoning="Rule: aggregate FinBERT score vs +/-0.25 threshold.",
                                extra={"aggregate_sentiment": r2(agg, 3), "headline_count": len(hs)})

    def finalize(self, ctx: StockContext, parsed: AgentOutput, settings: "Settings | None" = None,
                 det: AgentReport | None = None) -> AgentReport:
        det = det if det is not None else self.deterministic_report(ctx, settings)
        rep = self.report_from_output(ctx, parsed, extra={
            "aggregate_sentiment": det.extra.get("aggregate_sentiment"),
            "headline_count": det.extra.get("headline_count"),
        })
        return self.crosscheck(rep, det)
