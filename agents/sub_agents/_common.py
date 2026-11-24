"""Shared deterministic helpers for the Layer-1 sub-agents (pure functions, no I/O)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from core.enums import Signal
from core.types import Levels, StockContext, TimeframeFeatures

from agents.base import r2


@dataclass(frozen=True)
class Vote:
    """One rule-based observation: a fact sentence and its vote (-1 .. +1, 0 = informational)."""

    fact: str
    vote: float
    weight: float = 1.0


def score_votes(votes: Sequence[Vote]) -> float:
    """Weighted mean of the non-informational votes in [-1, 1] (0.0 when there are none).

    Informational facts (vote == 0 with weight 0) do not dilute the score; neutral votes
    (vote == 0, weight > 0) do.
    """
    num = sum(v.vote * v.weight for v in votes)
    den = sum(v.weight for v in votes if v.weight > 0)
    if den <= 0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def signal_from_score(score: float, threshold: float = 0.3) -> Signal:
    """Map a [-1, 1] score to a signal."""
    if score >= threshold:
        return Signal.BULLISH
    if score <= -threshold:
        return Signal.BEARISH
    return Signal.NEUTRAL


def confidence_from_score(score: float, signal: Signal, *, base: int = 35, span: int = 55,
                          neutral_conf: int = 40, cap: int = 90) -> int:
    """Deterministic confidence: grows with |score| for directional signals."""
    if signal == Signal.NEUTRAL:
        return neutral_conf
    return int(min(cap, base + span * min(1.0, abs(score))))


def facts_of(votes: Iterable[Vote]) -> list[str]:
    """Fact sentences of ``votes``."""
    return [v.fact for v in votes]


def strongest_facts(votes: Sequence[Vote], signal: Signal, n: int = 5) -> list[str]:
    """Up to ``n`` facts supporting ``signal`` (then the rest), for key findings."""
    want = 1 if signal == Signal.BULLISH else -1 if signal == Signal.BEARISH else 0
    supporting = [v for v in votes if (v.vote > 0 and want > 0) or (v.vote < 0 and want < 0)]
    others = [v for v in votes if v not in supporting]
    ordered = sorted(supporting, key=lambda v: -abs(v.vote * v.weight)) + others
    return [v.fact for v in ordered[:n]]


def tf_values(tf: TimeframeFeatures, keys: Sequence[str], nd: int = 2) -> dict[str, float | None]:
    """Rounded subset of ``tf.values`` (missing keys omitted)."""
    out: dict[str, float | None] = {}
    for k in keys:
        if k in tf.values and tf.values[k] is not None:
            out[k] = r2(tf.values[k], nd)
    return out


def levels_summary(levels: Levels | None, price: float | None, n: int = 3) -> dict[str, Any]:
    """Pivot/CPR/support/resistance summary with pre-computed distances from ``price``."""
    if levels is None:
        return {}
    out: dict[str, Any] = {}
    for k in ("pivot", "r1", "r2", "r3", "s1", "s2", "s3", "cpr_bc", "cpr_tc", "cpr_width_pct",
              "prev_high", "prev_low", "prev_close"):
        v = getattr(levels, k)
        if v is not None:
            out[k] = r2(v)

    def _with_dist(vals: Sequence[float]) -> list[dict[str, float | None]]:
        res = []
        for v in list(vals)[:n]:
            item: dict[str, float | None] = {"price": r2(v)}
            if price:
                item["distance_pct"] = r2((v - price) / price * 100)
            res.append(item)
        return res

    if levels.supports:
        out["supports"] = _with_dist(levels.supports)
    if levels.resistances:
        out["resistances"] = _with_dist(levels.resistances)
    if price is not None and levels.cpr_bc is not None and levels.cpr_tc is not None:
        lo, hi = sorted((levels.cpr_bc, levels.cpr_tc))
        out["price_vs_cpr"] = "ABOVE" if price > hi else "BELOW" if price < lo else "INSIDE"
    return out


def compare_fact(name_a: str, a: float | None, name_b: str, b: float | None) -> tuple[str, int] | None:
    """'close 101.00 is ABOVE EMA21 100.00 (+1.00%)' and +1/-1, or None if not comparable."""
    if a is None or b is None or b == 0:
        return None
    diff = (a - b) / b * 100
    side = "ABOVE" if a > b else "BELOW" if a < b else "AT"
    vote = 1 if a > b else -1 if a < b else 0
    return f"{name_a} {a:.2f} is {side} {name_b} {b:.2f} ({diff:+.2f}%)", vote


def window_rationale(ctx: StockContext) -> str:
    """'RELIANCE INTRADAY @ 2026-11-24T10:00' - for log / finding prefixes."""
    return f"{ctx.symbol} {ctx.mode.value} @ {ctx.as_of.isoformat(timespec='minutes')}"
