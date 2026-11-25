"""Agent calibration + suggested weights (advanced analytics, no automatic changes).

For every decision whose outcome is known - a traded position (net R) or a closed shadow
trade - the market direction after the setup is derived (+1 up / -1 down). Each agent's vote on
that decision is then scored:

* hit rate over its non-neutral votes,
* Brier score of its confidence (lower is better; 0.25 = coin flip at 50 %),
* suggested weight = current weight x clamp(hit_rate / 0.5, 0.25, 2.0), only once the agent
  has >= ``min_samples`` scored votes (otherwise the current weight is kept).

The owner applies suggested weights from the dashboard (and should validate them with a
walk-forward replay) - weights are never changed automatically, to avoid overfitting.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from storage.models import AgentReportRow, DecisionRow, ShadowTrade, Trade


def decision_outcomes(session: Session, batch_id: int) -> dict[int, float]:
    """decision_id -> realized R of the setup (traded position or shadow)."""
    out: dict[int, float] = {}
    trades = session.execute(select(Trade.decision_id, Trade.r_multiple, Trade.qty)
                             .where(Trade.batch_id == batch_id, Trade.decision_id.is_not(None))).all()
    agg: dict[int, tuple[float, int]] = {}
    for did, r, qty in trades:
        if r is None:
            continue
        s, n = agg.get(did, (0.0, 0))
        agg[did] = (s + float(r) * int(qty), n + int(qty))
    for did, (s, n) in agg.items():
        if n:
            out[int(did)] = s / n
    shadows = session.execute(select(ShadowTrade.decision_id, ShadowTrade.r_multiple)
                              .where(ShadowTrade.batch_id == batch_id, ShadowTrade.status == "CLOSED",
                                     ShadowTrade.decision_id.is_not(None), ShadowTrade.r_multiple.is_not(None))).all()
    for did, r in shadows:
        out.setdefault(int(did), float(r))
    return out


def agent_calibration(session: Session, batch_id: int, weights_by_mode: dict[str, dict[str, float]] | None = None,
                      *, min_samples: int = 20) -> pd.DataFrame:
    """Per (agent, mode): votes, hit rate, Brier score, current and suggested weight."""
    outcomes = decision_outcomes(session, batch_id)
    if not outcomes:
        return pd.DataFrame(columns=["agent", "mode", "votes", "scored", "hit_rate", "brier", "weight",
                                     "suggested_weight"])
    dirs = dict(session.execute(select(DecisionRow.id, DecisionRow.direction)
                                .where(DecisionRow.id.in_(list(outcomes)))).all())
    rows = session.execute(select(AgentReportRow.decision_id, AgentReportRow.agent, AgentReportRow.mode,
                                  AgentReportRow.signal, AgentReportRow.confidence, AgentReportRow.status)
                           .where(AgentReportRow.batch_id == batch_id,
                                  AgentReportRow.decision_id.in_(list(outcomes)))).all()
    recs: list[dict[str, Any]] = []
    for did, agent, mode, signal, conf, status in rows:
        r = outcomes.get(int(did))
        d = 1 if (dirs.get(did) or "LONG") == "LONG" else -1
        if r is None or r == 0:
            continue
        market = d if r > 0 else -d                     # +1 = price went up after the setup
        vote = 1 if signal == "BULLISH" else (-1 if signal == "BEARISH" else 0)
        recs.append({"agent": agent, "mode": mode, "vote": vote, "status": status,
                     "p": max(0.0, min(1.0, (conf or 0) / 100.0)), "correct": vote == market})
    if not recs:
        return pd.DataFrame(columns=["agent", "mode", "votes", "scored", "hit_rate", "brier", "weight",
                                     "suggested_weight"])
    df = pd.DataFrame(recs)
    out: list[dict[str, Any]] = []
    for (agent, mode), g in df.groupby(["agent", "mode"]):
        scored = g[(g["vote"] != 0) & (g["status"] == "OK")]
        hit = float(scored["correct"].mean()) if len(scored) else None
        brier = float(((scored["p"] - scored["correct"].astype(float)) ** 2).mean()) if len(scored) else None
        w = float((weights_by_mode or {}).get(str(mode), {}).get(str(agent), 1.0))
        suggested = w
        if hit is not None and len(scored) >= min_samples:
            suggested = round(w * max(0.25, min(2.0, hit / 0.5)), 2)
        out.append({"agent": agent, "mode": mode, "votes": int(len(g)), "scored": int(len(scored)),
                    "hit_rate": round(hit, 3) if hit is not None else None,
                    "brier": round(brier, 3) if brier is not None else None, "weight": w,
                    "suggested_weight": suggested})
    return pd.DataFrame(out).sort_values(["mode", "agent"]).reset_index(drop=True)
