"""Monte Carlo risk-of-ruin / drawdown simulation (advanced analytics).

Bootstraps the batch's realized trade outcomes (net P&L per position, or R-multiples x a
risk amount) into thousands of possible future trade sequences and reports:

* probability of ending in profit, median / 5th / 95th percentile final equity,
* median and 95th-percentile max drawdown,
* probability of hitting the capital-protection HALT (the stricter of the absolute floor
  ``start x (1 - hard_stop%)`` and the trailing floor ``HWM x (1 - hard_stop%)``) - paths stop
  trading once halted, exactly like the live system.

Deterministic for a given seed. Small samples give wide, unreliable bands - the result carries
``reliable = n_samples >= 30``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class MonteCarloResult:
    n_samples: int
    n_paths: int
    n_trades: int
    start_equity: float
    reliable: bool
    prob_profit: float
    prob_halt: float
    expected_final: float
    median_final: float
    p5_final: float
    p95_final: float
    median_max_dd_pct: float
    p95_max_dd_pct: float
    bands: dict[str, list[float]] = field(default_factory=dict)   # p5 / p50 / p95 equity per trade step

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def monte_carlo(
    outcomes: Sequence[float],
    start_equity: float,
    *,
    n_trades: int = 100,
    n_paths: int = 2000,
    hard_stop_pct: float = 9.0,
    starting_capital: float | None = None,
    seed: int = 42,
) -> MonteCarloResult | None:
    """Bootstrap ``outcomes`` (rupee P&L per trade) into ``n_paths`` sequences of ``n_trades``.

    Returns None if there are no outcomes.
    """
    x = np.asarray([float(v) for v in outcomes if v is not None and np.isfinite(v)], dtype="float64")
    if x.size == 0 or start_equity <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_paths, n_trades), replace=True)
    keep = 1.0 - hard_stop_pct / 100.0
    abs_floor = (starting_capital if starting_capital is not None else start_equity) * keep
    eq = np.empty((n_paths, n_trades + 1))
    eq[:, 0] = start_equity
    hwm = np.full(n_paths, start_equity)
    halted = np.zeros(n_paths, dtype=bool)
    max_dd = np.zeros(n_paths)
    cur = np.full(n_paths, start_equity)
    for i in range(n_trades):
        step = np.where(halted, 0.0, draws[:, i])
        cur = cur + step
        hwm = np.maximum(hwm, cur)
        dd = np.where(hwm > 0, (hwm - cur) / hwm * 100.0, 0.0)
        max_dd = np.maximum(max_dd, dd)
        floor = np.maximum(abs_floor, hwm * keep)
        halted = halted | (cur <= floor)
        eq[:, i + 1] = cur
    final = eq[:, -1]
    bands = {
        "p5": np.percentile(eq, 5, axis=0).round(2).tolist(),
        "p50": np.percentile(eq, 50, axis=0).round(2).tolist(),
        "p95": np.percentile(eq, 95, axis=0).round(2).tolist(),
    }
    return MonteCarloResult(
        n_samples=int(x.size), n_paths=n_paths, n_trades=n_trades, start_equity=round(start_equity, 2),
        reliable=bool(x.size >= 30),
        prob_profit=round(float((final > start_equity).mean()), 4),
        prob_halt=round(float(halted.mean()), 4),
        expected_final=round(float(final.mean()), 2),
        median_final=round(float(np.median(final)), 2),
        p5_final=round(float(np.percentile(final, 5)), 2),
        p95_final=round(float(np.percentile(final, 95)), 2),
        median_max_dd_pct=round(float(np.median(max_dd)), 2),
        p95_max_dd_pct=round(float(np.percentile(max_dd, 95)), 2),
        bands=bands,
    )


def outcomes_from_r(r_multiples: Sequence[float], risk_amount: float) -> list[float]:
    """Convert R-multiples into rupee outcomes at today's risk per trade."""
    return [float(r) * risk_amount for r in r_multiples if r is not None and np.isfinite(r)]
