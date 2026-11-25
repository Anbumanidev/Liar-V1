"""Walk-forward evaluation / tuning (sections 7.6, 17.12). Tuning ONLY happens here.

Rolling windows over trading days: for each window every parameter combination of the grid is
replayed on the TRAIN segment; the best (by net P&L, tie-break expectancy) is then replayed on
the following unseen TEST segment. Out-of-sample results are what count. Every parameter
change is logged with before/after metrics (JSONL ``walk_forward`` + returned report).

History is downloaded once and shared by every replay (no repeated network calls).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from config.loader import load_default_settings
from core.enums import Mode
from core.logging import log_event
from scheduler.replay import ReplayConfig, ReplayRunner, preload_frames, replay_settings, required_timeframes

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Small default grid (cheap): confluence minimum of each requested mode.
def default_grid(modes: list[Mode]) -> dict[str, list[Any]]:
    return {f"modes.modes.{m.value}.min_confluence": [55.0, 60.0, 70.0] for m in modes}


@dataclass
class WindowResult:
    train: tuple[str, str]
    test: tuple[str, str]
    chosen: dict[str, Any]
    train_net: float
    test_net: float
    test_trades: int
    test_expectancy: float | None
    baseline_test_net: float | None = None


@dataclass
class WalkForwardReport:
    windows: list[WindowResult] = field(default_factory=list)
    oos_net: float = 0.0
    oos_trades: int = 0
    baseline_oos_net: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]


def walk_forward(
    start: date,
    end: date,
    modes: list[Mode],
    *,
    symbols: list[str] | None = None,
    train_days: int = 20,
    test_days: int = 5,
    grid: dict[str, list[Any]] | None = None,
    llm_backend: str = "rules",
    name: str = "wf",
    frames: Any | None = None,
) -> WalkForwardReport:
    """Run the walk-forward procedure and return the out-of-sample report."""
    from data.calendar import load_calendar

    base = load_default_settings()
    grid = grid or default_grid(modes)
    cal = load_calendar(base.schedule)
    days = cal.trading_days_between(start, end)
    if len(days) < train_days + test_days:
        raise ValueError(f"need at least {train_days + test_days} trading days, got {len(days)}")
    probe = ReplayConfig(start=start, end=end, modes=modes, symbols=symbols, llm_backend=llm_backend)
    settings0 = replay_settings(base, probe)
    if frames is None:
        sym_list = symbols or ReplayRunner(probe)._symbols(settings0)  # noqa: SLF001
        frames = preload_frames(sym_list, required_timeframes(settings0, modes), start, end)
        symbols = sym_list
    report = WalkForwardReport()
    i, k = 0, 0
    baseline: dict[str, Any] = {}
    while i + train_days + test_days <= len(days):
        tr, te = days[i:i + train_days], days[i + train_days:i + train_days + test_days]
        best: tuple[float, float, dict[str, Any]] | None = None
        for combo in _combos(grid):
            cfg = ReplayConfig(start=tr[0], end=tr[-1], modes=modes, symbols=symbols, llm_backend=llm_backend,
                               name=f"{name}_w{k}_train", settings_changes=combo)
            rep = ReplayRunner(cfg, frames=frames, base_settings=base).run()
            score = (rep.net_pnl, rep.expectancy or 0.0, combo)
            if best is None or score[:2] > best[:2]:
                best = score
        assert best is not None
        chosen = best[2]
        test_cfg = ReplayConfig(start=te[0], end=te[-1], modes=modes, symbols=symbols, llm_backend=llm_backend,
                                name=f"{name}_w{k}_test", settings_changes=chosen)
        test_rep = ReplayRunner(test_cfg, frames=frames, base_settings=base).run()
        base_cfg = ReplayConfig(start=te[0], end=te[-1], modes=modes, symbols=symbols, llm_backend=llm_backend,
                                name=f"{name}_w{k}_baseline", settings_changes=baseline)
        base_rep = ReplayRunner(base_cfg, frames=frames, base_settings=base).run()
        wr = WindowResult(train=(str(tr[0]), str(tr[-1])), test=(str(te[0]), str(te[-1])), chosen=chosen,
                          train_net=best[0], test_net=test_rep.net_pnl, test_trades=test_rep.trades,
                          test_expectancy=test_rep.expectancy, baseline_test_net=base_rep.net_pnl)
        report.windows.append(wr)
        report.oos_net += test_rep.net_pnl
        report.oos_trades += test_rep.trades
        report.baseline_oos_net += base_rep.net_pnl
        log_event("walk_forward", {"window": k, "parameter_change": {"before": baseline, "after": chosen},
                                   "before_metrics": {"test_net": base_rep.net_pnl},
                                   "after_metrics": {"test_net": test_rep.net_pnl, "trades": test_rep.trades,
                                                     "expectancy": test_rep.expectancy}})
        logger.info(f"[walk-forward] window {k}: chose {chosen} -> OOS net {test_rep.net_pnl:.2f} "
                    f"(baseline {base_rep.net_pnl:.2f})")
        i += test_days
        k += 1
    out = PROJECT_ROOT / "runtime" / "replay" / f"{name}_walk_forward.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
    return report
