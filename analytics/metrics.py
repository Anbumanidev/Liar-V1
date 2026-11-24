"""Pure performance metrics (section 7.6), shared by the dashboard, the EOD email and the
strategy self-evaluation.

Conventions:
* ``win_rate`` is a FRACTION (0..1); summaries also expose ``win_rate_pct`` (percent units).
* R multiples come from ``trades.r_multiple`` (net-of-charges R as recorded by the paper
  engine); rupee figures from ``net_pnl``.
* The ``trades`` table stores one row per exit leg (partial exits are separate rows).
  :func:`positions_from_trades` merges legs into one row per position, which is the unit
  used for win rate / expectancy (a 50 % partial at 1R + a stop at breakeven is ONE trade).
* Every function handles empty input (returns ``None`` / 0 / an empty frame as documented).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

#: Columns of a trades frame understood by this module (missing ones are filled with NaN).
TRADE_COLUMNS: tuple[str, ...] = (
    "id", "batch_id", "position_id", "symbol", "mode", "direction", "qty", "entry_time", "entry_price",
    "exit_time", "exit_price", "stop_loss", "exit_reason", "is_partial", "gross_pnl", "charges", "net_pnl",
    "r_multiple", "source", "strategy", "regime", "grade", "holding_minutes", "holding_days",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "trades", "wins", "losses", "win_rate_pct", "avg_r", "expectancy_r", "expectancy_rs", "profit_factor",
    "gross_pnl", "charges", "net_pnl", "avg_win_rs", "avg_loss_rs", "max_drawdown", "max_drawdown_pct",
)


# =========================================================================== conversion


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    return {c: getattr(row, c, None) for c in TRADE_COLUMNS}


def trades_to_df(trades: pd.DataFrame | Iterable[Any] | None) -> pd.DataFrame:
    """Normalise trades (DataFrame, list of dicts or ORM ``Trade`` rows) into a DataFrame.

    Enum values are converted to plain strings; missing columns are added as NaN/None.
    """
    if trades is None:
        df = pd.DataFrame(columns=list(TRADE_COLUMNS))
    elif isinstance(trades, pd.DataFrame):
        df = trades.copy()
    else:
        df = pd.DataFrame([_row_to_dict(r) for r in trades])
    for c in TRADE_COLUMNS:
        if c not in df.columns:
            df[c] = None
    for c in ("net_pnl", "gross_pnl", "charges", "r_multiple", "qty", "entry_price", "exit_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("mode", "direction", "source", "strategy", "regime", "exit_reason", "grade"):
        df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    return df


def positions_from_trades(trades: pd.DataFrame | Iterable[Any] | None) -> pd.DataFrame:
    """Merge partial-exit legs into one row per position.

    ``net_pnl``/``gross_pnl``/``charges`` are summed, ``r_multiple`` is the quantity-weighted
    mean of the legs' R, ``qty`` is summed, ``exit_time`` is the last leg's exit. Rows without
    a ``position_id`` are kept as-is.
    """
    df = trades_to_df(trades)
    if df.empty:
        return df
    has_pos = df["position_id"].notna()
    singles = df[~has_pos]
    grouped = df[has_pos]
    if grouped.empty:
        return df.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, g in grouped.sort_values("exit_time", kind="stable").groupby("position_id", sort=False):
        last = g.iloc[-1].to_dict()
        qty = g["qty"].fillna(0)
        r = g["r_multiple"]
        if r.notna().any():
            w = qty.where(r.notna(), 0)
            r_val = float((r.fillna(0) * w).sum() / w.sum()) if w.sum() > 0 else float(r.dropna().mean())
        else:
            r_val = float("nan")
        last.update(
            qty=float(qty.sum()),
            net_pnl=float(g["net_pnl"].fillna(0).sum()),
            gross_pnl=float(g["gross_pnl"].fillna(0).sum()),
            charges=float(g["charges"].fillna(0).sum()),
            r_multiple=r_val,
            is_partial=False,
            entry_time=g["entry_time"].iloc[0],
        )
        rows.append(last)
    out = pd.concat([pd.DataFrame(rows), singles], ignore_index=True)
    if "exit_time" in out and out["exit_time"].notna().any():
        out = out.sort_values("exit_time", kind="stable", na_position="last").reset_index(drop=True)
    return out


def _clean(values: Iterable[float | None] | pd.Series | None) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    arr = pd.to_numeric(pd.Series(list(values) if not isinstance(values, pd.Series) else values), errors="coerce")
    arr = arr.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


# =========================================================================== scalar metrics


def win_rate(pnls: Iterable[float | None] | pd.Series | None) -> float | None:
    """Fraction (0..1) of strictly positive values; ``None`` for empty input."""
    a = _clean(pnls)
    if a.size == 0:
        return None
    return float((a > 0).sum() / a.size)


def avg_r(r_multiples: Iterable[float | None] | pd.Series | None) -> float | None:
    """Mean R multiple (NaN/None ignored); ``None`` for empty input."""
    a = _clean(r_multiples)
    return float(a.mean()) if a.size else None


def expectancy(values: Iterable[float | None] | pd.Series | None) -> float | None:
    """Expectancy per trade = mean outcome (works for R multiples or rupees).

    Equivalent to ``W * avg_win - (1 - W) * avg_loss``. ``None`` for empty input.
    """
    a = _clean(values)
    return float(a.mean()) if a.size else None


def expectancy_from_stats(win_rate_frac: float, avg_win: float, avg_loss: float) -> float:
    """``W * avg_win - (1 - W) * |avg_loss|``."""
    return win_rate_frac * avg_win - (1 - win_rate_frac) * abs(avg_loss)


def profit_factor(pnls: Iterable[float | None] | pd.Series | None) -> float | None:
    """Gross profit / gross loss. ``inf`` with profits and no losses, ``None`` when empty or
    when there is neither profit nor loss."""
    a = _clean(pnls)
    if a.size == 0:
        return None
    gp = float(a[a > 0].sum())
    gl = float(-a[a < 0].sum())
    if gl == 0:
        return math.inf if gp > 0 else None
    return gp / gl


def max_drawdown(equity: Sequence[float] | pd.Series | np.ndarray | None) -> tuple[float, float]:
    """Maximum peak-to-trough drawdown of an equity series.

    Returns:
        ``(absolute, pct)`` - absolute in the series' units (>= 0) and percent of the running
        peak (percent units; 0 when the peak is <= 0). ``(0.0, 0.0)`` for empty input.
    """
    a = _clean(equity)
    if a.size == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(a)
    dd = peaks - a
    i = int(np.argmax(dd))
    abs_dd = float(dd[i])
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_series = np.where(peaks > 0, dd / peaks * 100.0, 0.0)
    return abs_dd, float(np.max(pct_series)) if pct_series.size else 0.0


def equity_curve(trades: pd.DataFrame | Iterable[Any] | None, starting_capital: float = 0.0) -> pd.Series:
    """Cumulative realised net P&L (plus ``starting_capital``) indexed by exit time.

    Rows are ordered by ``exit_time`` (stable). Empty input -> empty float Series.
    """
    df = trades_to_df(trades)
    if df.empty:
        return pd.Series(dtype=float, name="equity")
    df = df.sort_values("exit_time", kind="stable", na_position="last")
    eq = starting_capital + df["net_pnl"].fillna(0.0).cumsum()
    idx = df["exit_time"] if df["exit_time"].notna().all() else pd.RangeIndex(len(df))
    return pd.Series(eq.to_numpy(dtype=float), index=idx, name="equity")


def kelly_fraction(win_rate_frac: float, payoff_ratio: float, fraction: float = 0.25) -> float:
    """Fractional Kelly: ``fraction * (W - (1 - W) / R)``, floored at 0.

    Args:
        win_rate_frac: W (0..1).
        payoff_ratio: R = average win / average loss (both positive). ``<= 0`` -> 0.
        fraction: Kelly fraction (0.25 = quarter Kelly).
    """
    if payoff_ratio <= 0 or not math.isfinite(payoff_ratio):
        return 0.0
    k = win_rate_frac - (1 - win_rate_frac) / payoff_ratio
    return max(0.0, k * fraction)


# =========================================================================== summaries


def _stats(df: pd.DataFrame) -> dict[str, float | int | None]:
    pnl = df["net_pnl"]
    rs = df["r_multiple"]
    a = _clean(pnl)
    wins = int((a > 0).sum())
    losses = int((a < 0).sum())
    wr = win_rate(pnl)
    eq = pnl.fillna(0.0).cumsum()
    dd_abs, dd_pct = max_drawdown(pd.concat([pd.Series([0.0]), eq], ignore_index=True))
    return {
        "trades": int(len(df)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wr * 100 if wr is not None else None,
        "avg_r": avg_r(rs),
        "expectancy_r": expectancy(rs),
        "expectancy_rs": expectancy(pnl),
        "profit_factor": profit_factor(pnl),
        "gross_pnl": float(df["gross_pnl"].fillna(0).sum()),
        "charges": float(df["charges"].fillna(0).sum()),
        "net_pnl": float(pnl.fillna(0).sum()),
        "avg_win_rs": float(a[a > 0].mean()) if wins else None,
        "avg_loss_rs": float(a[a < 0].mean()) if losses else None,
        "max_drawdown": dd_abs,
        "max_drawdown_pct": dd_pct,
    }


def summarize(
    trades_df: pd.DataFrame | Iterable[Any] | None,
    by: str | Sequence[str] | None = "mode",
    *,
    merge_partials: bool = True,
) -> pd.DataFrame:
    """Per-group metrics table.

    Args:
        trades_df: trades (DataFrame / dicts / ORM rows).
        by: grouping column(s), e.g. ``"mode"``, ``"strategy"``, ``"regime"``, ``"source"``;
            ``None`` -> one ``ALL`` row.
        merge_partials: merge partial-exit legs per position first (recommended).

    Returns:
        DataFrame with the group column(s) followed by :data:`SUMMARY_COLUMNS`
        (``max_drawdown`` is computed on the group's cumulative net P&L starting at 0;
        ``max_drawdown_pct`` is relative to that P&L peak and is 0 until it is positive).
        Empty input -> empty frame with those columns.
    """
    keys = [by] if isinstance(by, str) else list(by or [])
    df = positions_from_trades(trades_df) if merge_partials else trades_to_df(trades_df)
    cols = (keys or ["group"]) + list(SUMMARY_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=cols)
    if "exit_time" in df and df["exit_time"].notna().any():
        df = df.sort_values("exit_time", kind="stable", na_position="last")
    rows: list[dict[str, Any]] = []
    if not keys:
        rows.append({"group": "ALL", **_stats(df)})
    else:
        filled = df.copy()
        for k in keys:
            if k not in filled.columns:
                filled[k] = None
            filled[k] = filled[k].map(lambda v: "UNKNOWN" if v is None or (isinstance(v, float) and math.isnan(v)) else v)
        for name, g in filled.groupby(keys, sort=True, dropna=False):
            vals = name if isinstance(name, tuple) else (name,)
            rows.append({**dict(zip(keys, vals, strict=True)), **_stats(g)})
    return pd.DataFrame(rows, columns=cols)
