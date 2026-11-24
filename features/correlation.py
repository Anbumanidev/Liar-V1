"""Return correlation between two symbols (risk rule: no two positions with 20-day corr > 0.7).

Pearson correlation of daily close-to-close simple returns over the last ``lookback``
overlapping sessions. Series are aligned on calendar DATE (so differing bar timestamps of the
same day still match). Returns None when there are fewer than ``min_periods`` (default
``lookback``) overlapping returns or when either return series has zero variance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _by_date(s: pd.Series) -> pd.Series:
    s = s.astype("float64").dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        s = pd.Series(s.to_numpy(), index=pd.Index(s.index.date))
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def return_correlation(
    a_close: pd.Series,
    b_close: pd.Series,
    lookback: int = 20,
    *,
    min_periods: int | None = None,
) -> float | None:
    """Pearson correlation of daily returns over the last ``lookback`` common sessions."""
    if a_close is None or b_close is None or len(a_close) == 0 or len(b_close) == 0 or lookback < 2:
        return None
    need = lookback if min_periods is None else max(2, min(min_periods, lookback))
    joined = pd.concat([_by_date(a_close).rename("a"), _by_date(b_close).rename("b")], axis=1, join="inner")
    rets = joined.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna().iloc[-lookback:]
    if len(rets) < need:
        return None
    if rets["a"].std(ddof=0) == 0 or rets["b"].std(ddof=0) == 0:
        return None
    corr = float(np.corrcoef(rets["a"].to_numpy(), rets["b"].to_numpy())[0, 1])
    return corr if math.isfinite(corr) else None


def correlation_matrix(closes: dict[str, pd.Series], lookback: int = 20) -> dict[tuple[str, str], float | None]:
    """Pairwise :func:`return_correlation` for every unordered pair of symbols."""
    syms = sorted(closes)
    out: dict[tuple[str, str], float | None] = {}
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            out[(a, b)] = return_correlation(closes[a], closes[b], lookback)
    return out
