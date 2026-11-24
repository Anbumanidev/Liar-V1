"""Deterministic market-regime classification (spec section 7.1).

Order of evaluation (first match wins):

1. ``HIGH_VOLATILITY`` - India VIX elevated (computed by :mod:`features.market`) OR
   ``atr_expansion >= ATR_EXPANSION_HIGH_VOL`` (ATR(14) vs its 20-bar mean).
2. ``TRENDING_UP`` - ``ADX > ADX_TRENDING`` and close > EMA21 and (intraday) close > VWAP and
   the index trend is not BEARISH.
3. ``TRENDING_DOWN`` - mirror: ``ADX > ADX_TRENDING``, close < EMA21, (intraday) close < VWAP,
   index trend not BULLISH.
4. ``RANGING`` - ``ADX < ADX_RANGING`` or Bollinger squeeze or price inside the CPR.
5. ``UNDEFINED`` - anything else (including missing inputs).
"""

from __future__ import annotations

from core.enums import Regime, Signal
from core.types import TimeframeFeatures

ADX_TRENDING = 25.0
ADX_RANGING = 20.0
ATR_EXPANSION_HIGH_VOL = 1.5


def detect_regime(
    tf: TimeframeFeatures,
    *,
    index_trend: Signal | None = None,
    vix_elevated: bool = False,
    inside_cpr: bool = False,
    adx_trending: float = ADX_TRENDING,
    adx_ranging: float = ADX_RANGING,
    atr_expansion_high_vol: float = ATR_EXPANSION_HIGH_VOL,
) -> Regime:
    """Classify the regime of one timeframe's latest closed bar (see module docstring)."""
    v = tf.values
    atr_exp = v.get("atr_expansion")
    if vix_elevated or (atr_exp is not None and atr_exp >= atr_expansion_high_vol):
        return Regime.HIGH_VOLATILITY

    adx = v.get("adx_14")
    close = v.get("close")
    ema21 = v.get("ema_21")
    vwap = v.get("vwap")

    if adx is not None and close is not None and ema21 is not None and adx > adx_trending:
        above = close > ema21 and (vwap is None or close > vwap)
        below = close < ema21 and (vwap is None or close < vwap)
        if above and index_trend != Signal.BEARISH:
            return Regime.TRENDING_UP
        if below and index_trend != Signal.BULLISH:
            return Regime.TRENDING_DOWN

    squeeze = v.get("bb_squeeze")
    if (adx is not None and adx < adx_ranging) or (squeeze is not None and squeeze >= 1.0) or inside_cpr:
        return Regime.RANGING
    return Regime.UNDEFINED
