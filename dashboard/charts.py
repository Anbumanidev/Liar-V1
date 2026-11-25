"""Plotly figure builders (and number formatting) for the analytics pages (section 14.4).

Pure functions - DataFrame in, ``plotly.graph_objects.Figure`` out, no Streamlit - so each
is unit testable. Every builder returns a figure even for empty input (an annotated empty
figure). Phone friendly: compact height, small margins, horizontal legend above the plot,
horizontal bars for long category names.

Colour roles (validated reference palette):

* categorical series take slots in a fixed order (TRADING, RESERVE, TOTAL EQUITY, ...);
* polarity (profit / loss) uses the diverging blue <-> red pair - the bar direction carries
  the sign too, so colour is never the only cue;
* drawdown tiers use the status palette (good / warning / serious / critical) as light washes,
  always named in the legend.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go

#: Categorical slots, fixed order (never cycled past 8).
SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
#: Diverging poles for profit / loss.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
#: Muted ink for reference lines (HWM, entry, trade path).
NEUTRAL = "#898781"
INK = "#52514e"
SURFACE = "#fcfcfb"
#: Status palette for the drawdown tiers.
TIER_COLORS: dict[str, str] = {"NORMAL": "#0ca30c", "CAUTION": "#fab219", "DEFENSIVE": "#ec835a", "HALT": "#d03b3b"}
HEIGHT = 340
_MODE_ORDER = {"INTRADAY": 0, "BTST": 1, "SWING": 2, "POSITIONAL": 3}


# =========================================================================== formatting


def _bad(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def fmt_rs(value: Any, *, signed: bool = False, nd: int = 2) -> str:
    """``Rs 1,234.50`` (``-`` for missing); ``signed`` adds +/-."""
    if _bad(value):
        return "-"
    return f"Rs {float(value):+,.{nd}f}" if signed else f"Rs {float(value):,.{nd}f}"


def fmt_pct(value: Any, *, nd: int = 2, signed: bool = False) -> str:
    if _bad(value):
        return "-"
    return f"{float(value):+.{nd}f}%" if signed else f"{float(value):.{nd}f}%"


def fmt_num(value: Any, nd: int = 2) -> str:
    if _bad(value):
        return "-"
    return f"{float(value):,.{nd}f}"


def fmt_pf(value: Any) -> str:
    """Profit factor (``inf`` when there are no losses yet)."""
    if _bad(value):
        return "-"
    return "inf (no losses)" if math.isinf(float(value)) else f"{float(value):.2f}"


# =========================================================================== layout helpers


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _layout(fig: go.Figure, *, title: str | None = None, height: int = HEIGHT, legend: bool = True,
            hovermode: str | bool = "x unified") -> go.Figure:
    top = 24 + (32 if title else 0) + (24 if legend else 0)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)),
        hovermode=hovermode,
        barcornerradius=4,
        bargap=0.35,
    )
    return fig


def empty_figure(message: str = "No data yet", *, height: int = 200) -> go.Figure:
    """Placeholder figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=13, color=NEUTRAL))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, height=height, legend=False, hovermode=False)


def _signed_colors(values: Sequence[float]) -> list[str]:
    return [POSITIVE if (v or 0) >= 0 else NEGATIVE for v in values]


# =========================================================================== overview


def equity_curve_figure(series: pd.DataFrame, *, zoom: bool = True, title: str | None = None) -> go.Figure:
    """TRADING vs RESERVE stacked, TOTAL EQUITY line, HWM line, both floors and tier bands.

    ``series`` = ``analytics_queries.with_protection_levels(equity_series(...))``. With
    ``zoom`` the y-axis focuses on the floor-to-HWM region (where the tiers live); without it
    the axis starts at 0 so the stacked areas read as absolute amounts.
    """
    if series is None or series.empty:
        return empty_figure("No equity history yet")
    df = series.sort_values("ts", kind="stable").reset_index(drop=True)
    x = list(df["ts"])
    equity = df["equity"].astype(float)
    trading = df["trading"].astype(float)
    reserve = df["reserve"].astype(float)
    hwm = df["hwm"].astype(float)
    has_levels = all(c in df.columns for c in ("caution_level", "defensive_level", "trailing_floor"))
    has_abs = "abs_floor" in df.columns and df["abs_floor"].notna().any()
    top = float(max(hwm.max(), equity.max(), (trading + reserve).max()))
    if zoom:
        lows = [equity.min(), trading.min()]
        if has_levels:
            lows.append(df["trailing_floor"].min())
        if has_abs:
            lows.append(df["abs_floor"].min())
        low = float(min(lows))
        span = max(top - low, abs(top) * 0.01, 1.0)
        y0, y1 = low - span * 0.08, top + span * 0.08
    else:
        y0, y1 = 0.0, top * 1.05 if top > 0 else 1.0
    fig = go.Figure()
    if has_levels:
        first_hwm = float(hwm.iloc[0]) or 1.0

        def pct(col: str) -> float:
            return round((1.0 - float(df[col].iloc[0]) / first_hwm) * 100.0, 2)

        c_pct, d_pct, h_pct = pct("caution_level"), pct("defensive_level"), pct("trailing_floor")
        floor_line = pd.Series(y0, index=df.index)
        bands = (
            ("NORMAL", f"NORMAL (0-{c_pct:g}% DD)", hwm, df["caution_level"], 0.10),
            ("CAUTION", f"CAUTION ({c_pct:g}-{d_pct:g}%)", df["caution_level"], df["defensive_level"], 0.14),
            ("DEFENSIVE", f"DEFENSIVE ({d_pct:g}-{h_pct:g}%)", df["defensive_level"], df["trailing_floor"], 0.14),
            ("HALT", f"HALT (>= {h_pct:g}%)", df["trailing_floor"], floor_line, 0.08),
        )
        for tier, label, upper, lower, alpha in bands:
            fig.add_trace(go.Scatter(
                x=x + x[::-1], y=list(upper.astype(float)) + list(lower.astype(float))[::-1], fill="toself",
                fillcolor=_rgba(TIER_COLORS[tier], alpha), line=dict(width=0), hoverinfo="skip", name=label,
                legendgroup="tiers", mode="lines",
            ))
    fig.add_trace(go.Scatter(x=x, y=trading, name="TRADING", stackgroup="capital", mode="lines",
                             line=dict(width=2, color=SERIES[0]), fillcolor=_rgba(SERIES[0], 0.18),
                             hovertemplate="TRADING Rs %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=reserve, name="RESERVE", stackgroup="capital", mode="lines",
                             line=dict(width=2, color=SERIES[1]), fillcolor=_rgba(SERIES[1], 0.28),
                             hovertemplate="RESERVE Rs %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=equity, name="TOTAL EQUITY", mode="lines", line=dict(width=2, color=SERIES[2]),
                             hovertemplate="TOTAL EQUITY Rs %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x[-1]], y=[equity.iloc[-1]], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=9, color=SERIES[2], line=dict(width=2, color=SURFACE))))
    fig.add_trace(go.Scatter(x=x, y=hwm, name="HWM", mode="lines", line=dict(width=1.5, color=INK, dash="dash"),
                             hovertemplate="HWM Rs %{y:,.2f}<extra></extra>"))
    if has_abs:
        fig.add_trace(go.Scatter(x=x, y=df["abs_floor"].astype(float), name="Absolute floor", mode="lines",
                                 line=dict(width=1.5, color=TIER_COLORS["HALT"], dash="dash"),
                                 hovertemplate="Absolute floor Rs %{y:,.2f}<extra></extra>"))
    if has_levels:
        fig.add_trace(go.Scatter(x=x, y=df["trailing_floor"].astype(float), name="Trailing floor", mode="lines",
                                 line=dict(width=1.5, color=TIER_COLORS["HALT"], dash="dot"),
                                 hovertemplate="Trailing floor Rs %{y:,.2f}<extra></extra>"))
    fig.add_annotation(x=x[-1], y=float(equity.iloc[-1]), text=fmt_rs(equity.iloc[-1], nd=0), showarrow=False,
                       xanchor="right", yanchor="bottom", yshift=8, font=dict(size=11))
    fig.update_yaxes(range=[y0, y1], tickprefix="Rs ", tickformat=",.0f")
    fig.update_xaxes(type="date")
    return _layout(fig, title=title, height=400, legend=True)


def daily_pnl_figure(daily: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Realized net P&L per day (blue up = profit, red down = loss)."""
    if daily is None or daily.empty:
        return empty_figure("No closed trades yet")
    df = daily.sort_values("day")
    custom = df[["exits", "gross_pnl", "charges"]].to_numpy()
    fig = go.Figure(go.Bar(
        x=list(pd.to_datetime(df["day"])), y=df["net_pnl"], marker_color=_signed_colors(list(df["net_pnl"])),
        width=[86_400_000 * 0.6] * len(df), name="Net P&L", customdata=custom,
        hovertemplate="%{x|%d %b %Y}<br>net Rs %{y:,.2f}<br>gross Rs %{customdata[1]:,.2f}"
                      "<br>charges Rs %{customdata[2]:,.2f}<br>%{customdata[0]} exit(s)<extra></extra>",
    ))
    fig.update_yaxes(tickprefix="Rs ", tickformat=",.0f", zeroline=True)
    fig.update_xaxes(type="date")
    return _layout(fig, title=title, height=300, legend=False, hovermode="closest")


def group_bar_figure(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    *,
    title: str | None = None,
    money: bool = True,
    signed: bool = True,
    hover_extra: str | None = None,
) -> go.Figure:
    """Horizontal bars of one metric per group (mode / strategy / regime / agent ...).

    ``signed`` colours by sign (diverging pair); otherwise one series colour. ``hover_extra``
    names a column shown in the hover (e.g. ``trades``).
    """
    if df is None or df.empty or value_col not in df.columns:
        return empty_figure()
    data = df[[c for c in {label_col, value_col, hover_extra} if c]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").replace([math.inf, -math.inf], math.nan)
    data = data.dropna(subset=[value_col])
    if data.empty:
        return empty_figure()
    data[label_col] = data[label_col].fillna("UNKNOWN").astype(str)
    if label_col == "mode":
        data = data.sort_values(label_col, key=lambda s: s.map(lambda m: _MODE_ORDER.get(m, 9)), ascending=False)
    else:
        data = data.sort_values(value_col)
    values = list(data[value_col])
    colors = _signed_colors(values) if signed else [SERIES[0]] * len(values)
    text = [fmt_rs(v, signed=signed, nd=0) if money else fmt_num(v, 2) for v in values]
    value_fmt = "Rs %{x:,.2f}" if money else "%{x:,.2f}"
    extra = f"<br>{hover_extra}: %{{customdata}}" if hover_extra else ""
    fig = go.Figure(go.Bar(
        x=values, y=list(data[label_col]), orientation="h", marker_color=colors, text=text, textposition="outside",
        cliponaxis=False, customdata=list(data[hover_extra]) if hover_extra else None,
        hovertemplate=f"%{{y}}: {value_fmt}{extra}<extra></extra>", name=value_col,
    ))
    lo, hi = min(values + [0.0]), max(values + [0.0])
    pad = (hi - lo) * 0.25 or 1.0
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)], zeroline=True,
                     tickprefix="Rs " if money else "", tickformat=",.0f" if money else None)
    return _layout(fig, title=title, height=90 + 40 * len(data), legend=False, hovermode="closest")


def mode_split_figure(summary: pd.DataFrame, value_col: str = "net_pnl", *, title: str | None = None) -> go.Figure:
    """Net P&L (or another metric) per mode from ``analytics.metrics.summarize(by="mode")``."""
    return group_bar_figure(summary, "mode", value_col, title=title, money=value_col.endswith("pnl"),
                            hover_extra="trades" if summary is not None and "trades" in summary.columns else None)


# =========================================================================== ledger


def ledger_balances_figure(ledger: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Running TRADING, RESERVE and sizing capital after every ledger movement (step lines)."""
    if ledger is None or ledger.empty:
        return empty_figure("No ledger entries yet")
    df = ledger.sort_values(["ts", "id"], kind="stable")
    fig = go.Figure()
    for i, (col, name) in enumerate((("trading", "TRADING"), ("reserve", "RESERVE"),
                                     ("sizing_capital", "Sizing capital"))):
        fig.add_trace(go.Scatter(x=list(df["ts"]), y=df[col], name=name, mode="lines", line_shape="hv",
                                 line=dict(width=2, color=SERIES[i]),
                                 hovertemplate=f"{name} Rs %{{y:,.2f}}<extra></extra>"))
    fig.update_yaxes(tickprefix="Rs ", tickformat=",.0f", rangemode="tozero")
    fig.update_xaxes(type="date")
    return _layout(fig, title=title, height=320)


# =========================================================================== trades


def trade_figure(
    trade: dict[str, Any],
    candles: pd.DataFrame | None = None,
    *,
    legs: pd.DataFrame | None = None,
    title: str | None = None,
) -> go.Figure:
    """Candlestick (when candles are available) or a price axis with the trade's timestamps,
    plus entry / SL / T1 / T2 / exit levels and entry/exit markers."""
    entry_t = trade.get("entry_time")
    exit_t = trade.get("exit_time") or entry_t
    entry = trade.get("entry_price")
    exit_px = trade.get("exit_price")
    if not isinstance(entry_t, datetime) or entry is None:
        return empty_figure("Trade has no entry data")
    fig = go.Figure()
    if candles is not None and not candles.empty:
        fig.add_trace(go.Candlestick(
            x=list(candles["ts"]), open=candles["open"], high=candles["high"], low=candles["low"],
            close=candles["close"], name=str(trade.get("symbol") or "price"), showlegend=False,
            increasing=dict(line=dict(color=POSITIVE), fillcolor=POSITIVE),
            decreasing=dict(line=dict(color=NEGATIVE), fillcolor=NEGATIVE),
        ))
        x0, x1 = candles["ts"].min(), candles["ts"].max()
    else:
        span = exit_t - entry_t if isinstance(exit_t, datetime) else timedelta(0)
        pad = max(span * 0.15, timedelta(minutes=15))
        x0, x1 = entry_t - pad, exit_t + pad
        if exit_px is not None:
            fig.add_trace(go.Scatter(x=[entry_t, exit_t], y=[entry, exit_px], mode="lines", name="Trade path",
                                     line=dict(color=NEUTRAL, width=1.5, dash="dot"), hoverinfo="skip"))
    levels = (("Entry", entry, INK, "solid"), ("SL", trade.get("stop_loss"), TIER_COLORS["HALT"], "dash"),
              ("T1", trade.get("target_1"), POSITIVE, "dash"), ("T2", trade.get("target_2"), POSITIVE, "dot"),
              ("Exit", exit_px, SERIES[1], "solid"))
    for name, price, color, dash in levels:
        if price is None or _bad(price):
            continue
        fig.add_trace(go.Scatter(x=[x0, x1], y=[price, price], mode="lines", name=f"{name} {float(price):,.2f}",
                                 line=dict(color=color, width=1.5, dash=dash),
                                 hovertemplate=f"{name} {float(price):,.2f}<extra></extra>"))
    is_long = str(trade.get("direction")) == "LONG"
    fig.add_trace(go.Scatter(x=[entry_t], y=[entry], mode="markers", name="Entry fill",
                             marker=dict(symbol="triangle-up" if is_long else "triangle-down", size=13, color=INK,
                                         line=dict(width=2, color=SURFACE)),
                             hovertemplate="Entry %{y:,.2f} at %{x|%d %b %H:%M}<extra></extra>"))
    exits = legs if legs is not None and not legs.empty else None
    if exits is not None:
        xs, ys, labels = list(exits["exit_time"]), list(exits["exit_price"]), list(exits["exit_reason"])
    elif exit_px is not None:
        xs, ys, labels = [exit_t], [exit_px], [trade.get("exit_reason") or "EXIT"]
    else:
        xs, ys, labels = [], [], []
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="Exit fill(s)", customdata=labels,
                                 marker=dict(symbol="x", size=12, color=SERIES[1], line=dict(width=2, color=SURFACE)),
                                 hovertemplate="Exit %{y:,.2f} (%{customdata}) at %{x|%d %b %H:%M}<extra></extra>"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(type="date")
    fig.update_yaxes(tickformat=",.2f")
    return _layout(fig, title=title, height=380, hovermode="closest")


# =========================================================================== me vs system / rejected


def me_vs_system_figure(summary: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """My P&L vs the system-alone P&L per decision category (grouped horizontal bars)."""
    if summary is None or summary.empty:
        return empty_figure("No decided approvals or manual trades yet")
    df = summary.iloc[::-1]
    fig = go.Figure()
    for i, (col, name) in enumerate((("my_pnl", "Me"), ("system_pnl", "System alone"))):
        fig.add_trace(go.Bar(y=list(df["label"]), x=df[col], orientation="h", name=name, marker_color=SERIES[i],
                             hovertemplate=f"{name}: Rs %{{x:,.2f}}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickprefix="Rs ", tickformat=",.0f", zeroline=True)
    return _layout(fig, title=title, height=120 + 56 * len(df), hovermode="y unified")


def rejected_figure(summary: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Net hypothetical (shadow) P&L per rejection category: blue = profit missed,
    red = loss avoided."""
    if summary is None or summary.empty:
        return empty_figure("Nothing rejected yet")
    return group_bar_figure(summary, "label", "net_shadow_pnl", title=title, money=True, hover_extra="count")


# =========================================================================== performance / agents


def agent_latency_figure(health: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Average and p95 latency per agent (grouped horizontal bars, milliseconds)."""
    if health is None or health.empty:
        return empty_figure("No agent calls recorded yet")
    df = health.dropna(subset=["avg_latency_ms"]).sort_values("avg_latency_ms")
    if df.empty:
        return empty_figure("No latency recorded yet")
    fig = go.Figure()
    for i, (col, name) in enumerate((("avg_latency_ms", "Average"), ("p95_latency_ms", "p95"))):
        fig.add_trace(go.Bar(y=list(df["agent"]), x=df[col], orientation="h", name=name, marker_color=SERIES[i],
                             hovertemplate=f"{name}: %{{x:,.0f}} ms<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(ticksuffix=" ms", tickformat=",.0f")
    return _layout(fig, title=title, height=120 + 52 * len(df), hovermode="y unified")


def failure_rate_figure(health: pd.DataFrame, *, title: str | None = None) -> go.Figure:
    """Failure rate % per agent (single series)."""
    if health is None or health.empty:
        return empty_figure("No agent calls recorded yet")
    fig = group_bar_figure(health, "agent", "failure_rate_pct", title=title, money=False, signed=False,
                           hover_extra="calls")
    fig.update_xaxes(ticksuffix="%")
    return fig


# =========================================================================== batches


def batch_equity_figure(
    curves: pd.DataFrame, *, y: str = "return_pct", x: str = "day_n", title: str | None = None
) -> go.Figure:
    """Overlaid equity curves, one line per batch (colour follows the batch's position in the
    selection, max 8)."""
    if curves is None or curves.empty:
        return empty_figure("No equity history for the selected batches")
    fig = go.Figure()
    for i, (_bid, g) in enumerate(curves.groupby("batch_id", sort=False)):
        if i >= len(SERIES):
            break
        name = str(g["batch"].iloc[0])
        value_fmt = "%{y:+.2f}%" if y == "return_pct" else "Rs %{y:,.2f}"
        fig.add_trace(go.Scatter(x=list(g[x]), y=g[y], name=name, mode="lines", line=dict(width=2, color=SERIES[i]),
                                 hovertemplate=f"{name}: {value_fmt}<extra></extra>"))
    if y == "return_pct":
        fig.update_yaxes(ticksuffix="%", zeroline=True)
    else:
        fig.update_yaxes(tickprefix="Rs ", tickformat=",.0f")
    if x == "day_n":
        fig.update_xaxes(title_text="days since batch start")
    else:
        fig.update_xaxes(type="date")
    return _layout(fig, title=title, height=360)
