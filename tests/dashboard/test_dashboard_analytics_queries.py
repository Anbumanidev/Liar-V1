"""Unit tests for dashboard.analytics_queries and dashboard.charts (in-memory DB, no Streamlit)."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
from sqlalchemy.orm import Session

from core.clock import IST
from dashboard import analytics_queries as aq
from dashboard import charts as ch
from storage.models import Batch, DecisionRow, Trade
from tests.dashboard.test_dashboard_analytics_seed import NOW, seed_analytics_db, seed_days

DAYS = seed_days(NOW)
P = aq.RiskParams()  # hard stop 9 %, tiers 3 / 6 %, gap buffer 3 %, budget divisor 3


@pytest.fixture
def ids(session: Session) -> dict[str, int]:
    return seed_analytics_db(session, NOW)


# =========================================================================== protection math


def test_dashboard_analytics_protection_math() -> None:
    assert aq.floors(1000.0, 1034.0, 9.0) == pytest.approx((910.0, 940.94))
    assert aq.drawdown_pct(969.0, 1034.0) == pytest.approx(6.2862669)
    assert aq.drawdown_pct(1100.0, 1034.0) == 0.0
    assert aq.drawdown_pct(10.0, 0.0) == 0.0
    assert aq.tier_for(2.99, 1003.0, 940.0, P) == "NORMAL"
    assert aq.tier_for(3.0, 1003.0, 940.0, P) == "CAUTION"
    assert aq.tier_for(6.0, 972.0, 940.0, P) == "DEFENSIVE"
    assert aq.tier_for(9.0, 941.0, 900.0, P) == "HALT"
    assert aq.tier_for(1.0, 940.0, 940.0, P) == "HALT"  # floor reached wins over a small drawdown
    # LONG in profit: risk measured from the LTP too (gives back the open gain)
    assert aq.position_risk("LONG", 4, 100.0, 98.0, 103.0, 0.4) == pytest.approx(20.4)
    # SHORT with the SL trailed below entry: nothing at risk from the entry (only charges) ...
    assert aq.position_risk("SHORT", 10, 85.0, 84.0, None, 1.0) == pytest.approx(1.0)
    # ... but the open MTM gain down to the LTP can still be given back
    assert aq.position_risk("SHORT", 10, 85.0, 84.0, 80.0, 1.0) == pytest.approx(41.0)
    # overnight gap buffer dominates a tight stop
    assert aq.position_risk("LONG", 2, 450.0, 449.0, None, 0.0, overnight=True, gap_buffer_pct=3.0) == pytest.approx(27.0)
    assert aq.position_risk("LONG", 0, 450.0, 449.0) == 0.0
    params = aq.RiskParams.from_settings(None)
    assert (params.hard_stop_pct, params.caution_pct, params.defensive_pct) == (9.0, 3.0, 6.0)


def test_dashboard_analytics_risk_params_from_settings(settings) -> None:  # noqa: ANN001 - conftest fixture
    params = aq.RiskParams.from_settings(settings)
    assert params.hard_stop_pct == settings.risk.hard_stop_pct
    assert params.caution_pct == settings.risk.drawdown.caution_pct
    assert params.defensive_pct == settings.risk.drawdown.defensive_pct
    assert params.loss_budget_divisor == settings.risk.loss_budget_divisor


def test_dashboard_analytics_date_helpers() -> None:
    d1, d2 = date(2026, 11, 20), date(2026, 11, 23)
    assert aq.normalize_date_range((d2, d1)) == (d1, d2)
    assert aq.normalize_date_range((d1,)) == (d1, None)
    assert aq.normalize_date_range(d1) == (d1, d1)
    assert aq.normalize_date_range(()) == (None, None)
    assert aq.normalize_date_range(None) == (None, None)
    ts = pd.Series([datetime(2026, 11, 19, 23, 59), datetime(2026, 11, 20, 0, 0), datetime(2026, 11, 23, 23, 59),
                    datetime(2026, 11, 24, 0, 0), None])
    assert list(aq.date_mask(ts, d1, d2)) == [False, True, True, False, False]
    assert aq.date_mask(ts, None, None).all()
    lo, hi = aq.date_bounds(d1, d2)
    assert lo == datetime(2026, 11, 20, tzinfo=IST) and hi == datetime(2026, 11, 24, tzinfo=IST)
    assert aq.naive_ist(datetime(2026, 11, 20, 4, 0, tzinfo=IST).astimezone(tz=None).astimezone(IST)) == \
        datetime(2026, 11, 20, 4, 0)


# =========================================================================== capital & equity


def test_dashboard_analytics_capital_snapshot(session: Session, ids: dict[str, int]) -> None:
    snap = aq.capital_snapshot(session, ids["batch"], P)
    assert snap is not None
    assert snap["trading"] == pytest.approx(988.01)            # cash 588.01 + blocked 400
    assert snap["reserve"] == pytest.approx(8.29)
    assert snap["unrealized"] == pytest.approx(11.6)            # (103 - 100) x 4 - 0.4 entry charges
    assert snap["equity"] == pytest.approx(1007.9)              # TRADING + unrealized + RESERVE
    assert snap["hwm"] == pytest.approx(1034.0)
    assert snap["drawdown_pct"] == pytest.approx(2.524, abs=1e-3)
    assert snap["tier"] == "NORMAL"
    assert (snap["abs_floor"], snap["trailing_floor"], snap["effective_floor"]) == pytest.approx((910.0, 940.94, 940.94))
    assert snap["open_risk"] == pytest.approx(20.4)
    assert snap["loss_budget"] == pytest.approx(1007.9 - 20.4 - 940.94)
    assert snap["max_new_trade_risk"] == pytest.approx((1007.9 - 20.4 - 940.94) / 3, abs=0.01)
    assert snap["budget_left_pct"] == pytest.approx(46.56 / 66.96 * 100, abs=0.1)
    assert snap["net_since_start"] == pytest.approx(7.9) and snap["return_pct"] == pytest.approx(0.79)
    assert aq.capital_snapshot(session, None) is None
    assert aq.capital_snapshot(session, 999) is None
    point = aq.live_point(snap, NOW)
    assert point is not None and point["ts"] == NOW.replace(tzinfo=None) and point["equity"] == pytest.approx(1007.9)
    assert aq.live_point(None, NOW) is None


def test_dashboard_analytics_halted_account_is_halt_tier(session: Session, ids: dict[str, int]) -> None:
    from storage.models import CapitalAccount

    session.get(CapitalAccount, ids["batch"]).halted = True
    session.flush()
    assert aq.capital_snapshot(session, ids["batch"], P)["tier"] == "HALT"


def test_dashboard_analytics_equity_series_eod_live_and_levels(session: Session, ids: dict[str, int]) -> None:
    snap = aq.capital_snapshot(session, ids["batch"], P)
    series = aq.equity_series(session, ids["batch"], live=aq.live_point(snap, NOW))
    assert list(series["source"]) == ["START", "EOD", "EOD", "EOD", "LIVE"]
    assert list(series["equity"]) == pytest.approx([1000.0, 1034.0, 969.0, 996.3, 1007.9])
    assert list(series["trading"]) == pytest.approx([1000.0, 1023.8, 963.9, 983.01, 988.01])
    assert list(series["reserve"]) == pytest.approx([0.0, 10.2, 5.1, 13.29, 8.29])
    assert list(series["hwm"]) == pytest.approx([1000.0, 1034.0, 1034.0, 1034.0, 1034.0])
    assert series["ts"].is_monotonic_increasing
    assert series["ts"].iloc[1] == datetime.combine(DAYS["D3"], aq.EOD_TIME)
    levels = aq.with_protection_levels(series, 1000.0, P)
    assert list(levels["tier"]) == ["NORMAL", "NORMAL", "DEFENSIVE", "CAUTION", "NORMAL"]
    assert levels["abs_floor"].eq(910.0).all()
    assert list(levels["trailing_floor"]) == pytest.approx([910.0, 940.94, 940.94, 940.94, 940.94])
    assert levels["caution_level"].iloc[-1] == pytest.approx(1034.0 * 0.97)
    assert levels["defensive_level"].iloc[-1] == pytest.approx(1034.0 * 0.94)
    assert list(levels["drawdown_pct"]) == pytest.approx([0.0, 0.0, 6.286, 3.646, 2.524], abs=1e-3)
    # a live point stamped before the last EOD point is moved after it (never draws backwards)
    early = aq.equity_series(session, ids["batch"], live={**aq.live_point(snap, NOW),
                                                         "ts": datetime.combine(DAYS["D3"], aq.EOD_TIME)})
    assert early["ts"].is_monotonic_increasing and early["source"].iloc[-1] == "LIVE"


def test_dashboard_analytics_equity_series_fallbacks(session: Session, ids: dict[str, int]) -> None:
    old = aq.equity_series(session, ids["old_batch"])
    assert set(old["source"]) == {"LEDGER"}
    assert list(old["equity"]) == pytest.approx([1000.0, 1050.0, 1050.0, 1070.0, 1070.0])
    assert list(old["hwm"]) == pytest.approx([1000.0, 1050.0, 1050.0, 1070.0, 1070.0])
    # a batch with trades only (no EOD rows, no ledger): starting capital + cumulative net
    start = NOW - timedelta(days=3)
    b = Batch(name="trades-only", starting_capital=500.0, status="CLOSED", created_at=start, started_at=start)
    session.add(b)
    session.flush()
    for k, pnl in enumerate((10.0, -4.0)):
        t = start + timedelta(days=1 + k)
        session.add(Trade(batch_id=b.id, position_id=50 + k, symbol="IOC", mode="INTRADAY", product="MIS",
                          direction="LONG", qty=1, entry_time=t, entry_price=100.0, exit_time=t + timedelta(hours=1),
                          exit_price=100.0 + pnl, stop_loss=95.0, exit_reason="TARGET", gross_pnl=pnl, charges=0.0,
                          net_pnl=pnl, source="SYSTEM"))
    session.flush()
    only = aq.equity_series(session, b.id)
    assert list(only["source"]) == ["START", "TRADES", "TRADES"]
    assert list(only["equity"]) == pytest.approx([500.0, 510.0, 506.0])
    assert list(only["hwm"]) == pytest.approx([500.0, 510.0, 510.0])
    assert aq.equity_series(session, None).empty
    empty_levels = aq.with_protection_levels(aq.equity_series(session, None), 1000.0, P)
    assert empty_levels.empty and set(aq.LEVEL_COLUMNS) <= set(empty_levels.columns)


def test_dashboard_analytics_open_positions_and_regime(session: Session, ids: dict[str, int]) -> None:
    pos = aq.open_positions_frame(session, ids["batch"], now=NOW, params=P)
    assert len(pos) == 1
    row = pos.iloc[0]
    assert row["symbol"] == "PNB" and row["unrealized"] == pytest.approx(11.6)
    assert row["r_now"] == pytest.approx(1.5)                  # +3 per share on a 2-point initial risk
    assert row["risk_to_sl"] == pytest.approx(20.4) and row["price_age_s"] == 20 and not row["overnight"]
    assert aq.open_positions_frame(session, None).empty
    regime = aq.regime_rows({"INTRADAY": {"regime": "TRENDING_UP", "adx": 28.5}, "SWING": {"regime": "RANGING"},
                             "vix": 13.2})
    assert regime.to_dict("records") == [
        {"scope": "INTRADAY", "regime": "TRENDING_UP", "detail": '{"adx": 28.5}'},
        {"scope": "SWING", "regime": "RANGING", "detail": ""},
        {"scope": "vix", "regime": "13.2", "detail": ""},
    ]
    flat = aq.regime_rows({"regime": "RANGING", "adx": 18})
    assert flat.to_dict("records") == [{"scope": "market", "regime": "RANGING", "detail": '{"adx": 18}'}]
    assert aq.regime_rows(None).empty


# =========================================================================== ledger


def test_dashboard_analytics_ledger_running_balances(session: Session, ids: dict[str, int]) -> None:
    full = aq.ledger_frame(session, ids["batch"])
    assert len(full) == 22 and full["ts"].is_monotonic_increasing
    assert (full["trading"] == full["trading_cash"] + full["blocked"]).all()
    assert (full["total"] - (full["trading"] + full["reserve"])).abs().max() < 1e-9
    last = full.iloc[-1]
    assert (last["trading"], last["reserve"], last["total"]) == pytest.approx((988.01, 8.29, 996.3))
    assert (last["cum_locked"], last["cum_refilled"], last["cum_manual_transfer"], last["cum_realized"]) == \
        pytest.approx((18.39, 5.1, 5.0, -3.7))
    # start + realized + manual adjustments == TRADING + RESERVE after every row (section 6.4)
    assert ((1000.0 + full["cum_realized"]) - full["total"]).abs().max() < 1e-6
    summary = aq.ledger_summary(full)
    assert summary["rows"] == 22 and summary["sizing_changes"] == 1
    assert (summary["locked"], summary["refilled"], summary["manual_transfers"], summary["realized"]) == \
        pytest.approx((18.39, 5.1, 5.0, -3.7))
    assert summary["last"]["sizing_capital"] == pytest.approx(963.9)
    by_type = summary["by_type"].set_index("entry_type")
    assert by_type.loc["BLOCK", "count"] == 5 and by_type.loc["PROFIT_LOCK", "amount"] == pytest.approx(18.39)
    # filters keep the running totals computed on the whole ledger
    locks = aq.filter_ledger(full, entry_types=["PROFIT_LOCK", "REFILL", "MANUAL_TRANSFER"])
    assert list(locks["entry_type"]) == ["PROFIT_LOCK", "REFILL", "PROFIT_LOCK", "MANUAL_TRANSFER"]
    assert list(locks["cum_locked"]) == pytest.approx([10.2, 10.2, 18.39, 18.39])
    d2 = aq.filter_ledger(full, start=DAYS["D2"], end=DAYS["D2"])
    assert len(d2) == 6 and set(d2["ts"].dt.date) == {DAYS["D2"]}
    assert list(aq.filter_ledger(full, search="TOP UP")["entry_type"]) == ["MANUAL_TRANSFER"]
    assert list(aq.filter_ledger(full, modes=["SWING"])["entry_type"]) == ["BLOCK", "RELEASE", "REALIZED_PNL"]
    old = aq.ledger_frame(session, ids["old_batch"])
    assert list(old["cum_locked"]) == pytest.approx([0.0, 0.0, 15.0, 15.0, 21.0])
    empty = aq.ledger_summary(aq.ledger_frame(session, None))
    assert empty["rows"] == 0 and empty["last"] is None


# =========================================================================== trades & performance


def test_dashboard_analytics_trades_metrics_per_mode_strategy_regime(session: Session, ids: dict[str, int]) -> None:
    trades = aq.trades_frame(session, ids["batch"])
    assert len(trades) == 5
    itc = trades[trades["symbol"] == "ITC"].iloc[0]
    assert itc["regime"] == "RANGING"                          # filled from the decision
    m = aq.overview_metrics(trades)
    assert (m["trades"], m["wins"], m["losses"]) == (4, 3, 1)   # partial legs merged per position
    assert m["net_pnl"] == pytest.approx(-3.7) and m["gross_pnl"] == pytest.approx(2.0)
    assert m["charges"] == pytest.approx(5.7) and m["win_rate_pct"] == pytest.approx(75.0)
    assert m["profit_factor"] == pytest.approx(61.3 / 65.0)
    assert m["expectancy_rs"] == pytest.approx(-0.925)
    assert m["expectancy_r"] == pytest.approx((1.89 - 1.08 + (1.57 - 0.1) / 2 + 1.85) / 4)
    by_mode = aq.performance_table(trades, "mode").set_index("mode")
    assert by_mode.loc["INTRADAY", "net_pnl"] == pytest.approx(61.3) and by_mode.loc["INTRADAY", "trades"] == 3
    assert by_mode.loc["SWING", "net_pnl"] == pytest.approx(-65.0)
    by_strategy = aq.performance_table(trades, "strategy").set_index("strategy")
    assert by_strategy["net_pnl"].to_dict() == pytest.approx(
        {"orb": 34.0, "swing_breakout": -65.0, "vwap_pullback": 8.8, "UNKNOWN": 18.5})
    by_regime = aq.performance_table(trades, "regime").set_index("regime")
    assert by_regime["net_pnl"].to_dict() == pytest.approx({"RANGING": -65.0, "TRENDING_UP": 42.8, "UNKNOWN": 18.5})
    assert by_regime.loc["TRENDING_UP", "win_rate_pct"] == pytest.approx(100.0)
    # filters
    opts = aq.trade_filter_options(trades)
    assert opts["modes"] == ["INTRADAY", "SWING"] and opts["sources"] == ["MANUAL", "SYSTEM"]
    assert aq.NO_STRATEGY in opts["strategies"]
    assert list(aq.filter_trades(trades, modes=["SWING"])["symbol"]) == ["ITC"]
    assert list(aq.filter_trades(trades, strategies=[aq.NO_STRATEGY])["symbol"]) == ["NHPC"]
    assert list(aq.filter_trades(trades, sources=["MANUAL"])["symbol"]) == ["NHPC"]
    assert list(aq.filter_trades(trades, start=DAYS["D1"], end=DAYS["D1"])["symbol"]) == ["TATAPOWER", "TATAPOWER",
                                                                                           "NHPC"]
    assert aq.filter_trades(trades, modes=["BTST"]).empty
    daily = aq.daily_pnl(trades)
    assert list(daily["day"]) == [DAYS["D3"], DAYS["D2"], DAYS["D1"]]
    assert list(daily["net_pnl"]) == pytest.approx([34.0, -65.0, 27.3])
    assert list(daily["cum_net_pnl"]) == pytest.approx([34.0, -31.0, -3.7])
    assert list(daily["exits"]) == [1, 1, 3]
    choices = aq.trade_choices(trades)
    assert [c[0] for c in choices][:2] == [ids["t5"], ids["t4"]] and "NHPC" in choices[0][1]
    assert aq.overview_metrics(aq.trades_frame(session, None))["trades"] == 0
    assert aq.daily_pnl(aq.trades_frame(session, None)).empty


def test_dashboard_analytics_agent_accuracy_and_health(session: Session, ids: dict[str, int]) -> None:
    trades = aq.trades_frame(session, ids["batch"])
    acc = aq.agent_accuracy(session, ids["batch"], trades).set_index("agent")
    assert list(acc.index) == ["sentiment", "technical"]       # the FAILED news call is not a vote
    tech = acc.loc["technical"]
    assert (tech["votes"], tech["agreed"], tech["opposed"]) == (3, 3, 0)
    assert tech["hit_rate_pct"] == pytest.approx(66.7) and tech["agreed_win_rate_pct"] == pytest.approx(66.7)
    assert tech["agreed_net_pnl"] == pytest.approx(34.0 - 65.0 + 8.8)
    assert tech["avg_confidence"] == pytest.approx(75.0)
    sent = acc.loc["sentiment"]
    assert (sent["votes"], sent["agreed"], sent["opposed"]) == (2, 0, 2)  # SKIPPED vote ignored
    assert sent["hit_rate_pct"] == pytest.approx(50.0) and pd.isna(sent["agreed_win_rate_pct"])
    health = aq.agent_health(session, ids["batch"]).set_index("agent")
    assert set(health.index) == {"technical", "sentiment", "news", "debate", "meta (decision)"}
    assert health.loc["news", "failure_rate_pct"] == pytest.approx(100.0)
    assert (health.loc["sentiment", "calls"], health.loc["sentiment", "skipped"],
            health.loc["sentiment", "failure_rate_pct"]) == (3, 1, 0.0)
    assert health.loc["technical", "avg_latency_ms"] == pytest.approx(1233.3)
    assert health.loc["technical", "p95_latency_ms"] == pytest.approx(1470.0)
    assert health.loc["technical", "max_latency_ms"] == pytest.approx(1500.0)
    assert health.loc["meta (decision)", "avg_latency_ms"] == pytest.approx(2750.0)
    assert health.loc["debate", "models"] == "qwen3:8b"
    assert aq.agent_health(session, None).empty
    assert aq.agent_accuracy(session, ids["batch"], aq.trades_frame(session, None)).empty


# =========================================================================== rejected & me vs system


def test_dashboard_analytics_rejected_merging(session: Session, ids: dict[str, int]) -> None:
    df = aq.rejected_frame(session, ids["batch"])
    got = sorted(zip(df["symbol"], df["category"], strict=True))
    assert got == sorted([
        ("WIPRO", "USER"), ("LT", "TIMEOUT"), ("GAIL", "NOT_FILLED"), ("AXISBANK", "META"),
        ("KOTAKBANK", "RISK"), ("SUNPHARMA", "SIGNALS_ONLY"), ("TITAN", "FILTER"), ("IRFC", "RISK"),
        ("HAL", "FILTER"),
    ])  # pending BEL, traded TATASTEEL/TATAPOWER and the modified ITC are not listed
    rows = df.set_index("symbol")
    assert rows.loc["WIPRO", "shadow_id"] == ids["sh_a3"] and rows.loc["WIPRO", "shadow_pnl"] == pytest.approx(24.0)
    assert "too extended" in rows.loc["WIPRO", "reason"]
    assert rows.loc["LT", "shadow_outcome"] == "SL" and rows.loc["LT", "shadow_pnl"] == pytest.approx(-30.0)
    assert rows.loc["AXISBANK", "shadow_outcome"] == "OPEN" and not rows.loc["AXISBANK", "shadow_closed"]
    assert "Conflicting signals" in rows.loc["AXISBANK", "reason"]
    assert rows.loc["KOTAKBANK", "rule"] == "MAX_OPEN_POSITIONS" and pd.isna(rows.loc["KOTAKBANK", "shadow_id"])
    assert rows.loc["IRFC", "rule"] == "RR_BELOW_MIN"
    assert rows.loc["SUNPHARMA", "shadow_pnl"] == 0.0           # NOT_TRIGGERED counts as 0
    assert rows.loc["TITAN", "reason"] == "Filtered: confluence 42 < 60"
    assert rows.loc["HAL", "shadow_id"] == ids["sh_orphan"]     # shadow without decision / approval
    assert df["ts"].is_monotonic_decreasing
    summary = aq.rejected_summary(df).set_index("category")
    assert summary.loc["FILTER", "count"] == 2 and summary.loc["FILTER", "missed_profit"] == pytest.approx(12.0)
    assert summary.loc["USER", "missed_profit"] == pytest.approx(24.0)
    assert summary.loc["TIMEOUT", "avoided_loss"] == pytest.approx(30.0)
    assert summary.loc["TIMEOUT", "net_shadow_pnl"] == pytest.approx(-30.0)
    assert summary.loc["META", "shadow_tracked"] == 1 and summary.loc["META", "shadow_closed"] == 0
    assert summary["missed_profit"].sum() == pytest.approx(36.0) and summary["avoided_loss"].sum() == pytest.approx(30.0)
    assert (summary["avoided_loss"] >= 0).all()
    today = aq.rejected_frame(session, ids["batch"], start=DAYS["D0"], end=DAYS["D0"])
    assert sorted(today["symbol"]) == ["AXISBANK", "IRFC", "KOTAKBANK", "WIPRO"]
    filtered = aq.filtered_stocks_frame(session, ids["batch"])
    assert list(filtered["symbol"]) == ["YESBANK"] and filtered.iloc[0]["reason"].startswith("illiquid")
    assert aq.rejected_frame(session, None).empty and aq.rejected_summary(aq.rejected_frame(session, None)).empty


def test_dashboard_analytics_me_vs_system(session: Session, ids: dict[str, int]) -> None:
    detail = aq.me_vs_system(session, ids["batch"])
    rows = {(r["symbol"], r["category"]): r for r in detail.to_dict("records")}
    assert set(rows) == {("TATASTEEL", "APPROVED"), ("ITC", "MODIFIED"), ("WIPRO", "REJECTED"), ("LT", "TIMEOUT"),
                         ("IRFC", "BLOCKED"), ("GAIL", "APPROVED"), ("NHPC", "MANUAL")}
    assert (rows[("TATASTEEL", "APPROVED")]["my_pnl"], rows[("TATASTEEL", "APPROVED")]["system_pnl"]) == (34.0, 34.0)
    itc = rows[("ITC", "MODIFIED")]
    assert (itc["my_pnl"], itc["system_pnl"], itc["edge"]) == pytest.approx((-65.0, -40.0, -25.0))
    assert rows[("WIPRO", "REJECTED")]["edge"] == pytest.approx(-24.0)   # rejecting a winner cost 24
    assert rows[("LT", "TIMEOUT")]["edge"] == pytest.approx(30.0)        # the timeout avoided a 30 loss
    assert rows[("NHPC", "MANUAL")]["edge"] == pytest.approx(18.5)
    gail = rows[("GAIL", "APPROVED")]
    assert gail["trade_status"] == "NO FILL" and gail["edge"] == 0.0
    irfc = rows[("IRFC", "BLOCKED")]
    assert not irfc["comparable"] and irfc["note"] == "not shadow-tracked"
    totals = aq.me_vs_system_totals(detail)
    assert totals == {"items": 7, "compared": 6, "my_pnl": pytest.approx(-12.5), "system_pnl": pytest.approx(-12.0),
                      "edge": pytest.approx(-0.5), "pending": 1}
    summary = aq.me_vs_system_summary(detail).set_index("category")
    assert list(summary.index) == ["APPROVED", "MODIFIED", "REJECTED", "TIMEOUT", "BLOCKED", "MANUAL"]
    assert summary.loc["APPROVED", "items"] == 2 and summary.loc["APPROVED", "edge"] == 0.0
    assert summary.loc["BLOCKED", "compared"] == 0
    assert summary["edge"].sum() == pytest.approx(totals["edge"])
    assert aq.me_vs_system(session, None).empty
    assert aq.me_vs_system_totals(aq.me_vs_system(session, None))["items"] == 0


def test_dashboard_analytics_me_vs_system_open_and_untracked(session: Session, ids: dict[str, int]) -> None:
    """An approved trade still open, and a modified proposal whose shadow is still open, are not compared."""
    from storage.models import Position, ShadowTrade

    session.get(Position, ids["p5"]).approval_id = ids["a7"]       # GAIL approval now owns an OPEN position
    sh = session.get(ShadowTrade, ids["sh_a2"])
    sh.status, sh.outcome, sh.hypothetical_net_pnl = "OPEN", "OPEN", None
    session.flush()
    rows = {r["symbol"]: r for r in aq.me_vs_system(session, ids["batch"]).to_dict("records")}
    assert rows["GAIL"]["trade_status"] == "OPEN" and not rows["GAIL"]["comparable"]
    assert not rows["ITC"]["comparable"] and rows["ITC"]["note"] == "original proposal shadow still open"


# =========================================================================== day-wise timeline


def test_dashboard_analytics_day_timeline(session: Session, ids: dict[str, int]) -> None:
    assert aq.activity_days(session, ids["batch"]) == [DAYS["D0"], DAYS["D1"], DAYS["D2"], DAYS["D3"]]
    d1 = aq.day_timeline(session, ids["batch"], DAYS["D1"])
    assert d1["ts"].is_monotonic_increasing
    assert set(d1["kind"]) == {"DECISION", "SHADOW", "ENTRY", "CAPITAL", "EXIT", "APPROVAL", "EOD"}
    exits = d1[d1["kind"] == "EXIT"]
    assert list(exits["symbol"]) == ["TATAPOWER", "TATAPOWER", "NHPC"]
    assert "(partial)" in exits.iloc[0]["detail"]
    eod = d1[d1["kind"] == "EOD"]
    assert eod["ref"].str.startswith("run:").sum() == 1         # the EOD run
    assert any("TOTAL: 2 trade(s)" in d for d in eod["detail"])
    assert any("equity 996.30" in d for d in eod["detail"])
    approvals = d1[d1["kind"] == "APPROVAL"]
    assert any(d.startswith("queued") for d in approvals["detail"]) and any("APPROVED by admin" in d
                                                                           for d in approvals["detail"])
    d0 = aq.day_timeline(session, ids["batch"], DAYS["D0"])
    assert {"RUN", "SCAN", "DECISION", "APPROVAL", "ORDER", "ENTRY", "CAPITAL", "SHADOW"} <= set(d0["kind"])
    scan = d0[d0["kind"] == "SCAN"]
    assert len(scan) == 3 and any("excluded: illiquid" in d for d in scan["detail"])
    no_scans = aq.day_timeline(session, ids["batch"], DAYS["D0"], include_scans=False)
    assert "SCAN" not in set(no_scans["kind"]) and len(no_scans) == len(d0) - 3
    d2 = aq.day_timeline(session, ids["batch"], DAYS["D2"])
    timeout = d2[(d2["kind"] == "APPROVAL") & (d2["symbol"] == "LT")]
    assert list(timeout["detail"]) == ["TIMEOUT"]               # stamped at its expiry
    assert aq.day_timeline(session, ids["batch"], date(2020, 1, 1)).empty
    assert aq.day_timeline(session, None, DAYS["D0"]).empty and aq.activity_days(session, None) == []


# =========================================================================== trade drill-down & candles


def test_dashboard_analytics_trade_detail_history(session: Session, ids: dict[str, int]) -> None:
    detail = aq.trade_detail(session, ids["t2"])
    assert detail is not None
    assert detail["trade"]["regime"] == "RANGING" and detail["trade"]["approval_id"] == ids["a2"]
    assert detail["approval"]["status"] == "MODIFIED" and detail["position"]["id"] == ids["p2"]
    assert [r["agent"] for r in detail["decision"].agent_reports] == ["sentiment", "technical"]
    events = list(detail["history"]["event"])
    assert events == ["QUEUED", "DECISION MODIFIED", "COMMAND MODIFY_APPROVE", "AUDIT APPROVAL_MODIFY",
                      "COMMAND TIGHTEN_SL", "SL CHANGE"]
    assert "wider stop below base" in detail["history"].iloc[1]["detail"]
    assert detail["history"]["ts"].is_monotonic_increasing
    legs = aq.trade_detail(session, ids["t3"])["legs"]
    assert list(legs["net_pnl"]) == pytest.approx([9.4, -0.6])
    t1 = aq.trade_detail(session, ids["t1"])
    assert len(t1["orders"]) == 2 and t1["decision"].debates[0]["rounds"] == 2
    assert [e for e in t1["history"]["event"]] == ["QUEUED", "DECISION APPROVED", "COMMAND APPROVE"]
    manual = aq.trade_detail(session, ids["t5"])
    assert manual["approval"] is None and manual["decision"].decision is None and manual["history"].empty
    assert aq.trade_detail(session, 99_999) is None


def test_dashboard_analytics_trade_candles(session: Session, ids: dict[str, int], tmp_path: Path) -> None:
    t1 = aq.trade_detail(session, ids["t1"])["trade"]
    candles, source = aq.trade_candles(session, t1, timeframe="5m", cache_dir=tmp_path)
    assert candles is None and source == "none"
    # candles stored with the decision (aware ISO strings, mixed key case) are used first
    entry = t1["entry_time"]
    bars = [{"Time": (entry + timedelta(minutes=5 * k)).replace(tzinfo=IST).isoformat(), "open": 150 + k,
             "high": 151 + k, "low": 149 + k, "close": 150.5 + k} for k in range(-3, 6)]
    dec = session.get(DecisionRow, ids["d_t1"])
    dec.context_json = {**(dec.context_json or {}), "candles": bars}
    session.flush()
    candles, source = aq.trade_candles(session, t1, timeframe="5m")
    assert source == "db" and len(candles) == 9 and candles["ts"].iloc[3] == entry
    assert list(candles.columns) == list(aq.CANDLE_COLUMNS)
    # daily candles from the market-data disk cache (never the network) for a SWING trade
    t2 = aq.trade_detail(session, ids["t2"])["trade"]
    idx = pd.date_range(end=pd.Timestamp(DAYS["D1"]), periods=60, freq="B", tz="Asia/Kolkata", name="ts")
    frame = pd.DataFrame({"Open": 440.0, "High": 455.0, "Low": 430.0, "Close": 445.0, "Volume": 1e5}, index=idx)
    folder = tmp_path / "candles" / "yfinance"
    folder.mkdir(parents=True)
    frame.to_parquet(folder / f"ITC_1d_{DAYS['D1']:%Y%m%d}_20260801.parquet")
    (folder / "ITC_1d_29991231_20260101.parquet").write_bytes(b"corrupt")   # skipped
    candles, source = aq.trade_candles(session, t2, timeframe="1d", cache_dir=tmp_path)
    assert source == "disk cache" and not candles.empty
    assert candles["ts"].min() >= t2["entry_time"] - timedelta(days=31)
    assert aq.trade_candles(session, t2, timeframe="5m", cache_dir=tmp_path) == (None, "none")
    assert aq.candles_from_records([{"x": 1}]) is None and aq.candles_from_records(None) is None


# =========================================================================== batches


def test_dashboard_analytics_batch_comparison_and_settings_diff(session: Session, ids: dict[str, int]) -> None:
    comp = aq.batch_comparison(session, [ids["batch"], ids["old_batch"]], now=NOW).set_index("batch")
    assert list(comp.index) == ["ana-cur", "ana-old"]
    cur, old = comp.loc["ana-cur"], comp.loc["ana-old"]
    assert cur["equity"] == pytest.approx(1007.9) and cur["return_pct"] == pytest.approx(0.79)
    assert cur["max_drawdown_pct"] == pytest.approx(6.29) and cur["trades"] == 4
    assert old["equity"] == pytest.approx(1070.0) and old["return_pct"] == pytest.approx(7.0)
    assert math.isinf(old["profit_factor"]) and old["win_rate_pct"] == pytest.approx(100.0)
    assert old["days"] == pytest.approx(20.0) and old["max_drawdown_pct"] == 0.0
    curves = aq.batch_equity_curves(session, [ids["batch"], ids["old_batch"]], now=NOW)
    assert list(curves["batch_id"].unique()) == [ids["batch"], ids["old_batch"]]
    first = curves.groupby("batch_id").first()
    assert (first["day_n"] == 0).all() and (first["return_pct"] == 0).all()
    assert curves[curves["batch_id"] == ids["old_batch"]]["return_pct"].iloc[-1] == pytest.approx(7.0)
    assert curves[curves["batch_id"] == ids["batch"]]["return_pct"].iloc[-1] == pytest.approx(0.79)
    snap_cur, src_cur = aq.batch_settings(session, ids["batch"])
    snap_old, src_old = aq.batch_settings(session, ids["old_batch"])
    assert src_cur == "batch snapshot" and src_old == f"settings version {ids['settings_v1']}"
    diff = aq.settings_diff_frame(snap_old, snap_cur)
    assert list(diff["path"]) == ["capital.profit_lock_pct"]
    assert float(diff.iloc[0]["b"]) == float(diff.iloc[0]["a"]) + 10.0
    assert aq.settings_diff_frame(snap_old, snap_old).empty
    assert aq.settings_diff_frame(None, snap_cur).empty
    assert aq.batch_settings(session, 99_999) == (None, "none")
    assert aq.batch_comparison(session, [99_999]).empty and aq.batch_equity_curves(session, []).empty


# =========================================================================== empty database


def test_dashboard_analytics_empty_database(session: Session) -> None:
    b = Batch(name="bare", starting_capital=1000.0, status="ACTIVE", created_at=NOW)
    session.add(b)
    session.flush()
    for bid in (None, b.id):
        assert aq.capital_snapshot(session, bid) is None
        assert aq.trades_frame(session, bid).empty
        assert aq.ledger_frame(session, bid).empty
        assert aq.rejected_frame(session, bid).empty
        assert aq.me_vs_system(session, bid).empty
        assert aq.agent_health(session, bid).empty
        assert aq.open_positions_frame(session, bid).empty
        assert aq.filtered_stocks_frame(session, bid).empty
        assert aq.day_timeline(session, bid, NOW.date()).empty
    only_start = aq.equity_series(session, b.id)
    assert list(only_start["source"]) == ["START"] and only_start["equity"].iloc[0] == 1000.0
    assert aq.batch_comparison(session, [b.id], now=NOW).iloc[0]["trades"] == 0
    assert aq.trade_choices(aq.trades_frame(session, b.id)) == []


# =========================================================================== charts


def _names(fig: go.Figure) -> list[str]:
    return [t.name for t in fig.data if t.name]


def test_dashboard_analytics_charts_on_seeded_data(session: Session, ids: dict[str, int]) -> None:
    snap = aq.capital_snapshot(session, ids["batch"], P)
    levels = aq.with_protection_levels(aq.equity_series(session, ids["batch"], live=aq.live_point(snap, NOW)),
                                       1000.0, P)
    fig = ch.equity_curve_figure(levels)
    names = _names(fig)
    for expected in ("TRADING", "RESERVE", "TOTAL EQUITY", "HWM", "Absolute floor", "Trailing floor"):
        assert expected in names
    assert [n for n in names if n.split()[0] in ("NORMAL", "CAUTION", "DEFENSIVE", "HALT")] == [
        "NORMAL (0-3% DD)", "CAUTION (3-6%)", "DEFENSIVE (6-9%)", "HALT (>= 9%)"]
    stacked = [t for t in fig.data if getattr(t, "stackgroup", None) == "capital"]
    assert [t.name for t in stacked] == ["TRADING", "RESERVE"]
    y0, y1 = fig.layout.yaxis.range
    assert y0 < 910.0 and y1 > 1034.0 and y0 > 0               # zoomed on the floor-to-HWM region
    assert ch.equity_curve_figure(levels, zoom=False).layout.yaxis.range[0] == 0.0
    trades = aq.trades_frame(session, ids["batch"])
    daily = ch.daily_pnl_figure(aq.daily_pnl(trades))
    assert list(daily.data[0].marker.color) == [ch.POSITIVE, ch.NEGATIVE, ch.POSITIVE]
    modes = ch.mode_split_figure(aq.performance_table(trades, "mode"))
    assert list(modes.data[0].y) == ["SWING", "INTRADAY"]       # INTRADAY drawn on top
    figs = [
        ch.ledger_balances_figure(aq.ledger_frame(session, ids["batch"])),
        ch.group_bar_figure(aq.performance_table(trades, "strategy"), "strategy", "profit_factor", money=False),
        ch.me_vs_system_figure(aq.me_vs_system_summary(aq.me_vs_system(session, ids["batch"]))),
        ch.rejected_figure(aq.rejected_summary(aq.rejected_frame(session, ids["batch"]))),
        ch.agent_latency_figure(aq.agent_health(session, ids["batch"])),
        ch.failure_rate_figure(aq.agent_health(session, ids["batch"])),
        ch.batch_equity_figure(aq.batch_equity_curves(session, [ids["batch"], ids["old_batch"]], now=NOW)),
        ch.batch_equity_figure(aq.batch_equity_curves(session, [ids["batch"]], now=NOW), y="equity", x="ts"),
    ]
    assert all(isinstance(f, go.Figure) and f.data for f in figs)
    assert _names(figs[2]) == ["Me", "System alone"]
    assert _names(figs[6]) == [f"#{ids['batch']} ana-cur", f"#{ids['old_batch']} ana-old"]
    detail = aq.trade_detail(session, ids["t3"])
    level_fig = ch.trade_figure(detail["trade"], None, legs=detail["legs"])
    assert {"Trade path", "Entry fill", "Exit fill(s)"} <= set(_names(level_fig))
    assert any(n.startswith("SL ") for n in _names(level_fig)) and any(n.startswith("T1 ") for n in _names(level_fig))
    assert len([t for t in level_fig.data if t.name == "Exit fill(s)"][0].x) == 2
    candles = pd.DataFrame({"ts": pd.date_range(detail["trade"]["entry_time"], periods=5, freq="5min"),
                            "open": 200.0, "high": 206.0, "low": 199.0, "close": 204.0})
    candle_fig = ch.trade_figure(detail["trade"], candles)
    assert isinstance(candle_fig.data[0], go.Candlestick) and "Trade path" not in _names(candle_fig)


def test_dashboard_analytics_charts_on_empty_data() -> None:
    empty = pd.DataFrame()
    figs = [
        ch.equity_curve_figure(empty), ch.daily_pnl_figure(empty), ch.mode_split_figure(empty),
        ch.group_bar_figure(empty, "mode", "net_pnl"), ch.ledger_balances_figure(empty),
        ch.trade_figure({}), ch.me_vs_system_figure(empty), ch.rejected_figure(empty),
        ch.agent_latency_figure(empty), ch.failure_rate_figure(empty), ch.batch_equity_figure(empty),
        ch.empty_figure("nothing"),
    ]
    for fig in figs:
        assert isinstance(fig, go.Figure)
        assert not fig.data and fig.layout.annotations          # placeholder text, no traces
    only_inf = pd.DataFrame({"mode": ["INTRADAY"], "profit_factor": [math.inf]})
    assert not ch.group_bar_figure(only_inf, "mode", "profit_factor", money=False).data


def test_dashboard_analytics_formatters() -> None:
    assert ch.fmt_rs(1234.5) == "Rs 1,234.50" and ch.fmt_rs(-3.7, signed=True) == "Rs -3.70"
    assert ch.fmt_rs(None) == "-" and ch.fmt_rs(float("nan")) == "-" and ch.fmt_rs(5, signed=True) == "Rs +5.00"
    assert ch.fmt_pct(2.5249) == "2.52%" and ch.fmt_pct(None) == "-" and ch.fmt_pct(1, signed=True) == "+1.00%"
    assert ch.fmt_pf(math.inf) == "inf (no losses)" and ch.fmt_pf(0.9431) == "0.94" and ch.fmt_pf(None) == "-"
    assert ch.fmt_num(1234.567, 1) == "1,234.6" and ch.fmt_num("x") == "-"
    assert aq.jsonish({"a": [1, 2]}) == '{"a": [1, 2]}' and aq.jsonish(None) == ""
    assert aq.jsonish("x" * 50, limit=10) == "xxxxxxx..."
