"""Advanced analytics: Monte Carlo risk-of-ruin, agent calibration, stocks-in-play ranking."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from analytics.calibration import agent_calibration, decision_outcomes
from analytics.montecarlo import monte_carlo, outcomes_from_r
from core.clock import IST, SimulatedClock
from core.enums import Timeframe
from storage.models import AgentReportRow, DecisionRow, ShadowTrade, Trade
from strategies.ranking import rank_in_play
from tests.helpers import make_daily_candles, make_intraday_candles
from tests.paper_engine.paper_fakes import FakeData

NOW = datetime(2026, 11, 25, 11, 0, tzinfo=IST)


# ---------------------------------------------------------------- Monte Carlo


def test_monte_carlo_all_winners_never_halts() -> None:
    r = monte_carlo([5.0, 10.0, 3.0], 1000.0, n_trades=50, n_paths=500)
    assert r is not None and r.prob_profit == 1.0 and r.prob_halt == 0.0
    assert r.median_final > 1000 and r.p95_max_dd_pct == 0.0
    assert len(r.bands["p50"]) == 51 and r.reliable is False


def test_monte_carlo_all_losers_halts_and_stops_trading() -> None:
    r = monte_carlo([-10.0], 1000.0, n_trades=100, n_paths=100, hard_stop_pct=9.0)
    assert r is not None and r.prob_halt == 1.0 and r.prob_profit == 0.0
    # trading stops at the first equity <= 910 (exactly like the live HALT): 9 losses of 10
    assert r.median_final == pytest.approx(910.0)


def test_monte_carlo_deterministic_and_empty() -> None:
    a = monte_carlo([-5, 8, -3, 12, -6], 1000.0, seed=7)
    b = monte_carlo([-5, 8, -3, 12, -6], 1000.0, seed=7)
    assert a == b
    assert monte_carlo([], 1000.0) is None
    assert outcomes_from_r([1.0, -1.0, None], 10.0) == [10.0, -10.0]  # type: ignore[list-item]


# ---------------------------------------------------------------- calibration


def _decision(session, did: int, direction: str = "LONG") -> None:  # type: ignore[no-untyped-def]
    session.add(DecisionRow(id=did, batch_id=1, symbol="X", mode="INTRADAY", decision="BUY", direction=direction,
                            created_at=NOW))


def _report(session, did: int, agent: str, signal: str, conf: int = 70) -> None:  # type: ignore[no-untyped-def]
    session.add(AgentReportRow(batch_id=1, decision_id=did, symbol="X", mode="INTRADAY", agent=agent, signal=signal,
                               confidence=conf, status="OK", report_json={}, created_at=NOW))


def test_agent_calibration(session) -> None:  # type: ignore[no-untyped-def]
    for did in range(1, 25):
        win = did % 4 != 0          # 18 winners, 6 losers (all LONG setups)
        _decision(session, did)
        session.add(Trade(batch_id=1, position_id=did, symbol="X", mode="INTRADAY", product="MIS", direction="LONG",
                          qty=1, entry_time=NOW, entry_price=100, exit_time=NOW, exit_price=101 if win else 99,
                          stop_loss=99, exit_reason="TARGET" if win else "SL", gross_pnl=1 if win else -1, charges=0.1,
                          net_pnl=0.9 if win else -1.1, r_multiple=1.0 if win else -1.0, source="SYSTEM",
                          decision_id=did))
        _report(session, did, "technical", "BULLISH", 80)       # right 75 % of the time
        _report(session, did, "news_sentiment", "BEARISH", 60)  # right 25 % of the time
    session.add(ShadowTrade(batch_id=1, decision_id=99, symbol="Y", mode="INTRADAY", direction="LONG",
                            entry_price=100, stop_loss=99, target_1=102, rejection_source="USER", created_at=NOW,
                            status="CLOSED", outcome="SL", r_multiple=-1.0))
    session.flush()
    out = decision_outcomes(session, 1)
    assert out[1] == 1.0 and out[4] == -1.0 and out[99] == -1.0
    df = agent_calibration(session, 1, {"INTRADAY": {"technical": 1.5, "news_sentiment": 0.6}})
    tech = df[df.agent == "technical"].iloc[0]
    news = df[df.agent == "news_sentiment"].iloc[0]
    assert tech.hit_rate == pytest.approx(0.75) and tech.suggested_weight == pytest.approx(2.25)
    assert news.hit_rate == pytest.approx(0.25) and news.suggested_weight == pytest.approx(0.3)
    assert tech.brier < news.brier


def test_calibration_needs_min_samples(session) -> None:  # type: ignore[no-untyped-def]
    _decision(session, 1)
    session.add(ShadowTrade(batch_id=1, decision_id=1, symbol="X", mode="INTRADAY", direction="LONG",
                            entry_price=100, stop_loss=99, target_1=102, rejection_source="META", created_at=NOW,
                            status="CLOSED", outcome="TARGET_1", r_multiple=2.0))
    _report(session, 1, "technical", "BEARISH")
    session.flush()
    df = agent_calibration(session, 1, {"INTRADAY": {"technical": 1.5}})
    assert df.iloc[0].hit_rate == 0.0 and df.iloc[0].suggested_weight == 1.5   # too few samples: unchanged


# ---------------------------------------------------------------- stocks in play


def test_rank_in_play_prefers_movers() -> None:
    clock = SimulatedClock(NOW)
    data = FakeData(clock)
    days = [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 25)]
    calm = make_intraday_candles(days, 5, 100.0, seed=1, vol=0.0005)
    hot = make_intraday_candles(days, 5, 100.0, seed=2, vol=0.0005)
    today = hot.index.date == date(2026, 11, 25)
    hot.loc[today, ["open", "high", "low", "close"]] *= 1.03            # gapped up 3 %
    hot.loc[today, "volume"] *= 4                                         # 4x volume
    for sym, df in (("CALM", calm), ("HOT", hot), ("NIFTY50", calm)):
        data.add(sym, Timeframe.M5, df)
        data.add(sym, Timeframe.D1, make_daily_candles(date(2026, 11, 23), 30, 100.0, seed=3, vol=0.01))
    ranked = rank_in_play(["CALM", "HOT", "NODATA"], data, NOW)
    assert [r.symbol for r in ranked][:2] == ["HOT", "CALM"] and ranked[-1].symbol == "NODATA"
    assert ranked[0].score > ranked[1].score and ranked[-1].score == 0
    assert rank_in_play(["CALM", "HOT"], data, NOW, top_n=1)[0].symbol == "HOT"
