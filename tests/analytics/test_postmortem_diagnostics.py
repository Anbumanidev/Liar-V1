"""Trade post-mortem and daily log diagnostics (built from the 2026-11-25 losses)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from analytics.diagnostics import diagnose
from analytics.postmortem import CATALOG, analyze_day, analyze_position
from core.clock import IST
from storage.models import AgentReportRow, Batch, DebateRow, DecisionRow, Order, Position, Trade

DAY = date(2026, 11, 25)


def at(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2026, 11, 25, hh, mm, ss, tzinfo=IST)


def _seed_bel(session, *, opened: datetime, closed: datetime, entry: float, sl: float, t1: float, exit_px: float,
              data_at: datetime, valid_until: datetime, failed: int, bull: int, bear: int, room: float,
              batch_id: int) -> int:
    d = DecisionRow(batch_id=batch_id, symbol="BEL", mode="INTRADAY", strategy="ema_9_21_pullback", decision="BUY",
                    direction="LONG", confidence=70, confluence_score=73.4, grade="B", entry_price=404.3,
                    stop_loss=sl, target_1=t1, final_action="EXECUTED", latency_ms=1_500_000, created_at=opened,
                    meta_json={"reasons_for": ["trend"], "reasons_against_considered": ["all agents failed"]},
                    context_json={
                        "signals": [{"strategy": "ema_9_21_pullback", "valid_until": valid_until.isoformat()}],
                        "confluence": {"score": 73.4, "notes": [f"room: nearest opposing level 404.40 = {room}R"]},
                        "vote": {"bullish": 0, "bearish": 0, "neutral": 10, "failed": failed, "skipped": 1}})
    session.add(d)
    session.flush()
    for i in range(9):
        session.add(AgentReportRow(decision_id=d.id, batch_id=batch_id, symbol="BEL", mode="INTRADAY",
                                   agent=f"a{i}", signal="NEUTRAL", confidence=0,
                                   status="FAILED" if i < failed else "OK", report_json={}, created_at=opened))
    session.add(DebateRow(decision_id=d.id, batch_id=batch_id, symbol="BEL", mode="INTRADAY", rounds=2,
                          bull_json={"conviction": bull, "stance": "take it"}, bear_json={"conviction": bear,
                                                                                         "stance": "skip it"},
                          transcript_json=[], status="OK", created_at=opened))
    p = Position(batch_id=batch_id, symbol="BEL", mode="INTRADAY", product="MIS", direction="LONG", status="CLOSED",
                 qty=0, initial_qty=1, avg_entry=entry, stop_loss=sl, initial_stop_loss=sl, target_1=t1,
                 target_2=412.25, opened_at=opened, closed_at=closed, strategy="ema_9_21_pullback", grade="B",
                 margin_blocked=0.0, highest_price=entry, lowest_price=exit_px, decision_id=d.id,
                 sl_history_json=[{"ts": opened.isoformat(), "old": None, "new": sl, "by": "ENTRY"}])
    session.add(p)
    session.flush()
    session.add(Order(batch_id=batch_id, position_id=p.id, decision_id=d.id, symbol="BEL", mode="INTRADAY",
                      product="MIS", side="BUY", order_type="MARKET", purpose="ENTRY", qty=1, status="FILLED",
                      created_at=opened, filled_at=opened, fill_price=entry, is_paper=True,
                      meta_json={"proposal": {"entry_price": 404.3, "atr": 0.8462, "created_at": data_at.isoformat()}}))
    session.add(Trade(batch_id=batch_id, position_id=p.id, symbol="BEL", mode="INTRADAY", product="MIS",
                      direction="LONG", qty=1, entry_time=opened, entry_price=entry, exit_time=closed,
                      exit_price=exit_px, stop_loss=sl, target_1=t1, exit_reason="SL", gross_pnl=exit_px - entry,
                      charges=0.31, net_pnl=exit_px - entry - 0.31, source="SYSTEM"))
    session.flush()
    return p.id


@pytest.fixture
def bel(session):  # type: ignore[no-untyped-def]
    b = Batch(name="B1", starting_capital=1000.0, status="ACTIVE", created_at=at(9, 0))
    session.add(b)
    session.flush()
    first = _seed_bel(session, opened=at(12, 40, 46), closed=at(12, 43, 47), entry=402.45, sl=401.95, t1=409.9,
                      exit_px=401.7, data_at=at(12, 15), valid_until=at(12, 30), failed=9, bull=70, bear=85,
                      room=0.04, batch_id=b.id)
    second = _seed_bel(session, opened=at(12, 56, 56), closed=at(13, 15, 10), entry=402.55, sl=401.55, t1=405.07,
                       exit_px=401.3, data_at=at(12, 35), valid_until=at(12, 50), failed=9, bull=70, bear=95,
                       room=0.13, batch_id=b.id)
    session.commit()
    return b.id, first, second


def test_first_bel_trade_diagnosis(session, bel) -> None:  # type: ignore[no-untyped-def]
    _, first, _ = bel
    pm = analyze_position(session, first)
    codes = [i["code"] for i in pm["diagnosis"]]
    assert {"STALE_SIGNAL", "TIGHT_STOP", "AGENTS_FAILED", "DEBATE_LOST", "NO_ROOM", "MIDDAY_ENTRY"} <= set(codes)
    assert "REVENGE_REENTRY" not in codes
    assert codes == sorted(codes, key=lambda c: {"high": 0, "medium": 1, "low": 2}[CATALOG[c][0]])
    stale = next(i for i in pm["diagnosis"] if i["code"] == "STALE_SIGNAL")
    assert "25.8 min" in stale["detail"] and "expired at 12:30" in stale["detail"] and stale["fixed_by"]
    tight = next(i for i in pm["diagnosis"] if i["code"] == "TIGHT_STOP")
    assert "0.59 ATR" in tight["detail"]
    assert pm["result"] == "LOSS" and pm["net_pnl"] == -1.06 and pm["exit_reasons"] == ["SL"]
    assert any("9/10 LLM agents failed" in s or "9/9 LLM agents failed" in s for s in pm["story"])
    assert pm["decision"]["debate"] == {"bull": 70, "bear": 85, "bull_stance": "take it", "bear_stance": "skip it",
                                        "bull_points": [], "bear_points": []}


def test_reentry_after_stop_is_flagged(session, bel) -> None:  # type: ignore[no-untyped-def]
    _, _, second = bel
    pm = analyze_position(session, second)
    revenge = next(i for i in pm["diagnosis"] if i["code"] == "REVENGE_REENTRY")
    assert "13 min earlier" in revenge["detail"] and revenge["fixed_by"].startswith("risk rule REENTRY_COOLDOWN")


class _Candles:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def candles(self, symbol, timeframe, lookback_bars=300, start=None):  # type: ignore[no-untyped-def]
        return self.df.loc[self.df.index >= pd.Timestamp(start)] if start is not None else self.df


def test_noise_stop_and_mfe_from_candles(session, bel) -> None:  # type: ignore[no-untyped-def]
    _, first, _ = bel
    idx = pd.date_range(at(12, 40), at(14, 0), freq="1min")
    df = pd.DataFrame({"open": 402.3, "high": 402.6, "low": 401.6, "close": 402.0, "volume": 100.0}, index=idx)
    df.loc[df.index >= pd.Timestamp(at(13, 30)), "high"] = 410.5    # target 1 409.9 reached after the stop
    pm = analyze_position(session, first, _Candles(df))
    assert pm["mfe"] == pytest.approx(0.15) and pm["mae"] == pytest.approx(0.85)
    noise = next(i for i in pm["diagnosis"] if i["code"] == "NOISE_STOP")
    assert "reached at 13:30" in noise["detail"]
    assert any("reached target 1 409.9 at 13:30" in s for s in pm["story"])


def test_day_summary_ranks_issues(session, bel) -> None:  # type: ignore[no-untyped-def]
    rep = analyze_day(session, DAY)
    assert rep["summary"]["trades"] == 2 and rep["summary"]["losses"] == 2
    assert rep["summary"]["net_pnl"] == pytest.approx(-1.06 - 1.56)
    top = {i["code"]: i for i in rep["top_issues"]}
    assert top["STALE_SIGNAL"]["count"] == 2 and top["AGENTS_FAILED"]["count"] == 2
    assert rep["top_issues"][0]["severity"] == "high"
    assert analyze_day(session, date(2026, 11, 24))["trades"] == []


def test_unknown_position(session) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(KeyError):
        analyze_position(session, 999)


# --------------------------------------------------------------------------- diagnostics


def _write(folder: Path, cat: str, rows: list[dict]) -> None:  # type: ignore[type-arg]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{cat}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_diagnose_overloaded_day(tmp_path: Path, session) -> None:  # type: ignore[no-untyped-def]
    folder = tmp_path / "2026-11-25"
    rows = [{"agent": "technical", "status": "FAILED", "model": "qwen3:4b", "latency_ms": 360000,
             "error": "LLM failed after 3 attempt(s): attempt 3: ReadTimeout: timed out"}] * 8
    rows += [{"agent": "technical", "status": "OK", "model": "qwen3:4b", "latency_ms": 40000}] * 2
    rows += [{"agent": "_summary", "wall_ms": 1}]
    _write(folder, "agents", rows)
    _write(folder, "decisions", [{"symbol": "BEL", "latency_ms": 900000, "filtered": None},
                                 {"symbol": "TCS", "latency_ms": 2000, "filtered": "no strategy setup"}])
    _write(folder, "errors", [{"component": "data", "severity": "WARNING", "context": {"symbol": "BEL", "timeframe": "1m"},
                               "message": "candles BEL 1m: yfinance: yfinance returned no 1m candles for BEL"}] * 12
           + [{"component": "data", "context": {"symbol": "INFY"}, "message": "candles INFY 1m: replay: empty"}])
    (folder / "engine.log").write_text("x database is locked y\n" * 2, encoding="utf-8")
    rep = diagnose(session, DAY, tmp_path)
    assert rep["llm"]["calls"] == 10 and rep["llm"]["failed"] == 8 and rep["llm"]["timeouts"] == 8
    assert rep["llm"]["fail_pct"] == 80.0 and rep["llm"]["avg_latency_s"] == 40.0
    assert rep["pipeline"]["analyses"] == 1 and rep["pipeline"]["slow"] == 1
    assert rep["data"]["errors"] == 12 and rep["data"]["by_symbol"] == {"BEL": 12}    # test leftovers ignored
    assert rep["db"]["locked"] == 2
    titles = [f["title"] for f in rep["findings"]]
    assert titles[0].startswith("LLM overloaded: 80%")
    assert any(t.startswith("Analysis too slow") for t in titles)
    assert any("Market data gaps" in t for t in titles) and any("locked" in t for t in titles)
    assert rep["errors_top"][0]["count"] == 12


def test_diagnose_empty_day(tmp_path: Path) -> None:
    rep = diagnose(None, DAY, tmp_path)
    assert rep["findings"] == [] and rep["llm"]["calls"] == 0
