"""Price-behaviour check (features.price_behaviour + risk rules CHASING_HIGH / HISTORY_FADES).

Owner's finding on 2026-11-25: all three losing trades were bought high in an already exhausted day
range. The numbers below are the real ones measured from that day's 1-minute candles.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from config.loader import load_default_settings, load_hard_caps, update_settings
from core.clock import IST
from core.enums import Direction, DrawdownTier, EntryType, Grade, Mode, RiskRule
from core.types import CapitalSnapshot, TradeProposal
from features.price_behaviour import compute
from risk.manager import RiskManager
from risk.state import RiskState

DAYS = pd.bdate_range("2026-10-26", "2026-11-25").date.tolist()


class _Data:
    def __init__(self, m5: pd.DataFrame, d1: pd.DataFrame) -> None:
        self.frames = {"5m": m5, "1d": d1}

    def candles(self, symbol, timeframe, lookback_bars=300, start=None):  # type: ignore[no-untyped-def]
        df = self.frames[str(timeframe)]
        return df.tail(lookback_bars) if start is None else df.loc[df.index >= pd.Timestamp(start)]


def _session(day: date, path: list[float], vol: float = 1000.0) -> pd.DataFrame:
    """75 five-minute bars following ``path`` (piecewise-linear closes), small wicks."""
    closes = np.interp(np.linspace(0, len(path) - 1, 75), range(len(path)), path)
    idx = [datetime.combine(day, time(9, 15), tzinfo=IST) + timedelta(minutes=5 * i) for i in range(75)]
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 0.05,
                         "low": np.minimum(opens, closes) - 0.05, "close": closes, "volume": vol},
                        index=pd.DatetimeIndex(idx))


def _spike_day(day: date, fade: bool) -> pd.DataFrame:
    """Quiet morning, one 5-point spike bar at 12:15 to the top of the day, then a collapse or a run."""
    closes = [100.0 + (0.3 if i % 2 else 0.0) for i in range(36)] + [105.0]
    after = np.linspace(104.6, 100.0, 38) if fade else np.linspace(105.4, 110.0, 38)
    closes = np.array(closes + list(after))
    idx = [datetime.combine(day, time(9, 15), tzinfo=IST) + timedelta(minutes=5 * i) for i in range(75)]
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 0.05,
                         "low": np.minimum(opens, closes) - 0.05, "close": closes, "volume": 1000.0},
                        index=pd.DatetimeIndex(idx))


def _make(fade: bool, today_path: list[float]) -> _Data:
    """20 past sessions with a midday spike to the top of the range that then fades or runs; plus today."""
    frames, daily = [], []
    for d in DAYS[:-1]:
        s = _spike_day(d, fade)
        frames.append(s)
        daily.append({"ts": datetime.combine(d, time(9, 15), tzinfo=IST), "open": s["open"].iloc[0],
                      "high": s["high"].max(), "low": s["low"].min(), "close": s["close"].iloc[-1], "volume": 1e5})
    today = _session(DAYS[-1], today_path)
    frames.append(today)
    d1 = pd.DataFrame(daily).set_index("ts")
    d1.index = pd.DatetimeIndex(d1.index)
    return _Data(pd.concat(frames), d1)


NOW = datetime(2026, 11, 25, 12, 0, tzinfo=IST)


def test_snapshot_measures_range_position_and_exhaustion() -> None:
    data = _make(True, [100, 106, 112, 112, 112])       # +12 today vs a ~8-16 ADR
    b = compute(data, "X", NOW, live_price=111.5, thresholds=load_default_settings().risk)
    assert b is not None and b["direction"] == "LONG"
    assert 0.9 < b["range_pos"] <= 1.0
    assert b["day_low"] == pytest.approx(99.95, abs=0.01) and b["day_high"] >= 111.5
    assert b["range_vs_adr"] > 1.0 and b["vwap_ext_atr"] > 0
    assert any("% of today's range" in n for n in b["notes"])


def test_history_detects_a_stock_whose_highs_fade() -> None:
    fading = compute(_make(True, [100, 104, 108, 108, 108]), "X", NOW, live_price=107.9)
    trending = compute(_make(False, [100, 104, 108, 108, 108]), "X", NOW, live_price=107.9)
    assert fading["samples"] >= 15 and trending["samples"] >= 15
    assert fading["continuation_rate"] < 0.2 < 0.8 < trending["continuation_rate"]


def test_no_lookahead_and_missing_data() -> None:
    data = _make(True, [100, 104, 120, 140, 160])       # huge later move must be invisible at 10:00
    early = compute(data, "X", datetime(2026, 11, 25, 10, 0, tzinfo=IST))
    assert early["day_high"] < 115
    assert compute(_Data(pd.DataFrame(), pd.DataFrame()), "X", NOW) is None


# --------------------------------------------------------------------------- risk rules


def _snap(settings, now):  # type: ignore[no-untyped-def]
    from capital.protection import CapitalProtection

    prot = CapitalProtection(settings.risk)
    abs_f, trail_f = prot.floors(1000.0, 1000.0)
    return CapitalSnapshot(
        batch_id=1, as_of=now, starting_capital=1000.0, base_capital=1000.0, trading_cash=1000.0, blocked=0.0,
        reserve=0.0, reserve_peak=0.0, unrealized_pnl=0.0, total_equity=1000.0, hwm=1000.0, drawdown_pct=0.0,
        tier=DrawdownTier.NORMAL, sizing_capital=1000.0,
        mode_sizing_capital={m.value: 1000.0 * settings.mode(m).allocation_pct / 100 for m in Mode},
        mode_blocked={}, absolute_floor=abs_f, trailing_floor=trail_f, open_risk=0.0,
        loss_budget=prot.loss_budget(1000.0, 0.0, (abs_f, trail_f)), halted=False)


def _prop(now, **kw):  # type: ignore[no-untyped-def]
    base = dict(batch_id=1, symbol="INFY", mode=Mode.INTRADAY, direction=Direction.LONG, entry_price=100.0,
                entry_type=EntryType.MARKET, stop_loss=99.0, target_1=101.6, strategy="vwap_pullback",
                confidence=80, confluence_score=78.0, grade=Grade.A, atr=1.0, tradeable=True, sector="IT",
                ltp=100.05, created_at=now - timedelta(minutes=2), valid_until=now + timedelta(minutes=10),
                failed_agents_pct=0.0, debate_bull=75, debate_bear=60, room_r=2.5)
    base.update(kw)
    return TradeProposal(**base)


@pytest.mark.parametrize("name,pos,rng,ext,refused", [
    ("ITC 11:24 (real)", 0.8644, 1.6797, 3.259, True),       # top of a 168 % range, 3.3 ATR above VWAP
    ("BEL 12:40 (real)", 0.7151, 1.1234, 0.977, True),       # 72 % of an exhausted (112 %) range
    ("BEL 12:56 (real)", 0.7258, 1.1234, 1.149, True),
    ("ITC 09:40 (real)", 0.7736, 0.7545, 0.096, False),      # early: range only 75 % of normal
    ("top but calm", 0.85, 0.60, 1.0, False),                # near the high, not stretched, not exhausted
    ("top and stretched", 0.85, 0.60, 3.0, True),
    ("exhausted but low", 0.40, 1.30, 0.0, False),
])
def test_chasing_high_rule(name: str, pos: float, rng: float, ext: float, refused: bool) -> None:
    s = load_default_settings()
    rm = RiskManager(s, load_hard_caps())
    now = datetime(2026, 11, 25, 10, 30, tzinfo=IST)
    v = rm.evaluate(_prop(now, range_pos=pos, range_vs_adr=rng, vwap_ext_atr=ext),
                    RiskState(now=now, batch_id=1, capital=_snap(s, now)))
    if refused:
        assert v.rule == RiskRule.CHASING_HIGH, (name, v.message)
    else:
        assert v.approved, (name, v.message)


def test_short_side_is_mirrored() -> None:
    s = load_default_settings()
    rm = RiskManager(s, load_hard_caps())
    now = datetime(2026, 11, 25, 10, 30, tzinfo=IST)
    short = _prop(now, direction=Direction.SHORT, entry_price=100.0, stop_loss=101.0, target_1=98.4, ltp=99.95,
                  range_pos=0.15, range_vs_adr=1.2, vwap_ext_atr=-1.0)
    v = rm.evaluate(short, RiskState(now=now, batch_id=1, capital=_snap(s, now)))
    assert v.rule == RiskRule.CHASING_HIGH and "bottom" in v.message


def test_history_fades_rule_and_switches() -> None:
    s = load_default_settings()
    now = datetime(2026, 11, 25, 10, 30, tzinfo=IST)
    st = RiskState(now=now, batch_id=1, capital=_snap(s, now))
    fades = _prop(now, range_pos=0.75, range_vs_adr=0.8, vwap_ext_atr=1.0, continuation_rate=0.0, behaviour_samples=8)
    assert RiskManager(s, load_hard_caps()).evaluate(fades, st).rule == RiskRule.HISTORY_FADES
    few = fades.model_copy(update={"behaviour_samples": 3})               # too little history: no veto
    assert RiskManager(s, load_hard_caps()).evaluate(few, st).approved
    off = update_settings(s, {"risk.behaviour_check_enabled": False})
    assert RiskManager(off, load_hard_caps()).evaluate(fades, st).approved
    unknown = _prop(now)                                                    # older proposals: rules skipped
    assert RiskManager(s, load_hard_caps()).evaluate(unknown, st).approved
