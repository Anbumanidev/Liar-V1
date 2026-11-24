"""Section 8 guarantee: no sequence of SL-hit trades sized with
``risk <= max_new_trade_risk(normal_risk, loss_budget)`` can breach either floor.

Model: positions are opened at ``entry`` (marked at entry until they resolve) and resolve by
(a) the stop filling exactly at SL, (b) an overnight gap fill below the SL but within the gap
buffer (``entry * (1 - gap_buffer_pct)``), or (c) a winning exit. Charges are the real MIS/CNC
model. Wins raise the HWM, so the trailing floor ratchets up during the sequence.

Scope note: favourable mark-to-market excursions that raise the HWM and then reverse all the
way to an un-trailed stop are NOT covered by the loss budget alone - that case is caught by the
HALT tier (equity <= effective floor), which the position monitor acts on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.orm import Session

from capital.accounts import CapitalManager
from capital.protection import CapitalProtection
from config.schema import Settings
from core.clock import IST, SimulatedClock
from core.enums import Direction, DrawdownTier, Mode, Product
from core.types import OpenPositionView
from paper_engine.charges import entry_side, exit_side, order_charges

OPENED = datetime(2026, 11, 24, 10, 0, tzinfo=IST)


@dataclass
class _Open:
    view: OpenPositionView
    sl_dist: float


def _make_view(pid: int, mode: Mode, direction: Direction, qty: int, entry: float, sl: float,
               settings: Settings) -> OpenPositionView:
    product = Product.MIS if mode == Mode.INTRADAY else Product.CNC
    entry_c = order_charges(product, entry_side(direction), qty, entry, settings.charges).total
    return OpenPositionView(
        position_id=pid, symbol=f"S{pid}", mode=mode, product=product, direction=direction, qty=qty,
        avg_entry=entry, stop_loss=sl, ltp=entry, margin_blocked=qty * entry, entry_charges=entry_c,
        opened_at=OPENED, overnight=mode != Mode.INTRADAY,
    )


def _size(prot: CapitalProtection, settings: Settings, pid: int, mode: Mode, direction: Direction,
          entry: float, sl: float, max_risk: float, max_qty: int | None = None) -> OpenPositionView | None:
    """Largest qty whose FULL position risk (incl. gap + charges) is <= max_risk."""
    gap = entry * settings.risk.gap_buffer_pct / 100.0 if mode != Mode.INTRADAY else 0.0
    per_share = max(abs(entry - sl), gap)
    if per_share <= 0 or max_risk <= 0:
        return None
    qty = int(max_risk // per_share)
    if max_qty is not None:
        qty = min(qty, max_qty)
    while qty >= 1:
        view = _make_view(pid, mode, direction, qty, entry, sl, settings)
        if prot.position_risk(view) <= max_risk + 1e-9:
            return view
        qty = qty - max(1, qty // 20)
    return None


def _exit_price(rng: np.random.Generator, o: _Open, settings: Settings) -> tuple[float, str]:
    v = o.view
    sign = 1.0 if v.direction == Direction.LONG else -1.0
    u = rng.random()
    if u < 0.55:
        return v.stop_loss, "SL"
    if u < 0.75 and v.overnight:
        worst = v.avg_entry * (1 - sign * settings.risk.gap_buffer_pct / 100.0)
        # gap fill somewhere between the SL and the gap-buffer worst case
        lo, hi = sorted((worst, v.stop_loss))
        return float(rng.uniform(lo, hi)), "GAP"
    return v.avg_entry + sign * o.sl_dist * float(rng.uniform(0.2, 3.0)), "WIN"


def _net_pnl(v: OpenPositionView, exit_price: float, settings: Settings) -> float:
    sign = 1.0 if v.direction == Direction.LONG else -1.0
    gross = sign * (exit_price - v.avg_entry) * v.qty
    exit_c = order_charges(v.product, exit_side(v.direction), v.qty, exit_price, settings.charges).total
    return gross - v.entry_charges - exit_c


def _new_trade_params(rng: np.random.Generator) -> tuple[Mode, Direction, float, float]:
    mode = Mode.INTRADAY if rng.random() < 0.6 else Mode.SWING
    direction = Direction.SHORT if (mode == Mode.INTRADAY and rng.random() < 0.3) else Direction.LONG
    entry = float(rng.uniform(20.0, 2500.0))
    sl_dist = entry * float(rng.uniform(0.002, 0.06))
    sl = entry - sl_dist if direction == Direction.LONG else entry + sl_dist
    return mode, direction, entry, sl


@pytest.mark.parametrize("block", list(range(4)))
def test_sl_hits_never_breach_either_floor(settings: Settings, block: int) -> None:
    """200 random sequences (4 x 50 seeds) of up to 120 events each."""
    prot = CapitalProtection(settings.risk, settings.charges)
    breaches: list[str] = []
    min_margin = math.inf
    closest_rel = math.inf  # smallest (equity - floor) / start seen: proves the floor was approached
    opened = losses = 0
    for seed in range(block * 50, block * 50 + 50):
        rng = np.random.default_rng(seed)
        start = float(rng.choice([1000.0, 5000.0, 50_000.0, 1_000_000.0]))
        realized = start            # cash + blocked + reserve
        hwm = start
        open_: list[_Open] = []
        pid = 0
        for _ in range(120):
            equity = realized - sum(o.view.entry_charges for o in open_)  # MTM at entry = -charges
            hwm = max(hwm, equity)
            floors = prot.floors(start, hwm)
            min_margin = min(min_margin, equity - max(floors))
            closest_rel = min(closest_rel, (equity - max(floors)) / start)
            if equity < max(floors) - 1e-6:
                breaches.append(f"seed {seed}: equity {equity:.4f} < floor {max(floors):.4f}")
                break
            dd, tier = prot.evaluate(start, hwm, equity)
            if (rng.random() < 0.5 or not open_) and tier != DrawdownTier.HALT:
                mode, direction, entry, sl = _new_trade_params(rng)
                # deliberately aggressive "normal" risk (0.5 - 8 % of equity) to stress the budget
                normal = equity * float(rng.uniform(0.005, 0.08)) * prot.tier_constraints(tier).size_mult
                budget = prot.loss_budget(equity, prot.open_risk(o.view for o in open_), floors)
                max_risk = prot.max_new_trade_risk(normal, budget)
                pid += 1
                view = _size(prot, settings, pid, mode, direction, entry, sl, max_risk)
                if view is not None:
                    assert prot.position_risk(view) <= max_risk + 1e-9
                    open_.append(_Open(view, abs(entry - sl)))
                    opened += 1
            elif open_:
                o = open_.pop(int(rng.integers(0, len(open_))))
                price, kind = _exit_price(rng, o, settings)
                losses += kind != "WIN"
                realized += _net_pnl(o.view, price, settings)
        # final: resolve everything at the stop / worst gap -> still above both floors
        equity = realized - sum(o.view.entry_charges for o in open_)
        worst = realized
        for o in open_:
            v = o.view
            sign = 1.0 if v.direction == Direction.LONG else -1.0
            gap_px = v.avg_entry * (1 - sign * settings.risk.gap_buffer_pct / 100.0) if v.overnight else v.stop_loss
            px = min(v.stop_loss, gap_px) if sign > 0 else max(v.stop_loss, gap_px)
            worst += _net_pnl(v, px, settings)
        if worst < max(prot.floors(start, max(hwm, equity))) - 1e-6:
            breaches.append(f"seed {seed}: all-stops worst case {worst:.4f} breaches floor")
    assert not breaches, breaches[:5]
    assert min_margin >= -1e-6
    # non-vacuous: many trades, many stop/gap losses, and equity pushed very close to the floor
    assert opened > 1000 and losses > 500
    assert closest_rel < 0.005


@pytest.mark.parametrize("seed", list(range(20)))
def test_floor_guarantee_end_to_end_with_capital_manager(session: Session, settings: Settings, seed: int) -> None:
    """Same guarantee through the real CapitalManager (block/release/EOD/snapshot budget)."""
    rng = np.random.default_rng(1000 + seed)
    clock = SimulatedClock(datetime(2026, 11, 24, 9, 15, tzinfo=IST))
    mgr = CapitalManager(settings, clock)
    batch = 1
    start = float(rng.choice([1000.0, 10_000.0, 100_000.0]))
    mgr.initialize_batch(session, batch, start)
    prot = mgr.protection
    open_: list[_Open] = []
    day = date(2026, 11, 24)
    pid = 0
    for step in range(80):
        snap = mgr.snapshot(session, batch, [o.view for o in open_])
        assert snap.total_equity >= snap.absolute_floor - 1e-6, (seed, step)
        assert snap.total_equity >= snap.trailing_floor - 1e-6, (seed, step)
        if step % 10 == 9:
            mgr.run_eod(session, batch, day, None, snap.total_equity, snap.tier)
            day += timedelta(days=1)
            clock.set(datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST))
            continue
        if (rng.random() < 0.5 or not open_) and not snap.halted:
            mode, direction, entry, sl = _new_trade_params(rng)
            sizing = snap.mode_sizing_capital[str(mode)]
            normal = sizing * float(rng.uniform(0.01, 0.10)) * prot.tier_constraints(snap.tier).size_mult
            max_risk = prot.max_new_trade_risk(normal, snap.loss_budget)
            cash = mgr.available_cash(session, batch)
            pid += 1
            view = _size(prot, settings, pid, mode, direction, entry, sl, max_risk, max_qty=int(cash // entry))
            if view is not None:
                mgr.block(session, batch, view.margin_blocked, position_id=pid, mode=mode)
                open_.append(_Open(view, abs(entry - sl)))
        elif open_:
            o = open_.pop(int(rng.integers(0, len(open_))))
            price, _kind = _exit_price(rng, o, settings)
            mgr.release(session, batch, o.view.margin_blocked, _net_pnl(o.view, price, settings),
                        position_id=o.view.position_id, mode=o.view.mode)
    ok, details = mgr.check_ledger_balance(session, batch)
    assert ok, details["problems"]


def test_documented_gap_mtm_giveback_is_caught_by_halt_tier(settings: Settings) -> None:
    """An un-trailed winner giving back MTM gains trips HALT at the trailing floor."""
    prot = CapitalProtection(settings.risk)
    start = 1000.0
    hwm = 1200.0  # equity peaked with open MTM gains
    eq_at_floor = hwm * 0.91
    _dd, tier = prot.evaluate(start, hwm, eq_at_floor)
    assert tier == DrawdownTier.HALT
    _dd, tier = prot.evaluate(start, hwm, eq_at_floor + 1.0)
    assert tier == DrawdownTier.DEFENSIVE
