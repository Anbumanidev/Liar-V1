"""Deterministic risk manager - FINAL VETO (sections 5, 7.5, 8).

``RiskManager.evaluate(proposal, state)`` is a pure function: it checks every rule, records
each check (passed or not) in ``RiskVerdict.checks`` and returns the FIRST failing rule as
``verdict.rule``. The same function vets system trades, the owner's manual trades and every
approval-queue modification, so the owner is blocked by exactly the same rules (the dashboard
shows ``verdict.rule`` and ``verdict.message``).

Sizing (section 5 + 8)::

    normal_risk  = risk_per_trade_pct% x mode sizing capital x tier size_mult (x0.5 profit lock)
    qty_normal   = floor(normal_risk / |entry - SL|)
    qty_budget   = max q with  q x max(|entry-SL|, gap%) + round-trip charges(q) <= loss_budget / 3
    qty_capital  = floor(min(mode capital free, TRADING cash, max position %) / (entry / leverage))
    qty_overnight= floor(overnight exposure room / entry)                  (CNC modes)
    max_qty      = min(all of the above);  reject QTY_BELOW_ONE / the binding rule if < 1

``qty_budget`` uses the gap-adjusted, charges-inclusive risk so that no sequence of stop fills
(within the gap buffer) can push TOTAL EQUITY below either capital floor.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from config.schema import HardCaps, Settings
from core.clock import parse_hhmm, to_ist
from core.enums import (
    BatchStatus,
    Direction,
    DrawdownTier,
    EventType,
    Grade,
    Mode,
    Product,
    RiskRule,
    TradeSource,
)
from core.logging import log_event
from core.types import ChargesBreakdown, MarketEvent, OpenPositionView, RiskCheck, RiskVerdict, TradeProposal
from capital.protection import CapitalProtection
from modes.base import conversion_allowed, direction_allowed, is_overnight
from paper_engine.charges import expected_net_profit, round_trip_charges
from risk.guards import check_add_to_position
from risk.state import RiskState

GRADE_RANK: dict[Grade, int] = {Grade.C: 0, Grade.B: 1, Grade.A: 2, Grade.A_PLUS: 3}
_EPS = 1e-9


class _StateClock:
    """Adapter so JSONL risk logs land in the (possibly simulated) day of the state."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now


class _Checks:
    """Collects ``RiskCheck`` rows; remembers the first failure."""

    def __init__(self) -> None:
        self.rows: list[RiskCheck] = []
        self.first: tuple[RiskRule, str] | None = None

    def add(self, rule: RiskRule, passed: bool, detail: str = "") -> bool:
        self.rows.append(RiskCheck(rule=rule, passed=bool(passed), detail=detail))
        if not passed and self.first is None:
            self.first = (rule, detail)
        return bool(passed)

    @property
    def failed(self) -> bool:
        return self.first is not None


def grade_at_least(grade: Grade | None, minimum: Grade) -> bool:
    """True if ``grade`` ranks at or above ``minimum`` (None never qualifies)."""
    if grade is None:
        return False
    return GRADE_RANK[Grade(grade)] >= GRADE_RANK[Grade(minimum)]


class RiskManager:
    """Final veto over meta-agent decisions AND the owner's manual actions."""

    def __init__(self, settings: Settings, caps: HardCaps) -> None:
        self.settings = settings
        self.caps = caps
        self.protection = CapitalProtection(settings.risk, settings.charges)

    def update_settings(self, settings: Settings) -> None:
        """Hot-reload settings (new trades only; open positions keep their own)."""
        self.settings = settings
        self.protection = CapitalProtection(settings.risk, settings.charges)

    # ------------------------------------------------------------------ helpers

    def leverage_for(self, mode: Mode) -> float:
        """MIS leverage for INTRADAY (clamped to the hard cap); CNC never leveraged."""
        if Mode(mode) != Mode.INTRADAY:
            return 1.0
        return max(1.0, min(self.settings.risk.mis_leverage, self.caps.max_mis_leverage))

    def _small_capital(self, state: RiskState) -> bool:
        return state.capital.trading_balance < self.settings.risk.small_capital_threshold

    def _max_open(self, state: RiskState) -> int:
        r = self.settings.risk
        return r.small_max_open_positions if self._small_capital(state) else r.max_open_positions

    def _max_position_pct(self, state: RiskState) -> float:
        r = self.settings.risk
        pct = r.small_max_position_pct if self._small_capital(state) else r.max_position_pct
        return min(pct, self.caps.max_position_pct)

    def _mode_free_capital(self, state: RiskState, mode: Mode) -> float:
        key = Mode(mode).value
        sizing = state.capital.mode_sizing_capital.get(key, 0.0)
        used = state.capital.mode_blocked.get(key, 0.0)
        return max(0.0, sizing - used)

    def capital_for_position(self, state: RiskState, mode: Mode) -> float:
        """Rupees of margin a new position in ``mode`` may use (never includes RESERVE)."""
        pos_cap = self._max_position_pct(state) / 100.0 * max(0.0, state.capital.trading_balance)
        return max(0.0, min(self._mode_free_capital(state, mode), max(0.0, state.capital.trading_cash), pos_cap))

    def max_affordable_price(self, state: RiskState, mode: Mode) -> float:
        """Highest share price for which at least 1 share can be funded (universe filter)."""
        return self.capital_for_position(state, mode) * self.leverage_for(mode)

    def _overnight_exposure(self, positions: list[OpenPositionView]) -> float:
        total = 0.0
        for p in positions:
            if p.overnight or is_overnight(p.mode):
                total += p.qty * (p.ltp if p.ltp else p.avg_entry)
        return total

    def _tier(self, state: RiskState) -> DrawdownTier:
        return DrawdownTier(state.capital.tier)

    def _charges(self, product: Product, direction: Direction, qty: int, entry: float, exit_: float) -> ChargesBreakdown:
        return round_trip_charges(product, direction, qty, entry, exit_, self.settings.charges)

    def _fit_budget_qty(self, upper: int, per_share: float, budget: float, product: Product,
                        direction: Direction, entry: float, sl: float) -> int:
        """Largest q <= upper with ``q * per_share + round-trip charges at SL <= budget``."""
        if upper < 1 or budget <= 0:
            return 0

        def fits(q: int) -> bool:
            return q * per_share + self._charges(product, direction, q, entry, sl).total <= budget + _EPS

        if fits(upper):
            return upper
        lo, hi = 0, upper  # invariant: fits(lo) (or lo == 0), not fits(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid):
                lo = mid
            else:
                hi = mid
        return lo

    def _in_entry_window(self, mode: Mode, now: datetime) -> tuple[bool, str]:
        t = to_ist(now).time()
        mcfg = self.settings.mode(mode)
        if Mode(mode) == Mode.INTRADAY:
            lo = parse_hhmm(self.settings.risk.intraday_no_entry_before)
            hi = parse_hhmm(self.settings.risk.intraday_no_entry_after)
            if not (lo <= t <= hi):
                return False, f"INTRADAY entries only {lo:%H:%M}-{hi:%H:%M} (now {t:%H:%M})"
        if mcfg.entry_windows:
            for w in mcfg.entry_windows:
                if parse_hhmm(w.start) <= t <= parse_hhmm(w.end):
                    return True, ""
            windows = ", ".join(f"{w.start}-{w.end}" for w in mcfg.entry_windows)
            return False, f"{Mode(mode).value} entry windows {windows} (now {t:%H:%M})"
        return True, ""

    # ------------------------------------------------------------------ main entry

    def evaluate(self, proposal: TradeProposal, state: RiskState, *, check_timing: bool = True) -> RiskVerdict:
        """Vet a trade proposal. Never raises.

        Args:
            proposal: the fully specified trade (system or manual).
            state: capital / positions / mode state snapshot.
            check_timing: apply market-hours, entry-window and stale-price rules. The evening
                SWING/POSITIONAL scan passes False (orders are re-vetted with timing at execution).
        """
        try:
            verdict = self._evaluate(proposal, state, check_timing=check_timing)
        except Exception as exc:  # noqa: BLE001 - graceful degradation: any failure = NO TRADE
            verdict = RiskVerdict.reject(RiskRule.INVALID_PRICE, f"risk evaluation error: {exc}")
        log_event(
            "risk",
            {
                "symbol": proposal.symbol, "mode": proposal.mode.value, "source": proposal.source.value,
                "direction": proposal.direction.value, "entry": proposal.entry_price, "sl": proposal.stop_loss,
                "t1": proposal.target_1, "approved": verdict.approved,
                "rule": verdict.rule.value if verdict.rule else None, "message": verdict.message,
                "qty": verdict.qty, "max_qty": verdict.max_qty, "tier": verdict.tier.value,
            },
            clock=_StateClock(state.now),
        )
        return verdict

    def _evaluate(self, p: TradeProposal, state: RiskState, *, check_timing: bool) -> RiskVerdict:
        s = self.settings
        r = s.risk
        mode = Mode(p.mode)
        mcfg = s.mode(mode)
        direction = Direction(p.direction)
        product = p.product
        tier = self._tier(state)
        tc = self.protection.tier_constraints(tier)
        c = _Checks()
        system = p.source == TradeSource.SYSTEM

        # ---- hard / global gates
        c.add(RiskRule.PAPER_ONLY, self.caps.paper_only is True, "paper trading only")
        c.add(RiskRule.BATCH_NOT_ACTIVE, state.batch_status in (BatchStatus.ACTIVE, BatchStatus.HALTED),
              f"batch status {state.batch_status.value}")
        c.add(RiskRule.HALTED, not (state.capital.halted or tc.halted or state.batch_status == BatchStatus.HALTED),
              f"capital protection HALT (tier {tier.value}); resume only from the dashboard")
        if system:
            c.add(RiskRule.ENGINE_PAUSED, not state.engine_paused, "engine paused")
        c.add(RiskRule.MODE_DISABLED, mcfg.enabled, f"{mode.value} is disabled")
        ms = state.mode_state(mode)
        c.add(RiskRule.MODE_PAUSED, not ms.paused, f"{mode.value} paused by owner")
        auto_paused = ms.paused_until is not None and to_ist(ms.paused_until) > state.now
        c.add(RiskRule.CONSECUTIVE_LOSS_PAUSE, not auto_paused,
              f"{mode.value} paused until {ms.paused_until} ({ms.pause_reason or 'consecutive losses'})")
        stopped = ms.stopped_for_day is not None and ms.stopped_for_day == state.now.date()
        c.add(RiskRule.DAILY_PROFIT_LOCK, not stopped, f"{mode.value} stopped for the day (profit give-back)")
        c.add(RiskRule.NO_OVERNIGHT_SHORT, direction_allowed(mode, direction),
              f"{direction.value} not allowed in {mode.value} (CNC modes are long-only)")
        c.add(RiskRule.BLACKLISTED, p.symbol not in set(s.universe.blacklist), f"{p.symbol} is blacklisted")
        if system:
            c.add(RiskRule.STRATEGY_DISABLED, not state.strategy_disabled(p.strategy, mode),
                  f"strategy {p.strategy} auto-disabled in {mode.value}")
        leverage = self.leverage_for(mode)
        c.add(RiskRule.LEVERAGE_NOT_ALLOWED, product == Product.MIS or leverage == 1.0, "no leverage for CNC")

        # ---- timing
        if check_timing:
            c.add(RiskRule.MARKET_CLOSED, state.market_open, "market closed")
            ok, detail = self._in_entry_window(mode, state.now)
            c.add(RiskRule.ENTRY_WINDOW, ok, detail)
            if state.quote_age_seconds is not None:
                c.add(RiskRule.STALE_PRICE, state.quote_age_seconds <= r.stale_price_seconds,
                      f"quote is {state.quote_age_seconds:.0f}s old (> {r.stale_price_seconds}s)")

        # ---- daily limits
        done = state.trades_today.get(mode, 0)
        c.add(RiskRule.MAX_TRADES_PER_DAY, done < mcfg.max_trades_per_day,
              f"{done}/{mcfg.max_trades_per_day} {mode.value} trades today")
        trading_balance = max(0.0, state.capital.trading_balance)
        day_pnl = state.realized_today_total + min(0.0, state.intraday_open_pnl)
        loss_limit = r.daily_loss_limit_pct / 100.0 * trading_balance
        c.add(RiskRule.DAILY_LOSS_LIMIT, day_pnl > -loss_limit + _EPS,
              f"day P&L {day_pnl:.2f} reached the daily loss limit -{loss_limit:.2f} ({r.daily_loss_limit_pct}%)")

        # ---- agent / quality gates
        c.add(RiskRule.TRADEABLE_FALSE, p.tradeable is not False, "fundamental agent: TRADEABLE=false")
        if system:
            c.add(RiskRule.CONFIDENCE_BELOW_MIN, p.confidence >= mcfg.confidence_threshold,
                  f"confidence {p.confidence} < {mcfg.confidence_threshold}")
            if p.confluence_score is not None:
                c.add(RiskRule.CONFLUENCE_BELOW_MIN, p.confluence_score >= mcfg.min_confluence,
                      f"confluence {p.confluence_score:.1f} < {mcfg.min_confluence}")
            else:
                c.add(RiskRule.CONFLUENCE_BELOW_MIN, False, "confluence score missing")
            c.add(RiskRule.GRADE_NOT_ALLOWED, grade_at_least(p.grade, mcfg.min_grade),
                  f"grade {p.grade.value if p.grade else None} below mode minimum {mcfg.min_grade.value}")
        if tier != DrawdownTier.NORMAL and not tc.halted:
            ok = p.grade is not None and tc.grade_allowed(p.grade)
            allowed = "/".join(g.value for g in tc.allowed_grades)
            c.add(RiskRule.DRAWDOWN_TIER, ok, f"tier {tier.value} allows only {allowed} setups "
                                              f"(this one: {p.grade.value if p.grade else 'ungraded'})")
        profit_lock_half = False
        if mode == Mode.INTRADAY:
            intraday_realized = state.realized_today.get(Mode.INTRADAY, 0.0)
            lock_level = r.intraday_daily_profit_lock_pct / 100.0 * trading_balance
            if trading_balance > 0 and intraday_realized >= lock_level:
                profit_lock_half = True
                c.add(RiskRule.DAILY_PROFIT_LOCK, p.grade == Grade.A_PLUS,
                      f"intraday profit lock (+{intraday_realized:.2f} >= {lock_level:.2f}): only A+ at half size")

        # ---- price / SL / targets
        entry, sl = p.entry_price, p.stop_loss
        c.add(RiskRule.INVALID_PRICE, entry > 0, f"entry price {entry}")
        c.add(RiskRule.SL_MISSING, sl is not None and sl > 0, "stop-loss is mandatory")
        long_ = direction == Direction.LONG
        sl_ok_side = (sl < entry) if long_ else (sl > entry)
        c.add(RiskRule.SL_WRONG_SIDE, sl_ok_side, f"SL {sl} on the wrong side of entry {entry} for {direction.value}")
        sl_dist = abs(entry - sl)
        atr = p.atr if p.atr else state.atr
        if atr and atr > 0:
            c.add(RiskRule.SL_TOO_TIGHT, sl_dist >= r.sl_min_atr * atr - _EPS,
                  f"SL distance {sl_dist:.2f} < {r.sl_min_atr} x ATR {atr:.2f}")
            c.add(RiskRule.SL_TOO_WIDE, sl_dist <= r.sl_max_atr * atr + _EPS,
                  f"SL distance {sl_dist:.2f} > {r.sl_max_atr} x ATR {atr:.2f}")
        else:
            c.add(RiskRule.SL_TOO_TIGHT, sl_dist > 0, "SL distance is zero (ATR unknown: ATR bounds not checked)")
        t1_ok = (p.target_1 > entry) if long_ else (p.target_1 < entry)
        t2_ok = p.target_2 is None or ((p.target_2 >= p.target_1) if long_ else (p.target_2 <= p.target_1))
        c.add(RiskRule.TARGET_INVALID, t1_ok and t2_ok, f"targets T1={p.target_1} T2={p.target_2} invalid for {direction.value}")
        rr = p.rr
        c.add(RiskRule.RR_BELOW_MIN, rr + _EPS >= mcfg.min_rr, f"R:R {rr:.2f} < mode minimum {mcfg.min_rr}")
        ref = p.ltp or entry
        buf = r.circuit_buffer_pct / 100.0
        near_upper = p.upper_circuit is not None and ref >= p.upper_circuit * (1 - buf)
        near_lower = p.lower_circuit is not None and ref <= p.lower_circuit * (1 + buf)
        c.add(RiskRule.NEAR_CIRCUIT, not (near_upper or near_lower),
              f"price {ref} within {r.circuit_buffer_pct}% of circuit ({p.lower_circuit}-{p.upper_circuit})")

        # ---- portfolio constraints
        add_rule = check_add_to_position(p.symbol, direction, state.open_positions)
        if add_rule is not None:
            c.add(add_rule, False, f"already holding {p.symbol}")
        else:
            c.add(RiskRule.DUPLICATE_POSITION, True)
        max_open = self._max_open(state)
        if tc.max_open_positions is not None:
            max_open = min(max_open, tc.max_open_positions)
        n_open = len(state.open_positions)
        c.add(RiskRule.MAX_OPEN_POSITIONS, n_open < max_open, f"{n_open} open positions (max {max_open})")
        if p.sector:
            same = sum(1 for o in state.open_positions if o.sector and o.sector == p.sector)
            c.add(RiskRule.SECTOR_LIMIT, same < r.max_positions_per_sector,
                  f"{same} open position(s) in sector {p.sector} (max {r.max_positions_per_sector})")
        too_corr = {k: v for k, v in state.correlations.items() if v > r.max_correlation}
        c.add(RiskRule.CORRELATION_LIMIT, not too_corr,
              f"20d correlation > {r.max_correlation} with {', '.join(f'{k} ({v:.2f})' for k, v in too_corr.items())}")
        overnight = is_overnight(mode)
        if overnight:
            c.add(RiskRule.DRAWDOWN_TIER, tc.allow_new_overnight, f"tier {tier.value}: no new overnight positions")
            if r.avoid_events_in_window:
                avoid = {t.upper() for t in r.avoid_event_types}
                blocking = [e for e in p.events_in_window if _event_blocks(e, avoid, p.symbol)]
                c.add(RiskRule.EVENT_IN_WINDOW, not blocking,
                      "event(s) inside the holding window: "
                      + ", ".join(f"{e.event_type.value} {e.event_date}" for e in blocking))

        # ---- sizing
        verdict_kw: dict[str, Any] = {"tier": tier, "checks": c.rows}
        if sl_dist <= 0 or entry <= 0:
            return self._finish(c, p, qty=0, max_qty=0, **verdict_kw)
        base_pct = r.risk_per_trade_pct
        kelly_key = f"{p.strategy}:{mode.value}" if p.strategy else None
        if s.evaluation.kelly_enabled and kelly_key and kelly_key in state.kelly_fractions:
            base_pct = max(0.0, min(state.kelly_fractions[kelly_key] * 100.0, self.caps.max_risk_per_trade_pct))
        base_pct = min(base_pct, self.caps.max_risk_per_trade_pct)
        mode_sizing = state.capital.mode_sizing_capital.get(mode.value, 0.0)
        size_mult = tc.size_mult * (0.5 if profit_lock_half else 1.0)
        normal_risk = base_pct / 100.0 * mode_sizing * size_mult
        budget_cap = max(0.0, state.capital.loss_budget) / self.settings.risk.loss_budget_divisor
        risk_budget = self.protection.max_new_trade_risk(normal_risk, state.capital.loss_budget)
        gap_dist = entry * r.gap_buffer_pct / 100.0 if overnight else 0.0
        eff_per_share = max(sl_dist, gap_dist)

        qty_normal = math.floor(normal_risk / sl_dist + _EPS) if normal_risk > 0 else 0
        margin_per_share = entry / leverage
        mode_free = self._mode_free_capital(state, mode)
        qty_alloc = math.floor(mode_free / margin_per_share + _EPS)
        qty_cash = math.floor(max(0.0, state.capital.trading_cash) / margin_per_share + _EPS)
        pos_cap = self._max_position_pct(state) / 100.0 * trading_balance
        qty_poscap = math.floor(pos_cap / margin_per_share + _EPS)
        qty_capital = min(qty_alloc, qty_cash, qty_poscap)
        qty_overnight = 10**9
        if overnight:
            room = r.max_overnight_exposure_pct / 100.0 * trading_balance - self._overnight_exposure(state.open_positions)
            qty_overnight = max(0, math.floor(room / entry + _EPS))
        upper = max(0, min(qty_normal, qty_capital, qty_overnight))
        qty_budget = self._fit_budget_qty(max(upper, 1), eff_per_share, budget_cap, product, direction, entry, sl)
        max_qty = max(0, min(upper, qty_budget))

        verdict_kw.update(
            risk_budget=risk_budget,
            size_multiplier=size_mult,
        )
        if max_qty < 1:
            if qty_budget < 1:
                c.add(RiskRule.LOSS_BUDGET_EXHAUSTED, False,
                      f"loss budget {state.capital.loss_budget:.2f} / {r.loss_budget_divisor:g} = {budget_cap:.2f} "
                      f"cannot fund 1 share (risk/share {eff_per_share:.2f} + charges)")
            elif qty_alloc < 1:
                c.add(RiskRule.ALLOCATION_EXHAUSTED, False,
                      f"{mode.value} capital free {mode_free:.2f} < 1 share margin {margin_per_share:.2f}")
            elif qty_cash < 1:
                c.add(RiskRule.INSUFFICIENT_CASH, False,
                      f"TRADING cash {state.capital.trading_cash:.2f} < 1 share margin {margin_per_share:.2f}")
            elif qty_poscap < 1:
                c.add(RiskRule.MAX_POSITION_PCT, False,
                      f"max position {self._max_position_pct(state):g}% = {pos_cap:.2f} < 1 share {margin_per_share:.2f}")
            elif qty_overnight < 1:
                c.add(RiskRule.OVERNIGHT_EXPOSURE, False,
                      f"overnight exposure would exceed {r.max_overnight_exposure_pct}% of TRADING")
            else:
                c.add(RiskRule.QTY_BELOW_ONE, False,
                      f"risk {normal_risk:.2f} / SL distance {sl_dist:.2f} < 1 share")
            return self._finish(c, p, qty=0, max_qty=0, **verdict_kw)
        c.add(RiskRule.QTY_BELOW_ONE, True, f"max qty {max_qty}")

        qty = max_qty
        if p.requested_qty is not None:
            ok = 1 <= p.requested_qty <= max_qty
            c.add(RiskRule.QTY_ABOVE_MAX, ok, f"requested {p.requested_qty} > risk-manager maximum {max_qty}")
            qty = p.requested_qty if ok else max_qty

        # ---- charges vs edge (with the final qty)
        charges_t1 = self._charges(product, direction, qty, entry, p.target_1)
        net_t1 = expected_net_profit(product, direction, qty, entry, p.target_1, s.charges)
        c.add(RiskRule.CHARGES_EXCEED_EDGE, net_t1 > 0,
              f"expected net at T1 {net_t1:.2f} <= 0 after charges {charges_t1.total:.2f}")
        value = qty * entry
        pct = charges_t1.total / value * 100.0 if value > 0 else math.inf
        c.add(RiskRule.CHARGES_EXCEED_EDGE, pct <= r.max_charges_pct_of_value + _EPS,
              f"round-trip charges {charges_t1.total:.2f} = {pct:.2f}% of position value "
              f"(> {r.max_charges_pct_of_value}%)")

        verdict_kw.update(
            risk_amount=qty * sl_dist,
            effective_risk=qty * eff_per_share + self._charges(product, direction, qty, entry, sl).total,
            position_value=value,
            charges_estimate=charges_t1,
            expected_net_at_t1=net_t1,
        )
        return self._finish(c, p, qty=qty, max_qty=max_qty, **verdict_kw)

    def _finish(self, c: _Checks, p: TradeProposal, *, qty: int, max_qty: int, **kw: Any) -> RiskVerdict:
        if c.failed:
            rule, detail = c.first  # type: ignore[misc]
            return RiskVerdict(approved=False, rule=rule, message=f"{rule.value}: {detail}", qty=0,
                               max_qty=max_qty, **kw)
        return RiskVerdict(approved=True, rule=None, message="approved", qty=qty, max_qty=max_qty, **kw)

    # ------------------------------------------------------------------ conversion

    def evaluate_conversion(
        self,
        position: OpenPositionView,
        to_mode: Mode,
        state: RiskState,
        *,
        events: list[MarketEvent] | None = None,
    ) -> RiskVerdict:
        """Vet converting an open INTRADAY position to SWING/POSITIONAL (section 3)."""
        s, r = self.settings, self.settings.risk
        to_mode = Mode(to_mode)
        tier = self._tier(state)
        tc = self.protection.tier_constraints(tier)
        c = _Checks()
        c.add(RiskRule.CONVERSION_NOT_ALLOWED, conversion_allowed(position.mode, to_mode, position.direction),
              f"only LONG INTRADAY -> SWING/POSITIONAL conversions are allowed ({position.mode.value} "
              f"{position.direction.value} -> {to_mode.value})")
        cutoff = parse_hhmm(r.conversion_cutoff)
        c.add(RiskRule.CONVERSION_NOT_ALLOWED, to_ist(state.now).time() < cutoff,
              f"conversions only before {r.conversion_cutoff}")
        c.add(RiskRule.MODE_DISABLED, s.mode(to_mode).enabled, f"{to_mode.value} is disabled")
        c.add(RiskRule.HALTED, not (state.capital.halted or tc.halted), "capital protection HALT")
        c.add(RiskRule.DRAWDOWN_TIER, tc.allow_new_overnight, f"tier {tier.value}: no new overnight positions")
        ltp = position.ltp or position.avg_entry
        required = position.qty * position.avg_entry          # CNC: full value, no leverage
        extra = max(0.0, required - position.margin_blocked)
        free = self._mode_free_capital(state, to_mode)
        c.add(RiskRule.ALLOCATION_EXHAUSTED, free + _EPS >= required,
              f"{to_mode.value} capital free {free:.2f} < position value {required:.2f}")
        c.add(RiskRule.INSUFFICIENT_CASH, max(0.0, state.capital.trading_cash) + _EPS >= extra,
              f"extra margin {extra:.2f} > TRADING cash {state.capital.trading_cash:.2f}")
        trading_balance = max(0.0, state.capital.trading_balance)
        room = r.max_overnight_exposure_pct / 100.0 * trading_balance - self._overnight_exposure(state.open_positions)
        c.add(RiskRule.OVERNIGHT_EXPOSURE, position.qty * ltp <= room + _EPS,
              f"overnight exposure room {room:.2f} < position value {position.qty * ltp:.2f}")
        now_risk = self.protection.position_risk(position, include_gap=False)
        new_risk = self.protection.position_risk(position, include_gap=True)
        c.add(RiskRule.LOSS_BUDGET_EXHAUSTED, new_risk - now_risk <= state.capital.loss_budget + _EPS,
              f"gap-adjusted risk increase {new_risk - now_risk:.2f} > loss budget {state.capital.loss_budget:.2f}")
        if r.avoid_events_in_window and events:
            avoid = {t.upper() for t in r.avoid_event_types}
            blocking = [e for e in events if _event_blocks(e, avoid, position.symbol)]
            c.add(RiskRule.EVENT_IN_WINDOW, not blocking,
                  "event(s) inside the holding window: " + ", ".join(f"{e.event_type.value} {e.event_date}" for e in blocking))
        verdict = (
            RiskVerdict(approved=False, rule=c.first[0], message=f"{c.first[0].value}: {c.first[1]}",
                        qty=0, max_qty=position.qty, tier=tier, checks=c.rows)
            if c.failed
            else RiskVerdict(approved=True, message="conversion approved", qty=position.qty, max_qty=position.qty,
                             tier=tier, checks=c.rows, position_value=required, effective_risk=new_risk)
        )
        log_event("risk", {"conversion": True, "symbol": position.symbol, "to_mode": to_mode.value,
                           "approved": verdict.approved, "rule": verdict.rule.value if verdict.rule else None,
                           "message": verdict.message}, clock=_StateClock(state.now))
        return verdict


def _event_blocks(event: MarketEvent, avoid_types: set[str], symbol: str) -> bool:
    """An event blocks a new overnight entry if its type is avoided and it concerns the symbol
    (or is a market-wide MACRO event of an avoided type)."""
    etype = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
    if etype.upper() not in avoid_types:
        return False
    return event.symbol is None or event.symbol.strip().upper() == symbol.strip().upper()

