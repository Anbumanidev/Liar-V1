"""Virtual capital accounts + ledger per batch (section 6).

Accounts (one ``capital_accounts`` row per batch):

* ``trading_cash`` - free cash. ``blocked`` - margin of open positions (cash -> blocked on
  entry). TRADING balance = ``trading_cash + blocked``.
* ``reserve`` - locked profits. NEVER available for trading: :meth:`CapitalManager.block`
  raises :class:`~capital.errors.InsufficientCashError` (a ``ReserveAccessError``) rather than
  touch it, and :meth:`CapitalManager.available_cash` excludes it.
* ``reserve_peak`` - max RESERVE ever (refills never take RESERVE below
  ``reserve_protected_pct`` % of it). ``hwm`` - high-water mark of TOTAL EQUITY.
* ``sizing_capital`` - capital the risk manager sizes from; grows slowly, shrinks fast (6.3).

Every movement writes one ``capital_ledger`` row carrying the balances AFTER the movement and
a JSONL ``capital`` event. Signed ``amount`` semantics per entry type (used by the replay in
:meth:`CapitalManager.check_ledger_balance`):

==================  ==========================================  ==============================
entry_type          amount                                      effect
==================  ==========================================  ==============================
INITIAL_CAPITAL     starting capital (> 0)                      cash += a; sizing = a
BLOCK               margin (> 0)                                cash -= a; blocked += a
RELEASE             margin (> 0)                                blocked -= a; cash += a
REALIZED_PNL        net P&L of the closed leg (signed)          cash += a
DIVIDEND            dividend (> 0), counted as realized P&L     cash += a
MANUAL_ADJUSTMENT   confirmed owner adjustment (signed)         cash += a
PROFIT_LOCK         locked profit (> 0)                         cash -= a; reserve += a
REFILL              capped refill (> 0)                         reserve -= a; cash += a
MANUAL_TRANSFER     confirmed owner transfer (> 0)              reserve -= a; cash += a
SIZING_CHANGE       delta of sizing capital (signed)            sizing += a (no money moves)
LOSS_ABSORB         0 (marker; the loss is in ``note``)         none
CORPORATE_ACTION    informational                               none
==================  ==========================================  ==============================

Invariant (6.4): ``starting + cum_realized_net + cum_manual_adjustments == cash + blocked +
reserve``. A manual RESERVE -> TRADING transfer moves money between accounts and keeps it.

Transactions: methods ``flush`` but never ``commit`` - the caller owns the unit of work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from capital.errors import AccountNotFoundError, CapitalError, InsufficientCashError, UnconfirmedTransferError
from capital.protection import CapitalProtection
from config.schema import Settings
from core.clock import IST, Clock
from core.enums import CapitalAccountName as Acc
from core.enums import DrawdownTier, LedgerEntryType as L, Mode
from core.errors import LedgerImbalanceError, ReserveAccessError
from core.logging import log_event
from core.types import CapitalSnapshot, OpenPositionView
from storage.audit import audit
from storage.models import CapitalAccount, CapitalLedgerEntry, DailySummary

__all__ = ["BALANCE_TOLERANCE", "CapitalManager", "EODResult"]

#: Absolute rupee tolerance for balance checks (one paisa).
BALANCE_TOLERANCE: float = 0.01

#: Movement rules for the ledger replay: entry type -> (d_cash, d_blocked, d_reserve) per +1 amount.
_MOVES: dict[str, tuple[int, int, int]] = {
    L.INITIAL_CAPITAL: (1, 0, 0),
    L.BLOCK: (-1, 1, 0),
    L.RELEASE: (1, -1, 0),
    L.REALIZED_PNL: (1, 0, 0),
    L.DIVIDEND: (1, 0, 0),
    L.MANUAL_ADJUSTMENT: (1, 0, 0),
    L.PROFIT_LOCK: (-1, 0, 1),
    L.REFILL: (1, 0, -1),
    L.MANUAL_TRANSFER: (1, 0, -1),
    L.SIZING_CHANGE: (0, 0, 0),
    L.LOSS_ABSORB: (0, 0, 0),
    L.CORPORATE_ACTION: (0, 0, 0),
}


@dataclass
class EODResult:
    """Outcome of :meth:`CapitalManager.run_eod` for one batch and day."""

    batch_id: int
    day: date
    skipped: bool
    realized_net: float = 0.0
    profit_locked: float = 0.0
    profit_lock_shortfall: float = 0.0
    loss_absorbed: float = 0.0
    refilled: float = 0.0
    refill_needed: float = 0.0
    sizing_before: float = 0.0
    sizing_after: float = 0.0
    sizing_up_streak: int = 0
    trading_cash: float = 0.0
    blocked: float = 0.0
    trading_balance: float = 0.0
    reserve: float = 0.0
    reserve_peak: float = 0.0
    equity: float = 0.0
    hwm: float = 0.0
    drawdown_pct: float = 0.0
    tier: DrawdownTier = DrawdownTier.NORMAL
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict."""
        d = asdict(self)
        d["day"] = self.day.isoformat()
        d["tier"] = str(self.tier)
        return d


class CapitalManager:
    """Owns all ``capital_accounts`` / ``capital_ledger`` writes for every batch.

    Args:
        settings: active ``Settings`` (``capital``, ``risk``, ``charges``, ``modes`` are used).
        clock: time source (ledger timestamps, JSONL folders).
    """

    def __init__(self, settings: Settings, clock: Clock) -> None:
        self.settings = settings
        self.clock = clock
        self.protection = CapitalProtection(settings.risk, settings.charges)

    def update_settings(self, settings: Settings) -> None:
        """Hot-reload settings (RELOAD_SETTINGS)."""
        self.settings = settings
        self.protection = CapitalProtection(settings.risk, settings.charges)

    # ================================================================== basics

    def account(self, session: Session, batch_id: int) -> CapitalAccount:
        """The batch's account row (raises ``AccountNotFoundError``)."""
        acct = session.get(CapitalAccount, batch_id)
        if acct is None:
            raise AccountNotFoundError(f"no capital account for batch {batch_id}")
        return acct

    def initialize_batch(self, session: Session, batch_id: int, starting_capital: float) -> CapitalAccount:
        """Create the account (all cash in TRADING, RESERVE 0) + INITIAL_CAPITAL ledger row."""
        if not starting_capital > 0:
            raise CapitalError(f"starting capital must be > 0, got {starting_capital}")
        if session.get(CapitalAccount, batch_id) is not None:
            raise CapitalError(f"capital account for batch {batch_id} already initialized")
        start = float(starting_capital)
        acct = CapitalAccount(
            batch_id=batch_id,
            starting_capital=start,
            base_capital=start,
            trading_cash=start,
            blocked=0.0,
            reserve=0.0,
            reserve_peak=0.0,
            hwm=start,
            sizing_capital=start,
            sizing_up_streak=0,
            cum_realized_net=0.0,
            cum_manual_adjustments=0.0,
            last_eod_date=None,
            halted=False,
            halt_reason=None,
            tier=DrawdownTier.NORMAL.value,
            updated_at=self.clock.now(),
        )
        session.add(acct)
        session.flush()
        self._post(session, acct, L.INITIAL_CAPITAL, start, None, Acc.TRADING, note="batch initialized")
        return acct

    def available_cash(self, session: Session, batch_id: int) -> float:
        """Free TRADING cash. RESERVE and blocked margin are never included."""
        return max(0.0, self.account(session, batch_id).trading_cash)

    @staticmethod
    def trading_balance(acct: CapitalAccount) -> float:
        """TRADING = cash + blocked (cost basis/margin of open positions)."""
        return acct.trading_cash + acct.blocked

    # ================================================================== ledger plumbing

    def _post(
        self,
        session: Session,
        acct: CapitalAccount,
        entry_type: L,
        amount: float,
        from_account: Acc | None,
        to_account: Acc | None,
        *,
        position_id: int | None = None,
        trade_id: int | None = None,
        mode: Mode | str | None = None,
        actor: str = "SYSTEM",
        confirmed: bool = True,
        note: str | None = None,
    ) -> CapitalLedgerEntry:
        """Write one ledger row with the balances AFTER the movement (already applied)."""
        now = self.clock.now()
        acct.updated_at = now
        row = CapitalLedgerEntry(
            batch_id=acct.batch_id,
            ts=now,
            entry_type=L(entry_type).value,
            amount=float(amount),
            from_account=from_account.value if from_account is not None else None,
            to_account=to_account.value if to_account is not None else None,
            trading_cash_after=acct.trading_cash,
            blocked_after=acct.blocked,
            reserve_after=acct.reserve,
            sizing_capital_after=acct.sizing_capital,
            position_id=position_id,
            trade_id=trade_id,
            mode=str(mode) if mode is not None else None,
            actor=actor,
            confirmed=confirmed,
            note=note,
        )
        session.add(row)
        session.flush()
        log_event(
            "capital",
            {
                "event": "ledger",
                "batch_id": acct.batch_id,
                "ledger_id": row.id,
                "entry_type": row.entry_type,
                "amount": row.amount,
                "from": row.from_account,
                "to": row.to_account,
                "trading_cash": acct.trading_cash,
                "blocked": acct.blocked,
                "reserve": acct.reserve,
                "reserve_peak": acct.reserve_peak,
                "sizing_capital": acct.sizing_capital,
                "position_id": position_id,
                "trade_id": trade_id,
                "mode": row.mode,
                "actor": actor,
                "note": note,
            },
            clock=self.clock,
        )
        return row

    def _set_sizing(self, session: Session, acct: CapitalAccount, new_sizing: float, reason: str) -> float:
        """Change sizing capital with a SIZING_CHANGE ledger row. Returns the delta."""
        delta = float(new_sizing) - acct.sizing_capital
        if abs(delta) < 1e-9:
            return 0.0
        acct.sizing_capital = float(new_sizing)
        self._post(session, acct, L.SIZING_CHANGE, delta, None, Acc.SIZING, note=reason)
        logger.info(f"[capital] batch {acct.batch_id} sizing capital -> {new_sizing:.2f} ({reason})")
        return delta

    def _apply_sizing_decrease(self, session: Session, acct: CapitalAccount) -> bool:
        """Section 6.3 'shrink fast': sizing = TRADING the moment TRADING < sizing."""
        trading = self.trading_balance(acct)
        if trading < acct.sizing_capital - 1e-9:
            self._set_sizing(session, acct, max(0.0, trading), "decrease: TRADING below sizing capital")
            acct.sizing_up_streak = 0
            return True
        return False

    # ================================================================== position money

    def block(
        self,
        session: Session,
        batch_id: int,
        amount: float,
        *,
        position_id: int | None = None,
        mode: Mode | str | None = None,
        note: str | None = None,
    ) -> CapitalLedgerEntry:
        """Move a position's margin from TRADING cash to BLOCKED.

        Raises:
            CapitalError: ``amount <= 0``.
            InsufficientCashError: ``amount`` exceeds TRADING cash. RESERVE is never used, even
                when ``cash + reserve`` would cover it.
        """
        if not amount > 0:
            raise CapitalError(f"block amount must be > 0, got {amount}")
        acct = self.account(session, batch_id)
        if amount > acct.trading_cash + 1e-9:
            raise InsufficientCashError(
                f"cannot block Rs {amount:.2f}: TRADING cash is Rs {acct.trading_cash:.2f} "
                f"(RESERVE Rs {acct.reserve:.2f} is never available for trading)"
            )
        acct.trading_cash -= amount
        acct.blocked += amount
        return self._post(
            session, acct, L.BLOCK, amount, Acc.TRADING, Acc.BLOCKED, position_id=position_id, mode=mode, note=note
        )

    def release(
        self,
        session: Session,
        batch_id: int,
        margin: float,
        net_pnl: float,
        *,
        position_id: int | None = None,
        trade_id: int | None = None,
        mode: Mode | str | None = None,
        note: str | None = None,
    ) -> list[CapitalLedgerEntry]:
        """Close (part of) a position: RELEASE ``margin`` then book REALIZED_PNL ``net_pnl``.

        ``net_pnl`` = gross - (accrued entry charges for the closed qty + exit charges). A
        REALIZED_PNL row is always written (even for 0) so every closed leg is in the ledger.
        A release up to one paisa above ``blocked`` (float residue of proportional releases) is
        clamped; anything larger raises ``LedgerImbalanceError``.
        """
        if margin < 0:
            raise CapitalError(f"release margin must be >= 0, got {margin}")
        acct = self.account(session, batch_id)
        rows: list[CapitalLedgerEntry] = []
        if margin > 0:
            if margin > acct.blocked + BALANCE_TOLERANCE:
                raise LedgerImbalanceError(
                    f"cannot release Rs {margin:.4f}: only Rs {acct.blocked:.4f} blocked (batch {batch_id})"
                )
            amount = min(margin, acct.blocked)
            acct.blocked -= amount
            acct.trading_cash += amount
            rows.append(
                self._post(session, acct, L.RELEASE, amount, Acc.BLOCKED, Acc.TRADING,
                           position_id=position_id, trade_id=trade_id, mode=mode, note=note)
            )
        acct.trading_cash += net_pnl
        acct.cum_realized_net += net_pnl
        rows.append(
            self._post(session, acct, L.REALIZED_PNL, net_pnl, None, Acc.TRADING,
                       position_id=position_id, trade_id=trade_id, mode=mode, note=note)
        )
        if acct.trading_cash < -1e-9:
            logger.warning(
                f"[capital] batch {batch_id} TRADING cash negative ({acct.trading_cash:.2f}) after a loss "
                "(leveraged loss exceeded margin); RESERVE is NOT used to cover it"
            )
        self._apply_sizing_decrease(session, acct)
        return rows

    def credit_dividend(
        self,
        session: Session,
        batch_id: int,
        amount: float,
        *,
        position_id: int | None = None,
        symbol: str | None = None,
        mode: Mode | str | None = None,
        note: str | None = None,
    ) -> CapitalLedgerEntry:
        """Credit a cash dividend to TRADING cash (counted as realized net P&L)."""
        if not amount > 0:
            raise CapitalError(f"dividend amount must be > 0, got {amount}")
        acct = self.account(session, batch_id)
        acct.trading_cash += amount
        acct.cum_realized_net += amount
        text = note or (f"dividend {symbol}" if symbol else "dividend")
        return self._post(session, acct, L.DIVIDEND, amount, None, Acc.TRADING,
                          position_id=position_id, mode=mode, note=text)

    # ================================================================== HWM / halt

    def update_hwm(self, session: Session, batch_id: int, equity: float) -> float:
        """Raise the high-water mark to ``equity`` if higher. Returns the HWM."""
        acct = self.account(session, batch_id)
        if equity > acct.hwm:
            acct.hwm = float(equity)
            acct.updated_at = self.clock.now()
        return acct.hwm

    def set_halted(self, session: Session, batch_id: int, reason: str, *, actor: str = "SYSTEM") -> None:
        """Mark the account HALTED (section 8). Exiting positions is the engine's job."""
        acct = self.account(session, batch_id)
        if acct.halted:
            return
        acct.halted = True
        acct.halt_reason = reason
        acct.tier = DrawdownTier.HALT.value
        acct.updated_at = self.clock.now()
        audit(session, actor, "CAPITAL_HALT", f"batch:{batch_id}", {"reason": reason},
              batch_id=batch_id, clock=self.clock)
        log_event("capital", {"event": "halt", "batch_id": batch_id, "reason": reason}, clock=self.clock)

    def clear_halt(self, session: Session, batch_id: int, actor: str, *, note: str | None = None) -> None:
        """Owner resume from HALT (dashboard only). Audit-logged."""
        acct = self.account(session, batch_id)
        was = acct.halt_reason
        acct.halted = False
        acct.halt_reason = None
        acct.updated_at = self.clock.now()
        audit(session, actor, "CAPITAL_RESUME_HALT", f"batch:{batch_id}",
              {"previous_reason": was, "note": note}, batch_id=batch_id, clock=self.clock)
        log_event("capital", {"event": "resume_halt", "batch_id": batch_id, "actor": actor}, clock=self.clock)

    # ================================================================== EOD (6.2 / 6.3)

    def realized_net_for_day(self, session: Session, batch_id: int, day: date) -> float:
        """Sum of REALIZED_PNL + DIVIDEND ledger amounts timestamped on ``day`` (IST)."""
        start = datetime.combine(day, time(0, 0), tzinfo=IST)
        end = start + timedelta(days=1)
        total = session.execute(
            select(func.coalesce(func.sum(CapitalLedgerEntry.amount), 0.0)).where(
                CapitalLedgerEntry.batch_id == batch_id,
                CapitalLedgerEntry.entry_type.in_([L.REALIZED_PNL.value, L.DIVIDEND.value]),
                CapitalLedgerEntry.ts >= start,
                CapitalLedgerEntry.ts < end,
            )
        ).scalar_one()
        return float(total)

    def run_eod(
        self,
        session: Session,
        batch_id: int,
        day: date,
        realized_net_today: float | None = None,
        equity: float | None = None,
        tier: DrawdownTier | None = None,
    ) -> EODResult:
        """Section 6.2 / 6.3 EOD processing. Idempotent per day (``last_eod_date``).

        Args:
            day: the trading day being closed.
            realized_net_today: REALIZED net P&L of the day (incl. dividends). ``None`` -> summed
                from today's REALIZED_PNL + DIVIDEND ledger rows.
            equity: TOTAL EQUITY incl. MTM of open positions (for HWM / drawdown). ``None`` ->
                ``cash + blocked + reserve`` (no MTM).
            tier: drawdown tier from the latest snapshot. ``None`` -> computed here.

        Steps: profit > 0 -> PROFIT_LOCK ``profit_lock_pct`` % to RESERVE (capped at free cash;
        any shortfall is reported, not deferred). Loss -> LOSS_ABSORB marker, then if
        ``cash + blocked < BASE`` a REFILL from RESERVE up to BASE, never taking RESERVE below
        ``reserve_protected_pct`` % of its peak. (No refill on flat/profit days - refills are
        tied to absorbing a loss, per 6.2.) Then HWM, sizing step (6.3), daily_summary TOTAL row.
        """
        acct = self.account(session, batch_id)
        cap = self.settings.capital
        if acct.last_eod_date is not None and day <= acct.last_eod_date:
            logger.info(f"[capital] EOD for batch {batch_id} on {day} already processed (last {acct.last_eod_date})")
            return self._eod_result(acct, day, skipped=True, equity=equity)

        realized = self.realized_net_for_day(session, batch_id, day) if realized_net_today is None \
            else float(realized_net_today)
        res = EODResult(batch_id=batch_id, day=day, skipped=False, realized_net=realized,
                        sizing_before=acct.sizing_capital)

        # ---- 6.2 profit lock / loss absorb / capped refill
        if realized > 0:
            lock = realized * cap.profit_lock_pct / 100.0
            free = max(0.0, acct.trading_cash)
            if lock > free:
                res.profit_lock_shortfall = lock - free
                res.notes.append(f"profit lock capped by free cash: wanted {lock:.2f}, locked {free:.2f}")
                logger.warning(f"[capital] batch {batch_id} {res.notes[-1]}")
                lock = free
            if lock > 0:
                acct.trading_cash -= lock
                acct.reserve += lock
                acct.reserve_peak = max(acct.reserve_peak, acct.reserve)
                self._post(session, acct, L.PROFIT_LOCK, lock, Acc.TRADING, Acc.RESERVE,
                           note=f"EOD {day}: lock {cap.profit_lock_pct:g}% of realized {realized:.2f}")
                res.profit_locked = lock
        elif realized < 0:
            res.loss_absorbed = -realized
            self._post(session, acct, L.LOSS_ABSORB, 0.0, None, Acc.TRADING,
                       note=f"EOD {day}: TRADING absorbed realized loss {realized:.2f}")
            trading = self.trading_balance(acct)
            if trading < acct.base_capital - 1e-9:
                need = acct.base_capital - trading
                protected = acct.reserve_peak * cap.reserve_protected_pct / 100.0
                available = max(0.0, acct.reserve - protected)
                refill = min(need, available)
                res.refill_needed = need
                if refill > 1e-9:
                    acct.reserve -= refill
                    acct.trading_cash += refill
                    self._post(session, acct, L.REFILL, refill, Acc.RESERVE, Acc.TRADING,
                               note=f"EOD {day}: refill toward BASE {acct.base_capital:.2f} "
                                    f"(need {need:.2f}, reserve floor {protected:.2f})")
                    res.refilled = refill
                if refill < need - 1e-9:
                    res.notes.append(f"refill capped: needed {need:.2f}, refilled {refill:.2f}")

        # ---- HWM / tier
        eq = float(equity) if equity is not None else acct.trading_cash + acct.blocked + acct.reserve
        acct.hwm = max(acct.hwm, eq)
        dd_pct = self.protection.drawdown_pct(eq, acct.hwm)
        if tier is None:
            tier = self.protection.tier_for(dd_pct, eq, self.protection.effective_floor(acct.starting_capital, acct.hwm))
        tier = DrawdownTier(tier)
        acct.tier = tier.value

        # ---- 6.3 sizing: shrink fast, grow slowly
        trading = self.trading_balance(acct)
        if trading < acct.sizing_capital - 1e-9:
            self._apply_sizing_decrease(session, acct)
        elif tier == DrawdownTier.NORMAL and trading >= acct.sizing_capital * (1 + cap.sizing_increase_threshold_pct / 100.0) - 1e-9:
            acct.sizing_up_streak += 1
            if acct.sizing_up_streak >= cap.sizing_increase_days:
                if cap.sizing_step_mode == "STEP_PCT":
                    target = min(trading, acct.sizing_capital * (1 + cap.sizing_step_pct / 100.0))
                else:
                    target = trading
                self._set_sizing(session, acct, target,
                                 f"increase after {acct.sizing_up_streak} qualifying days ({cap.sizing_step_mode})")
                acct.sizing_up_streak = 0
        else:
            acct.sizing_up_streak = 0

        acct.last_eod_date = day
        acct.updated_at = self.clock.now()
        session.flush()

        out = self._eod_result(acct, day, skipped=False, equity=eq)
        for name in ("realized_net", "profit_locked", "profit_lock_shortfall", "loss_absorbed", "refilled",
                     "refill_needed", "sizing_before", "notes"):
            setattr(out, name, getattr(res, name))
        out.tier = tier
        out.drawdown_pct = dd_pct
        self._upsert_daily_total(session, out)
        log_event("capital", {"event": "eod", **out.as_dict()}, clock=self.clock)
        return out

    def _eod_result(self, acct: CapitalAccount, day: date, *, skipped: bool, equity: float | None) -> EODResult:
        eq = float(equity) if equity is not None else acct.trading_cash + acct.blocked + acct.reserve
        return EODResult(
            batch_id=acct.batch_id,
            day=day,
            skipped=skipped,
            sizing_before=acct.sizing_capital,
            sizing_after=acct.sizing_capital,
            sizing_up_streak=acct.sizing_up_streak,
            trading_cash=acct.trading_cash,
            blocked=acct.blocked,
            trading_balance=self.trading_balance(acct),
            reserve=acct.reserve,
            reserve_peak=acct.reserve_peak,
            equity=eq,
            hwm=acct.hwm,
            drawdown_pct=self.protection.drawdown_pct(eq, acct.hwm),
            tier=DrawdownTier(acct.tier),
        )

    def _upsert_daily_total(self, session: Session, res: EODResult) -> DailySummary:
        """Write the capital-owned fields of the ``daily_summary`` TOTAL row."""
        row = session.execute(
            select(DailySummary).where(
                DailySummary.batch_id == res.batch_id, DailySummary.day == res.day, DailySummary.mode == "TOTAL"
            )
        ).scalar_one_or_none()
        if row is None:
            row = DailySummary(batch_id=res.batch_id, day=res.day, mode="TOTAL")
            session.add(row)
        row.trading_balance = res.trading_balance
        row.trading_cash = res.trading_cash
        row.reserve = res.reserve
        row.equity = res.equity
        row.hwm = res.hwm
        row.drawdown_pct = res.drawdown_pct
        row.tier = res.tier.value
        row.sizing_capital = res.sizing_after
        row.profit_locked = res.profit_locked
        row.refilled = res.refilled
        session.flush()
        return row

    # ================================================================== owner actions

    def manual_reserve_transfer(
        self,
        session: Session,
        batch_id: int,
        amount: float,
        actor: str,
        confirmed: bool,
        *,
        note: str | None = None,
    ) -> CapitalLedgerEntry:
        """Owner-confirmed RESERVE -> TRADING transfer (the only non-refill way RESERVE shrinks).

        Raises ``UnconfirmedTransferError`` (and audit-logs the rejected attempt) unless
        ``confirmed`` is True; ``CapitalError`` for an amount <= 0 or above RESERVE.
        """
        acct = self.account(session, batch_id)
        if not confirmed:
            audit(session, actor, "RESERVE_TRANSFER_REJECTED", f"batch:{batch_id}",
                  {"amount": amount, "reason": "not confirmed"}, batch_id=batch_id, clock=self.clock)
            raise UnconfirmedTransferError("RESERVE -> TRADING transfer requires explicit confirmation")
        if not amount > 0:
            raise CapitalError(f"transfer amount must be > 0, got {amount}")
        if amount > acct.reserve + 1e-9:
            raise ReserveAccessError(f"cannot transfer Rs {amount:.2f}: RESERVE holds Rs {acct.reserve:.2f}")
        amount = min(float(amount), acct.reserve)
        acct.reserve -= amount
        acct.trading_cash += amount
        audit(session, actor, "RESERVE_TRANSFER", f"batch:{batch_id}",
              {"amount": amount, "reserve_after": acct.reserve, "note": note}, batch_id=batch_id, clock=self.clock)
        return self._post(session, acct, L.MANUAL_TRANSFER, amount, Acc.RESERVE, Acc.TRADING,
                          actor=actor, confirmed=True, note=note or "owner-confirmed RESERVE -> TRADING transfer")

    def manual_adjustment(
        self,
        session: Session,
        batch_id: int,
        amount: float,
        actor: str,
        reason: str,
        *,
        confirmed: bool = False,
    ) -> CapitalLedgerEntry:
        """Owner-confirmed signed adjustment of TRADING cash (counted in the invariant)."""
        acct = self.account(session, batch_id)
        if not confirmed:
            audit(session, actor, "CAPITAL_ADJUSTMENT_REJECTED", f"batch:{batch_id}",
                  {"amount": amount, "reason": reason}, batch_id=batch_id, clock=self.clock)
            raise UnconfirmedTransferError("manual capital adjustment requires explicit confirmation")
        if amount == 0:
            raise CapitalError("adjustment amount must be non-zero")
        if acct.trading_cash + amount < -1e-9:
            raise CapitalError(
                f"adjustment {amount:.2f} would make TRADING cash negative ({acct.trading_cash:.2f})"
            )
        acct.trading_cash += amount
        acct.cum_manual_adjustments += amount
        audit(session, actor, "CAPITAL_ADJUSTMENT", f"batch:{batch_id}",
              {"amount": amount, "reason": reason}, batch_id=batch_id, clock=self.clock)
        row = self._post(session, acct, L.MANUAL_ADJUSTMENT, amount, None, Acc.TRADING,
                         actor=actor, confirmed=True, note=reason)
        self._apply_sizing_decrease(session, acct)
        return row

    # ================================================================== sizing / views

    def mode_sizing_capital(self, account: CapitalAccount, settings: Settings | None = None) -> dict[Mode, float]:
        """Per-mode sizing capital = sizing_capital * allocation_pct / 100 (every mode)."""
        s = settings or self.settings
        return {m: account.sizing_capital * cfg.allocation_pct / 100.0 for m, cfg in s.modes.modes.items()}

    def check_ledger_balance(self, session: Session, batch_id: int) -> tuple[bool, dict[str, Any]]:
        """Verify the 6.4 invariant AND that replaying the ledger reproduces every balance.

        Returns ``(ok, details)``; ``details['problems']`` lists every mismatch found.
        """
        acct = self.account(session, batch_id)
        rows = list(session.execute(
            select(CapitalLedgerEntry).where(CapitalLedgerEntry.batch_id == batch_id).order_by(CapitalLedgerEntry.id)
        ).scalars())
        problems: list[str] = []
        lhs = acct.starting_capital + acct.cum_realized_net + acct.cum_manual_adjustments
        rhs = acct.trading_cash + acct.blocked + acct.reserve
        if abs(lhs - rhs) > BALANCE_TOLERANCE:
            problems.append(f"invariant: start+realized+adjustments={lhs:.4f} != cash+blocked+reserve={rhs:.4f}")

        cash = blocked = reserve = sizing = 0.0
        realized = adjustments = 0.0
        for r in rows:
            moves = _MOVES.get(r.entry_type)
            if moves is None:
                problems.append(f"row {r.id}: unknown entry_type {r.entry_type!r}")
                continue
            dc, db, dr = moves
            cash += dc * r.amount
            blocked += db * r.amount
            reserve += dr * r.amount
            if r.entry_type == L.INITIAL_CAPITAL:
                sizing = r.amount
            elif r.entry_type == L.SIZING_CHANGE:
                sizing += r.amount
            if r.entry_type in (L.REALIZED_PNL, L.DIVIDEND):
                realized += r.amount
            elif r.entry_type == L.MANUAL_ADJUSTMENT:
                adjustments += r.amount
            for name, replayed, stored in (
                ("cash", cash, r.trading_cash_after),
                ("blocked", blocked, r.blocked_after),
                ("reserve", reserve, r.reserve_after),
                ("sizing", sizing, r.sizing_capital_after),
            ):
                if abs(replayed - stored) > BALANCE_TOLERANCE:
                    problems.append(f"row {r.id} ({r.entry_type}): replayed {name} {replayed:.4f} != stored {stored:.4f}")
            if r.reserve_after < -BALANCE_TOLERANCE or r.blocked_after < -BALANCE_TOLERANCE:
                problems.append(f"row {r.id}: negative reserve/blocked")

        for name, replayed, stored in (
            ("cash", cash, acct.trading_cash),
            ("blocked", blocked, acct.blocked),
            ("reserve", reserve, acct.reserve),
            ("sizing", sizing, acct.sizing_capital),
            ("cum_realized_net", realized, acct.cum_realized_net),
            ("cum_manual_adjustments", adjustments, acct.cum_manual_adjustments),
        ):
            if abs(replayed - stored) > BALANCE_TOLERANCE:
                problems.append(f"final {name}: ledger {replayed:.4f} != account {stored:.4f}")

        details: dict[str, Any] = {
            "batch_id": batch_id,
            "rows": len(rows),
            "lhs": lhs,
            "rhs": rhs,
            "difference": lhs - rhs,
            "replayed": {"cash": cash, "blocked": blocked, "reserve": reserve, "sizing": sizing},
            "problems": problems,
        }
        if problems:
            logger.error(f"[capital] ledger check failed for batch {batch_id}: {problems[:5]}")
            log_event("capital", {"event": "ledger_check_failed", **details}, clock=self.clock)
        return not problems, details

    def assert_ledger_balanced(self, session: Session, batch_id: int) -> None:
        """Raise ``LedgerImbalanceError`` if :meth:`check_ledger_balance` fails."""
        ok, details = self.check_ledger_balance(session, batch_id)
        if not ok:
            raise LedgerImbalanceError("; ".join(details["problems"][:10]))

    def snapshot(
        self,
        session: Session,
        batch_id: int,
        open_positions: list[OpenPositionView],
        realized_today: float = 0.0,
        *,
        update_hwm: bool = True,
    ) -> CapitalSnapshot:
        """Point-in-time capital state (sections 6 and 8).

        Side effects (both intentional, 6.3 / 6.1): applies an immediate sizing decrease when
        TRADING < sizing (SIZING_CHANGE row), and - unless ``update_hwm=False`` - raises the
        persisted HWM when TOTAL EQUITY (incl. MTM) exceeds it. Also stores the tier.

        Unrealized P&L per position = gross MTM at ``ltp`` (0 when no quote) minus accrued
        entry charges.
        """
        acct = self.account(session, batch_id)
        self._apply_sizing_decrease(session, acct)
        unrealized = sum(p.unrealized_gross - p.entry_charges for p in open_positions)
        equity = acct.trading_cash + acct.blocked + unrealized + acct.reserve
        if update_hwm and equity > acct.hwm:
            acct.hwm = equity
        hwm = max(acct.hwm, equity)
        abs_floor, trail_floor = self.protection.floors(acct.starting_capital, hwm)
        dd = self.protection.drawdown_pct(equity, hwm)
        tier = self.protection.tier_for(dd, equity, max(abs_floor, trail_floor))
        open_risk = self.protection.open_risk(open_positions)
        budget = self.protection.loss_budget(equity, open_risk, (abs_floor, trail_floor))
        acct.tier = tier.value
        acct.updated_at = self.clock.now()
        mode_blocked: dict[str, float] = {}
        for p in open_positions:
            mode_blocked[str(p.mode)] = mode_blocked.get(str(p.mode), 0.0) + p.margin_blocked
        session.flush()
        return CapitalSnapshot(
            batch_id=batch_id,
            as_of=self.clock.now(),
            starting_capital=acct.starting_capital,
            base_capital=acct.base_capital,
            trading_cash=acct.trading_cash,
            blocked=acct.blocked,
            reserve=acct.reserve,
            reserve_peak=acct.reserve_peak,
            unrealized_pnl=unrealized,
            total_equity=equity,
            hwm=hwm,
            drawdown_pct=dd,
            tier=tier,
            sizing_capital=acct.sizing_capital,
            mode_sizing_capital={str(m): v for m, v in self.mode_sizing_capital(acct).items()},
            mode_blocked=mode_blocked,
            realized_today=realized_today,
            absolute_floor=abs_floor,
            trailing_floor=trail_floor,
            open_risk=open_risk,
            loss_budget=budget,
            halted=acct.halted or tier == DrawdownTier.HALT,
        )
