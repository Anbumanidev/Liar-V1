"""Deterministic brokerage + statutory charges model for paper trades (section 9).

Two product models are configured in ``config/charges.yaml`` (``ChargesSettings``):

* **MIS** (INTRADAY): brokerage ``min(Rs 20, 0.03 %)`` per executed order by default, STT on the
  sell side only, exchange transaction charge, SEBI fee, stamp duty on the buy side, GST.
* **CNC** (BTST / SWING / POSITIONAL): brokerage configurable (default ZERO), STT on buy *and*
  sell, exchange transaction charge, SEBI fee, higher stamp duty on the buy side, GST, and a DP
  charge per scrip on the sell day.

Formulas (``value = qty * price``; every ``*_pct`` is in PERCENT units, so ``0.025`` = 0.025 %):

==================  =====================================================================
brokerage           FLAT: ``brokerage_flat`` per executed order;
                    PCT_CAP: ``min(brokerage_cap, brokerage_pct/100 * value)``;
                    ZERO: ``0``.
STT                 ``stt_buy_pct/100 * value`` (BUY) or ``stt_sell_pct/100 * value`` (SELL)
exchange_txn        ``exchange_txn_pct/100 * value``
SEBI fee            ``sebi_per_crore * value / 1e7``
stamp duty          ``stamp_buy_pct/100 * value`` on BUY orders only
IPFT                ``ipft_pct/100 * value``
DP charge           ``dp_charge_per_scrip`` on a CNC SELL (one per scrip per sell day)
GST                 ``gst_pct/100 * (brokerage + exchange_txn + SEBI fee + DP charge)``
                    (no GST on STT, stamp duty or IPFT)
==================  =====================================================================

Rounding: when ``round_stt_stamp`` is true, STT and stamp duty are each rounded to the nearest
whole rupee **half-up** (0.50 -> 1, 0.49 -> 0), which is what contract notes do. Python's
built-in ``round`` is banker's rounding (half-even: ``round(0.5) == 0``, ``round(2.5) == 2``)
and is deliberately NOT used. The raw value is first quantised to 8 decimals so binary-float
noise (``0.49999999999`` for a true ``0.5``) cannot flip the result. All other components are
kept at full float precision (no paise rounding) so per-leg accrual stays proportional.

The rounding is applied per order. That matches the exchange for CNC (buy and sell are on
different days, i.e. different contract notes) and for MIS (STT is sell-only, stamp is buy-only).

Everything here is pure: no I/O, no clock, no globals.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from config.schema import ChargesSettings, ProductCharges
from core.enums import Direction, OrderSide, Product
from core.types import ChargesBreakdown

__all__ = [
    "ChargesCalculator",
    "charges_pct_of_value",
    "entry_side",
    "estimate_round_trip",
    "exit_side",
    "expected_net_profit",
    "gross_pnl",
    "order_charges",
    "product_charges",
    "round_half_up",
    "round_trip_charges",
]

#: SEBI turnover fee is quoted per crore (1 crore = 10,000,000 rupees).
ONE_CRORE: float = 1e7


def round_half_up(value: float, ndigits: int = 0) -> float:
    """Round ``value`` half-up (away from zero on .5) to ``ndigits`` decimals.

    ``round_half_up(0.5) == 1.0`` and ``round_half_up(2.5) == 3.0`` (unlike Python's
    half-even ``round``). The float is quantised to 8 decimals first to strip binary noise.
    """
    clean = Decimal(repr(value)).quantize(Decimal("1e-8"), rounding=ROUND_HALF_UP)
    step = Decimal(1).scaleb(-ndigits)
    return float(clean.quantize(step, rounding=ROUND_HALF_UP))


def product_charges(product: Product, cfg: ChargesSettings) -> ProductCharges:
    """The charges model for ``product`` (MIS or CNC)."""
    return cfg.mis if Product(product) == Product.MIS else cfg.cnc


def entry_side(direction: Direction) -> OrderSide:
    """Order side that OPENS a position in ``direction``."""
    return OrderSide.BUY if Direction(direction) == Direction.LONG else OrderSide.SELL


def exit_side(direction: Direction) -> OrderSide:
    """Order side that CLOSES a position in ``direction``."""
    return OrderSide.SELL if Direction(direction) == Direction.LONG else OrderSide.BUY


def _brokerage(pc: ProductCharges, value: float) -> float:
    model = pc.brokerage_model
    if model == "ZERO":
        return 0.0
    if model == "FLAT":
        return pc.brokerage_flat
    if model == "PCT_CAP":
        return min(pc.brokerage_cap, pc.brokerage_pct / 100.0 * value)
    # The schema regex makes this unreachable; fail loudly rather than under-charge.
    raise ValueError(f"unknown brokerage_model {model!r}")


def order_charges(
    product: Product,
    side: OrderSide,
    qty: int,
    price: float,
    cfg: ChargesSettings,
    *,
    include_dp: bool | None = None,
) -> ChargesBreakdown:
    """Charges for ONE executed order of ``qty`` shares at ``price``.

    Args:
        product: MIS or CNC (selects the charges model).
        side: BUY or SELL.
        qty: executed quantity. ``qty <= 0`` -> all-zero breakdown (nothing executed, so no
            per-order brokerage either).
        price: fill price (must be >= 0).
        cfg: ``settings.charges``.
        include_dp: force the DP charge on/off. ``None`` (default) applies it to CNC SELL
            orders only. Pass ``False`` for the 2nd+ sell of the same scrip on the same day
            (DP is charged once per scrip per sell day).

    Returns:
        ``ChargesBreakdown`` (``.total`` is the sum of every component).
    """
    if price < 0:
        raise ValueError(f"price must be >= 0, got {price}")
    if qty <= 0:
        return ChargesBreakdown()
    product = Product(product)
    side = OrderSide(side)
    pc = product_charges(product, cfg)
    value = float(qty) * float(price)

    brokerage = _brokerage(pc, value)
    stt_pct = pc.stt_buy_pct if side == OrderSide.BUY else pc.stt_sell_pct
    stt = stt_pct / 100.0 * value
    exchange_txn = pc.exchange_txn_pct / 100.0 * value
    sebi = pc.sebi_per_crore * value / ONE_CRORE
    stamp = pc.stamp_buy_pct / 100.0 * value if side == OrderSide.BUY else 0.0
    ipft = pc.ipft_pct / 100.0 * value
    if include_dp is None:
        include_dp = product == Product.CNC and side == OrderSide.SELL
    dp = pc.dp_charge_per_scrip if include_dp else 0.0
    if pc.round_stt_stamp:
        stt = round_half_up(stt)
        stamp = round_half_up(stamp)
    gst = pc.gst_pct / 100.0 * (brokerage + exchange_txn + sebi + dp)
    return ChargesBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange_txn,
        sebi=sebi,
        stamp_duty=stamp,
        gst=gst,
        dp=dp,
        ipft=ipft,
    )


def round_trip_charges(
    product: Product,
    direction: Direction,
    qty: int,
    entry_price: float,
    exit_price: float,
    cfg: ChargesSettings,
) -> ChargesBreakdown:
    """Entry order + exit order charges for a full round trip.

    LONG = BUY at ``entry_price`` then SELL at ``exit_price``; SHORT = SELL then BUY. The DP
    charge follows ``order_charges``' default (CNC SELL only).
    """
    entry = order_charges(product, entry_side(direction), qty, entry_price, cfg)
    exit_ = order_charges(product, exit_side(direction), qty, exit_price, cfg)
    return entry + exit_


def estimate_round_trip(
    product: Product,
    direction: Direction,
    qty: int,
    entry: float,
    exit: float,  # noqa: A002 - name mirrors the contract
    cfg: ChargesSettings,
) -> float:
    """Total round-trip charges in rupees (convenience for the risk manager)."""
    return round_trip_charges(product, direction, qty, entry, exit, cfg).total


def gross_pnl(direction: Direction, qty: int, entry: float, exit: float) -> float:  # noqa: A002
    """Gross (pre-charges) P&L of a round trip."""
    sign = 1.0 if Direction(direction) == Direction.LONG else -1.0
    return sign * (float(exit) - float(entry)) * qty


def expected_net_profit(
    product: Product,
    direction: Direction,
    qty: int,
    entry: float,
    target: float,
    cfg: ChargesSettings,
) -> float:
    """Gross profit at ``target`` minus round-trip charges (section 5 charges check).

    The risk manager rejects with ``CHARGES_EXCEED_EDGE`` when this is ``<= 0``.
    """
    return gross_pnl(direction, qty, entry, target) - estimate_round_trip(
        product, direction, qty, entry, target, cfg
    )


def charges_pct_of_value(
    product: Product,
    direction: Direction,
    qty: int,
    entry_price: float,
    exit_price: float,
    cfg: ChargesSettings,
) -> float:
    """Round-trip charges as a PERCENT of the position value (``qty * entry_price``).

    Compared against ``risk.max_charges_pct_of_value`` (default 1 %). Returns ``inf`` for a
    zero-value position so such a trade can never pass the check.
    """
    value = float(qty) * float(entry_price)
    if value <= 0:
        return float("inf")
    return estimate_round_trip(product, direction, qty, entry_price, exit_price, cfg) / value * 100.0


class ChargesCalculator:
    """Thin object wrapper so callers can hold one configured instance.

    Example::

        calc = ChargesCalculator(settings.charges)
        calc.estimate_round_trip(Product.MIS, Direction.LONG, 10, 500.0, 510.0)
    """

    def __init__(self, cfg: ChargesSettings) -> None:
        self.cfg = cfg

    def order(
        self, product: Product, side: OrderSide, qty: int, price: float, *, include_dp: bool | None = None
    ) -> ChargesBreakdown:
        """See :func:`order_charges`."""
        return order_charges(product, side, qty, price, self.cfg, include_dp=include_dp)

    def round_trip(
        self, product: Product, direction: Direction, qty: int, entry_price: float, exit_price: float
    ) -> ChargesBreakdown:
        """See :func:`round_trip_charges`."""
        return round_trip_charges(product, direction, qty, entry_price, exit_price, self.cfg)

    def estimate_round_trip(
        self, product: Product, direction: Direction, qty: int, entry: float, exit: float  # noqa: A002
    ) -> float:
        """See :func:`estimate_round_trip`."""
        return estimate_round_trip(product, direction, qty, entry, exit, self.cfg)

    def expected_net_profit(
        self, product: Product, direction: Direction, qty: int, entry: float, target: float
    ) -> float:
        """See :func:`expected_net_profit`."""
        return expected_net_profit(product, direction, qty, entry, target, self.cfg)

    def pct_of_value(
        self, product: Product, direction: Direction, qty: int, entry_price: float, exit_price: float
    ) -> float:
        """See :func:`charges_pct_of_value`."""
        return charges_pct_of_value(product, direction, qty, entry_price, exit_price, self.cfg)
