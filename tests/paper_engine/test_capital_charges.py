"""Charges model tests (section 9) with hand-computed examples."""

from __future__ import annotations

import pytest

from config.schema import ChargesSettings, Settings
from core.enums import Direction, OrderSide, Product
from paper_engine.charges import (
    ChargesCalculator,
    charges_pct_of_value,
    estimate_round_trip,
    expected_net_profit,
    order_charges,
    round_half_up,
    round_trip_charges,
)


@pytest.fixture
def cfg(settings: Settings) -> ChargesSettings:
    return settings.charges


def _with(cfg: ChargesSettings, product: str, **changes: object) -> ChargesSettings:
    pc = getattr(cfg, product).model_copy(update=changes)
    return cfg.model_copy(update={product: pc})


# ----------------------------------------------------------------------------- rounding


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.5, 1.0), (1.5, 2.0), (2.5, 3.0), (0.49, 0.0), (0.15, 0.0), (1.275, 1.0), (0.4999, 0.0),
     # binary noise below 1e-8 is stripped: a computed 0.4999999999999 is a true 0.5
     (0.4999999999999, 1.0)],
)
def test_round_half_up_is_not_bankers(value: float, expected: float) -> None:
    assert round_half_up(value) == expected


def test_round_half_up_handles_float_noise() -> None:
    # 2000 * 0.025 / 100 is a "true" 0.5 that may carry binary noise.
    assert round_half_up(2000 * 0.025 / 100.0) == 1.0
    assert round_half_up(0.1 + 0.2, 2) == 0.3
    assert round(2.5) == 2  # documents why Python's round() is not used


# ----------------------------------------------------------------------------- MIS


def test_mis_buy_order_hand_computed(cfg: ChargesSettings) -> None:
    # Buy 10 @ 500 = 5000. brokerage min(20, 0.03% = 1.5) = 1.5; STT buy 0;
    # exch 0.00297% = 0.1485; SEBI 10*5000/1e7 = 0.005; stamp 0.003% = 0.15 -> rounds to 0;
    # IPFT 0.0001% = 0.005; GST 18% of (1.5 + 0.1485 + 0.005) = 0.29763.
    b = order_charges(Product.MIS, OrderSide.BUY, 10, 500.0, cfg)
    assert b.brokerage == pytest.approx(1.5)
    assert b.stt == 0.0
    assert b.exchange_txn == pytest.approx(0.1485)
    assert b.sebi == pytest.approx(0.005)
    assert b.stamp_duty == 0.0
    assert b.ipft == pytest.approx(0.005)
    assert b.gst == pytest.approx(0.29763)
    assert b.dp == 0.0
    assert b.total == pytest.approx(1.95613, abs=1e-4)


def test_mis_round_trip_10_at_500_to_510(cfg: ChargesSettings) -> None:
    # Sell 10 @ 510 = 5100: brokerage 1.53; STT 0.025% = 1.275 -> 1; exch 0.15147;
    # SEBI 0.0051; IPFT 0.0051; GST 0.18*(1.53+0.15147+0.0051) = 0.3035826 -> 2.9952526.
    sell = order_charges(Product.MIS, OrderSide.SELL, 10, 510.0, cfg)
    assert sell.stt == 1.0
    assert sell.stamp_duty == 0.0
    assert sell.total == pytest.approx(2.9952526, abs=1e-4)
    rt = round_trip_charges(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg)
    assert rt.total == pytest.approx(1.95613 + 2.9952526, abs=1e-4)  # 4.9514
    assert estimate_round_trip(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg) == pytest.approx(rt.total)
    # gross 100 - 4.95 = 95.05
    assert expected_net_profit(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg) == pytest.approx(
        100.0 - rt.total
    )
    assert charges_pct_of_value(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg) < 1.0


def test_mis_brokerage_cap_applies_on_large_orders(cfg: ChargesSettings) -> None:
    # 0.03% of 1,00,000 = 30 -> capped at 20
    b = order_charges(Product.MIS, OrderSide.BUY, 100, 1000.0, cfg)
    assert b.brokerage == 20.0
    # stamp 0.003% of 1e5 = 3.0 exactly
    assert b.stamp_duty == 3.0


def test_mis_short_round_trip_sell_first(cfg: ChargesSettings) -> None:
    rt = round_trip_charges(Product.MIS, Direction.SHORT, 10, 510.0, 500.0, cfg)
    # same two orders as the LONG example, just reversed in time -> same total
    assert rt.total == pytest.approx(round_trip_charges(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg).total)
    assert expected_net_profit(Product.MIS, Direction.SHORT, 10, 510.0, 500.0, cfg) == pytest.approx(100.0 - rt.total)


# ----------------------------------------------------------------------------- CNC


def test_cnc_round_trip_1_at_600_to_640(cfg: ChargesSettings) -> None:
    # Buy 1 @ 600: brokerage 0; STT 0.1% = 0.6 -> 1; exch 0.01782; SEBI 0.0006; stamp 0.015% = 0.09 -> 0;
    #   IPFT 0.0006; GST 0.18*(0.01782+0.0006) = 0.0033156 -> 1.0223356
    buy = order_charges(Product.CNC, OrderSide.BUY, 1, 600.0, cfg)
    assert buy.stt == 1.0 and buy.stamp_duty == 0.0 and buy.dp == 0.0 and buy.brokerage == 0.0
    assert buy.total == pytest.approx(1.0223356, abs=1e-4)  # .total is rounded to 4 dp
    # Sell 1 @ 640: STT 0.64 -> 1; exch 0.019008; SEBI 0.00064; IPFT 0.00064; DP 13;
    #   GST 0.18*(13 + 0.019008 + 0.00064) = 2.34353664 -> 16.36382464
    sell = order_charges(Product.CNC, OrderSide.SELL, 1, 640.0, cfg)
    assert sell.dp == 13.0
    assert sell.gst == pytest.approx(2.34353664, abs=1e-8)
    assert sell.total == pytest.approx(16.36382464, abs=1e-4)
    rt = round_trip_charges(Product.CNC, Direction.LONG, 1, 600.0, 640.0, cfg)
    assert rt.total == pytest.approx(17.38616024, abs=1e-4)
    assert rt.dp == 13.0


def test_cnc_small_trade_exceeds_one_percent(cfg: ChargesSettings, settings: Settings) -> None:
    """A ~Rs 600 CNC trade costs ~2.9 % round trip (DP charge + GST) -> fails the 1 % rule."""
    pct = charges_pct_of_value(Product.CNC, Direction.LONG, 1, 600.0, 600.0, cfg)
    assert pct == pytest.approx(17.3846712 / 600 * 100, abs=1e-3)
    assert pct > settings.risk.max_charges_pct_of_value
    # whereas a Rs 60,000 CNC trade is well under 1 %
    assert charges_pct_of_value(Product.CNC, Direction.LONG, 100, 600.0, 600.0, cfg) < 1.0


def test_cnc_dp_override(cfg: ChargesSettings) -> None:
    second_sell = order_charges(Product.CNC, OrderSide.SELL, 1, 600.0, cfg, include_dp=False)
    assert second_sell.dp == 0.0
    forced = order_charges(Product.CNC, OrderSide.BUY, 1, 600.0, cfg, include_dp=True)
    assert forced.dp == 13.0
    assert order_charges(Product.MIS, OrderSide.SELL, 1, 600.0, cfg).dp == 0.0


# ----------------------------------------------------------------------------- models / edge cases


def test_flat_and_zero_brokerage_models(cfg: ChargesSettings) -> None:
    flat = _with(cfg, "mis", brokerage_model="FLAT", brokerage_flat=20.0)
    assert order_charges(Product.MIS, OrderSide.BUY, 1, 100.0, flat).brokerage == 20.0
    zero = _with(cfg, "mis", brokerage_model="ZERO")
    assert order_charges(Product.MIS, OrderSide.BUY, 1, 100.0, zero).brokerage == 0.0
    pct = _with(cfg, "cnc", brokerage_model="PCT_CAP", brokerage_pct=0.1, brokerage_cap=5.0)
    assert order_charges(Product.CNC, OrderSide.BUY, 1, 1000.0, pct).brokerage == pytest.approx(1.0)
    assert order_charges(Product.CNC, OrderSide.BUY, 10, 1000.0, pct).brokerage == 5.0


def test_gst_base_includes_brokerage_exchange_sebi_dp_only(cfg: ChargesSettings) -> None:
    b = order_charges(Product.CNC, OrderSide.SELL, 7, 333.0, cfg)
    assert b.gst == pytest.approx(0.18 * (b.brokerage + b.exchange_txn + b.sebi + b.dp))


def test_no_rounding_when_disabled(cfg: ChargesSettings) -> None:
    raw = _with(cfg, "mis", round_stt_stamp=False)
    b = order_charges(Product.MIS, OrderSide.SELL, 10, 510.0, raw)
    assert b.stt == pytest.approx(1.275)
    b2 = order_charges(Product.MIS, OrderSide.BUY, 10, 500.0, raw)
    assert b2.stamp_duty == pytest.approx(0.15)


def test_zero_qty_and_invalid_price(cfg: ChargesSettings) -> None:
    assert order_charges(Product.MIS, OrderSide.BUY, 0, 100.0, cfg).total == 0.0
    with pytest.raises(ValueError):
        order_charges(Product.MIS, OrderSide.BUY, 1, -1.0, cfg)
    assert charges_pct_of_value(Product.MIS, Direction.LONG, 0, 100.0, 100.0, cfg) == float("inf")


def test_expected_net_profit_negative_when_target_too_close(cfg: ChargesSettings) -> None:
    # 1 share, target Rs 1 away: gross 1 < CNC charges ~17 -> charges exceed edge
    assert expected_net_profit(Product.CNC, Direction.LONG, 1, 600.0, 601.0, cfg) < 0


def test_calculator_wrapper_matches_functions(cfg: ChargesSettings) -> None:
    calc = ChargesCalculator(cfg)
    assert calc.estimate_round_trip(Product.MIS, Direction.LONG, 10, 500.0, 510.0) == pytest.approx(
        estimate_round_trip(Product.MIS, Direction.LONG, 10, 500.0, 510.0, cfg)
    )
    assert calc.order(Product.CNC, OrderSide.SELL, 1, 600.0).dp == 13.0
    assert calc.round_trip(Product.CNC, Direction.LONG, 1, 600.0, 640.0).total == pytest.approx(17.3862, abs=1e-4)
    assert calc.pct_of_value(Product.CNC, Direction.LONG, 1, 600.0, 600.0) > 1.0
    assert calc.expected_net_profit(Product.MIS, Direction.LONG, 10, 500.0, 510.0) > 90.0
