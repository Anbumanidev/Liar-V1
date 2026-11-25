"""Tax estimates: classification, set-off rules, exemption, cess, pre-trade scenarios, tips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from config.loader import update_settings
from config.schema import Settings, TaxSettings
from core.clock import IST
from core.enums import Direction, Mode, Product, TaxHead
from tax.engine import (
    classify,
    effective_rate_pct,
    fy_bounds,
    next_advance_tax_due,
    pre_trade_estimate,
    summarize,
    tax_tips,
    trade_tax_lines,
)


def dt(y: int, m: int, d: int, hh: int = 10) -> datetime:
    return datetime(y, m, d, hh, 0, tzinfo=IST)


@dataclass
class T:
    symbol: str
    mode: str
    product: str
    direction: str
    qty: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    gross_pnl: float
    charges: float
    id: int | None = None


def test_fy_bounds() -> None:
    assert fy_bounds(date(2026, 11, 25)) == (date(2026, 4, 1), date(2027, 3, 31), "FY2026-27")
    assert fy_bounds(date(2027, 2, 1))[2] == "FY2026-27"
    assert fy_bounds(date(2026, 3, 31))[2] == "FY2025-26"


def test_classification() -> None:
    cfg = TaxSettings()
    assert classify(Product.MIS, dt(2026, 11, 25), dt(2026, 11, 25, 15), cfg) == TaxHead.SPECULATIVE
    assert classify(Product.CNC, dt(2026, 11, 25), dt(2026, 11, 25, 15), cfg) == TaxHead.SPECULATIVE  # same-day CNC
    assert classify(Product.CNC, dt(2026, 11, 25), dt(2026, 12, 5), cfg) == TaxHead.STCG
    assert classify(Product.CNC, dt(2025, 11, 20), dt(2026, 11, 25), cfg) == TaxHead.LTCG
    assert classify(Product.CNC, dt(2026, 11, 25), dt(2026, 12, 5), TaxSettings(delivery_as_business=True)) \
        == TaxHead.BUSINESS


def test_rates_with_cess() -> None:
    cfg = TaxSettings()
    assert effective_rate_pct(TaxHead.STCG, cfg) == pytest.approx(20.8)
    assert effective_rate_pct(TaxHead.LTCG, cfg) == pytest.approx(13.0)
    assert effective_rate_pct(TaxHead.SPECULATIVE, cfg) == pytest.approx(31.2)


def test_stt_not_deductible_for_capital_gains(settings: Settings) -> None:
    t = T("INFY", "SWING", "CNC", "LONG", 10, 1000.0, 1100.0, dt(2026, 11, 2), dt(2026, 11, 12), 1000.0, 40.0)
    line = trade_tax_lines([t], settings.charges, TaxSettings())[0]
    assert line.head == TaxHead.STCG and line.stt == pytest.approx(21.0)   # 0.1% of 10000 + 0.1% of 11000
    assert line.taxable == pytest.approx(1000.0 - (40.0 - 21.0))
    assert line.standalone_tax == pytest.approx(line.taxable * 0.208, abs=0.01)
    biz = trade_tax_lines([t], settings.charges, TaxSettings(delivery_as_business=True))[0]
    assert biz.taxable == pytest.approx(960.0)


def _line(head: TaxHead, taxable: float, d: date = date(2026, 11, 25)):  # type: ignore[no-untyped-def]
    from tax.engine import TradeTaxLine

    return TradeTaxLine(None, "X", "SWING", head, d, 5, taxable, 0.0, 0.0, taxable, 0.0, taxable)


def test_set_off_rules_and_exemption() -> None:
    cfg = TaxSettings()
    lines = [_line(TaxHead.STCG, 50_000), _line(TaxHead.STCG, -70_000), _line(TaxHead.LTCG, 200_000),
             _line(TaxHead.SPECULATIVE, -5_000), _line(TaxHead.SPECULATIVE, 2_000)]
    s = summarize(lines, cfg, date(2026, 11, 25))
    # net STCL 20k offsets LTCG 200k -> 180k; exemption 125k -> 55k taxable at 12.5%
    assert s.stcg_taxable == 0 and s.ltcg_after_setoff == pytest.approx(180_000)
    assert s.ltcg_taxable == pytest.approx(55_000) and s.ltcg_exemption_left == 0
    assert s.carry_forward_speculative_loss == pytest.approx(3_000)       # intraday loss stays intraday
    assert s.tax_ltcg == pytest.approx(6_875) and s.cess == pytest.approx(275)
    assert s.total_tax == pytest.approx(7_150)


def test_long_term_loss_only_offsets_ltcg() -> None:
    s = summarize([_line(TaxHead.STCG, 10_000), _line(TaxHead.LTCG, -8_000)], TaxSettings(), date(2026, 11, 25))
    assert s.stcg_taxable == pytest.approx(10_000) and s.carry_forward_lt_capital_loss == pytest.approx(8_000)
    assert s.tax_stcg == pytest.approx(2_000)


def test_basic_exemption_absorbs_stcg_first() -> None:
    s = summarize([_line(TaxHead.STCG, 10_000)], TaxSettings(basic_exemption_unused=4_000), date(2026, 11, 25))
    assert s.stcg_taxable == pytest.approx(6_000) and s.basic_exemption_used == pytest.approx(4_000)


def test_other_fy_excluded() -> None:
    s = summarize([_line(TaxHead.STCG, 10_000, date(2026, 3, 30))], TaxSettings(), date(2026, 11, 25))
    assert s.n_trades == 0 and s.total_tax == 0


def test_pre_trade_estimate(settings: Settings) -> None:
    est = pre_trade_estimate(Mode.SWING, Direction.LONG, 2, 500.0, 485.0, 530.0, 545.0, settings.charges, settings.tax)
    assert est.head == TaxHead.STCG
    labels = [x.label for x in est.scenarios]
    assert labels == ["Stop-loss", "Target 1", "Target 2"]
    sl, t1 = est.scenarios[0], est.scenarios[1]
    assert sl.tax == 0 and sl.net_after_tax < 0
    assert t1.gross == pytest.approx(60.0) and t1.tax > 0 and t1.net_after_tax == pytest.approx(60 - t1.charges - t1.tax)
    assert est.post_tax_rr is not None and est.post_tax_rr > 0
    intraday = pre_trade_estimate(Mode.INTRADAY, Direction.SHORT, 5, 200.0, 202.0, 197.0, None, settings.charges,
                                  settings.tax)
    assert intraday.head == TaxHead.SPECULATIVE and len(intraday.scenarios) == 2
    assert intraday.scenarios[1].gross == pytest.approx(15.0)


def test_advance_tax_dates() -> None:
    assert next_advance_tax_due(date(2026, 11, 25)) == (date(2026, 12, 15), 75)
    assert next_advance_tax_due(date(2027, 1, 10)) == (date(2027, 3, 15), 100)
    assert next_advance_tax_due(date(2026, 4, 2)) == (date(2026, 6, 15), 15)
    assert next_advance_tax_due(date(2027, 3, 20)) == (date(2027, 6, 15), 15)


def test_tips(settings: Settings) -> None:
    cfg = settings.tax
    s = summarize([_line(TaxHead.STCG, 80_000), _line(TaxHead.SPECULATIVE, -2_000)], cfg, date(2027, 3, 1))
    pos = SimpleNamespace(symbol="LT", product="CNC", direction="LONG", qty=10, avg_entry=100.0, last_price=90.0,
                          opened_at=dt(2027, 2, 1))
    tips = tax_tips(s, [pos], date(2027, 3, 1), cfg, settings.charges)
    titles = [t.title for t in tips]
    assert "Tax-loss harvesting: LT" in titles
    harvest = next(t for t in tips if t.title.startswith("Tax-loss"))
    assert harvest.est_saving == pytest.approx(100 * 0.208, abs=0.01) and "31 March" in harvest.detail
    assert "Advance tax" in titles and "Intraday loss can only offset intraday profit" in titles
    assert titles[-1] == "Disclaimer"
