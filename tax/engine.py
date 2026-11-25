"""Indian income-tax ESTIMATES for equity trades (NOT tax advice).

Deterministic Python (no LLM). Rates come from ``config/tax.yaml`` (FY 2026-27 defaults):

* Intraday equity (bought and sold the same day) -> SPECULATIVE business income at the
  owner's slab rate. Speculative losses only offset speculative income; carry forward 4 years.
* Delivery held <= 12 months -> STCG 20 % (old s.111A, new Income-tax Act 2025 s.196).
* Delivery held  > 12 months -> LTCG 12.5 % above Rs 1.25 lakh per FY (old s.112A, new s.198).
* Short-term capital loss offsets STCG and then LTCG; long-term loss only LTCG; 8-year carry
  forward (only if the return is filed on time).
* Capital gains: transaction costs (brokerage, exchange, SEBI, stamp, GST, DP) are deductible,
  STT is NOT. Business / speculative income: every charge incl. STT is a deductible expense.
* + surcharge (if configured) and 4 % health & education cess on the tax.

Used before a trade (``pre_trade_estimate`` - shown in the approval queue / manual trade
preview) and after trades (``trade_tax_lines`` + ``summarize`` - FY liability, carry-forwards)
and by ``tax_tips`` (rule-based, explainable suggestions to reduce tax / charges).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from config.schema import ChargesSettings, TaxSettings
from core.enums import Direction, Mode, Product, TaxHead
from paper_engine.charges import order_charges, entry_side, exit_side, round_trip_charges

ADVANCE_TAX_THRESHOLD = 10_000.0
DISCLAIMER = ("Estimates only - not tax advice. Paper trades are simulated; tax applies only to real trades. "
              "Confirm with a chartered accountant before filing.")


# =========================================================================== helpers


def fy_bounds(d: date) -> tuple[date, date, str]:
    """Indian financial year containing ``d``: (1 April, 31 March, 'FY2026-27')."""
    start_year = d.year if d.month >= 4 else d.year - 1
    return date(start_year, 4, 1), date(start_year + 1, 3, 31), f"FY{start_year}-{str(start_year + 1)[-2:]}"


def classify(product: Product | str, entry_time: datetime, exit_time: datetime, cfg: TaxSettings) -> TaxHead:
    """Tax head of a closed trade leg."""
    if entry_time.date() == exit_time.date():
        return TaxHead.SPECULATIVE
    if cfg.delivery_as_business:
        return TaxHead.BUSINESS
    held = (exit_time.date() - entry_time.date()).days
    return TaxHead.LTCG if held >= cfg.ltcg_min_holding_days else TaxHead.STCG


def head_for_mode(mode: Mode, cfg: TaxSettings) -> TaxHead:
    """Expected tax head of a NEW trade in ``mode`` (all CNC modes hold < 12 months)."""
    if Mode(mode) == Mode.INTRADAY:
        return TaxHead.SPECULATIVE
    return TaxHead.BUSINESS if cfg.delivery_as_business else TaxHead.STCG


def base_rate_pct(head: TaxHead, cfg: TaxSettings) -> float:
    if head in (TaxHead.SPECULATIVE, TaxHead.BUSINESS):
        return cfg.slab_rate_pct
    return cfg.ltcg_rate_pct if head == TaxHead.LTCG else cfg.stcg_rate_pct


def effective_rate_pct(head: TaxHead, cfg: TaxSettings) -> float:
    """Marginal rate incl. surcharge and cess (percent)."""
    return base_rate_pct(head, cfg) * (1 + cfg.surcharge_pct / 100.0) * (1 + cfg.cess_pct / 100.0)


def deductible_charges(head: TaxHead, total_charges: float, stt: float) -> float:
    """Charges deductible from the gain: STT is not deductible for capital gains."""
    if head in (TaxHead.STCG, TaxHead.LTCG):
        return max(0.0, total_charges - stt)
    return total_charges


# =========================================================================== per trade


@dataclass
class TradeTaxLine:
    trade_id: int | None
    symbol: str
    mode: str
    head: TaxHead
    exit_date: date
    holding_days: int
    gross: float
    charges: float
    stt: float
    taxable: float          # signed: gross - deductible charges
    standalone_tax: float   # tax on this leg alone at the marginal rate (0 for losses)
    net_after_tax: float    # gross - charges - standalone_tax

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["head"] = self.head.value
        return d


def _stt_of_leg(product: Product, direction: Direction, qty: int, entry: float, exit_: float,
                charges_cfg: ChargesSettings) -> float:
    return (order_charges(product, entry_side(direction), qty, entry, charges_cfg).stt
            + order_charges(product, exit_side(direction), qty, exit_, charges_cfg).stt)


def trade_tax_lines(trades: Iterable[Any], charges_cfg: ChargesSettings, cfg: TaxSettings) -> list[TradeTaxLine]:
    """Tax lines for ``trades`` rows (storage.models.Trade or objects with the same fields)."""
    out: list[TradeTaxLine] = []
    for t in trades:
        head = classify(t.product, t.entry_time, t.exit_time, cfg)
        stt = _stt_of_leg(Product(t.product), Direction(t.direction), int(t.qty), float(t.entry_price),
                          float(t.exit_price), charges_cfg)
        charges = float(t.charges or 0.0)
        taxable = float(t.gross_pnl) - deductible_charges(head, charges, stt)
        tax = max(0.0, taxable) * effective_rate_pct(head, cfg) / 100.0
        out.append(TradeTaxLine(
            trade_id=getattr(t, "id", None), symbol=t.symbol, mode=t.mode, head=head, exit_date=t.exit_time.date(),
            holding_days=(t.exit_time.date() - t.entry_time.date()).days, gross=round(float(t.gross_pnl), 2),
            charges=round(charges, 2), stt=round(stt, 2), taxable=round(taxable, 2), standalone_tax=round(tax, 2),
            net_after_tax=round(float(t.gross_pnl) - charges - tax, 2),
        ))
    return out


# =========================================================================== FY summary


@dataclass
class TaxSummary:
    fy: str
    start: date
    end: date
    n_trades: int = 0
    speculative_net: float = 0.0
    business_net: float = 0.0
    stcg_gains: float = 0.0
    stcg_losses: float = 0.0
    ltcg_gains: float = 0.0
    ltcg_losses: float = 0.0
    stcg_taxable: float = 0.0
    ltcg_after_setoff: float = 0.0
    ltcg_exemption_used: float = 0.0
    ltcg_exemption_left: float = 0.0
    ltcg_taxable: float = 0.0
    basic_exemption_used: float = 0.0
    carry_forward_speculative_loss: float = 0.0
    carry_forward_business_loss: float = 0.0
    carry_forward_st_capital_loss: float = 0.0
    carry_forward_lt_capital_loss: float = 0.0
    tax_speculative: float = 0.0
    tax_business: float = 0.0
    tax_stcg: float = 0.0
    tax_ltcg: float = 0.0
    surcharge: float = 0.0
    cess: float = 0.0
    total_tax: float = 0.0
    stt_paid: float = 0.0
    other_charges: float = 0.0
    speculative_turnover: float = 0.0   # sum of |P&L| of intraday trades (ICAI guidance)
    net_profit_before_tax: float = 0.0
    net_profit_after_tax: float = 0.0
    effective_tax_pct: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(lines: list[TradeTaxLine], cfg: TaxSettings, fy_date: date) -> TaxSummary:
    """FY tax estimate with the set-off rules described in the module docstring."""
    start, end, label = fy_bounds(fy_date)
    rows = [ln for ln in lines if start <= ln.exit_date <= end]
    s = TaxSummary(fy=label, start=start, end=end, n_trades=len(rows))
    for ln in rows:
        s.stt_paid += ln.stt
        s.other_charges += ln.charges - ln.stt
        s.net_profit_before_tax += ln.gross - ln.charges
        if ln.head == TaxHead.SPECULATIVE:
            s.speculative_net += ln.taxable
            s.speculative_turnover += abs(ln.gross)
        elif ln.head == TaxHead.BUSINESS:
            s.business_net += ln.taxable
        elif ln.head == TaxHead.STCG:
            if ln.taxable >= 0:
                s.stcg_gains += ln.taxable
            else:
                s.stcg_losses += -ln.taxable
        else:
            if ln.taxable >= 0:
                s.ltcg_gains += ln.taxable
            else:
                s.ltcg_losses += -ln.taxable
    # capital gains set-off: STCL vs STCG then LTCG; LTCL only vs LTCG
    net_st = s.stcg_gains - s.stcg_losses
    net_lt = s.ltcg_gains - s.ltcg_losses
    if net_st < 0 and net_lt > 0:
        used = min(-net_st, net_lt)
        net_st += used
        net_lt -= used
        s.notes.append(f"Rs {used:,.0f} of short-term capital loss set off against LTCG.")
    s.carry_forward_st_capital_loss = max(0.0, -net_st)
    s.carry_forward_lt_capital_loss = max(0.0, -net_lt)
    stcg = max(0.0, net_st)
    ltcg = max(0.0, net_lt)
    s.ltcg_after_setoff = ltcg
    s.ltcg_exemption_used = min(ltcg, cfg.ltcg_exemption)
    s.ltcg_exemption_left = cfg.ltcg_exemption - s.ltcg_exemption_used
    ltcg_taxable = ltcg - s.ltcg_exemption_used
    # unused basic exemption (residents) absorbs STCG first (higher rate), then LTCG
    basic = cfg.basic_exemption_unused
    use = min(basic, stcg)
    stcg -= use
    basic -= use
    use2 = min(basic, ltcg_taxable)
    ltcg_taxable -= use2
    s.basic_exemption_used = use + use2
    s.stcg_taxable, s.ltcg_taxable = stcg, ltcg_taxable
    # speculative / business
    spec = s.speculative_net
    if spec < 0:
        s.carry_forward_speculative_loss = -spec
        spec = 0.0
    biz = s.business_net
    if biz < 0:
        s.carry_forward_business_loss = -biz
        biz = 0.0
    s.tax_speculative = spec * cfg.slab_rate_pct / 100.0
    s.tax_business = biz * cfg.slab_rate_pct / 100.0
    s.tax_stcg = stcg * cfg.stcg_rate_pct / 100.0
    s.tax_ltcg = ltcg_taxable * cfg.ltcg_rate_pct / 100.0
    base = s.tax_speculative + s.tax_business + s.tax_stcg + s.tax_ltcg
    s.surcharge = base * cfg.surcharge_pct / 100.0
    s.cess = (base + s.surcharge) * cfg.cess_pct / 100.0
    s.total_tax = base + s.surcharge + s.cess
    s.net_profit_after_tax = s.net_profit_before_tax - s.total_tax
    if s.net_profit_before_tax > 0:
        s.effective_tax_pct = round(s.total_tax / s.net_profit_before_tax * 100.0, 2)
    for attr in ("speculative_net", "business_net", "stcg_gains", "stcg_losses", "ltcg_gains", "ltcg_losses",
                 "stcg_taxable", "ltcg_after_setoff", "ltcg_exemption_used", "ltcg_exemption_left", "ltcg_taxable",
                 "basic_exemption_used", "carry_forward_speculative_loss", "carry_forward_business_loss",
                 "carry_forward_st_capital_loss", "carry_forward_lt_capital_loss", "tax_speculative",
                 "tax_business", "tax_stcg", "tax_ltcg", "surcharge", "cess", "total_tax", "stt_paid",
                 "other_charges", "speculative_turnover", "net_profit_before_tax", "net_profit_after_tax"):
        setattr(s, attr, round(getattr(s, attr), 2))
    return s


# =========================================================================== before a trade


@dataclass
class Scenario:
    label: str
    exit_price: float
    gross: float
    charges: float
    stt: float
    taxable: float
    tax: float
    net_after_tax: float


@dataclass
class PreTradeTax:
    head: TaxHead
    rate_pct: float
    scenarios: list[Scenario]
    breakeven_move_pct: float          # price move needed just to cover round-trip charges
    post_tax_rr: float | None          # after-tax profit at T1 / net loss at SL
    charges_pct_of_value: float
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["head"] = self.head.value
        return d


def pre_trade_estimate(
    mode: Mode,
    direction: Direction,
    qty: int,
    entry: float,
    stop_loss: float,
    target_1: float,
    target_2: float | None,
    charges_cfg: ChargesSettings,
    cfg: TaxSettings,
) -> PreTradeTax:
    """Charges + tax + net for SL / T1 / T2 exits of a planned trade (standalone, marginal rate)."""
    mode = Mode(mode)
    direction = Direction(direction)
    product = Product.MIS if mode == Mode.INTRADAY else Product.CNC
    head = head_for_mode(mode, cfg)
    rate = effective_rate_pct(head, cfg)
    sign = 1.0 if direction == Direction.LONG else -1.0
    scenarios: list[Scenario] = []
    for label, px in (("Stop-loss", stop_loss), ("Target 1", target_1), ("Target 2", target_2)):
        if px is None or qty < 1:
            continue
        ch = round_trip_charges(product, direction, qty, entry, px, charges_cfg)
        gross = sign * (px - entry) * qty
        taxable = gross - deductible_charges(head, ch.total, ch.stt)
        tax = max(0.0, taxable) * rate / 100.0
        scenarios.append(Scenario(label, round(px, 2), round(gross, 2), round(ch.total, 2), round(ch.stt, 2),
                                  round(taxable, 2), round(tax, 2), round(gross - ch.total - tax, 2)))
    value = qty * entry
    rt = round_trip_charges(product, direction, max(qty, 1), entry, entry, charges_cfg).total
    breakeven = rt / value * 100.0 if value > 0 else float("inf")
    sl = next((x for x in scenarios if x.label == "Stop-loss"), None)
    t1 = next((x for x in scenarios if x.label == "Target 1"), None)
    post_rr = (t1.net_after_tax / abs(sl.net_after_tax)) if (sl and t1 and sl.net_after_tax < 0) else None
    notes: list[str] = []
    if head == TaxHead.SPECULATIVE:
        notes.append("Intraday = speculative income at your slab rate; a loss only offsets intraday profits "
                     "(carry forward 4 years).")
    elif head == TaxHead.STCG:
        notes.append(f"Delivery held <= 12 months = STCG {cfg.stcg_rate_pct:g}% (+cess). STT is not deductible.")
    if sl and sl.net_after_tax < 0:
        saving = abs(sl.taxable) * rate / 100.0 if sl.taxable < 0 else 0.0
        if saving > 0:
            notes.append(f"If stopped out, the loss can save about Rs {saving:,.2f} of tax when set off "
                         "against same-head gains.")
    return PreTradeTax(head, round(rate, 3), scenarios, round(breakeven, 3),
                       round(post_rr, 2) if post_rr is not None else None,
                       round(rt / value * 100.0, 3) if value > 0 else float("inf"), notes)


# =========================================================================== tips


@dataclass
class TaxTip:
    severity: str          # info | saving | warning
    title: str
    detail: str
    est_saving: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def next_advance_tax_due(today: date) -> tuple[date, int]:
    """Next advance-tax instalment date of the current FY and the cumulative % due by then."""
    y = fy_bounds(today)[0].year
    for d, pct in ((date(y, 6, 15), 15), (date(y, 9, 15), 45), (date(y, 12, 15), 75), (date(y + 1, 3, 15), 100)):
        if d >= today:
            return d, pct
    return date(y + 1, 6, 15), 15


def tax_tips(summary: TaxSummary, open_positions: Iterable[Any], today: date, cfg: TaxSettings,
             charges_cfg: ChargesSettings) -> list[TaxTip]:
    """Deterministic, explainable tips to reduce tax and charges."""
    tips: list[TaxTip] = []
    st_rate = effective_rate_pct(TaxHead.STCG, cfg)
    spec_rate = effective_rate_pct(TaxHead.SPECULATIVE, cfg)
    fy_end = summary.end
    days_left = (fy_end - today).days
    # 1. tax-loss harvesting on open delivery positions
    for p in open_positions:
        ltp = getattr(p, "last_price", None) or getattr(p, "ltp", None)
        if ltp is None or getattr(p, "product", "CNC") != Product.CNC.value:
            continue
        sign = 1.0 if getattr(p, "direction", "LONG") == Direction.LONG.value else -1.0
        unreal = sign * (ltp - p.avg_entry) * p.qty
        if unreal < 0 and (summary.stcg_taxable > 0 or summary.ltcg_taxable > 0):
            offset = min(-unreal, summary.stcg_taxable + summary.ltcg_taxable)
            saving = offset * st_rate / 100.0
            urgency = " Do it before 31 March to count in this FY." if days_left <= 45 else ""
            tips.append(TaxTip("saving", f"Tax-loss harvesting: {p.symbol}",
                               f"{p.symbol} has an unrealized loss of Rs {-unreal:,.2f}. If you were going to exit "
                               f"anyway, booking it this FY offsets Rs {offset:,.2f} of capital gains.{urgency}",
                               round(saving, 2)))
        held = (today - p.opened_at.date()).days if getattr(p, "opened_at", None) else 0
        if unreal > 0 and 300 <= held < cfg.ltcg_min_holding_days:
            diff = unreal * (st_rate - effective_rate_pct(TaxHead.LTCG, cfg)) / 100.0
            tips.append(TaxTip("saving", f"Near long-term: {p.symbol}",
                               f"Held {held} days; after {cfg.ltcg_min_holding_days} days the gain becomes LTCG "
                               f"({cfg.ltcg_rate_pct:g}% above Rs {cfg.ltcg_exemption:,.0f}) instead of STCG "
                               f"({cfg.stcg_rate_pct:g}%).", round(diff, 2)))
    # 2. speculative losses
    if summary.carry_forward_speculative_loss > 0:
        tips.append(TaxTip("warning", "Intraday loss can only offset intraday profit",
                           f"Rs {summary.carry_forward_speculative_loss:,.2f} speculative loss this FY cannot reduce "
                           "capital gains or salary. File ITR-3 before the due date to carry it forward 4 years."))
    if summary.carry_forward_st_capital_loss > 0 or summary.carry_forward_lt_capital_loss > 0:
        tips.append(TaxTip("info", "Carry forward capital losses",
                           f"Short-term loss Rs {summary.carry_forward_st_capital_loss:,.2f} / long-term loss "
                           f"Rs {summary.carry_forward_lt_capital_loss:,.2f} can be carried forward 8 years - only "
                           "if the return is filed on time."))
    # 3. LTCG exemption
    if summary.ltcg_exemption_left > 0 and summary.ltcg_after_setoff > 0:
        tips.append(TaxTip("info", "LTCG exemption remaining",
                           f"Rs {summary.ltcg_exemption_left:,.0f} of the Rs {cfg.ltcg_exemption:,.0f} LTCG exemption "
                           "is unused this FY; long-term gains up to that amount are tax-free."))
    # 4. advance tax
    if summary.total_tax >= ADVANCE_TAX_THRESHOLD:
        due, pct = next_advance_tax_due(today)
        tips.append(TaxTip("warning", "Advance tax",
                           f"Estimated liability Rs {summary.total_tax:,.0f} >= Rs {ADVANCE_TAX_THRESHOLD:,.0f}: pay "
                           f"{pct}% cumulatively by {due:%d %b %Y} to avoid interest (s.234B/C). Capital gains made "
                           "after an instalment date can be paid in the remaining instalments."))
    # 5. charges hygiene
    if summary.stt_paid > 0 and summary.stcg_gains + summary.ltcg_gains > 0 and not cfg.delivery_as_business:
        tips.append(TaxTip("info", "STT is not deductible from capital gains",
                           f"Rs {summary.stt_paid:,.2f} STT paid this FY. If you trade delivery very frequently, ask "
                           "a CA whether business-income treatment (STT and all charges deductible) fits you."))
    dp = charges_cfg.cnc.dp_charge_per_scrip * (1 + charges_cfg.cnc.gst_pct / 100.0)
    if dp > 0:
        tips.append(TaxTip("info", "Cut delivery DP charges",
                           f"Each delivery sell day costs ~Rs {dp:.2f} per scrip regardless of size. Sell the full "
                           "quantity in one go, avoid tiny delivery positions, or prefer brokers with lower DP fees."))
    if summary.speculative_turnover > 0:
        tips.append(TaxTip("info", "Intraday turnover & ITR form",
                           f"Speculative turnover (sum of |P&L|) Rs {summary.speculative_turnover:,.2f}. Intraday "
                           f"income needs ITR-3; it is taxed at your slab ({cfg.slab_rate_pct:g}%, "
                           f"~{spec_rate:.1f}% with cess). Keep contract notes and the broker's tax P&L."))
    if cfg.basic_exemption_unused == 0:
        tips.append(TaxTip("info", "Unused basic exemption",
                           "Residents whose other income is below the basic exemption limit can absorb STCG/LTCG "
                           "with the unused part - set tax.basic_exemption_unused to include it in estimates."))
    tips.append(TaxTip("info", "Disclaimer", DISCLAIMER))
    return tips


def fy_trades(trades: Iterable[Any], d: date) -> list[Any]:
    start, end, _ = fy_bounds(d)
    return [t for t in trades if start <= t.exit_time.date() <= end]


__all__ = [
    "DISCLAIMER", "PreTradeTax", "Scenario", "TaxSummary", "TaxTip", "TradeTaxLine", "classify",
    "effective_rate_pct", "fy_bounds", "fy_trades", "head_for_mode", "next_advance_tax_due",
    "pre_trade_estimate", "summarize", "tax_tips", "trade_tax_lines", "timedelta",
]
