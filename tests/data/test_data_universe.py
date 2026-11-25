"""UniverseService: static lists, metadata, build_universe and filter_universe (every exclusion has a reason)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from config.schema import Settings, UniverseSettings
from core.clock import SimulatedClock
from core.enums import Mode, Timeframe
from core.logging import read_events
from core.types import Quote, empty_ohlcv
from data.universe import UniverseService
from tests.helpers import ist, make_daily_candles

CSV_50 = """# test list - verify
symbol,company,sector
AAA,Alpha Tech Ltd.,Information Technology
BBB,Beta Bank Ltd.,Financial Services
CCC,Gamma Tobacco Ltd.,Fast Moving Consumer Goods
DDD,Delta Power Ltd.,Power
"""
CSV_100 = CSV_50 + "EEE,Epsilon Metals Ltd.,Metals & Mining\n"


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    (tmp_path / "nifty50.csv").write_text(CSV_50, encoding="utf-8")
    (tmp_path / "nifty100.csv").write_text(CSV_100, encoding="utf-8")
    return tmp_path                                     # nifty200.csv / fno.csv deliberately missing


def _with_universe(settings: Settings, **kw: Any) -> Settings:
    return settings.model_copy(update={"universe": UniverseSettings(**kw)})


def _with_mode_universe(settings: Settings, mode: Mode, universe: str) -> Settings:
    modes = dict(settings.modes.modes)
    modes[mode] = modes[mode].model_copy(update={"universe": universe})
    return settings.model_copy(update={"modes": settings.modes.model_copy(update={"modes": modes})})


# ------------------------------------------------------------------ static data


def test_static_lists_and_metadata() -> None:
    u = UniverseService()
    n50, n100, n200 = (set(u.symbols_for(n)) for n in ("NIFTY50", "NIFTY100", "NIFTY200"))
    assert len(n50) == 50 and len(n100) == 100 and len(n200) == 200
    assert n50 <= n100 <= n200
    assert {"RELIANCE", "TCS", "HDFCBANK", "INFY"} <= n50
    assert "RELIANCE" in set(u.symbols_for("fno"))
    assert u.symbols_for("CUSTOM") == [] and u.symbols_for("NIFTY999") == []
    assert u.company_name("reliance") == "Reliance Industries Ltd."
    assert u.sector_of("RELIANCE") == "Oil Gas & Consumable Fuels"
    assert u.company_name("NOPE") is None and u.sector_of("NOPE") is None
    assert "CUSTOM" in u.universe_names()


def test_parse_csv_skips_comments_and_blank_lines() -> None:
    rows = UniverseService.parse_csv("# c\n\nsymbol,company,sector\n abc ,A Ltd., IT \n,No symbol,X\n")
    assert rows == [{"symbol": "abc", "company": "A Ltd.", "sector": "IT"}]


def test_missing_files_are_tolerated(static_dir: Path) -> None:
    u = UniverseService(static_dir)
    assert u.symbols_for("NIFTY200") == [] and u.symbols_for("FNO") == []
    assert u.symbols_for("NIFTY 50") == ["AAA", "BBB", "CCC", "DDD"]
    assert u.all_symbols() == ["AAA", "BBB", "CCC", "DDD", "EEE"]


# ------------------------------------------------------------------ build


def test_build_universe_default_and_mode_source(static_dir: Path, settings: Settings) -> None:
    u = UniverseService(static_dir)
    s = _with_universe(settings)
    assert u.build_universe(Mode.INTRADAY, s) == ["AAA", "BBB", "CCC", "DDD"]          # NIFTY50
    assert u.build_universe(Mode.SWING, s) == ["AAA", "BBB", "CCC", "DDD", "EEE"]      # NIFTY100


def test_build_universe_blacklist_sectors_custom_whitelist(static_dir: Path, settings: Settings) -> None:
    u = UniverseService(static_dir)
    s = _with_universe(settings, blacklist=["bbb"], excluded_sectors=["fast moving consumer goods"],
                       custom_symbols=["zzz"], whitelist=["YYY"])
    included, excluded = u.build_universe_detailed(Mode.INTRADAY, s)
    assert included == ["AAA", "DDD", "YYY", "ZZZ"]
    assert dict(excluded) == {"BBB": "BLACKLISTED", "CCC": "EXCLUDED_SECTOR: Fast Moving Consumer Goods"}


def test_build_universe_whitelist_only_and_custom_mode(static_dir: Path, settings: Settings) -> None:
    u = UniverseService(static_dir)
    s = _with_universe(settings, whitelist_only=True, whitelist=["AAA", "EEE", "BBB"], blacklist=["BBB"],
                       custom_symbols=["ZZZ"])
    assert u.build_universe(Mode.INTRADAY, s) == ["AAA", "EEE"]                        # blacklist still wins
    custom = _with_mode_universe(_with_universe(settings, custom_symbols=["ZZZ"], whitelist=["AAA"]),
                                 Mode.POSITIONAL, "CUSTOM")
    assert u.build_universe(Mode.POSITIONAL, custom) == ["AAA", "ZZZ"]


# ------------------------------------------------------------------ filter


class FakeData:
    """Minimal MarketDataService stand-in: closed daily candles + quotes."""

    def __init__(self, daily: dict[str, pd.DataFrame], quotes: dict[str, Quote]) -> None:
        self.clock = SimulatedClock(ist(2026, 11, 23, 9, 0))
        self.daily, self.quote_map = daily, quotes
        self.candle_calls: list[tuple[str, Timeframe]] = []

    def candles(self, symbol: str, timeframe: Timeframe, lookback_bars: int = 300,
                start: datetime | None = None) -> pd.DataFrame:
        self.candle_calls.append((symbol, timeframe))
        return self.daily.get(symbol, empty_ohlcv()).tail(lookback_bars)

    def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        return {s: self.quote_map[s] for s in symbols if s in self.quote_map}


def _daily(price: float, volume: float, seed: int) -> pd.DataFrame:
    return make_daily_candles(date(2026, 11, 20), 25, start_price=price, seed=seed, drift=0.0, vol=1e-5,
                              base_volume=volume)


def _quote(sym: str, ltp: float, spread_pct: float = 0.05) -> Quote:
    half = ltp * spread_pct / 200
    return Quote(symbol=sym, ltp=ltp, timestamp=ist(2026, 11, 23, 9, 0), bid=ltp - half, ask=ltp + half)


def test_filter_universe_reasons(static_dir: Path, settings: Settings) -> None:
    u = UniverseService(static_dir)
    s = _with_universe(settings, blacklist=["BBB"], excluded_sectors=["Power"], min_price=20.0, max_price=5000.0,
                       min_avg_daily_turnover_cr=5.0, max_spread_pct=0.3)
    daily = {"AAA": _daily(500, 200_000, 1), "BBB": _daily(500, 200_000, 2), "DDD": _daily(500, 200_000, 3),
             "PENNY": _daily(10, 50_000_000, 4), "PRICEY": _daily(6000, 100_000, 5),
             "MRF": _daily(3000, 100_000, 6), "ILLIQ": _daily(100, 1_000, 7), "WIDE": _daily(200, 1_000_000, 8),
             "NOQUOTE": _daily(300, 500_000, 9)}
    quotes = {"AAA": _quote("AAA", 500), "PENNY": _quote("PENNY", 10), "PRICEY": _quote("PRICEY", 6000),
              "MRF": _quote("MRF", 3000), "ILLIQ": _quote("ILLIQ", 100), "WIDE": _quote("WIDE", 200, 2.0)}
    data = FakeData(daily, quotes)
    symbols = ["AAA", "BBB", "DDD", "PENNY", "PRICEY", "MRF", "ILLIQ", "WIDE", "NODATA", "noquote"]
    included, excluded = u.filter_universe(symbols, data, s, Mode.INTRADAY,  # type: ignore[arg-type]
                                           max_affordable_price=1000.0)
    reasons = dict(excluded)
    assert included == ["AAA", "NOQUOTE"]                     # NOQUOTE priced from its last daily close
    assert reasons["BBB"] == "BLACKLISTED"
    assert reasons["DDD"] == "EXCLUDED_SECTOR: Power"
    assert reasons["PENNY"].startswith("PRICE_BELOW_MIN")
    assert reasons["PRICEY"].startswith("PRICE_ABOVE_MAX")
    assert reasons["MRF"].startswith("UNAFFORDABLE") and "1000.00" in reasons["MRF"]
    assert reasons["ILLIQ"].startswith("LOW_LIQUIDITY")
    assert reasons["WIDE"].startswith("WIDE_SPREAD")
    assert reasons["NODATA"].startswith("NO_DATA")
    assert sorted(included + list(reasons)) == sorted(s.upper() for s in symbols)   # every symbol accounted for
    assert all(r for r in reasons.values())
    assert ("BBB", Timeframe.D1) not in data.candle_calls     # cheap rules first: no data fetched
    logged = read_events(date(2026, 11, 23), "scans")
    assert logged and logged[-1]["event"] == "universe_filter"
    assert {e["symbol"] for e in logged[-1]["excluded"]} == set(reasons)


def test_filter_universe_without_affordability_limit(static_dir: Path, settings: Settings) -> None:
    u = UniverseService(static_dir)
    s = _with_universe(settings, min_avg_daily_turnover_cr=0.0)
    data = FakeData({"MRF": _daily(3000, 100_000, 6)}, {})
    included, excluded = u.filter_universe(["MRF"], data, s, Mode.SWING)  # type: ignore[arg-type]
    assert included == ["MRF"] and excluded == []
