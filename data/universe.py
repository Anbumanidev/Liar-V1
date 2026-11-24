"""Stock universe per mode + liquidity / price / spread filters (section 7.5).

Static constituent lists live in ``data/static/<name>.csv`` with columns
``symbol,company,sector`` (lines starting with ``#`` are comments). Index membership changes
semi-annually: ``refresh_from_web()`` re-downloads the niftyindices.com lists.

Every exclusion carries a reason string, e.g. ``("ZOMATO", "BLACKLISTED")`` or
``("MRF", "UNAFFORDABLE: price 140000.00 > max affordable 600.00 (cannot size >= 1 share)")``.
"""

from __future__ import annotations

import csv
import io
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.enums import Mode, Timeframe
from core.logging import log_event

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings
    from data.service import MarketDataService

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: universe name -> static file
UNIVERSE_FILES: dict[str, str] = {
    "NIFTY50": "nifty50.csv",
    "NIFTY100": "nifty100.csv",
    "NIFTY200": "nifty200.csv",
    "FNO": "fno.csv",
}

NIFTYINDICES_URLS: dict[str, str] = {
    "NIFTY50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "NIFTY100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "NIFTY200": "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
}

CRORE = 1e7
TURNOVER_LOOKBACK = 20


def _norm_sym(s: str) -> str:
    return s.strip().upper()


class UniverseService:
    """Static universes, symbol metadata, universe building and filtering.

    Args:
        static_dir: folder with the universe CSVs.
        cache_dir: optional cache root (used by ``refresh_from_web`` when not writing static).
    """

    def __init__(self, static_dir: Path = STATIC_DIR, cache_dir: Path | None = None) -> None:
        self.static_dir = Path(static_dir)
        self.cache_dir = cache_dir
        self._lists: dict[str, list[str]] = {}
        self._meta: dict[str, dict[str, str]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    @staticmethod
    def parse_csv(text: str) -> list[dict[str, str]]:
        """Parse a ``symbol,company,sector`` CSV (``#`` comment lines ignored)."""
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        rows: list[dict[str, str]] = []
        for row in csv.DictReader(io.StringIO("\n".join(lines))):
            norm = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
            if norm.get("symbol"):
                rows.append(norm)
        return rows

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            # NIFTY200 first so the broadest list provides metadata; others fill gaps.
            for name in ("NIFTY200", "NIFTY100", "NIFTY50", "FNO"):
                path = self.static_dir / UNIVERSE_FILES[name]
                try:
                    rows = self.parse_csv(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    logger.warning(f"[universe] cannot read {path}: {exc}")
                    rows = []
                syms: list[str] = []
                for r in rows:
                    sym = _norm_sym(r["symbol"])
                    syms.append(sym)
                    meta = self._meta.setdefault(sym, {})
                    if r.get("company") and not meta.get("company"):
                        meta["company"] = r["company"]
                    if r.get("sector") and not meta.get("sector"):
                        meta["sector"] = r["sector"]
                self._lists[name] = sorted(dict.fromkeys(syms))
            self._loaded = True

    def reload(self) -> None:
        """Forget cached lists (after the CSVs changed)."""
        with self._lock:
            self._lists.clear()
            self._meta.clear()
            self._loaded = False

    # ------------------------------------------------------------------ lookups
    def universe_names(self) -> list[str]:
        """Available universe names (plus CUSTOM)."""
        return [*UNIVERSE_FILES, "CUSTOM"]

    def symbols_for(self, name: str) -> list[str]:
        """Constituents of ``NIFTY50`` / ``NIFTY100`` / ``NIFTY200`` / ``FNO`` (``CUSTOM`` -> [])."""
        self._load()
        key = name.strip().upper().replace(" ", "")
        if key == "CUSTOM":
            return []
        if key not in self._lists:
            logger.warning(f"[universe] unknown universe {name!r}")
            return []
        return list(self._lists[key])

    def all_symbols(self) -> list[str]:
        """Every symbol known from any static list."""
        self._load()
        return sorted(self._meta)

    def sector_of(self, symbol: str) -> str | None:
        """Sector/industry of ``symbol`` (None if unknown)."""
        self._load()
        return self._meta.get(_norm_sym(symbol), {}).get("sector") or None

    def company_name(self, symbol: str) -> str | None:
        """Company name of ``symbol`` (None if unknown)."""
        self._load()
        return self._meta.get(_norm_sym(symbol), {}).get("company") or None

    # ------------------------------------------------------------------ build
    def build_universe_detailed(self, mode: Mode, settings: "Settings") -> tuple[list[str], list[tuple[str, str]]]:
        """Like ``build_universe`` but also returns ``(symbol, reason)`` exclusions."""
        u = settings.universe
        cfg = settings.mode(mode)
        whitelist = [_norm_sym(s) for s in u.whitelist]
        if u.whitelist_only:
            base = list(whitelist)
        else:
            source = cfg.universe.strip().upper()
            base = self.symbols_for(source) if source != "CUSTOM" else []
            base += [_norm_sym(s) for s in u.custom_symbols] + whitelist
        candidates = list(dict.fromkeys(base))
        black = {_norm_sym(s) for s in u.blacklist}
        excluded_sectors = {s.strip().lower() for s in u.excluded_sectors if s.strip()}
        included: list[str] = []
        exclusions: list[tuple[str, str]] = []
        for sym in candidates:
            if sym in black:
                exclusions.append((sym, "BLACKLISTED"))
                continue
            sector = self.sector_of(sym)
            if sector and sector.lower() in excluded_sectors:
                exclusions.append((sym, f"EXCLUDED_SECTOR: {sector}"))
                continue
            included.append(sym)
        return sorted(included), exclusions

    def build_universe(self, mode: Mode, settings: "Settings") -> list[str]:
        """Universe source of the mode + custom + whitelist (or whitelist only) - blacklist - excluded sectors."""
        return self.build_universe_detailed(mode, settings)[0]

    # ------------------------------------------------------------------ filter
    def filter_universe(
        self,
        symbols: Iterable[str],
        data_service: "MarketDataService",
        settings: "Settings",
        mode: Mode,
        max_affordable_price: float | None = None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """Apply price / affordability / liquidity / spread filters.

        Args:
            symbols: candidate symbols (typically ``build_universe``).
            data_service: source of closed daily candles and quotes.
            settings: ``settings.universe`` thresholds.
            mode: for logging.
            max_affordable_price: highest price the risk manager can size to >= 1 share
                (computed by the caller); ``None`` disables the check.

        Returns:
            ``(included, [(symbol, reason), ...])``.
        """
        u = settings.universe
        black = {_norm_sym(s) for s in u.blacklist}
        excluded_sectors = {s.strip().lower() for s in u.excluded_sectors if s.strip()}
        syms = list(dict.fromkeys(_norm_sym(s) for s in symbols))
        quotes: dict[str, Any] = {}
        try:
            quotes = data_service.quotes(syms) if syms else {}
        except Exception as exc:  # noqa: BLE001 - fall back to last daily close
            logger.warning(f"[universe] bulk quotes failed: {exc}")
        included: list[str] = []
        exclusions: list[tuple[str, str]] = []
        for sym in syms:
            reason = self._exclusion_reason(sym, data_service, u, quotes.get(sym), black,
                                            excluded_sectors, max_affordable_price)
            if reason:
                exclusions.append((sym, reason))
            else:
                included.append(sym)
        try:
            log_event("scans", {"event": "universe_filter", "mode": mode.value, "included": included,
                                "excluded": [{"symbol": s, "reason": r} for s, r in exclusions]},
                      clock=data_service.clock)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[universe] could not log filter result: {exc}")
        return included, exclusions

    def _exclusion_reason(
        self,
        sym: str,
        data_service: "MarketDataService",
        u: Any,
        quote: Any,
        black: set[str],
        excluded_sectors: set[str],
        max_affordable_price: float | None,
    ) -> str | None:
        if sym in black:
            return "BLACKLISTED"
        sector = self.sector_of(sym)
        if sector and sector.lower() in excluded_sectors:
            return f"EXCLUDED_SECTOR: {sector}"
        daily = data_service.candles(sym, Timeframe.D1, lookback_bars=TURNOVER_LOOKBACK)
        if daily is None or daily.empty:
            return "NO_DATA: no closed daily candles"
        price = float(quote.ltp) if quote is not None and quote.ltp else float(daily["close"].iloc[-1])
        if price < u.min_price:
            return f"PRICE_BELOW_MIN: {price:.2f} < {u.min_price:.2f}"
        if u.max_price is not None and price > u.max_price:
            return f"PRICE_ABOVE_MAX: {price:.2f} > {u.max_price:.2f}"
        if max_affordable_price is not None and price > max_affordable_price:
            return (f"UNAFFORDABLE: price {price:.2f} > max affordable {max_affordable_price:.2f} "
                    f"(cannot size >= 1 share)")
        recent = daily.tail(TURNOVER_LOOKBACK)
        turnover_cr = float((recent["close"] * recent["volume"]).mean()) / CRORE
        if turnover_cr < u.min_avg_daily_turnover_cr:
            return (f"LOW_LIQUIDITY: avg {len(recent)}-day turnover Rs {turnover_cr:.2f} cr < "
                    f"{u.min_avg_daily_turnover_cr:.2f} cr")
        spread = quote.spread_pct if quote is not None else None
        if spread is not None and spread > u.max_spread_pct:
            return f"WIDE_SPREAD: {spread:.3f}% > {u.max_spread_pct:.3f}%"
        return None

    # ------------------------------------------------------------------ refresh
    def refresh_from_web(self, dest_dir: Path | None = None, timeout: float = 20.0) -> dict[str, int]:
        """Re-download NIFTY 50/100/200 constituents from niftyindices.com (network, failure-safe).

        Writes ``symbol,company,sector`` CSVs into ``dest_dir`` (default: the static dir) and
        returns ``{universe: count}`` for the lists that were refreshed.
        """
        import requests

        dest = Path(dest_dir) if dest_dir else self.static_dir
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
        done: dict[str, int] = {}
        for name, url in NIFTYINDICES_URLS.items():
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                reader = csv.DictReader(io.StringIO(resp.content.decode("utf-8-sig")))
                rows = sorted({(r["Symbol"].strip().upper(), r["Company Name"].strip(), r["Industry"].strip())
                               for r in reader if r.get("Symbol")})
                if len(rows) < 40:
                    raise ValueError(f"only {len(rows)} rows")
                buf = io.StringIO()
                buf.write(f"# {name} constituents. Source: {url}. Refreshed by UniverseService.refresh_from_web - verify.\n")
                w = csv.writer(buf, lineterminator="\n")
                w.writerow(["symbol", "company", "sector"])
                w.writerows(rows)
                dest.mkdir(parents=True, exist_ok=True)
                (dest / UNIVERSE_FILES[name]).write_text(buf.getvalue(), encoding="utf-8")
                done[name] = len(rows)
            except Exception as exc:  # noqa: BLE001 - keep the existing file
                logger.warning(f"[universe] refresh of {name} failed, keeping existing list: {exc}")
        if done and dest == self.static_dir:
            self.reload()
        return done
