"""REPLAY mode: historical trading days minute-by-minute (sections 9, 17.9).

The SAME engine code runs against a ``SimulatedClock`` and a ``ReplayDataProvider`` that only
serves bars closed at the simulated time (no look-ahead). Each replay uses its own SQLite file
(``runtime/replay/<name>.db``) so the live paper account is never touched. Overnight holds
(BTST / SWING / POSITIONAL) are carried across days, gap fills happen at the next open.

Replay specifics:
* No human is present: MANUAL_APPROVE is treated as AUTO (configurable via
  ``ReplayConfig.execution_policy``).
* BTST T+1 short-delivery risk is simulated (replay only): with probability
  ``settings.replay.btst_short_delivery_probability`` a next-day BTST sale is closed out in the
  auction at ``SHORT_DELIVERY_PENALTY_PCT`` worse than the market.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import func, select

from config.loader import load_default_settings, update_settings
from config.schema import Settings
from core.clock import IST, SimulatedClock
from core.enums import ExecutionPolicy, ExitReason, Mode, Timeframe
from core.logging import log_event

SHORT_DELIVERY_PENALTY_PCT = 5.0
INDEX_SYMBOLS = ("NIFTY50", "BANKNIFTY", "INDIAVIX")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ReplayConfig:
    start: date
    end: date
    modes: list[Mode]
    symbols: list[str] | None = None
    name: str = "replay"
    llm_backend: str = "rules"
    execution_policy: ExecutionPolicy = ExecutionPolicy.AUTO
    starting_capital: float | None = None
    settings_changes: dict[str, Any] = field(default_factory=dict)
    db_url: str | None = None
    overwrite: bool = True
    max_symbols: int = 15
    step_minutes: int | None = None


@dataclass
class ReplayReport:
    name: str
    start: str
    end: str
    days: int
    trades: int
    net_pnl: float
    gross_pnl: float
    charges: float
    win_rate: float | None
    expectancy: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    final_equity: float | None
    by_mode: dict[str, dict[str, Any]]
    ledger_ok: bool
    short_delivery_events: int
    db_url: str
    settings_changes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay_settings(base: Settings, cfg: ReplayConfig) -> Settings:
    """Settings for a replay: only the requested modes enabled, policy, backend, overrides."""
    changes: dict[str, Any] = {"engine.llm_backend": cfg.llm_backend, "modes.autonomy": "PER_MODE"}
    for m in Mode:
        changes[f"modes.modes.{m.value}.enabled"] = m in cfg.modes
        changes[f"modes.modes.{m.value}.execution_policy"] = cfg.execution_policy.value
    if cfg.starting_capital:
        changes["capital.starting_capital"] = float(cfg.starting_capital)
    if cfg.step_minutes:
        changes["replay.step_minutes"] = int(cfg.step_minutes)
    # capital allocation: keep the configured split unless a requested mode has 0 % -> equal split
    if any(base.mode(m).allocation_pct <= 0 for m in cfg.modes):
        share = round(100.0 / max(1, len(cfg.modes)), 4)
        for m in Mode:
            changes[f"modes.modes.{m.value}.allocation_pct"] = share if m in cfg.modes else 0.0
    s = update_settings(base, changes)
    if cfg.settings_changes:
        s = update_settings(s, cfg.settings_changes)
    return s


def required_timeframes(settings: Settings, modes: list[Mode]) -> list[Timeframe]:
    tfs: set[Timeframe] = {Timeframe.D1}
    for m in modes:
        tfs |= set(settings.mode(m).timeframes)
        if m in (Mode.INTRADAY, Mode.BTST):
            tfs |= {Timeframe.M5}
    tfs.discard(Timeframe.W1)   # resampled from daily by the data layer
    return sorted(tfs, key=lambda t: ["1m", "5m", "15m", "1h", "1d", "1wk"].index(t.value))


def preload_frames(symbols: list[str], timeframes: list[Timeframe], start: date, end: date,
                   provider: Any | None = None) -> dict[tuple[str, Timeframe], pd.DataFrame]:
    """Download history once (live provider, default yfinance) for every symbol/timeframe."""
    from data.calendar import load_calendar
    from data.providers.replay_provider import ReplayDataProvider

    if provider is None:
        from data.providers.yfinance_provider import YFinanceProvider

        provider = YFinanceProvider(clock=None)
    from features.market import DEFAULT_GLOBAL_CUES, DEFAULT_SECTOR_INDICES

    t0 = datetime.combine(start - timedelta(days=420), time(9, 15), tzinfo=IST)
    t1 = datetime.combine(end + timedelta(days=1), time(15, 30), tzinfo=IST)
    clock = SimulatedClock(t1 + timedelta(days=1))
    cal = load_calendar()
    main = list(dict.fromkeys(list(symbols) + list(INDEX_SYMBOLS)))
    # market-context symbols (sector indices, global cues) only need daily bars
    extras = [s for s in list(DEFAULT_SECTOR_INDICES.values()) + list(DEFAULT_GLOBAL_CUES) if s not in main]
    frames: dict[tuple[str, Timeframe], pd.DataFrame] = {}
    for syms, tfs in ((main, timeframes), (extras, [Timeframe.D1])):
        rp = ReplayDataProvider.from_provider(provider, syms, tfs, t0, t1, clock, cal)
        for sym in rp.symbols():
            for tf in rp.timeframes_for(sym):
                df = rp._frames.get((sym, tf))  # noqa: SLF001 - one-off export of the preloaded history
                if df is not None and not df.empty:
                    frames[(sym, tf)] = df
    return frames


class ReplayRunner:
    """Runs the engine's daily plan over historical days with a simulated clock."""

    def __init__(self, cfg: ReplayConfig, *, frames: dict[tuple[str, Timeframe], pd.DataFrame] | None = None,
                 base_settings: Settings | None = None, provider: Any | None = None) -> None:
        self.cfg = cfg
        self.frames = frames
        self.base_settings = base_settings
        self.provider = provider
        self.short_delivery_events = 0

    def _db_url(self) -> str:
        if self.cfg.db_url:
            return self.cfg.db_url
        path = PROJECT_ROOT / "runtime" / "replay" / f"{self.cfg.name}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.cfg.overwrite:
            for suffix in ("", "-wal", "-shm"):
                p = Path(str(path) + suffix)
                if p.exists():
                    p.unlink()
        return f"sqlite:///{path.as_posix()}"

    def _symbols(self, settings: Settings) -> list[str]:
        if self.cfg.symbols:
            return [s.strip().upper() for s in self.cfg.symbols][: self.cfg.max_symbols]
        from data.universe import UniverseService

        uni = UniverseService()
        out: list[str] = []
        for m in self.cfg.modes:
            for sym in uni.build_universe(m, settings):
                if sym not in out:
                    out.append(sym)
        return out[: self.cfg.max_symbols]

    def _install_short_delivery(self, svc: Any, rng: random.Random) -> None:
        """Wrap broker.exit_position: simulated BTST T+1 short delivery (replay only)."""
        prob = svc.settings.replay.btst_short_delivery_probability
        broker = svc.broker
        original = broker.exit_position
        runner = self

        def exit_with_short_delivery(session: Any, position: Any, qty: int | None = None,
                                     reason: ExitReason = ExitReason.MANUAL, **kw: Any) -> Any:
            if position.mode == Mode.BTST.value and position.holding_days <= 1 and prob > 0 and rng.random() < prob:
                price = kw.get("price")
                if price is None:
                    mp = broker.market_price(position.symbol)
                    price = mp[0] if mp else position.last_price
                if price:
                    kw["price"] = round(price * (1 - SHORT_DELIVERY_PENALTY_PCT / 100.0), 2)
                    kw["apply_slippage"] = False
                    runner.short_delivery_events += 1
                    log_event("replay", {"event": "btst_short_delivery", "symbol": position.symbol,
                                         "penalty_pct": SHORT_DELIVERY_PENALTY_PCT, "auction_price": kw["price"]},
                              clock=svc.clock)
            return original(session, position, qty, reason, **kw)

        broker.exit_position = exit_with_short_delivery  # type: ignore[method-assign]

    def run(self) -> ReplayReport:
        from control.bootstrap import bootstrap
        from data.calendar import load_calendar
        from data.providers.replay_provider import ReplayDataProvider
        from scheduler.jobs import run_job
        from scheduler.plan import DailyPlan

        cfg = self.cfg
        base = self.base_settings or load_default_settings()
        settings = replay_settings(base, cfg)
        calendar = load_calendar(settings.schedule)
        days = calendar.trading_days_between(cfg.start, cfg.end)
        if not days:
            raise ValueError(f"no NSE trading days between {cfg.start} and {cfg.end}")
        symbols = self._symbols(settings)
        frames = self.frames
        if frames is None:
            logger.info(f"[replay] downloading history for {len(symbols)} symbols ...")
            frames = preload_frames(symbols, required_timeframes(settings, cfg.modes), cfg.start, cfg.end,
                                    self.provider)
        clock = SimulatedClock(datetime.combine(days[0], time(8, 0), tzinfo=IST))
        provider = ReplayDataProvider(frames, clock, calendar)
        db_url = self._db_url()
        svc = bootstrap(process="replay", clock=clock, db_url=db_url, primary_provider=provider, is_replay=True,
                        llm_backend=cfg.llm_backend, with_notifier=False, offline=True, settings_override=settings,
                        console_log=False)
        if cfg.symbols or self.frames is not None:
            # restrict the universe to the replayed symbols
            wanted = set(symbols)
            orig_build = svc.universe.build_universe
            svc.universe.build_universe = lambda mode, st: [s for s in orig_build(mode, st) if s in wanted] or sorted(wanted)  # type: ignore[method-assign]
        rng = random.Random(settings.replay.seed)
        self._install_short_delivery(svc, rng)
        plan = DailyPlan(settings)
        step = timedelta(minutes=settings.replay.step_minutes)
        logger.info(f"[replay] {cfg.name}: {len(days)} days, modes {[m.value for m in cfg.modes]}, "
                    f"{len(symbols)} symbols, llm={cfg.llm_backend}")
        for d in days:
            t = datetime.combine(d, time(8, 25), tzinfo=IST)
            end_t = datetime.combine(d, time(16, 5), tzinfo=IST)
            clock.set(t)
            prev = t
            while t <= end_t:
                t = t + step
                clock.set(t)
                for job in plan.jobs_due(prev, t):
                    run_job(svc, job)
                if time(9, 15) < t.time() <= time(15, 31):
                    run_job(svc, "monitor")
                    if t.minute % 15 == 0:
                        run_job(svc, "shadow_update")
                prev = t
            logger.info(f"[replay] {d} done")
        report = self._report(svc, days, db_url)
        out = PROJECT_ROOT / "runtime" / "replay" / f"{cfg.name}_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
        log_event("replay", {"event": "finished", **report.as_dict()}, clock=clock)
        return report

    def _report(self, svc: Any, days: list[date], db_url: str) -> ReplayReport:
        from analytics.metrics import expectancy, max_drawdown, profit_factor, win_rate
        from storage.models import DailySummary, Trade

        with svc.session() as s:
            bid = svc.ctx.batch_id
            trades = s.execute(select(Trade).where(Trade.batch_id == bid)).scalars().all()
            nets = [t.net_pnl for t in trades]
            df = pd.DataFrame([{"net_pnl": t.net_pnl, "gross_pnl": t.gross_pnl, "charges": t.charges,
                                "mode": t.mode, "r_multiple": t.r_multiple} for t in trades])
            by_mode: dict[str, dict[str, Any]] = {}
            if not df.empty:
                for m, g in df.groupby("mode"):
                    by_mode[str(m)] = {"trades": int(len(g)), "net_pnl": round(float(g["net_pnl"].sum()), 2),
                                       "win_rate": _safe(win_rate, g["net_pnl"])}
            eq = s.execute(select(DailySummary.day, DailySummary.equity).where(
                DailySummary.batch_id == bid, DailySummary.mode == "TOTAL").order_by(DailySummary.day)).all()
            equity = pd.Series([e for _, e in eq if e is not None], dtype="float64")
            ok, _details = svc.capital.check_ledger_balance(s, bid)
            acct = svc.capital.account(s, bid)
            final_equity = acct.trading_cash + acct.blocked + acct.reserve
            mdd = None
            if len(equity) > 1:
                res = _safe(max_drawdown, equity)
                mdd = res[1] if isinstance(res, tuple) else res
        return ReplayReport(
            name=self.cfg.name, start=str(days[0]), end=str(days[-1]), days=len(days), trades=len(trades),
            net_pnl=round(sum(nets), 2), gross_pnl=round(float(df["gross_pnl"].sum()), 2) if not df.empty else 0.0,
            charges=round(float(df["charges"].sum()), 2) if not df.empty else 0.0,
            win_rate=_safe(win_rate, df["net_pnl"]) if not df.empty else None,
            expectancy=_safe(expectancy, df["net_pnl"]) if not df.empty else None,
            profit_factor=_safe(profit_factor, df["net_pnl"]) if not df.empty else None,
            max_drawdown_pct=mdd, final_equity=round(final_equity, 2), by_mode=by_mode, ledger_ok=ok,
            short_delivery_events=self.short_delivery_events, db_url=db_url,
            settings_changes=self.cfg.settings_changes,
        )


def _safe(fn: Any, arg: Any) -> Any:
    """Call an analytics metric tolerantly (DataFrame or Series inputs; None on failure)."""
    try:
        val = fn(arg)
    except Exception as exc:  # noqa: BLE001 - metrics are informational
        logger.debug(f"[replay] metric {getattr(fn, '__name__', fn)} failed: {exc}")
        return None
    if isinstance(val, float):
        return round(val, 4)
    return val


__all__ = ["ReplayConfig", "ReplayReport", "ReplayRunner", "preload_frames", "replay_settings",
           "required_timeframes", "func"]
