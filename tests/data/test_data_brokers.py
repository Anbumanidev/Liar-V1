"""Angel One / Upstox providers: unavailable without credentials, paper-only guards installed,
and (with fake SDK / HTTP clients) correct candle + quote normalisation. No network."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

import core.guard as guard
import data.providers.angelone_provider as angel_mod
import data.providers.upstox_provider as upstox_mod
from core.clock import SimulatedClock
from core.enums import Timeframe
from core.errors import DataUnavailableError
from core.types import validate_ohlcv
from data.providers.angelone_provider import AngelOneProvider, parse_instrument_master
from data.providers.upstox_provider import UpstoxProvider, parse_upstox_instruments
from tests.helpers import ist

CLOCK_NOW = ist(2026, 11, 23, 10, 0)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(CLOCK_NOW)


def _never_called(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("must not be called without credentials")


# ------------------------------------------------------------------ guards + missing credentials


@pytest.mark.parametrize("module, cls", [(angel_mod, AngelOneProvider), (upstox_mod, UpstoxProvider)])
def test_constructor_installs_order_guards(monkeypatch: pytest.MonkeyPatch, module: Any, cls: type) -> None:
    calls: list[int] = []
    monkeypatch.setattr(module, "install_order_guards", lambda: calls.append(1) or [])
    cls(None)
    assert calls, f"{cls.__name__} must install the paper-trading order guards"


def test_guards_are_really_active_after_construction(clock: SimulatedClock) -> None:
    AngelOneProvider(None, clock=clock)
    UpstoxProvider(None, clock=clock)
    assert guard.guards_installed()
    httpx = pytest.importorskip("httpx")
    assert getattr(httpx.HTTPTransport.handle_request, "__paper_guard__", False)
    with pytest.raises(Exception, match="PAPER TRADING ONLY"):
        guard.check_url("https://api.upstox.com/v2/order/place")


def test_angelone_unavailable_without_credentials(clock: SimulatedClock) -> None:
    for secrets in (None, SimpleNamespace(angel_api_key=SecretStr("k"), angel_client_code=None, angel_pin=None,
                                          angel_totp_secret=None)):
        p = AngelOneProvider(secrets, clock=clock, smart_factory=_never_called, instrument_loader=_never_called)
        assert not p.configured and p.healthcheck() is False
        with pytest.raises(DataUnavailableError):
            p.get_candles("RELIANCE", Timeframe.M5, CLOCK_NOW - timedelta(days=1), CLOCK_NOW)
        assert p.get_quote("RELIANCE") is None
        assert p.get_quotes(["RELIANCE", "TCS"]) == {}


def test_upstox_unavailable_without_token_and_no_network(clock: SimulatedClock) -> None:
    for secrets in (None, SimpleNamespace(upstox_access_token=SecretStr("  "))):
        p = UpstoxProvider(secrets, clock=clock, client=SimpleNamespace(get=_never_called),
                           instrument_loader=_never_called)
        assert not p.configured and p.healthcheck() is False
        with pytest.raises(DataUnavailableError):
            p.get_candles("RELIANCE", Timeframe.D1, CLOCK_NOW - timedelta(days=30), CLOCK_NOW)
        assert p.get_quote("RELIANCE") is None            # regression: no instrument-file download first


# ------------------------------------------------------------------ Angel One with a fake SmartConnect

ANGEL_SECRETS = SimpleNamespace(angel_api_key=SecretStr("key"), angel_client_code="C123",
                                angel_pin=SecretStr("1234"), angel_totp_secret=SecretStr("JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP"))
MASTER = [{"exch_seg": "NSE", "symbol": "RELIANCE-EQ", "token": "2885", "instrumenttype": ""},
          {"exch_seg": "NSE", "symbol": "Nifty 50", "token": "99926000", "instrumenttype": "AMXIDX"},
          {"exch_seg": "BSE", "symbol": "RELIANCE", "token": "500325", "instrumenttype": ""},
          {"exch_seg": "NSE", "symbol": "RELIANCE-BL", "token": "1", "instrumenttype": ""}]


class FakeSmart:
    def __init__(self, login_ok: bool = True) -> None:
        self.login_ok = login_ok
        self.candle_params: list[dict[str, Any]] = []

    def generateSession(self, client: str, pin: str, totp: str) -> dict[str, Any]:  # noqa: N802 - SDK name
        assert client == "C123" and pin == "1234" and len(totp) == 6
        return {"status": self.login_ok, "message": "ok" if self.login_ok else "Invalid totp"}

    def getCandleData(self, params: dict[str, Any]) -> dict[str, Any]:  # noqa: N802
        self.candle_params.append(params)
        rows = [["2026-11-23T09:15:00+05:30", 100, 102, 99, 101, 1000],
                ["2026-11-23T09:20:00+05:30", 101, 103, 100, 102, 1500]]
        return {"status": True, "data": rows}

    def ltpData(self, exchange: str, tsym: str, token: str) -> dict[str, Any]:  # noqa: N802
        return {"status": True, "data": {"ltp": 2500.5, "open": 2490, "high": 2510, "low": 2480, "close": 2470}}


def test_parse_instrument_master() -> None:
    assert parse_instrument_master(MASTER) == {"RELIANCE": ("RELIANCE-EQ", "2885"), "NIFTY50": ("Nifty 50", "99926000")}


def test_angelone_candles_quotes_and_token_cache(monkeypatch: pytest.MonkeyPatch, clock: SimulatedClock,
                                                 tmp_path: Path) -> None:
    monkeypatch.setattr(angel_mod, "_REQUEST_GAP_SECONDS", 0.0)
    smart = FakeSmart()
    p = AngelOneProvider(ANGEL_SECRETS, cache_dir=tmp_path, clock=clock, smart_factory=lambda key: smart,
                         instrument_loader=lambda: MASTER)
    assert p.configured and p.healthcheck()
    df = p.get_candles("reliance", Timeframe.M5, ist(2026, 11, 23, 9, 15), CLOCK_NOW)
    validate_ohlcv(df)
    assert list(df.index) == [ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 20)]
    assert df["close"].tolist() == [101.0, 102.0]
    assert smart.candle_params[0]["symboltoken"] == "2885" and smart.candle_params[0]["interval"] == "FIVE_MINUTE"
    assert (tmp_path / "angelone" / "tokens_20261123.json").exists()
    smart.candle_params.clear()
    p.get_candles("RELIANCE", Timeframe.D1, CLOCK_NOW - timedelta(days=3000), CLOCK_NOW)   # > 1900-day limit
    assert len(smart.candle_params) == 2 and smart.candle_params[0]["interval"] == "ONE_DAY"
    q = p.get_quote("RELIANCE")
    assert q is not None and q.ltp == 2500.5 and q.prev_close == 2470.0 and q.timestamp == CLOCK_NOW
    assert q.source == "angelone"
    with pytest.raises(DataUnavailableError):
        p.get_candles("UNKNOWN", Timeframe.M5, ist(2026, 11, 23, 9, 15), CLOCK_NOW)
    with pytest.raises(DataUnavailableError):
        p.get_candles("RELIANCE", Timeframe.W1, ist(2026, 11, 1), CLOCK_NOW)


def test_angelone_rejected_login_is_unavailable(clock: SimulatedClock) -> None:
    p = AngelOneProvider(ANGEL_SECRETS, clock=clock, smart_factory=lambda key: FakeSmart(login_ok=False),
                         instrument_loader=lambda: MASTER)
    assert p.healthcheck() is False
    assert p.last_error is not None and "Invalid totp" in p.last_error
    assert p.get_quote("RELIANCE") is None


# ------------------------------------------------------------------ Upstox with a fake HTTP client


class FakeResp:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any] | None, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> FakeResp:
        self.requests.append((url, params, headers or {}))
        if "/market-quote/ltp" in url:
            return FakeResp({"status": "success", "data": {"NSE_EQ:RELIANCE": {"last_price": 2501.0}}})
        if "/intraday/" in url:
            return FakeResp({"status": "success", "data": {"candles": [
                ["2026-11-23T09:20:00+05:30", 101, 103, 100, 102, 1500, 0]]}})
        return FakeResp({"status": "success", "data": {"candles": [
            ["2026-11-23T09:15:00+05:30", 100, 102, 99, 101, 1000, 0]]}})


UPSTOX_ROWS = [{"segment": "NSE_EQ", "trading_symbol": "RELIANCE", "instrument_key": "NSE_EQ|INE002A01018",
                "instrument_type": "EQ"},
               {"segment": "NSE_FO", "trading_symbol": "RELIANCE FUT", "instrument_key": "NSE_FO|1"}]


def test_parse_upstox_instruments() -> None:
    assert parse_upstox_instruments(UPSTOX_ROWS) == {"RELIANCE": "NSE_EQ|INE002A01018"}


def test_upstox_candles_include_today_and_quote(clock: SimulatedClock, tmp_path: Path) -> None:
    client = FakeClient()
    p = UpstoxProvider(SimpleNamespace(upstox_access_token=SecretStr("tok")), cache_dir=tmp_path, clock=clock,
                       client=client, instrument_loader=lambda: UPSTOX_ROWS)
    df = p.get_candles("RELIANCE", Timeframe.M5, ist(2026, 11, 23, 9, 15), CLOCK_NOW)
    validate_ohlcv(df)
    assert list(df.index) == [ist(2026, 11, 23, 9, 15), ist(2026, 11, 23, 9, 20)]      # history + intraday
    urls = [u for u, _, _ in client.requests]
    assert any("/v3/historical-candle/NSE_EQ%7CINE002A01018/minutes/5/" in u for u in urls)
    assert any("/intraday/" in u for u in urls)
    assert all(h.get("Authorization") == "Bearer tok" for _, _, h in client.requests)
    assert (tmp_path / "upstox" / "instruments_20261123.json").exists()
    q = p.get_quote("RELIANCE")
    assert q is not None and q.ltp == 2501.0 and q.timestamp == CLOCK_NOW
    assert p.instrument_key("NIFTY50") == "NSE_INDEX|Nifty 50"
    assert p.healthcheck() is True
