"""Paper-trading guard tests (rule 0.1)."""

from __future__ import annotations

import pytest

from config.schema import HardCaps
from core import guard
from core.errors import RealTradingForbiddenError


def test_paper_constant_is_true() -> None:
    assert guard.PAPER_TRADING_ONLY is True


def test_assert_paper_only_accepts_default_caps(caps: HardCaps) -> None:
    guard.assert_paper_only(caps)


def test_assert_paper_only_rejects_non_paper_object() -> None:
    class Fake:
        paper_only = False

    with pytest.raises(RealTradingForbiddenError):
        guard.assert_paper_only(Fake())


def test_hard_caps_cannot_disable_paper_only() -> None:
    with pytest.raises(ValueError):
        HardCaps(paper_only=False)


@pytest.mark.parametrize(
    "url",
    [
        "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder",
        "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/modifyOrder",
        "https://apiconnect.angelone.in/rest/secure/angelbroking/gtt/v1/createRule",
        "https://api.upstox.com/v2/order/place",
        "https://api-hft.upstox.com/v3/order/place",
        "https://api.kite.trade/orders/regular",
        "https://api.upstox.com/v2/order/positions/exit",
    ],
)
def test_order_urls_blocked(url: str) -> None:
    assert guard.is_forbidden_url(url)
    with pytest.raises(RealTradingForbiddenError):
        guard.check_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://apiconnect.angelone.in/rest/secure/angelbroking/historical/v1/getCandleData",
        "https://apiconnect.angelone.in/rest/secure/angelbroking/market/v1/quote/",
        "https://api.upstox.com/v2/historical-candle/NSE_EQ%7CINE002A01018/day/2026-01-01",
        "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS",
        "http://127.0.0.1:11434/api/chat",
        "https://www.nseindia.com/api/corporate-announcements?index=equities",
    ],
)
def test_data_urls_allowed(url: str) -> None:
    assert not guard.is_forbidden_url(url)


def test_install_guards_patches_smartapi_and_http() -> None:
    patched = guard.startup_safety_check(HardCaps())
    assert guard.guards_installed()
    # SmartAPI is a pinned dependency: its order methods must now raise.
    try:
        from SmartApi.smartConnect import SmartConnect
    except Exception:  # pragma: no cover - SDK import failure in exotic envs
        pytest.skip("SmartApi not importable")
    assert any("placeOrder" in p for p in patched) or SmartConnect.placeOrder is guard.forbidden_order_call
    with pytest.raises(RealTradingForbiddenError):
        SmartConnect.placeOrder(None, {})  # type: ignore[arg-type]


def test_http_layer_blocks_order_endpoint_via_requests() -> None:
    guard.install_order_guards()
    import requests

    with pytest.raises(RealTradingForbiddenError):
        requests.post("https://api.upstox.com/v2/order/place", json={}, timeout=1)


def test_http_layer_blocks_order_endpoint_via_httpx() -> None:
    guard.install_order_guards()
    import httpx

    with pytest.raises(RealTradingForbiddenError):
        httpx.post("https://api.upstox.com/v2/order/place", json={}, timeout=1)
