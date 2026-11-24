"""PAPER-TRADING GUARD (non-negotiable rule 0.1).

This module makes real order placement impossible at three layers:

1. ``PAPER_TRADING_ONLY`` is a hard-coded constant; ``assert_paper_only`` refuses to
   start the system unless both the constant and ``hard_caps.yaml: paper_only`` are True.
2. Broker SDK order methods (Angel One SmartAPI, Upstox) are monkey-patched to raise
   ``RealTradingForbiddenError``.
3. The HTTP layer (urllib3 - used by requests and the broker SDKs - and httpx) refuses
   any request whose URL looks like an order / GTT / position-conversion endpoint.

Call ``startup_safety_check(caps)`` as the very first thing in every process
(engine, dashboard, replay, scripts). It is idempotent.
"""

from __future__ import annotations

import functools
import importlib
import re
import threading
from typing import Any, Callable

from core.errors import RealTradingForbiddenError

#: Hard-coded. There is no code path that flips this.
PAPER_TRADING_ONLY: bool = True

#: URL fragments that identify order-placement-like endpoints of any broker.
FORBIDDEN_URL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"placeorder",
        r"modifyorder",
        r"cancelorder",
        r"convertposition",
        r"place_order",
        r"/orders?(/|$|\?)",          # /order/..., /orders/..., Kite /orders/regular, Upstox /v2/order/place
        r"/gtt",                        # GTT rules (Angel, Upstox, Kite)
        r"createrule|modifyrule|cancelrule",
        r"positions/exit",
        r"/exit(all)?positions?",
    )
)

#: Broker SDK classes/methods that place or alter real orders.
_SDK_TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "SmartApi.smartConnect": {
        "SmartConnect": (
            "placeOrder",
            "placeOrderFullResponse",
            "modifyOrder",
            "cancelOrder",
            "convertPosition",
            "gttCreateRule",
            "gttModifyRule",
            "gttCancelRule",
        ),
    },
}

_UPSTOX_MODULE = "upstox_client"
_install_lock = threading.Lock()
_installed = False


def is_forbidden_url(url: str) -> bool:
    """Return True if ``url`` looks like an order-placement endpoint."""
    return any(p.search(url or "") for p in FORBIDDEN_URL_PATTERNS)


def check_url(url: str) -> None:
    """Raise ``RealTradingForbiddenError`` if ``url`` is an order endpoint."""
    if is_forbidden_url(url):
        raise RealTradingForbiddenError(
            f"PAPER TRADING ONLY: blocked HTTP call to order endpoint: {url}"
        )


def forbidden_order_call(*_args: Any, **_kwargs: Any) -> None:
    """Replacement for every real order method. Always raises."""
    raise RealTradingForbiddenError(
        "PAPER TRADING ONLY: real order placement/modification/cancellation is forbidden."
    )


def assert_paper_only(caps: Any | None = None) -> None:
    """Startup assertion: refuse to run unless paper-only is enforced everywhere.

    Args:
        caps: a ``config.schema.HardCaps`` instance (or anything with ``paper_only``).
    """
    if PAPER_TRADING_ONLY is not True:  # pragma: no cover - constant
        raise RealTradingForbiddenError("PAPER_TRADING_ONLY constant was modified.")
    if caps is not None and getattr(caps, "paper_only", None) is not True:
        raise RealTradingForbiddenError(
            "config/hard_caps.yaml must contain 'paper_only: true'. Refusing to start."
        )


def _patch_sdk_methods() -> list[str]:
    patched: list[str] = []
    for module_name, classes in _SDK_TARGETS.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:  # SDK not installed / import side effects failed: nothing to patch
            continue
        for class_name, methods in classes.items():
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            for method in methods:
                if hasattr(cls, method):
                    setattr(cls, method, forbidden_order_call)
                    patched.append(f"{module_name}.{class_name}.{method}")
    try:
        upstox = importlib.import_module(_UPSTOX_MODULE)
    except Exception:
        upstox = None
    if upstox is not None:
        for attr_name in dir(upstox):
            if "order" not in attr_name.lower() and "gtt" not in attr_name.lower():
                continue
            cls = getattr(upstox, attr_name, None)
            if not isinstance(cls, type):
                continue
            for method in dir(cls):
                low = method.lower()
                if any(k in low for k in ("place", "modify", "cancel", "exit", "convert")):
                    setattr(cls, method, forbidden_order_call)
                    patched.append(f"{_UPSTOX_MODULE}.{attr_name}.{method}")
    return patched


def _wrap(original: Callable[..., Any], url_getter: Callable[..., str]) -> Callable[..., Any]:
    @functools.wraps(original)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        check_url(url_getter(*args, **kwargs))
        return original(*args, **kwargs)

    guarded.__paper_guard__ = True  # type: ignore[attr-defined]
    return guarded


def _wrap_async(original: Callable[..., Any], url_getter: Callable[..., str]) -> Callable[..., Any]:
    @functools.wraps(original)
    async def guarded(*args: Any, **kwargs: Any) -> Any:
        check_url(url_getter(*args, **kwargs))
        return await original(*args, **kwargs)

    guarded.__paper_guard__ = True  # type: ignore[attr-defined]
    return guarded


def _patch_http_layer() -> list[str]:
    patched: list[str] = []
    try:
        import urllib3.connectionpool as cp

        original = cp.HTTPConnectionPool.urlopen
        if not getattr(original, "__paper_guard__", False):

            def _urllib3_url(self: Any, method: str, url: str, *a: Any, **k: Any) -> str:
                return f"{getattr(self, 'host', '')}{url}"

            cp.HTTPConnectionPool.urlopen = _wrap(original, _urllib3_url)  # type: ignore[method-assign]
            patched.append("urllib3.HTTPConnectionPool.urlopen")
    except ImportError:
        pass
    try:
        import httpx

        def _httpx_url(self: Any, request: Any, *a: Any, **k: Any) -> str:
            return str(getattr(request, "url", ""))

        sync_original = httpx.HTTPTransport.handle_request
        if not getattr(sync_original, "__paper_guard__", False):
            httpx.HTTPTransport.handle_request = _wrap(sync_original, _httpx_url)  # type: ignore[method-assign]
            patched.append("httpx.HTTPTransport.handle_request")
        async_original = httpx.AsyncHTTPTransport.handle_async_request
        if not getattr(async_original, "__paper_guard__", False):
            httpx.AsyncHTTPTransport.handle_async_request = _wrap_async(  # type: ignore[method-assign]
                async_original, _httpx_url
            )
            patched.append("httpx.AsyncHTTPTransport.handle_async_request")
    except ImportError:
        pass
    return patched


def install_order_guards() -> list[str]:
    """Install SDK + HTTP guards. Idempotent. Returns the list of patched targets."""
    global _installed
    with _install_lock:
        patched = _patch_sdk_methods() + _patch_http_layer()
        _installed = True
        return patched


def guards_installed() -> bool:
    """True once ``install_order_guards`` has run in this process."""
    return _installed


def startup_safety_check(caps: Any | None = None) -> list[str]:
    """Run the paper-only assertion and install every guard. Call at process start."""
    assert_paper_only(caps)
    return install_order_guards()
