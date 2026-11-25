"""Fixtures for the web dashboard tests: temp-file SQLite DB seeded with a realistic day, a fake
market data service with deterministic candles/quotes, an auth file and a logged-in TestClient."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dashboard.auth import build_auth_document, hash_password, write_auth_config
from storage.db import create_all_for_url
from tests.webui.webui_kit import DAY, OPTIONAL_MODULES, PASSWORD, login, make_ctx, seed
from webui.context import WebContext
from webui.server import create_app


@pytest.fixture
def no_optional_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the parallel-developed modules unimportable (None in sys.modules -> ImportError)."""
    for name in OPTIONAL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> Callable[..., types.ModuleType]:
    def _install(name: str, **funcs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in funcs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod
    return _install


@pytest.fixture
def web_env(tmp_path: Path) -> dict[str, Any]:
    url = f"sqlite:///{(tmp_path / 'trading.db').as_posix()}"
    factory = create_all_for_url(url)
    ids = seed(factory)
    auth_file = tmp_path / "secrets" / "dashboard_auth.yaml"
    write_auth_config(auth_file, build_auth_document("admin", "Owner", hash_password(PASSWORD, rounds=4)))
    log_dir = tmp_path / "logs"
    folder = log_dir / DAY.isoformat()
    folder.mkdir(parents=True)
    (folder / "engine.log").write_text(
        "2026-11-25 13:59:00.123 +05:30 | INFO     | control.engine:_fire:89 - [engine] job intraday_cycle start\n"
        "2026-11-25 13:59:05.000 +05:30 | ERROR    | data.service:_fail:400 - [data] boom\n"
        "Traceback (most recent call last):\n  File x\n"
        "2026-11-25 13:59:10.000 +05:30 | INFO     | control.engine:_fire:99 - [engine] job intraday_cycle done\n",
        encoding="utf-8")
    (folder / "decisions.jsonl").write_text('{"ts": "2026-11-25T13:00:00+05:30", "category": "decisions", '
                                           '"symbol": "TCS", "decision": "BUY"}\nnot json\n', encoding="utf-8")
    return {"tmp": tmp_path, "factory": factory, "ids": ids, "auth_file": auth_file, "log_dir": log_dir}


@pytest.fixture
def ctx(web_env: dict[str, Any]) -> Iterator[WebContext]:
    c = make_ctx(web_env)
    yield c
    c.close()


@pytest.fixture
def client(ctx: WebContext) -> Iterator[TestClient]:
    with TestClient(create_app(ctx)) as c:
        yield c


@pytest.fixture
def authed(client: TestClient) -> TestClient:
    login(client)
    return client
