"""Shared pytest fixtures for the whole suite.

Fixtures:
    caps            - HardCaps from config/hard_caps.yaml
    settings        - default Settings from config/*.yaml
    session_factory - in-memory SQLite with all tables created
    session         - a Session from session_factory (rolled back / closed after test)
    sim_clock       - SimulatedClock at 2026-11-24 09:15 IST (a Tuesday trading day)
    ctx             - AppContext wired with the above (no data/llm/notifier)
JSONL logs are redirected to a temp dir for every test (autouse).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from config.loader import load_default_settings, load_hard_caps
from config.schema import HardCaps, Settings
from core.clock import IST, SimulatedClock
from core.context import AppContext
from core.logging import set_log_dir
from storage.db import create_all_for_url


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_network = os.environ.get("RUN_NETWORK_TESTS") == "1"
    run_ollama = os.environ.get("RUN_OLLAMA_TESTS") == "1"
    for item in items:
        if "network" in item.keywords and not run_network:
            item.add_marker(pytest.mark.skip(reason="network test (set RUN_NETWORK_TESTS=1)"))
        if "ollama" in item.keywords and not run_ollama:
            item.add_marker(pytest.mark.skip(reason="needs Ollama (set RUN_OLLAMA_TESTS=1)"))


@pytest.fixture(autouse=True)
def _tmp_logs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    set_log_dir(tmp_path_factory.mktemp("logs"))
    yield


@pytest.fixture
def caps() -> HardCaps:
    return load_hard_caps()


@pytest.fixture
def settings() -> Settings:
    return load_default_settings()


@pytest.fixture
def session_factory() -> sessionmaker[Session]:
    return create_all_for_url("sqlite://")


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    s = session_factory()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture
def sim_clock() -> SimulatedClock:
    return SimulatedClock(datetime(2026, 11, 24, 9, 15, tzinfo=IST))


@pytest.fixture
def ctx(settings: Settings, caps: HardCaps, session_factory: sessionmaker[Session], sim_clock: SimulatedClock) -> AppContext:
    return AppContext(settings=settings, caps=caps, session_factory=session_factory, clock=sim_clock)
