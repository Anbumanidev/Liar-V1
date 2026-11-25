"""Fixtures for the agents tests (FakeLLMClient / RulesClient only - Ollama is never needed)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from loguru import logger
from sqlalchemy.orm import Session, sessionmaker

from agents.llm import RulesClient
from agents.prompt_store import PromptStore
from core.clock import SimulatedClock


@pytest.fixture
def prompts(session_factory: sessionmaker[Session], sim_clock: SimulatedClock) -> PromptStore:
    """Seeded prompt store without caching (every get hits the DB)."""
    store = PromptStore(session_factory, clock=sim_clock, cache_ttl_seconds=0)
    store.seed()
    return store


@pytest.fixture
def rules() -> RulesClient:
    return RulesClient()


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    """Captured loguru messages (WARNING and above) for the duration of a test."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING", format="{level} {message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)
