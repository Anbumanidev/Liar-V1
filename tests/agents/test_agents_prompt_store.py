"""PromptStore: seed / get / pinned version / save_version / revert / render (StrictUndefined)."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from agents.llm import FakeLLMClient
from agents.prompt_store import (
    DEBATE_VARIABLES,
    DEFAULT_PROMPTS_DIR,
    META_VARIABLES,
    SUB_AGENT_VARIABLES,
    PromptNotFoundError,
    PromptRenderError,
    PromptStore,
    allowed_variables,
)
from agents.registry import get_agent
from config.schema import AGENT_NAMES, ALL_LLM_AGENTS, Settings
from core.clock import SimulatedClock
from core.enums import AgentStatus
from tests.agents.agents_kit import good_answers, intraday_ctx


def test_seed_loads_every_agent_including_debate_and_meta(prompts: PromptStore) -> None:
    for agent in ALL_LLM_AGENTS:
        tpl = prompts.get(agent)
        assert tpl.version == 1 and tpl.author == "seed" and tpl.active
        assert tpl.system_prompt.strip() and tpl.user_template.strip()
    assert {p.stem for p in DEFAULT_PROMPTS_DIR.glob("*.yaml")} == set(ALL_LLM_AGENTS)
    assert prompts.seed() == 0  # idempotent


def test_get_auto_seeds_missing_agent(session_factory: sessionmaker[Session], sim_clock: SimulatedClock) -> None:
    store = PromptStore(session_factory, clock=sim_clock, cache_ttl_seconds=0)
    assert store.get("bear").version == 1
    assert [t.agent for t in store.versions("bear")] == ["bear"]
    assert store.versions("technical") == []  # only the requested agent was seeded
    with pytest.raises(PromptNotFoundError):
        store.get("no_such_agent")


def test_save_version_pinned_get_and_revert(prompts: PromptStore) -> None:
    v1 = prompts.get("technical")
    v2 = prompts.save_version("technical", "system v2 {{ mode_rules }}", "user v2 {{ payload_json }}", "owner", "edit")
    assert v2.version == 2 and v2.author == "owner" and v2.note == "edit"
    assert prompts.get("technical").version == 2
    assert prompts.get("technical", 1).system_prompt == v1.system_prompt  # pinned
    assert prompts.get("technical", 99).version == 2  # missing pin -> latest
    v3 = prompts.revert("technical", 1, "owner")
    assert v3.version == 3 and v3.system_prompt == v1.system_prompt and v3.note == "revert to v1"
    assert [t.version for t in prompts.versions("technical")] == [3, 2, 1]
    with pytest.raises(PromptNotFoundError):
        prompts.revert("technical", 42, "owner")


def test_save_version_rejects_bad_syntax_and_unknown_variables(prompts: PromptStore) -> None:
    with pytest.raises(ValueError, match="invalid prompt template"):
        prompts.save_version("technical", "{% if %}", "u", "owner")
    with pytest.raises(ValueError, match="unknown variable"):
        prompts.save_version("technical", "{{ payload_jsn }}", "u", "owner")
    with pytest.raises(ValueError, match="options_json"):
        prompts.save_version("bull", "{{ options_json }}", "u", "owner")  # meta-only variable
    assert prompts.save_version("meta", "{{ options_json }} {{ facts_json }}", "{{ enabled_modes }}", "o").version == 2


def test_allowed_variables_per_agent_kind() -> None:
    assert allowed_variables("bull") == allowed_variables("bear") == DEBATE_VARIABLES
    assert allowed_variables("meta") == META_VARIABLES
    for agent in AGENT_NAMES + ("future_agent",):
        assert allowed_variables(agent) == SUB_AGENT_VARIABLES


def test_render_strict_undefined_and_sandbox(prompts: PromptStore) -> None:
    assert prompts.render("Hi {{ name }}  ", name="X") == "Hi X"
    with pytest.raises(PromptRenderError, match="undefined"):
        prompts.render("Hi {{ nme }}", name="X")
    with pytest.raises(PromptRenderError):
        prompts.render("{{ ''.__class__.__mro__ }}")
    assert issubclass(PromptRenderError, ValueError)
    tpl = prompts.get("technical")
    with pytest.raises(PromptRenderError, match="prompt technical v1"):
        prompts.render_prompt(tpl, agent="technical")  # most variables missing
    assert prompts.template_variables("{{ a }}{% for x in range(2) %}{{ x }}{{ loop.index }}{% endfor %}") == {"a"}


def test_cache_ttl_and_invalidation(session_factory: sessionmaker[Session], sim_clock: SimulatedClock) -> None:
    store = PromptStore(session_factory, clock=sim_clock, cache_ttl_seconds=3600)
    store.seed()
    assert store.get("technical").version == 1
    store.save_version("technical", "s {{ symbol }}", "u", "owner")  # invalidates this process' cache
    assert store.get("technical").version == 2
    other = PromptStore(session_factory, clock=sim_clock, cache_ttl_seconds=3600)
    other.save_version("technical", "s3", "u3", "dashboard")  # another writer: cached until TTL / invalidate
    assert store.get("technical").version == 2
    store.invalidate()
    assert store.get("technical").version == 3


def test_concurrent_reads_on_single_connection_engine(prompts: PromptStore) -> None:
    """Sub-agents read prompts from parallel threads; the in-memory test DB has ONE connection."""
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(40):
                for agent in ("technical", "stop_loss", "meta"):
                    tpl = prompts.get(agent)
                    assert tpl.agent == agent and isinstance(tpl.system_prompt, str)
        except BaseException as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []


def test_seed_skips_invalid_files(tmp_path: Path, session_factory: sessionmaker[Session]) -> None:
    good = "agent: technical\nversion: 1\nsystem: |\n  S {{ mode_rules }}\nuser_template: |\n  U {{ payload_json }}\n"
    (tmp_path / "technical.yaml").write_text(good, encoding="utf-8")
    (tmp_path / "price_action.yaml").write_text("agent: price_action\nsystem: ''\nuser_template: u\n", encoding="utf-8")
    (tmp_path / "news_sentiment.yaml").write_text(good.replace("technical", "fundamental"), encoding="utf-8")
    (tmp_path / "stop_loss.yaml").write_text(good.replace("technical", "stop_loss").replace("mode_rules", "typo"),
                                             encoding="utf-8")
    (tmp_path / "market_context.yaml").write_text(": [unbalanced", encoding="utf-8")
    store = PromptStore(session_factory, prompts_dir=tmp_path, cache_ttl_seconds=0)
    assert store.seed() == 1
    assert store.get("technical").system_prompt.startswith("S ")
    assert store.versions("stop_loss") == [] and store.versions("price_action") == []


def test_agent_uses_pinned_prompt_version(settings: Settings, prompts: PromptStore) -> None:
    prompts.save_version("technical", "NEW SYSTEM {{ mode_rules }}", "NEW USER {{ payload_json }}", "owner")
    fake = FakeLLMClient(good_answers())
    rep = get_agent("technical").run(intraday_ctx(), fake, settings, prompts)
    assert rep.prompt_version == 2 and fake.calls[-1]["system"].startswith("NEW SYSTEM")
    settings.agents.prompt_versions["technical"] = 1
    rep = get_agent("technical").run(intraday_ctx(), fake, settings, prompts)
    assert rep.status == AgentStatus.OK and rep.prompt_version == 1
    assert not fake.calls[-1]["system"].startswith("NEW SYSTEM")


def test_broken_dashboard_prompt_fails_agent_not_pipeline(settings: Settings, prompts: PromptStore,
                                                          session_factory: sessionmaker[Session]) -> None:
    # The dashboard writes prompt_templates directly (no validation): simulate a typo there.
    from storage.models import PromptTemplate

    with session_factory() as s:
        s.add(PromptTemplate(agent="technical", version=2, system_prompt="{{ mode_ruels }}", user_template="u",
                             created_at=intraday_ctx().as_of, author="dashboard", active=True))
        s.commit()
    fake = FakeLLMClient(good_answers())
    rep = get_agent("technical").run(intraday_ctx(), fake, settings, prompts)
    assert rep.status == AgentStatus.FAILED and rep.confidence == 0
    assert "prompt technical v2 failed to render" in (rep.error or "") and "mode_ruels" in (rep.error or "")
    assert rep.prompt_version == 2 and fake.calls == []
