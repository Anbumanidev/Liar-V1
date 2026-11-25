"""LLM client layer: OllamaClient retry loop (stubbed ollama client), FakeLLMClient, RulesClient, build_llm."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, field_validator

from agents.llm import (
    FakeLLMClient,
    LLMClient,
    OllamaClient,
    RulesClient,
    build_llm,
    extract_json_text,
    field_of,
)
from config.schema import Settings


class Out(BaseModel):
    signal: str
    confidence: int


class Exploding(BaseModel):
    x: int

    @field_validator("x")
    @classmethod
    def _boom(cls, v: int) -> int:
        raise TypeError("not a ValueError: escapes pydantic")


VALID = json.dumps({"signal": "BULLISH", "confidence": 70})


def resp(content: str, tin: int | None = 10, tout: int | None = 5) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": content}, "prompt_eval_count": tin, "eval_count": tout}


class StubOllama:
    """Minimal ``ollama.Client`` stand-in: ``chat`` replays a script (last item reused)."""

    def __init__(self, script: list[Any], models: tuple[str, ...] = ("qwen3:4b", "qwen3:8b"),
                 list_error: Exception | None = None, delay: float = 0.0, reject_think: bool = False) -> None:
        self.script = list(script)
        self.models = models
        self.list_error = list_error
        self.delay = delay
        self.reject_think = reject_think
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> Any:
        if self.reject_think and "think" in kwargs:
            raise TypeError("Client.chat() got an unexpected keyword argument 'think'")
        self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        if self.delay:
            time.sleep(self.delay)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def list(self) -> Any:
        if self.list_error:
            raise self.list_error
        return {"models": [{"model": m} for m in self.models]}


def client(stub: StubOllama, retries: int = 2) -> OllamaClient:
    return OllamaClient(host="http://stub:11434", timeout=5, max_retries=retries, keep_alive="5m", num_ctx=4096,
                        client=stub)


# --------------------------------------------------------------------------- OllamaClient


def test_invalid_json_then_valid_reprompts_with_validation_error() -> None:
    stub = StubOllama([resp("this is not json"), resp(VALID)])
    res = client(stub).generate("technical", "qwen3:4b", "SYS", "USER", Out, temperature=0.3)

    assert res.ok and isinstance(res.parsed, Out) and res.parsed.confidence == 70
    assert res.attempts == 2 and len(res.errors) == 1 and "invalid output" in res.errors[0]
    assert res.tokens_in == 20 and res.tokens_out == 10  # summed over attempts
    assert res.model == "qwen3:4b" and res.raw == VALID and res.error is None
    first, second = stub.calls
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == "this is not json"
    assert "rejected by the validator" in second["messages"][3]["content"]
    assert first["format"] == Out.model_json_schema()
    assert first["options"] == {"temperature": 0.3, "num_ctx": 4096}
    assert first["keep_alive"] == "5m" and first["think"] is False and first["model"] == "qwen3:4b"


def test_schema_violation_is_retried_then_exhausted() -> None:
    stub = StubOllama([resp(json.dumps({"signal": "BULLISH"}))])  # missing confidence, every time
    res = client(stub, retries=2).generate("technical", "m", "S", "U", Out)

    assert not res.ok and res.parsed is None
    assert res.attempts == 3 and len(stub.calls) == 3 and len(res.errors) == 3
    assert res.error is not None and res.error.startswith("LLM failed after 3 attempt(s)")
    assert "confidence" in res.error
    assert res.raw is not None and res.tokens_in == 30


def test_network_error_is_retried() -> None:
    stub = StubOllama([ConnectionError("connection refused"), resp(VALID)])
    res = client(stub).generate("technical", "m", "S", "U", Out)
    assert res.ok and res.attempts == 2
    assert "ConnectionError" in res.errors[0]
    # transport errors do not append a corrective turn
    assert len(stub.calls[1]["messages"]) == 2


def test_network_errors_exhausted_never_raise() -> None:
    stub = StubOllama([TimeoutError("read timed out")])
    res = client(stub, retries=1).generate("technical", "m", "S", "U", Out)
    assert not res.ok and res.attempts == 2 and "TimeoutError" in (res.error or "")
    assert res.tokens_in is None and res.tokens_out is None


def test_unexpected_exception_inside_validation_never_raises() -> None:
    stub = StubOllama([resp(json.dumps({"x": 1}))])
    res = client(stub).generate("technical", "m", "S", "U", Exploding)
    assert not res.ok and "unexpected" in (res.error or "")


def test_zero_retries_single_attempt() -> None:
    stub = StubOllama([resp("nope")])
    res = client(stub, retries=0).generate("technical", "m", "S", "U", Out)
    assert not res.ok and res.attempts == 1 and len(stub.calls) == 1


def test_empty_content_is_an_error_and_retried() -> None:
    stub = StubOllama([resp("   "), resp(VALID)])
    res = client(stub).generate("technical", "m", "S", "U", Out)
    assert res.ok and res.attempts == 2 and "empty response" in res.errors[0]


def test_latency_and_tokens_recorded() -> None:
    stub = StubOllama([resp(VALID, tin=123, tout=45)], delay=0.03)
    res = client(stub).generate("technical", "m", "S", "U", Out)
    assert res.ok and res.latency_ms >= 25 and res.tokens_in == 123 and res.tokens_out == 45


def test_attribute_style_response_and_unknown_token_counts() -> None:
    obj = SimpleNamespace(message=SimpleNamespace(content=VALID), prompt_eval_count=None, eval_count=None)
    res = client(StubOllama([obj])).generate("technical", "m", "S", "U", Out)
    assert res.ok and res.tokens_in is None and res.tokens_out is None


def test_think_kwarg_fallback_for_old_ollama_libs() -> None:
    stub = StubOllama([resp(VALID)], reject_think=True)
    res = client(stub).generate("meta", "m", "S", "U", Out, think=True)
    assert res.ok and res.attempts == 1 and "think" not in stub.calls[0]


def test_fenced_and_thinking_answers_are_salvaged() -> None:
    noisy = "<think>let me reason {not json}</think>\n```json\n" + VALID + "\n```"
    res = client(StubOllama([resp(noisy)])).generate("technical", "m", "S", "U", Out)
    assert res.ok and res.attempts == 1
    assert extract_json_text("Sure! Here it is: " + VALID + " Hope it helps") == VALID


def test_list_models_and_healthcheck() -> None:
    ok = client(StubOllama([resp(VALID)], models=("b:1", "a:1")))
    assert ok.list_models() == ["a:1", "b:1"] and ok.healthcheck() is True
    bad = client(StubOllama([resp(VALID)], list_error=ConnectionError("down")))
    assert bad.list_models() == [] and bad.healthcheck() is False


def test_field_of_handles_dicts_objects_and_missing() -> None:
    assert field_of({"a": 1}, "a") == 1
    assert field_of(SimpleNamespace(a=2), "a") == 2
    assert field_of(None, "a", 3) == 3
    assert field_of(SimpleNamespace(), "missing", "d") == "d"


# --------------------------------------------------------------------------- fake / rules / factory


def test_fake_client_scripts_lists_defaults_callables_and_exceptions() -> None:
    fake = FakeLLMClient({
        "technical": ["bad", {"signal": "BULLISH", "confidence": 61}],
        "single": {"signal": "NEUTRAL", "confidence": 1},
        "call": lambda system, user: {"signal": system, "confidence": len(user)},
        "boom": [ConnectionError("x"), VALID],
        "*": [{"signal": "BEARISH", "confidence": 9}],
    })
    r1 = fake.generate("technical", "m", "S", "U", Out)
    assert r1.ok and r1.attempts == 2 and r1.parsed.confidence == 61
    assert fake.generate("technical", "m", "S", "U", Out).parsed.confidence == 61  # last answer reused
    assert fake.generate("single", "m", "S", "U", Out).parsed.signal == "NEUTRAL"
    assert fake.generate("call", "m", "SYS", "four", Out).parsed.model_dump() == {"signal": "SYS", "confidence": 4}
    assert fake.generate("boom", "m", "S", "U", Out).attempts == 2
    assert fake.generate("anything", "m", "S", "U", Out).parsed.signal == "BEARISH"
    assert len(fake.calls_for("technical")) == 3
    call = fake.calls_for("call")[0]
    assert call["system"] == "SYS" and call["user"] == "four" and call["schema"] == "Out"
    assert isinstance(fake, LLMClient) and fake.healthcheck() and fake.list_models() == ["fake-model"]


def test_fake_client_without_script_fails_gracefully() -> None:
    res = FakeLLMClient({}).generate("technical", "m", "S", "U", Out)
    assert not res.ok and "no scripted response" in (res.error or "")


def test_rules_client_skips() -> None:
    rc = RulesClient()
    res = rc.generate("technical", "m", "S", "U", Out)
    assert not res.ok and res.skipped and res.attempts == 0 and res.model == "rules"
    assert rc.healthcheck() and rc.list_models() == [] and isinstance(rc, LLMClient)


def test_build_llm_backends(settings: Settings) -> None:
    llm = build_llm(settings)
    assert isinstance(llm, OllamaClient)
    assert llm.host == settings.models.ollama_host and llm.max_retries == settings.models.max_retries
    assert llm.num_ctx == settings.models.num_ctx and llm.keep_alive == settings.models.keep_alive
    assert llm.timeout == float(settings.models.request_timeout_seconds)
    settings.engine.llm_backend = "rules"
    assert isinstance(build_llm(settings), RulesClient)


@pytest.mark.ollama
def test_real_ollama_roundtrip(settings: Settings) -> None:  # pragma: no cover - needs a local server
    llm = build_llm(settings)
    res = llm.generate("technical", settings.models.model_for("technical"), "Answer JSON.",
                       'Return {"signal": "NEUTRAL", "confidence": 50}', Out)
    assert res.ok
