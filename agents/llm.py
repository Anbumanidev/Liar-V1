"""LLM client layer (section 2, rule 0.4).

Every LLM call in the system goes through an ``LLMClient``:

* ``generate(agent, model, system, user, schema, *, temperature, think)`` returns an
  ``LLMResult`` and **never raises**. The answer is validated with ``schema``
  (``model_validate_json``); invalid JSON / schema violations are re-prompted up to
  ``max_retries`` times by appending the bad answer plus a corrective user message that
  quotes the validation error. Network / timeout errors are retried the same number of
  times. After that ``ok=False`` with ``error`` set and the caller degrades (FAILED agent
  -> neutral vote, NO_TRADE meta).
* ``OllamaClient`` talks to a local Ollama server (open-weight models only, rule 0.2).
* ``FakeLLMClient`` replays scripted answers for tests (no network).
* ``RulesClient`` is the ``llm_backend: rules`` backend: it never calls a model and returns
  ``skipped=True`` so agents use their deterministic interpretation instead.

Debate / meta agents reuse the same clients with their own Pydantic schemas.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:  # pragma: no cover
    from config.schema import Settings

#: Max characters of a validation error quoted back to the model in a corrective prompt.
_MAX_ERROR_CHARS = 1500
#: Max characters of a bad answer echoed back to the model (keeps the context small).
_MAX_ECHO_CHARS = 4000

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*(.*?)\s*```$", re.DOTALL)


class LLMResult(BaseModel):
    """Outcome of one ``LLMClient.generate`` call (including all retries)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    parsed: Any = None                 # instance of the requested schema when ok
    raw: str | None = None             # last raw answer text
    model: str = ""
    latency_ms: int = 0                # wall time of all attempts
    tokens_in: int | None = None       # summed over attempts (None when unknown)
    tokens_out: int | None = None
    attempts: int = 0                  # number of model calls made
    error: str | None = None
    skipped: bool = False              # True for the "rules" backend (no model was called)
    errors: list[str] = Field(default_factory=list)  # one entry per failed attempt


@runtime_checkable
class LLMClient(Protocol):
    """Structural interface every LLM backend implements."""

    backend: str

    def generate(
        self,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        temperature: float = 0.2,
        think: bool = False,
    ) -> LLMResult:
        """Run one structured generation. Never raises."""
        ...

    def healthcheck(self) -> bool:
        """True when the backend can serve requests."""
        ...

    def list_models(self) -> list[str]:
        """Names of installed models (empty when unknown / unavailable)."""
        ...


# --------------------------------------------------------------------------- helpers


def field_of(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a dict-like or attribute-style object (ollama lib versions differ)."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    try:
        return getattr(obj, name)
    except AttributeError:
        pass
    try:
        return obj[name]  # SubscriptableBaseModel
    except (KeyError, TypeError, IndexError):
        return default


def extract_json_text(text: str) -> str:
    """Strip ``<think>`` blocks and Markdown fences; isolate the outermost JSON object.

    Small models occasionally wrap JSON in ```json fences or prepend a sentence even when
    ``format`` is set; this salvages such answers without a retry.
    """
    t = _THINK_RE.sub("", text or "").strip()
    m = _FENCE_RE.match(t)
    if m:
        t = m.group(1).strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return t[start : end + 1]
    return t


def corrective_message(error: str) -> str:
    """User message sent after an invalid answer (quotes the validation error)."""
    err = error if len(error) <= _MAX_ERROR_CHARS else error[:_MAX_ERROR_CHARS] + " ..."
    return (
        "Your previous answer was rejected by the validator:\n"
        f"{err}\n\n"
        "Reply again with ONLY one JSON object that matches the required schema exactly. "
        "No prose, no Markdown, no code fences. Do not invent numbers: only use values "
        "present in the provided facts."
    )


def _add(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


class _ChatError(Exception):
    """Transport / server error from a backend (retried, then reported)."""


class _RetryingClient:
    """Shared validate-and-re-prompt loop. Subclasses implement ``_chat``."""

    backend: str = "base"

    def __init__(self, max_retries: int = 2, *, timeout_retries: int | None = None,
                 max_concurrent: int | None = None) -> None:
        """
        Args:
            max_retries: re-prompts after an INVALID answer (schema / JSON errors).
            timeout_retries: retries after a transport error (timeout, connection). None = same
                as ``max_retries``. Kept low on purpose: a timed-out local model is overloaded,
                and retrying 3x only multiplies the queue (2026-11-25: 360 s per failed agent).
            max_concurrent: process-wide cap on in-flight calls (None = unlimited). Calls wait
                for a slot BEFORE the request starts, so the HTTP timeout measures generation
                time, not time spent queueing behind other agents inside the model server.
        """
        self.max_retries = max(0, int(max_retries))
        self.timeout_retries = self.max_retries if timeout_retries is None else max(0, int(timeout_retries))
        # Overall attempt cap: legacy behaviour (max_retries + 1 for any mix of errors) unless a
        # separate transport-retry budget was given.
        self.max_attempts = (self.max_retries + 1 if timeout_retries is None
                             else self.max_retries + self.timeout_retries + 1)
        self.max_concurrent = int(max_concurrent) if max_concurrent else None
        self._slots = threading.BoundedSemaphore(self.max_concurrent) if self.max_concurrent else None

    def _call(
        self,
        agent: str,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float,
        think: bool,
    ) -> tuple[str, int | None, int | None]:
        """``_chat`` inside a concurrency slot (if capped)."""
        if self._slots is None:
            return self._chat(agent, model, messages, schema, temperature, think)
        with self._slots:
            return self._chat(agent, model, messages, schema, temperature, think)

    # Subclasses return (content, tokens_in, tokens_out) or raise.
    def _chat(
        self,
        agent: str,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float,
        think: bool,
    ) -> tuple[str, int | None, int | None]:
        raise NotImplementedError

    def generate(
        self,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        temperature: float = 0.2,
        think: bool = False,
    ) -> LLMResult:
        """Generate, validate with ``schema``, re-prompt on errors. Never raises."""
        t0 = time.perf_counter()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        tokens_in: int | None = None
        tokens_out: int | None = None
        raw: str | None = None
        errors: list[str] = []
        attempts = 0
        transport_failures = 0
        invalid_answers = 0
        try:
            while True:
                attempts += 1
                attempt = attempts
                try:
                    content, t_in, t_out = self._call(agent, model, messages, schema, temperature, think)
                except Exception as exc:  # noqa: BLE001 - any transport error is retried then reported
                    msg = f"attempt {attempt}: {type(exc).__name__}: {exc}"
                    errors.append(msg)
                    logger.warning(f"LLM call failed agent={agent} model={model} {msg}")
                    transport_failures += 1
                    if transport_failures > self.timeout_retries or attempts >= self.max_attempts:
                        break
                    continue
                tokens_in, tokens_out = _add(tokens_in, t_in), _add(tokens_out, t_out)
                raw = content
                try:
                    parsed = schema.model_validate_json(extract_json_text(content))
                except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                    msg = f"attempt {attempt}: invalid output: {exc}"
                    errors.append(msg)
                    logger.warning(f"LLM output invalid agent={agent} model={model} {msg[:300]}")
                    invalid_answers += 1
                    if invalid_answers > self.max_retries or attempts >= self.max_attempts:
                        break
                    echo = content if len(content) <= _MAX_ECHO_CHARS else content[:_MAX_ECHO_CHARS]
                    messages.append({"role": "assistant", "content": echo})
                    messages.append({"role": "user", "content": corrective_message(str(exc))})
                    continue
                return LLMResult(
                    ok=True, parsed=parsed, raw=raw, model=model,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    tokens_in=tokens_in, tokens_out=tokens_out, attempts=attempts, errors=errors,
                )
        except Exception as exc:  # noqa: BLE001 - generate() must never raise (rule 0.4)
            logger.exception(f"Unexpected LLM client error agent={agent}: {exc}")
            errors.append(f"unexpected: {type(exc).__name__}: {exc}")
        error = f"LLM failed after {attempts} attempt(s): " + (errors[-1] if errors else "unknown error")
        return LLMResult(
            ok=False, parsed=None, raw=raw, model=model,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            tokens_in=tokens_in, tokens_out=tokens_out, attempts=attempts, error=error, errors=errors,
        )


# --------------------------------------------------------------------------- Ollama


class OllamaClient(_RetryingClient):
    """Structured-output client for a local Ollama server.

    Args:
        host: Ollama base URL, e.g. ``http://127.0.0.1:11434``.
        timeout: per-request timeout in seconds.
        max_retries: re-prompts after the first attempt (settings.models.max_retries).
        keep_alive: how long Ollama keeps the model loaded (``"15m"``).
        num_ctx: context window passed in ``options``.
        client: optional pre-built ``ollama.Client``-compatible object (tests inject a stub).
    """

    backend = "ollama"

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        max_retries: int = 2,
        keep_alive: str | float | None = "15m",
        num_ctx: int = 8192,
        client: Any = None,
        *,
        timeout_retries: int | None = None,
        max_concurrent: int | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries, timeout_retries=timeout_retries, max_concurrent=max_concurrent)
        self.host = host
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client = client
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        """Lazily build the ``ollama.Client`` (import deferred so tests need no server)."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    import ollama  # local import: optional at test time

                    self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def _chat(
        self,
        agent: str,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float,
        think: bool,
    ) -> tuple[str, int | None, int | None]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "options": {"temperature": float(temperature), "num_ctx": int(self.num_ctx)},
            "keep_alive": self.keep_alive,
            "think": bool(think),
        }
        try:
            resp = client.chat(**kwargs)
        except TypeError as exc:
            # Older ollama libs have no ``think`` kwarg: retry once without it.
            if "think" not in str(exc):
                raise
            kwargs.pop("think", None)
            resp = client.chat(**kwargs)
        message = field_of(resp, "message")
        content = field_of(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise _ChatError("empty response content")
        t_in = field_of(resp, "prompt_eval_count")
        t_out = field_of(resp, "eval_count")
        return content, (int(t_in) if t_in is not None else None), (int(t_out) if t_out is not None else None)

    def list_models(self) -> list[str]:
        """Installed model names via ``client.list()``; empty list on failure (logged)."""
        try:
            resp = self._get_client().list()
        except Exception as exc:  # noqa: BLE001 - dashboard / health page must not crash
            logger.warning(f"Ollama list() failed at {self.host}: {exc}")
            return []
        models = field_of(resp, "models") or []
        names: list[str] = []
        for m in models:
            name = field_of(m, "model") or field_of(m, "name")
            if name:
                names.append(str(name))
        return sorted(names)

    def healthcheck(self) -> bool:
        """True if the Ollama server answers ``list()``."""
        try:
            self._get_client().list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Ollama healthcheck failed at {self.host}: {exc}")
            return False


# --------------------------------------------------------------------------- fakes / rules

FakeResponse = Mapping[str, Any] | str | BaseException | Callable[[str, str], Any]


class FakeLLMClient(_RetryingClient):
    """Scripted client for tests: same retry / validation loop as ``OllamaClient``.

    ``responses`` maps agent name -> a list of answers consumed in order (the last one is
    reused when the list runs out), a single answer, or a single callable. Each answer is a
    ``dict`` (sent as JSON), a ``str`` (sent verbatim, may be invalid JSON), an exception
    instance (raised as a transport error) or ``callable(system, user) -> dict | str``. The
    key ``"*"`` is the default for agents without their own entry. Every call is recorded in
    ``calls``.
    """

    backend = "fake"

    def __init__(
        self,
        responses: Mapping[str, Sequence[FakeResponse] | Callable[[str, str], Any]] | None = None,
        max_retries: int = 2,
        models: Sequence[str] = ("fake-model",),
        healthy: bool = True,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self._responses: dict[str, Any] = {}
        for key, value in (responses or {}).items():
            if isinstance(value, (list, tuple)):
                self._responses[key] = list(value)
            elif isinstance(value, (Mapping, str, BaseException)):
                self._responses[key] = [value]  # a single scripted answer
            else:
                self._responses[key] = value  # callable(system, user)
        self._positions: dict[str, int] = {}
        self._models = list(models)
        self._healthy = healthy
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    def _next(self, agent: str) -> Any:
        with self._lock:
            entry = self._responses.get(agent, self._responses.get("*"))
            if entry is None:
                raise _ChatError(f"FakeLLMClient has no scripted response for agent {agent!r}")
            if callable(entry) and not isinstance(entry, list):
                return entry
            if not entry:
                raise _ChatError(f"FakeLLMClient response list for {agent!r} is empty")
            pos = self._positions.get(agent, 0)
            self._positions[agent] = pos + 1
            return entry[min(pos, len(entry) - 1)]

    def _chat(
        self,
        agent: str,
        model: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        temperature: float,
        think: bool,
    ) -> tuple[str, int | None, int | None]:
        with self._lock:
            self.calls.append({
                "agent": agent, "model": model, "messages": [dict(m) for m in messages],
                "system": messages[0]["content"], "user": messages[1]["content"],
                "schema": schema.__name__, "temperature": temperature, "think": think,
            })
        item = self._next(agent)
        if callable(item) and not isinstance(item, BaseException):
            item = item(messages[0]["content"], messages[1]["content"])
        if isinstance(item, BaseException):
            raise item
        content = item if isinstance(item, str) else json.dumps(item)
        return content, len(messages[-1]["content"]) // 4, len(content) // 4

    def calls_for(self, agent: str) -> list[dict[str, Any]]:
        """Recorded calls for one agent."""
        return [c for c in self.calls if c["agent"] == agent]

    def healthcheck(self) -> bool:
        return self._healthy

    def list_models(self) -> list[str]:
        return list(self._models)


class RulesClient:
    """``llm_backend: rules`` - no model is ever called.

    ``generate`` returns ``ok=False, skipped=True`` so ``SubAgent.run`` returns its
    ``deterministic_report`` (status OK) instead of a FAILED report.
    """

    backend = "rules"

    def generate(
        self,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        *,
        temperature: float = 0.2,
        think: bool = False,
    ) -> LLMResult:
        return LLMResult(ok=False, skipped=True, model="rules", attempts=0,
                         error="rules backend: deterministic interpretation used")

    def healthcheck(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return []


def build_llm(settings: "Settings") -> LLMClient:
    """Build the configured backend (``settings.engine.llm_backend``)."""
    backend = settings.engine.llm_backend
    if backend == "rules":
        logger.info("LLM backend: rules (deterministic interpretation, no model calls)")
        return RulesClient()
    m = settings.models
    logger.info(f"LLM backend: ollama at {m.ollama_host} (profile {m.active_profile}, "
                f"max {m.max_concurrent_requests} concurrent request(s))")
    return OllamaClient(
        host=m.ollama_host,
        timeout=float(m.request_timeout_seconds),
        max_retries=m.max_retries,
        keep_alive=m.keep_alive,
        num_ctx=m.num_ctx,
        timeout_retries=m.timeout_retries,
        max_concurrent=m.max_concurrent_requests,
    )
