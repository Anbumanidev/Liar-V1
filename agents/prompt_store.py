"""Versioned prompt templates (section 16).

Prompts live in the ``prompt_templates`` table. ``agents/prompts/<agent>.yaml`` files are
only the *seed* (version 1). The dashboard edits prompts by inserting a new row with
``version = max + 1`` (``active=True``); a "revert" is a new version that copies an old one,
so history is never rewritten and every agent report records the exact version it used.

Resolution in ``get(agent, version)``: an explicit (pinned) version - from
``settings.agents.prompt_versions`` - wins; otherwise the highest version number.

Templates are Jinja2, rendered in a ``SandboxedEnvironment`` with ``StrictUndefined`` so a
typo in a dashboard-edited prompt fails loudly: ``render`` raises ``PromptRenderError`` and the
agent is FAILED (neutral vote / NO_TRADE), never silently fed an empty prompt. Seeds and
``save_version`` additionally reject templates that use variables the agent does not provide
(``allowed_variables``). Variables (system and user templates):

* Layer-1 sub-agents: ``agent, symbol, mode, as_of, payload_json, mode_rules, output_schema``
* ``bull`` / ``bear``: ``agent, symbol, mode, as_of, mode_rules, output_schema, facts_json,
  phase, round, total_rounds, opponent, opponent_last_json, transcript_json``
* ``meta``: ``agent, symbol, mode, as_of, mode_rules, output_schema, facts_json, options_json,
  selection_policy, enabled_modes``
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template, TemplateError, meta
from jinja2.sandbox import SandboxedEnvironment
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from config.schema import DEBATE_AGENTS, META_AGENT
from core.clock import Clock, SystemClock
from storage.models import PromptTemplate

DEFAULT_PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"
SEED_AUTHOR = "seed"
#: Max compiled templates kept per store (dashboard edits add new sources over time).
_MAX_COMPILED = 256
_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
#: Parsed + validated seed files keyed by path -> ((mtime_ns, size), data or None, error or None).
_SEED_CACHE: dict[str, tuple[tuple[int, int], dict[str, Any] | None, str | None]] = {}
_SEED_CACHE_LOCK = threading.Lock()

_COMMON_VARIABLES: frozenset[str] = frozenset({"agent", "symbol", "mode", "as_of", "mode_rules", "output_schema"})
#: Variables passed to every Layer-1 sub-agent prompt (``agents.base.SubAgent.prompt_vars``).
SUB_AGENT_VARIABLES: frozenset[str] = _COMMON_VARIABLES | {"payload_json"}
#: Variables passed to the bull / bear prompts (``agents.debate.debate``).
DEBATE_VARIABLES: frozenset[str] = _COMMON_VARIABLES | {
    "facts_json", "phase", "round", "total_rounds", "opponent", "opponent_last_json", "transcript_json",
}
#: Variables passed to the meta prompt (``agents.meta.meta``).
META_VARIABLES: frozenset[str] = _COMMON_VARIABLES | {"facts_json", "options_json", "selection_policy",
                                                      "enabled_modes"}


def allowed_variables(agent: str) -> frozenset[str]:
    """Template variables the runtime provides for ``agent``'s prompt."""
    if agent in DEBATE_AGENTS:
        return DEBATE_VARIABLES
    if agent == META_AGENT:
        return META_VARIABLES
    return SUB_AGENT_VARIABLES


class PromptNotFoundError(LookupError):
    """No prompt template exists for an agent (and none could be seeded)."""


class PromptRenderError(ValueError):
    """A template failed to render (undefined variable, sandbox violation, syntax error)."""


class PromptStore:
    """DB-backed, versioned prompt templates with a YAML seed.

    Args:
        session_factory: SQLAlchemy session factory of the trading DB.
        prompts_dir: directory of ``<agent>.yaml`` seed files.
        clock: time source for ``created_at`` (defaults to ``SystemClock``).
        cache_ttl_seconds: how long ``get`` results are cached in-process (0 disables).
            Dashboard edits become visible after at most this delay.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        prompts_dir: Path | str = DEFAULT_PROMPTS_DIR,
        clock: Clock | None = None,
        cache_ttl_seconds: float = 15.0,
    ) -> None:
        self._sf = session_factory
        self.prompts_dir = Path(prompts_dir)
        self.clock: Clock = clock or SystemClock()
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[tuple[str, int | None], tuple[float, PromptTemplate]] = {}
        self._lock = threading.Lock()
        # Sub-agents run in parallel threads; a single-connection engine (in-memory SQLite with
        # StaticPool: tests, replay) must never be used by two threads at once.
        self._db_lock = threading.RLock()
        self._compiled: dict[str, Template] = {}
        self._compile_lock = threading.Lock()
        self._env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False,
                                         keep_trailing_newline=False)

    # ------------------------------------------------------------------ seed

    def _parse_seed(self, path: Path) -> dict[str, Any]:
        """Parse and validate one seed file.

        Raises:
            ValueError: invalid content (wrong agent name, empty keys, bad template).
        """
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_YAML_LOADER) or {}
        if not isinstance(data, dict):
            raise ValueError("seed file must be a mapping")
        agent = str(data.get("agent") or path.stem)
        if agent != path.stem:
            raise ValueError(f"agent {agent!r} does not match file name {path.stem!r}")
        for key in ("system", "user_template"):
            if not isinstance(data.get(key), str) or not data[key].strip():
                raise ValueError(f"missing/empty key {key!r}")
        self.validate_template(data["system"], agent)
        self.validate_template(data["user_template"], agent)
        return data

    def _seed_files(self) -> dict[str, dict[str, Any]]:
        """Load and validate every ``*.yaml`` seed file; bad files are logged and skipped.

        Parsed files are cached per (mtime, size) so building many stores stays cheap.
        """
        out: dict[str, dict[str, Any]] = {}
        if not self.prompts_dir.is_dir():
            logger.warning(f"Prompt seed dir not found: {self.prompts_dir}")
            return out
        for path in sorted(self.prompts_dir.glob("*.yaml")):
            try:
                st = path.stat()
            except OSError as exc:
                logger.error(f"Cannot stat prompt seed {path.name}: {exc}")
                continue
            stamp = (st.st_mtime_ns, st.st_size)
            key = str(path.resolve())
            with _SEED_CACHE_LOCK:
                cached = _SEED_CACHE.get(key)
            if cached is None or cached[0] != stamp:
                try:
                    cached = (stamp, self._parse_seed(path), None)
                except Exception as exc:  # noqa: BLE001 - one bad seed must not block the others
                    cached = (stamp, None, f"{type(exc).__name__}: {exc}")
                with _SEED_CACHE_LOCK:
                    _SEED_CACHE[key] = cached
            _, data, error = cached
            if data is None:
                logger.error(f"Invalid prompt seed {path.name}: {error}")
                continue
            out[path.stem] = data
        return out

    def seed(self, agents: list[str] | None = None) -> int:
        """Insert YAML seeds as version 1 for every agent that has no rows yet.

        Args:
            agents: restrict seeding to these agents (default: every YAML file).

        Returns:
            Number of agents seeded.
        """
        files = self._seed_files()
        n = 0
        with self._db_lock, self._sf() as session:
            for agent, data in files.items():
                if agents is not None and agent not in agents:
                    continue
                exists = session.scalar(
                    select(func.count()).select_from(PromptTemplate).where(PromptTemplate.agent == agent)
                )
                if exists:
                    continue
                session.add(PromptTemplate(
                    agent=agent,
                    version=1,
                    system_prompt=data["system"].strip() + "\n",
                    user_template=data["user_template"].strip() + "\n",
                    created_at=self.clock.now(),
                    author=SEED_AUTHOR,
                    note=f"seed from {agent}.yaml (file version {data.get('version', 1)})",
                    active=True,
                ))
                n += 1
            try:
                session.commit()
            except IntegrityError:
                # Another process seeded concurrently: that's fine.
                session.rollback()
                logger.info("Prompt seeding raced with another writer; keeping existing rows")
                return 0
        if n:
            logger.info(f"Seeded {n} prompt template(s) from {self.prompts_dir}")
        self.invalidate()
        return n

    # ------------------------------------------------------------------ read

    def invalidate(self) -> None:
        """Drop the in-process cache (call after writes from this process)."""
        with self._lock:
            self._cache.clear()

    def get(self, agent: str, version: int | None = None) -> PromptTemplate:
        """Return the pinned ``version`` (when given and existing) else the highest version.

        A missing pinned version is logged and falls back to the latest. If the agent has no
        rows at all, its YAML seed is inserted first.

        Raises:
            PromptNotFoundError: no template exists and none could be seeded.
        """
        key = (agent, version)
        if self.cache_ttl_seconds > 0:
            with self._lock:
                hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < self.cache_ttl_seconds:
                return hit[1]
        tpl = self._load(agent, version)
        if tpl is None:
            self.seed([agent])
            tpl = self._load(agent, version)
        if tpl is None:
            raise PromptNotFoundError(f"no prompt template for agent {agent!r}")
        if self.cache_ttl_seconds > 0:
            with self._lock:
                self._cache[key] = (time.monotonic(), tpl)
        return tpl

    def _load(self, agent: str, version: int | None) -> PromptTemplate | None:
        with self._db_lock, self._sf() as session:
            if version is not None:
                row = session.scalar(select(PromptTemplate).where(
                    PromptTemplate.agent == agent, PromptTemplate.version == int(version)))
                if row is not None:
                    session.expunge(row)
                    return row
                logger.warning(f"Pinned prompt {agent} v{version} not found; using latest")
            row = session.scalar(
                select(PromptTemplate).where(PromptTemplate.agent == agent)
                .order_by(PromptTemplate.version.desc()).limit(1)
            )
            if row is not None:
                session.expunge(row)
            return row

    def versions(self, agent: str) -> list[PromptTemplate]:
        """All versions of ``agent``'s prompt, newest first."""
        with self._db_lock, self._sf() as session:
            rows = list(session.scalars(
                select(PromptTemplate).where(PromptTemplate.agent == agent)
                .order_by(PromptTemplate.version.desc())
            ))
            for r in rows:
                session.expunge(r)
            return rows

    # ------------------------------------------------------------------ write

    def save_version(
        self,
        agent: str,
        system: str,
        user_template: str,
        author: str,
        note: str | None = None,
    ) -> PromptTemplate:
        """Insert a new version (``max + 1``, active). Templates are syntax-checked first.

        Raises:
            ValueError: a template does not parse or uses variables ``agent`` does not provide.
        """
        self.validate_template(system, agent)
        self.validate_template(user_template, agent)
        for attempt in range(3):
            with self._db_lock, self._sf() as session:
                current = session.scalar(
                    select(func.max(PromptTemplate.version)).where(PromptTemplate.agent == agent)
                ) or 0
                row = PromptTemplate(
                    agent=agent, version=int(current) + 1, system_prompt=system,
                    user_template=user_template, created_at=self.clock.now(), author=author,
                    note=note, active=True,
                )
                session.add(row)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    logger.warning(f"Prompt version race for {agent} (attempt {attempt + 1}); retrying")
                    continue
                _ = row.version  # load attributes even if the factory expires on commit
                session.expunge(row)
                self.invalidate()
                logger.info(f"Saved prompt {agent} v{row.version} by {author}")
                return row
        raise RuntimeError(f"could not allocate a new prompt version for {agent!r}")

    def revert(self, agent: str, version: int, author: str) -> PromptTemplate:
        """Create a new version that copies ``version`` (history is never rewritten).

        Raises:
            PromptNotFoundError: ``version`` does not exist.
        """
        with self._db_lock, self._sf() as session:
            old = session.scalar(select(PromptTemplate).where(
                PromptTemplate.agent == agent, PromptTemplate.version == int(version)))
            if old is None:
                raise PromptNotFoundError(f"{agent} v{version} does not exist")
            system, user = old.system_prompt, old.user_template
        return self.save_version(agent, system, user, author, note=f"revert to v{version}")

    # ------------------------------------------------------------------ render

    def template_variables(self, template: str) -> set[str]:
        """Variables ``template`` reads from the render context (Jinja globals excluded).

        Raises:
            ValueError: the template does not parse.
        """
        try:
            ast = self._env.parse(template)
        except TemplateError as exc:
            raise ValueError(f"invalid prompt template: {exc}") from exc
        return set(meta.find_undeclared_variables(ast)) - set(self._env.globals)

    def validate_template(self, template: str, agent: str | None = None) -> None:
        """Raise ``ValueError`` if ``template`` is not valid Jinja2.

        With ``agent`` it also rejects variables the runtime does not provide for that agent
        (see ``allowed_variables``), so a typo is caught when saving instead of FAILING the
        agent at run time.
        """
        used = self.template_variables(template)
        if agent is not None:
            unknown = sorted(used - allowed_variables(agent))
            if unknown:
                raise ValueError(
                    f"prompt template for {agent!r} uses unknown variable(s) {unknown}; "
                    f"available: {sorted(allowed_variables(agent))}"
                )

    def render(self, template: str, **variables: Any) -> str:
        """Render ``template`` (sandboxed, StrictUndefined).

        Raises:
            PromptRenderError: missing variable, sandbox violation or template syntax error.
        """
        try:
            return self._compile(template).render(**variables).strip()
        except TemplateError as exc:
            raise PromptRenderError(f"{type(exc).__name__}: {exc}") from exc

    def _compile(self, template: str) -> Template:
        """Compiled ``template`` (cached; rendering a compiled template is thread-safe)."""
        compiled = self._compiled.get(template)
        if compiled is None:
            with self._compile_lock:
                compiled = self._compiled.get(template)
                if compiled is None:
                    compiled = self._env.from_string(template)
                    if len(self._compiled) >= _MAX_COMPILED:
                        self._compiled.clear()
                    self._compiled[template] = compiled
        return compiled

    def render_prompt(self, tpl: PromptTemplate, **variables: Any) -> tuple[str, str]:
        """Render both the system prompt and the user template of ``tpl``.

        Raises:
            PromptRenderError: naming the agent and prompt version that failed.
        """
        try:
            return self.render(tpl.system_prompt, **variables), self.render(tpl.user_template, **variables)
        except PromptRenderError as exc:
            raise PromptRenderError(f"prompt {tpl.agent} v{tpl.version} failed to render: {exc}") from exc
