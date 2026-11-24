"""Sub-agent registry (section 4: "Registry-based; add more agents later without code
changes elsewhere").

Agents register themselves with the ``@register_agent`` class decorator. Every module in
``agents/sub_agents/`` is auto-imported on first use, so dropping a new file there with a
registered ``SubAgent`` subclass is enough to add an agent (plus its prompt YAML and an
entry in the settings if it should be weighted / toggled).
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import TYPE_CHECKING, TypeVar

from loguru import logger

from config.schema import AGENT_NAMES
from core.enums import Mode

if TYPE_CHECKING:  # pragma: no cover
    from agents.base import SubAgent
    from config.schema import Settings

_REGISTRY: dict[str, "SubAgent"] = {}
_LOCK = threading.RLock()
_LOADED = False

T = TypeVar("T", bound=type)


def register_agent(cls: T) -> T:
    """Class decorator: instantiate ``cls`` and register it under ``cls.name``.

    Raises:
        ValueError: missing name or a *different* class already registered under that name.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} has no agent name")
    with _LOCK:
        existing = _REGISTRY.get(name)
        if existing is not None and type(existing).__qualname__ != cls.__qualname__:
            raise ValueError(f"agent name {name!r} already registered by {type(existing).__name__}")
        _REGISTRY[name] = cls()
    return cls


def load_builtin_agents() -> None:
    """Import every module of ``agents.sub_agents`` (idempotent). Import errors are logged."""
    global _LOADED
    with _LOCK:
        if _LOADED:
            return
        import agents.sub_agents as pkg

        for mod in pkgutil.iter_modules(pkg.__path__):
            if mod.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{pkg.__name__}.{mod.name}")
            except Exception as exc:  # noqa: BLE001 - one broken agent must not disable the rest
                logger.exception(f"Failed to import sub-agent module {mod.name}: {exc}")
        _LOADED = True
        missing = [n for n in AGENT_NAMES if n not in _REGISTRY]
        if missing:
            logger.error(f"Sub-agents missing from registry: {missing}")


def _order_key(name: str) -> tuple[int, str]:
    return (AGENT_NAMES.index(name) if name in AGENT_NAMES else len(AGENT_NAMES), name)


def all_agents() -> list["SubAgent"]:
    """Every registered agent, in ``AGENT_NAMES`` order (extra agents after, by name)."""
    load_builtin_agents()
    with _LOCK:
        return [_REGISTRY[n] for n in sorted(_REGISTRY, key=_order_key)]


def agent_names() -> list[str]:
    """Names of every registered agent (ordered like ``all_agents``)."""
    return [a.name for a in all_agents()]


def get_agent(name: str) -> "SubAgent":
    """Registered agent instance by name.

    Raises:
        KeyError: unknown agent.
    """
    load_builtin_agents()
    with _LOCK:
        if name not in _REGISTRY:
            raise KeyError(f"unknown sub-agent {name!r}")
        return _REGISTRY[name]


def agents_for_mode(mode: Mode, settings: "Settings") -> list["SubAgent"]:
    """Agents that are applicable to ``mode`` AND enabled (globally and for the mode)."""
    out: list["SubAgent"] = []
    for agent in all_agents():
        if not agent.applies_to(mode):
            continue
        enabled, _ = agent.enabled_for(settings, mode)
        if enabled:
            out.append(agent)
    return out
