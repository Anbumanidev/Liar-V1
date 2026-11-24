"""Settings versioning (section 14.2).

* Every settings change creates a new ``settings_versions`` row with a full snapshot and a
  diff against the previous version.
* The latest row is the active configuration. ``revert_to`` creates a NEW version that copies
  an older snapshot (history is append-only).
* Changes apply to NEW trades; open positions keep the settings frozen on them.
* Every save is validated against the hard caps and audit-logged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.loader import (
    assert_within_caps,
    load_default_settings,
    settings_from_dict,
    settings_to_dict,
    update_settings,
)
from config.schema import HardCaps, Settings
from core.clock import IST, Clock
from core.logging import log_event
from storage.audit import audit
from storage.models import SettingsVersion


def _now(clock: Clock | None) -> datetime:
    return clock.now() if clock is not None else datetime.now(IST)


def diff_settings(old: Any, new: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Recursive diff of two JSON-like structures -> ``[{path, old, new}]``."""
    changes: list[dict[str, Any]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in old:
                changes.append({"path": path, "old": None, "new": new[key]})
            elif key not in new:
                changes.append({"path": path, "old": old[key], "new": None})
            else:
                changes.extend(diff_settings(old[key], new[key], path))
    elif old != new:
        changes.append({"path": prefix, "old": old, "new": new})
    return changes


def latest_version(session: Session) -> SettingsVersion | None:
    """Most recent settings_versions row (the active config) or None."""
    return session.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc()).limit(1)).scalar_one_or_none()


def ensure_seeded(session: Session, caps: HardCaps, *, clock: Clock | None = None) -> SettingsVersion:
    """Create version 1 from the YAML files if the table is empty. Returns the active row."""
    row = latest_version(session)
    if row is not None:
        return row
    settings = load_default_settings()
    assert_within_caps(settings, caps)
    row = SettingsVersion(
        created_at=_now(clock),
        author="system",
        note="Seeded from config/*.yaml",
        settings_json=settings_to_dict(settings),
        diff_json=[],
    )
    session.add(row)
    session.flush()
    audit(session, "system", "SETTINGS_SEED", f"settings_version:{row.id}", {}, clock=clock)
    return row


def get_active_settings(session: Session, caps: HardCaps, *, clock: Clock | None = None) -> tuple[int, Settings]:
    """Return ``(version_id, Settings)`` of the active (latest) version, seeding if needed."""
    row = ensure_seeded(session, caps, clock=clock)
    return row.id, settings_from_dict(row.settings_json)


def get_version(session: Session, version_id: int) -> Settings:
    """Load a specific settings version."""
    row = session.get(SettingsVersion, version_id)
    if row is None:
        raise KeyError(f"settings version {version_id} not found")
    return settings_from_dict(row.settings_json)


def save_settings(
    session: Session,
    settings: Settings,
    caps: HardCaps,
    *,
    author: str,
    note: str | None = None,
    batch_id: int | None = None,
    reverted_from_id: int | None = None,
    clock: Clock | None = None,
) -> SettingsVersion:
    """Validate against caps and store a new version (no-op if nothing changed).

    Raises:
        HardCapViolationError: if any hard cap is exceeded.
    """
    assert_within_caps(settings, caps)
    current = ensure_seeded(session, caps, clock=clock)
    new_json = settings_to_dict(settings)
    changes = diff_settings(current.settings_json, new_json)
    if not changes and reverted_from_id is None:
        return current
    row = SettingsVersion(
        batch_id=batch_id,
        created_at=_now(clock),
        author=author,
        note=note,
        settings_json=new_json,
        diff_json=changes,
        reverted_from_id=reverted_from_id,
    )
    session.add(row)
    session.flush()
    audit(
        session,
        author,
        "SETTINGS_REVERT" if reverted_from_id else "SETTINGS_CHANGE",
        f"settings_version:{row.id}",
        {"note": note, "changes": changes, "reverted_from": reverted_from_id},
        batch_id=batch_id,
        clock=clock,
    )
    log_event("settings", {"version_id": row.id, "author": author, "note": note, "changes": changes}, clock=clock)
    return row


def apply_changes(
    session: Session,
    changes: dict[str, Any],
    caps: HardCaps,
    *,
    author: str,
    note: str | None = None,
    batch_id: int | None = None,
    clock: Clock | None = None,
) -> SettingsVersion:
    """Apply ``{dotted.path: value}`` changes to the active settings and save a new version."""
    _vid, current = get_active_settings(session, caps, clock=clock)
    updated = update_settings(current, changes)
    return save_settings(session, updated, caps, author=author, note=note, batch_id=batch_id, clock=clock)


def revert_to(
    session: Session,
    version_id: int,
    caps: HardCaps,
    *,
    author: str,
    batch_id: int | None = None,
    clock: Clock | None = None,
) -> SettingsVersion:
    """Create a new version that restores ``version_id``'s snapshot."""
    old = get_version(session, version_id)
    return save_settings(
        session, old, caps, author=author, note=f"Revert to version {version_id}",
        batch_id=batch_id, reverted_from_id=version_id, clock=clock,
    )


def list_versions(session: Session, limit: int = 100) -> list[SettingsVersion]:
    """Most recent versions first."""
    return list(
        session.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc()).limit(limit)).scalars()
    )
