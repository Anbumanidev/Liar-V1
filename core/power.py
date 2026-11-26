"""Keep the computer awake while the engine runs (Windows), so cycles and the dashboard never stop.

2026-11-26: the PC went to sleep around 10:20-10:47, so five 5-minute cycles were missed and the
dashboard stopped updating. ``SetThreadExecutionState`` with ES_CONTINUOUS tells Windows not to
sleep (and, optionally, not to turn the display off) for as long as the calling thread runs or
until :func:`release` is called. It does not override closing a laptop lid - set "When I close the
lid: Do nothing" in Windows power settings if you close it. No-op on other operating systems.
"""

from __future__ import annotations

import sys

from loguru import logger

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002
ES_AWAYMODE_REQUIRED = 0x00000040


def _set(flags: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        prev = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))  # type: ignore[attr-defined]
        return bool(prev)
    except Exception as exc:  # noqa: BLE001 - power management is best effort
        logger.warning(f"[power] SetThreadExecutionState failed: {exc}")
        return False


def keep_awake(*, display: bool = True) -> bool:
    """Prevent sleep (and the display turning off when ``display``) until :func:`release`.

    Must be called from a long-lived thread (the engine's main loop). Returns True if applied.
    """
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | (ES_DISPLAY_REQUIRED if display else 0)
    ok = _set(flags)
    if ok:
        logger.info(f"[power] keeping this PC awake while the engine runs (display {'on' if display else 'may sleep'})")
    return ok


def release() -> bool:
    """Allow normal sleep again."""
    return _set(ES_CONTINUOUS)


__all__ = ["keep_awake", "release"]
