"""Start the local Ollama server when the engine needs it and it is not running.

2026-11-26: the engine started at 08:33 before Ollama was up ("Failed to connect to Ollama"). With
``engine.auto_start_ollama`` the engine starts ``ollama serve`` itself (hidden, detached, output in
``runtime/ollama.log``) and waits until it answers. Local only - never a remote host.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def find_ollama() -> str | None:
    """Path of the ollama executable (PATH, then the default Windows install folder)."""
    exe = shutil.which("ollama")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cand = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if cand.exists():
            return str(cand)
    return None


def ensure_running(llm: Any, host: str, runtime_dir: Path, *, wait_seconds: float = 45.0) -> bool:
    """True if Ollama answers (starting it first when it is local and down). Never raises."""
    try:
        if llm.healthcheck():
            return True
    except Exception:  # noqa: BLE001
        pass
    if (urlparse(host).hostname or "") not in LOCAL_HOSTS:
        logger.warning(f"[ollama] {host} is not reachable and is not local - not starting anything")
        return False
    exe = find_ollama()
    if exe is None:
        logger.error("[ollama] not running and the ollama executable was not found - install it from ollama.com")
        return False
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log = open(runtime_dir / "ollama.log", "ab")  # noqa: SIM115 - handed to the child process
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen([exe, "serve"], stdout=log, stderr=log, stdin=subprocess.DEVNULL, creationflags=flags,
                         close_fds=True)
        logger.info(f"[ollama] was not running - started '{exe} serve' (log: {runtime_dir / 'ollama.log'})")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ollama] could not start ollama serve: {exc}")
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        time.sleep(1.5)
        try:
            if llm.healthcheck():
                logger.info("[ollama] is up")
                return True
        except Exception:  # noqa: BLE001
            pass
    logger.error(f"[ollama] still not answering after {wait_seconds:.0f}s")
    return False


__all__ = ["ensure_running", "find_ollama"]
