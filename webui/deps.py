"""Shared helpers for the route modules: context/user access, command enqueueing, JSON safety."""

from __future__ import annotations

import json
import math
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError

from core.enums import CommandType
from webui import queries as q
from webui.context import WebContext


def get_ctx(request: Request) -> WebContext:
    return request.app.state.ctx


def get_user(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if not user:  # the security middleware guarantees this for every /api route but /api/login
        raise HTTPException(status_code=401, detail="not authenticated")
    return str(user)


def sanitize(obj: Any) -> Any:
    """Replace NaN/inf floats with None recursively (strict JSON)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSON response that never emits NaN/Infinity (third-party analytics may contain them)."""

    def render(self, content: Any) -> bytes:
        return json.dumps(sanitize(content), ensure_ascii=False, allow_nan=False, default=str,
                          separators=(",", ":")).encode("utf-8")


def validation_message(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'payload'}: {e['msg']}" for e in exc.errors())


def enqueue(ctx: WebContext, user: str, command_type: CommandType, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate + enqueue a control command for the engine (audited by ``enqueue_command``)."""
    from control.commands import enqueue_command

    with ctx.session() as s:
        batch = q.current_batch(s)
        try:
            row = enqueue_command(s, command_type, payload, actor=user, batch_id=batch.id if batch else None,
                                  clock=ctx.clock)
        except ValidationError as exc:
            s.rollback()
            raise HTTPException(status_code=422, detail=f"Invalid {CommandType(command_type).value} request: "
                                                        f"{validation_message(exc)}") from exc
        s.commit()
        logger.info(f"[webui] {user} queued {row.command_type} #{row.id}")
        return {"command_id": row.id, "type": row.command_type, "status": row.status}
