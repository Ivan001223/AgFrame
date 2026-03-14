from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.infrastructure.utils.logging import get_logger

logger = get_logger("exception_handler")


def get_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id
    header_value = request.headers.get("X-Request-ID")
    request_id = header_value if header_value else str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


def build_error_content(
    *,
    request: Request,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": get_request_id(request),
        }
    }
    if details:
        payload["error"]["details"] = details
    return payload


async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"未捕获的异常: {type(exc).__name__}",
        extra={"path": request.url.path, "method": request.method, "request_id": get_request_id(request)},
    )
    return JSONResponse(
        status_code=500,
        content=build_error_content(
            request=request,
            code="internal_error",
            message="Internal server error",
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content=build_error_content(
            request=request,
            code="http_error",
            message=detail,
        ),
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {"loc": [str(part) for part in item.get("loc", [])], "msg": item.get("msg", ""), "type": item.get("type", "")}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=build_error_content(
            request=request,
            code="validation_error",
            message="Request validation failed",
            details=details,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
