"""
Shared request/response helpers for the API layer.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from app.logging_setup import StructLogger

logger = StructLogger()

# Client IDs must be safe, short tokens to avoid log injection.
X_REQUEST_ID_MAX_LENGTH = 64


def get_request_id(request: Request) -> str:
    """Return a validated request id or generate a new UUID."""
    header_value = request.headers.get("X-Request-ID", "").strip()
    if header_value and len(header_value) <= X_REQUEST_ID_MAX_LENGTH:
        # Allow only URI-safe characters to avoid log/header injection.
        safe = "".join(c for c in header_value if c.isalnum() or c in "-_.")
        if safe:
            return safe[:X_REQUEST_ID_MAX_LENGTH]
    return uuid.uuid4().hex


_ERROR_CODES_BY_STATUS = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "FILE_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    500: "INTERNAL_ERROR",
    503: "DATABASE_UNAVAILABLE",
    401: "UNAUTHORIZED",
}


def structured_error(status_code: int, message: str, detail: object = None,
                     request_id: Optional[str] = None) -> JSONResponse:
    code = _ERROR_CODES_BY_STATUS.get(status_code, "REQUEST_FAILED")
    error: Dict[str, object] = {"code": code, "message": message}
    if detail is not None:
        error["detail"] = detail
    content: Dict[str, object] = {"success": False, "error": error}
    if request_id:
        content["request_id"] = request_id
    return JSONResponse(status_code=status_code, content=content)
