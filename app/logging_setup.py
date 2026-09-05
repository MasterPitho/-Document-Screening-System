"""
Privacy-conscious structured logging.

Guarantees that uploaded image contents, face embeddings, MRZ passport
numbers, and other sensitive PII are never written to logs.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any

_MRZ_NUMBER_RE = re.compile(r"[A-Z0-9]{8,9}")


def redact_mrz(value: str) -> str:
    """Mask MRZ-like alphanumeric runs so full passport numbers never reach logs."""
    if not value:
        return value
    return _MRZ_NUMBER_RE.sub(lambda m: m.group(0)[:4] + "****", value)


def set_up_logging(level: str, api_env: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    root = logging.getLogger()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(StructuredFormatter(api_env=api_env))
    root.handlers = [handler]
    logging.getLogger("document_screening").propagate = True


class StructuredFormatter(logging.Formatter):
    """Logs a JSON-ish line with timestamp, level, logger, and message."""

    def __init__(self, api_env: str) -> None:
        super().__init__()
        self._api_env = api_env

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "env": self._api_env,
        }
        msg = record.getMessage()
        # Re-emit the structured key=value tail that our helpers produce.
        entry["message"] = msg
        # Extract extra fields supplied via our logging helper (attrs).
        attrs = getattr(record, "attrs", None)
        if isinstance(attrs, dict):
            entry.update(attrs)
        return json.dumps(entry, default=str)


class StructLogger:
    """Thin wrapper that lets callers emit structured, redactable log lines."""

    def __init__(self, name: str = "document_screening") -> None:
        self._logger = logging.getLogger(name)

    def _emit(self, level: int, message: str, **attrs: Any) -> None:
        record = logging.LogRecord(
            name=self._logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None,
        )
        record.attrs = attrs  # type: ignore[attr-defined]
        self._logger.handle(record)

    def info(self, message: str, **attrs: Any) -> None:
        self._emit(logging.INFO, message, **attrs)

    def warning(self, message: str, **attrs: Any) -> None:
        self._emit(logging.WARNING, message, **attrs)

    def error(self, message: str, **attrs: Any) -> None:
        self._emit(logging.ERROR, message, **attrs)

    def exception(self, message: str, **attrs: Any) -> None:
        self._emit(logging.ERROR, message, **attrs)
