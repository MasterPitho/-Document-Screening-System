"""Watchlist document-number normalization helpers."""

from __future__ import annotations

import re

_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]")


def normalize_document_number(value: str) -> str:
    """Normalize a document number for exact matching (case + separators)."""
    return _NON_ALPHANUMERIC.sub("", (value or "").upper())