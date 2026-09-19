"""
Append-only, privacy-preserving hash-chain ledger.

Records a SHA-256 digest of each screening's *risk metadata* (no names, no
document numbers, no raw MRZ text). The chain is deterministic: each entry
hashes its predecessor's hash plus canonical JSON of its payload, so any edit
to any past payload breaks verification.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from typing import Any

GENESIS_PREV_HASH = "0" * 64


def canonical_json(payload: dict) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_entry_hash(*, prev_hash: str, entry_type: str,
                       payload: dict, created_at: datetime.datetime) -> str:
    body = "|".join([prev_hash, entry_type, canonical_json(payload),
                     created_at.isoformat()])
    return sha256_hex(body)


def risk_payload_for_screening(screening: Any) -> dict:
    """Privacy-safe payload: risk metadata only, never PII or document numbers."""
    return {
        "risk_score": screening.risk_score,
        "risk_level": screening.risk_level,
        "decision": screening.decision,
        "status_color": screening.status_color,
        "document_type": screening.document_type,
        "mrz_status": screening.mrz_status,
        "face_status": screening.face_status,
        "tampering_status": screening.tampering_status,
        "liveness_status": screening.liveness_status,
    }