"""
Document Screening & Verification Engine - entry point.

The implementation lives in the ``app`` package:
  - app/main.py            ASGI app factory + HTTP routes
  - app/services/mrz.py    strategy-based document parsers (TD3/TD1) + OCR
  - app/services/face_recognition.py  ArcFace embedding face verification
  - app/services/liveness.py          passive presentation attack screening
  - app/services/tampering.py         multi-signal image tampering analysis
  - app/services/risk_engine.py      deterministic, explainable risk scoring

This module is a compatibility shim that re-exports the ASGI application and
the shared rule-based MRZ helpers for tooling that imports ``main`` directly.

Screening signals are heuristic: nothing here is a legally definitive identity
decision, and a trained human officer remains the final decision maker.
"""

from __future__ import annotations

from app.config import get_settings
from app.main import app, create_app
from app.services.mrz import (
    calculate_icao_checksum,
    extract_mrz_from_image,
    mrz_char_value,
    mrz_year_full,
    parse_td3_mrz,
    verify_mrz_field,
)

settings = get_settings()
MAX_IMAGE_BYTES = settings.max_image_bytes
MAX_IMAGE_WIDTH = settings.max_image_width
MAX_IMAGE_HEIGHT = settings.max_image_height
MAX_IMAGE_PIXELS = settings.max_image_pixels
MRZ_CONFIDENCE_THRESHOLD = settings.mrz_confidence_threshold
FACE_SIMILARITY_THRESHOLD = settings.face_similarity_threshold
FACE_MATCH_THRESHOLD = settings.face_similarity_threshold
TAMPERING_THRESHOLD = settings.tampering_threshold
RISK_THRESHOLDS = (settings.risk_review_threshold, settings.risk_reject_threshold)
RISK_WEIGHTS = settings.risk_weights

__all__ = [
    "app",
    "create_app",
    "settings",
    "calculate_icao_checksum",
    "extract_mrz_from_image",
    "mrz_char_value",
    "mrz_year_full",
    "parse_td3_mrz",
    "verify_mrz_field",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_WIDTH",
    "MAX_IMAGE_HEIGHT",
    "MAX_IMAGE_PIXELS",
    "MRZ_CONFIDENCE_THRESHOLD",
    "FACE_SIMILARITY_THRESHOLD",
    "FACE_MATCH_THRESHOLD",
    "TAMPERING_THRESHOLD",
    "RISK_THRESHOLDS",
    "RISK_WEIGHTS",
]
