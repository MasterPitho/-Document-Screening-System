"""
Configuration for the Document Screening Engine.

All tunables are read from environment variables (with sensible defaults) and
validated at startup so invalid configuration fails loudly instead of silently
behaving badly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int_value(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float_value(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _csv_value(name: str, legacy_name: str, default: str) -> list[str]:
    raw = os.getenv(name, os.getenv(legacy_name, default))
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_value(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str]
    api_env: str
    log_level: str

    # Upload security
    max_image_bytes: int
    max_image_pixels: int
    max_image_width: int
    max_image_height: int
    allowed_image_types: set[str]
    allowed_image_extensions: set[str]

    # MRZ
    mrz_confidence_threshold: float
    mrz_year_pivot: int

    # Face recognition
    face_similarity_threshold: float  # FACE_MATCH_THRESHOLD
    face_min_detection_confidence: float
    face_min_quality: float
    face_model_name: str
    face_det_size: int
    face_ctx_id: int
    face_models_dir: str

    # Passive liveness / presentation attack detection (PAD)
    liveness_enabled: bool
    liveness_model_path: str
    liveness_heuristic_enabled: bool
    liveness_spoof_threshold: float
    liveness_uncertain_threshold: float
    liveness_model_input_size: int
    liveness_model_ctx_id: int

    # Tampering
    tampering_threshold: float
    tampering_review_threshold: float

    # Risk
    risk_review_threshold: int
    risk_reject_threshold: int

    # Auth / DB
    auth_token_ttl_hours: int
    admin_username: str
    admin_email: str
    admin_password: str
    database_url: str

    # DB startup behaviour (PostgreSQL bring-up is not instant)
    db_connect_timeout: int
    db_connect_retries: int
    db_retry_delay: float

    risk_weights: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        default_size_mb = _int_value("MAX_FILE_SIZE_MB", 10)
        default_weights = {
            "TAMPERING_SUSPECTED": _int_value("RISK_TAMPERING", 40),
            "TAMPERING_INCONCLUSIVE": _int_value("RISK_TAMPERING_INCONCLUSIVE", 15),
            "FACE_MISMATCH": _int_value("RISK_FACE_MISMATCH", 35),
            "FACE_NOT_DETECTED": _int_value("RISK_FACE_NOT_DETECTED", 20),
            "FACE_LOW_CONFIDENCE": _int_value("RISK_FACE_LOW_CONFIDENCE", 15),
            "FACE_MULTIPLE": _int_value("RISK_FACE_MULTIPLE", 20),
            "MRZ_CHECKSUM_FAILURE": _int_value("RISK_MRZ_CHECKSUM", 20),
            "EXPIRED_DOCUMENT": _int_value("RISK_EXPIRED", 25),
            "MRZ_NOT_DETECTED": _int_value("RISK_MRZ_NOT_DETECTED", 20),
            "MRZ_LOW_CONFIDENCE": _int_value("RISK_MRZ_LOW_CONFIDENCE", 10),
            "IMAGE_QUALITY": _int_value("RISK_IMAGE_QUALITY", 10),
            "MODULE_ERROR": _int_value("RISK_MODULE_ERROR", 25),
            "UNKNOWN_MODULE": _int_value("RISK_UNKNOWN_MODULE", 15),
            "LIVENESS_FAILED": _int_value("RISK_LIVENESS_FAILED", 50),
            "LIVENESS_UNCERTAIN": _int_value("RISK_LIVENESS_UNCERTAIN", 15),
        }

        return cls(
            cors_origins=_csv_value(
                "CORS_ORIGINS", "ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173",
            ),
            api_env=os.getenv("API_ENV", "development").strip() or "development",
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
            max_image_bytes=_int_value("MAX_IMAGE_BYTES", default_size_mb * 1024 * 1024),
            max_image_pixels=_int_value("MAX_IMAGE_PIXELS", 40_000_000),
            max_image_width=_int_value("MAX_IMAGE_WIDTH", 10_000),
            max_image_height=_int_value("MAX_IMAGE_HEIGHT", 10_000),
            allowed_image_types={t.strip() for t in
                                 os.getenv("ALLOWED_IMAGE_TYPES",
                                           "image/jpeg,image/png,image/webp").split(",")
                                 if t.strip()},
            allowed_image_extensions={e.strip().lower() for e in
                                      os.getenv("ALLOWED_IMAGE_EXTENSIONS",
                                                "jpg,jpeg,png,webp").split(",")
                                      if e.strip()},
            mrz_confidence_threshold=_float_value("MRZ_CONFIDENCE_THRESHOLD", 0.70),
            mrz_year_pivot=_int_value("MRZ_YEAR_PIVOT", 50),
            face_similarity_threshold=_float_value(
                "FACE_MATCH_THRESHOLD",
                _float_value("FACE_SIMILARITY_THRESHOLD", 0.35),
            ),
            face_min_detection_confidence=_float_value("FACE_MIN_DETECTION_CONFIDENCE", 0.50),
            face_min_quality=_float_value("FACE_MIN_QUALITY", 0.20),
            face_model_name=os.getenv("FACE_MODEL_NAME", "buffalo_sc").strip() or "buffalo_sc",
            face_det_size=_int_value("FACE_DET_SIZE", 640),
            face_ctx_id=_int_value("FACE_CTX_ID", -1),
            face_models_dir=os.getenv("FACE_MODELS_DIR", "~/.insightface").strip(),
            liveness_enabled=_bool_value("LIVENESS_ENABLED", True),
            liveness_model_path=os.getenv("LIVENESS_MODEL_PATH", "").strip(),
            liveness_heuristic_enabled=_bool_value("LIVENESS_HEURISTIC_ENABLED", True),
            liveness_spoof_threshold=_float_value("LIVENESS_SPOOF_THRESHOLD", 0.40),
            liveness_uncertain_threshold=_float_value("LIVENESS_UNCERTAIN_THRESHOLD", 0.60),
            liveness_model_input_size=_int_value("LIVENESS_MODEL_INPUT_SIZE", 160),
            liveness_model_ctx_id=_int_value("LIVENESS_MODEL_CTX_ID", -1),
            tampering_threshold=_float_value("TAMPERING_THRESHOLD", 70.0),
            tampering_review_threshold=_float_value("TAMPERING_REVIEW_THRESHOLD", 45.0),
            risk_review_threshold=_int_value(
                "RISK_REVIEW_THRESHOLD",
                _int_value("RISK_MEDIUM_THRESHOLD", 35),
            ),
            risk_reject_threshold=_int_value(
                "RISK_REJECT_THRESHOLD",
                _int_value("RISK_HIGH_THRESHOLD", 65),
            ),
            risk_weights=default_weights,
            auth_token_ttl_hours=_int_value("AUTH_TOKEN_TTL_HOURS", 24),
            admin_username=os.getenv("ADMIN_USERNAME", "").strip(),
            admin_email=os.getenv("ADMIN_EMAIL", "").strip(),
            admin_password=os.getenv("ADMIN_PASSWORD", "").strip(),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./document_screening.db").strip(),
            db_connect_timeout=_int_value("DB_CONNECT_TIMEOUT", 5),
            db_connect_retries=_int_value("DB_CONNECT_RETRIES", 5),
            db_retry_delay=_float_value("DB_RETRY_DELAY", 1.0),
        )

    def validate(self) -> None:
        """Raise ValueError on invalid configuration; never silently proceed."""

        def _check(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(message)

        _check(self.max_image_bytes > 0, "MAX_IMAGE_BYTES must be greater than zero")
        _check(self.max_image_pixels > 0, "MAX_IMAGE_PIXELS must be greater than zero")
        _check(self.max_image_width > 0 and self.max_image_height > 0,
               "MAX_IMAGE_WIDTH and MAX_IMAGE_HEIGHT must be greater than zero")
        _check(bool(self.allowed_image_types), "ALLOWED_IMAGE_TYPES cannot be empty")
        _check(bool(self.allowed_image_extensions), "ALLOWED_IMAGE_EXTENSIONS cannot be empty")
        _check(0.0 <= self.mrz_confidence_threshold <= 1.0,
               "MRZ_CONFIDENCE_THRESHOLD must be between 0 and 1")
        _check(0 <= self.mrz_year_pivot <= 100, "MRZ_YEAR_PIVOT must be between 0 and 100")
        _check(0.0 <= self.face_similarity_threshold <= 1.0,
               "FACE_MATCH_THRESHOLD must be between 0 and 1")
        _check(0.0 <= self.face_min_detection_confidence <= 1.0,
               "FACE_MIN_DETECTION_CONFIDENCE must be between 0 and 1")
        _check(0.0 <= self.face_min_quality <= 1.0, "FACE_MIN_QUALITY must be between 0 and 1")
        _check(self.face_ctx_id >= 0 or self.face_ctx_id == -1,
               "FACE_CTX_ID must be a valid device id or -1 for CPU")
        _check(bool(self.face_models_dir), "FACE_MODELS_DIR cannot be empty")
        _check(0.0 <= self.liveness_spoof_threshold < self.liveness_uncertain_threshold <= 1.0,
               "Liveness thresholds must satisfy 0 <= spoof < uncertain <= 1")
        _check(self.liveness_model_input_size > 0,
               "LIVENESS_MODEL_INPUT_SIZE must be greater than zero")
        _check(self.liveness_model_ctx_id >= 0 or self.liveness_model_ctx_id == -1,
               "LIVENESS_MODEL_CTX_ID must be a valid device id or -1 for CPU")
        _check(0.0 <= self.tampering_threshold <= 100.0,
               "TAMPERING_THRESHOLD must be between 0 and 100")
        _check(0.0 <= self.tampering_review_threshold <= 100.0,
               "TAMPERING_REVIEW_THRESHOLD must be between 0 and 100")
        _check(0 <= self.risk_review_threshold < self.risk_reject_threshold <= 100,
               "Risk thresholds must satisfy 0 <= review < reject <= 100")
        _check(all(w >= 0 for w in self.risk_weights.values()),
               "Risk weights cannot be negative")
        _check(self.db_connect_timeout >= 0, "DB_CONNECT_TIMEOUT cannot be negative")
        _check(self.db_connect_retries >= 0, "DB_CONNECT_RETRIES cannot be negative")
        _check(self.db_retry_delay >= 0, "DB_RETRY_DELAY cannot be negative")

        required_factors = {
            "TAMPERING_SUSPECTED", "TAMPERING_INCONCLUSIVE", "FACE_MISMATCH",
            "FACE_NOT_DETECTED", "FACE_LOW_CONFIDENCE", "FACE_MULTIPLE",
            "MRZ_CHECKSUM_FAILURE", "EXPIRED_DOCUMENT", "MRZ_NOT_DETECTED",
            "MRZ_LOW_CONFIDENCE", "IMAGE_QUALITY", "MODULE_ERROR", "UNKNOWN_MODULE",
            "LIVENESS_FAILED", "LIVENESS_UNCERTAIN",
        }
        _check(required_factors.issubset(self.risk_weights.keys()),
               "Risk weight mapping is missing required factors")


def get_settings() -> Settings:
    return Settings.from_env()
