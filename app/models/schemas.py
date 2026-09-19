"""
Pydantic response/request schemas for the screening API.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FaceVerificationResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    operator_name: Optional[str] = None
    similarity_score: Optional[float] = None
    threshold: Optional[float] = None
    matched: Optional[bool] = None
    face_detected_in_document: Optional[bool] = None
    face_detected_in_live: Optional[bool] = None
    face_bounding_box: Optional[dict[str, int]] = None
    explanation: str = ""
    module_state: str = "NOT_AVAILABLE"


class PrivacySafeMRZResult(BaseModel):
    """Zero-PII MRZ verification response schema.

    Exposes structural validity and verification outcomes only. Never returns
    names, passport/document numbers, birth dates, nationality, or raw lines.
    """
    model_config = ConfigDict(extra="ignore")
    detected: bool
    valid: bool = False
    status: str = "NOT_DETECTED"
    source: str = "none"
    document_type: str = "UNKNOWN"
    checksum_valid: bool = False
    format_valid: bool = False
    fields_detected: int = 0
    issues: list[str] = Field(default_factory=list)
    module_state: str = "NOT_AVAILABLE"
    confidence: float = 0.0


class MRZValidationResult(PrivacySafeMRZResult):
    """Backward-compatible alias for privacy-safe MRZ results."""
    pass



class TamperingResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    score: float = 0.0
    confidence: float = 0.0
    signals: dict[str, Any] = Field(default_factory=dict)
    suspicious_regions: list[dict[str, Any]] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    operator_name: Optional[str] = None
    module_state: str = "NOT_AVAILABLE"


class LivenessResultSchema(BaseModel):
    """Passive liveness (PAD) result attached to a screening response.

    ``status`` / ``score`` / ``checked`` / ``detail`` are the canonical
    fields. The legacy ``liveness_*`` / ``is_live`` / ``module_state``
    keys are retained for backward compatibility with existing clients and
    are populated from the same service result (see ``from_service``).
    """

    model_config = ConfigDict(extra="allow")

    status: str = "NOT_CHECKED"
    score: float = Field(0.0, ge=0.0, le=1.0)
    checked: bool = False
    detail: Optional[str] = None

    is_live: Optional[bool] = None
    liveness_score: Optional[float] = None
    liveness_status: str = "NOT_CHECKED"
    method: str = "NOT_CHECKED"
    model_used: Optional[str] = None
    signals: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""
    module_state: str = "NOT_AVAILABLE"

    @classmethod
    def from_service(cls, result: dict[str, Any]) -> "LivenessResultSchema":
        status = str(result.get("liveness_status", "NOT_CHECKED"))
        return cls(
            status=status,
            score=float(result.get("liveness_score") or 0.0),
            checked=status in {"LIVE", "SPOOF_DETECTED", "UNCERTAIN"},
            detail=result.get("explanation"),
            **result,
        )


LivenessResult = LivenessResultSchema  # backward-compatible alias


class RiskAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: int = Field(ge=0, le=100)
    status: str
    level: str
    decision: str
    factors: list[dict[str, Any]]
    reasons: list[str]
    module_statuses: dict[str, str]
    risk_normalized: float
    confidence: Optional[float] = None
    explanation: str


class ScreenResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    request_id: str
    processing_time_ms: int
    document: dict[str, Any]
    modules: dict[str, Any]
    risk_assessment: RiskAssessment
    mrz: MRZValidationResult
    tampering_analysis: TamperingResult
    face_verification: FaceVerificationResult
    liveness: Optional[LivenessResultSchema] = Field(
        default_factory=lambda: LivenessResultSchema(
            status="NOT_CHECKED", score=0.0, checked=False))


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=5, max_length=255)
    full_name: str = Field(default="", max_length=120)
    role: str = Field(default="officer", max_length=20)
    officer_id: Optional[str] = Field(default=None, max_length=50)
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    officer_id: Optional[str] = None
    created_at: str


class RegisterResponse(UserOut):
    pass


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    token_type: str = "bearer"
    user: UserOut


class ScreeningRecordOut(BaseModel):
    id: int
    request_id: str
    risk_score: int
    risk_level: str
    decision: str
    status_color: str
    module_states: dict[str, str]
    factors: list[dict[str, Any]]
    processing_time_ms: int
    document_type: str
    mrz_status: Optional[str] = None
    face_status: Optional[str] = None
    face_similarity: Optional[float] = None
    tampering_status: Optional[str] = None
    tampering_score: Optional[float] = None
    liveness_status: Optional[str] = None
    liveness_score: Optional[float] = None
    mrz_source: str
    user_id: Optional[int] = None
    notes: Optional[str] = None
    created_at: str


class TimelinePoint(BaseModel):
    timestamp: str      # ISO datetime or formatted label like "06:00", "12:00"
    risk_score: float   # average risk score in this bucket
    count: int          # total screenings in this bucket


class DashboardTrendResponse(BaseModel):
    range: str
    today_total: int
    yesterday_total: int
    delta_today_pct: float            # e.g. +18.4
    avg_processing_time_ms: float
    avg_time_delta_pct: float         # e.g. -6.8
    average_risk_score: float         # average across the range
    high_risk_events_count: int       # count of events where risk_score >= threshold
    timeline: list[TimelinePoint]     # 5-10 time intervals for the graph


class DecisionUpdateRequest(BaseModel):
    decision: str                     # "CLEARED", "REVIEW", "HOLD", "SECONDARY_INSPECTION"
    notes: Optional[str] = None
    review_notes: Optional[str] = None

    @field_validator("notes", "review_notes")
    @classmethod
    def validate_no_pii_in_notes(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return v
        # 12-digit Aadhaar pattern (continuous or 4-4-4)
        if re.search(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", v):
            raise ValueError("Notes cannot contain sensitive Aadhaar identity numbers (PII protection).")
        # 10-char PAN pattern
        if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", v):
            raise ValueError("Notes cannot contain sensitive PAN card numbers (PII protection).")
        # 16-digit payment card numbers
        if re.search(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", v):
            raise ValueError("Notes cannot contain payment card numbers (PII protection).")
        return v


class NotificationItem(BaseModel):
    id: str
    type: str
    message: str
    created_at: str
    read: bool = False


class WatchlistItem(BaseModel):
    id: int
    name: str
    document_number: str
    reason: str
    severity: str
    created_at: str
    is_demo_data: bool = True
    source: str = "DEMO_DATA_NOT_FOR_OPERATIONAL_USE"


class WatchlistCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    document_number: str = Field(min_length=3, max_length=64)
    reason: str = Field(default="", max_length=500)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"


class WatchlistUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=2, max_length=120)
    document_number: Optional[str] = Field(default=None, min_length=3, max_length=64)
    reason: Optional[str] = Field(default=None, max_length=500)
    severity: Optional[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]] = None


class ScreeningFactorOut(BaseModel):
    id: int
    factor_name: str
    severity: str
    weight: int
    description: str


class ScreeningListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    decision: Optional[str] = None
    risk_level: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    records: list[ScreeningRecordOut]


class ScreeningStats(BaseModel):
    total: int
    cleared: int
    secondary_inspection: int
    high_risk: int
    mrz_failures: int
    face_mismatches: int
    suspicious_tampering: int
    by_decision: dict[str, int]
    by_risk_level: dict[str, int]


class ReportSummary(BaseModel):
    total_screenings: int
    cleared: int
    secondary_inspection: int
    high_risk: int
    avg_processing_time_ms: float
    by_risk_level: dict[str, int]
    by_decision: dict[str, int]


class PersistenceStatus(BaseModel):
    status: str
    screening_id: Optional[int] = None
