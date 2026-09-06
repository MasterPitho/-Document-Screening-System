"""
Pydantic response/request schemas for the screening API.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


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


class MRZValidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    detected: bool
    source: str
    status: str = "NOT_DETECTED"
    confidence: float = 0.0
    line1: Optional[str] = None
    line2: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    module_state: str = "NOT_AVAILABLE"


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


class LivenessResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    is_live: Optional[bool] = None
    liveness_score: Optional[float] = None
    liveness_status: str = "NOT_CHECKED"
    method: str = "NOT_CHECKED"
    model_used: Optional[str] = None
    signals: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    explanation: str = ""
    module_state: str = "NOT_AVAILABLE"


class RiskAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    status: str
    level: str
    decision: str
    factors: list[dict[str, Any]]
    reasons: list[str]
    module_statuses: dict[str, str]
    confidence: float
    explanation: str


class ScreenResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    request_id: str
    processing_time_ms: int
    document: dict[str, str]
    modules: dict[str, Any]
    risk_assessment: RiskAssessment
    mrz: MRZValidationResult
    tampering_analysis: TamperingResult
    face_verification: FaceVerificationResult
    liveness: LivenessResult = Field(default_factory=LivenessResult)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(min_length=5, max_length=255)
    full_name: str = Field(default="", max_length=120)
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
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
    user_id: Optional[int]
    created_at: str


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
