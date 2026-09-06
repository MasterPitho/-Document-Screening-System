"""
SQLAlchemy 2.x models for the Document Screening Engine.

Privacy rule: only screening risk metadata is persisted. Uploaded images, face
embeddings, raw MRZ strings, and passport numbers are NEVER stored. The only
document-identifying value kept is ``request_id`` (a correlation token), plus
redacted module state summaries in the ``factors`` JSON column.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    full_name = Column(String(120), nullable=False, default="")
    role = Column(String(20), nullable=False, default="officer")
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    tokens = relationship("AuthToken", back_populates="user",
                          cascade="all, delete-orphan")


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    user = relationship("User", back_populates="tokens")


class Screening(Base):
    """Persisted, privacy-preserving result of one document screening.

    The ``factors`` JSON column stores the risk-factor list (names, weights and
    short English details) as returned by the risk engine. This is screening
    metadata only - never raw MRZ text, names, or document numbers.
    """

    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True)
    request_id = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    processing_time_ms = Column(Integer, nullable=False)

    document_type = Column(String(32), nullable=False, default="UNKNOWN")
    mrz_status = Column(String(40), nullable=False)
    face_status = Column(String(40), nullable=False)
    face_similarity = Column(Float, nullable=True)
    tampering_status = Column(String(40), nullable=False)
    tampering_score = Column(Float, nullable=True)
    liveness_status = Column(String(40), nullable=False, default="NOT_CHECKED")
    liveness_score = Column(Float, nullable=True)

    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String(20), nullable=False)
    decision = Column(String(40), nullable=False)
    status_color = Column(String(10), nullable=False)

    module_states = Column(JSON, nullable=False)
    factors = Column(JSON, nullable=False, default=list)
    mrz_source = Column(String(10), nullable=False, default="none")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    factor_rows = relationship(
        "ScreeningFactor",
        back_populates="screening",
        cascade="all, delete-orphan",
        order_by="ScreeningFactor.id",
        passive_deletes=True,
    )
    audit_events = relationship(
        "AuditLog",
        back_populates="screening",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_screenings_request_id", request_id, unique=True),
        Index("ix_screenings_created_at", created_at),
        Index("ix_screenings_decision", decision),
        Index("ix_screenings_risk_level", risk_level),
    )


class ScreeningFactor(Base):
    """One risk factor surfaced by a screening, normalized for reporting."""

    __tablename__ = "screening_factors"

    id = Column(Integer, primary_key=True)
    screening_id = Column(
        Integer,
        ForeignKey("screenings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    factor_name = Column(String(40), nullable=False, index=True)
    severity = Column(String(12), nullable=False, default="MEDIUM")  # HIGH | MEDIUM | LOW
    weight = Column(Integer, nullable=False)
    description = Column(Text, nullable=False, default="")

    screening = relationship("Screening", back_populates="factor_rows")


class AuditLog(Base):
    """Audit trail events; privacy-safe, references screenings by id only."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    screening_id = Column(
        Integer,
        ForeignKey("screenings.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    event_type = Column(String(40), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
    request_id = Column(String(64), nullable=False)
    message = Column(Text, nullable=False, default="")

    screening = relationship("Screening", back_populates="audit_events")

    __table_args__ = (
        Index("ix_audit_logs_created_at", created_at),
        Index("ix_audit_logs_event_type", event_type),
    )
