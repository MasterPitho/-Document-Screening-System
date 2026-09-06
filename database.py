"""
Database compatibility shim (legacy import surface).

The persistence layer now lives in ``app/db`` (models, engine, repositories).
This module re-exports the models and a module-level ``engine``/``SessionLocal``
bound to the configured ``DATABASE_URL`` so older tooling and tests that import
from ``database`` keep working. New code should import from ``app.db``.
"""

from __future__ import annotations

from app.config import get_settings
from app.db.database import build_database, get_db
from app.db.models import (
    AuthToken,
    AuditLog,
    Base,
    Screening,
    ScreeningFactor,
    User,
)

_settings = get_settings()
_database = build_database(_settings.database_url)

engine = _database.engine
SessionLocal = _database.session_factory()

# Backward-compatible alias for the pre-PostgreSQL era.
ScreeningRecord = Screening

DATABASE_URL = _settings.database_url


def init_db() -> None:
    """Create tables from model metadata (development / test convenience)."""
    _database.create_all()


__all__ = [
    "AuthToken",
    "AuditLog",
    "Base",
    "DATABASE_URL",
    "Screening",
    "ScreeningFactor",
    "ScreeningRecord",
    "SessionLocal",
    "User",
    "engine",
    "get_db",
    "init_db",
]
