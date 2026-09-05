"""
Persistence layer for the Document Screening Engine.

The database is settings-driven and app-owned: ``create_app`` builds a
:class:`Database` from the configured ``DATABASE_URL`` and exposes it as
``app.state.database`` plus lightweight repositories. Sessions are created and
closed per operation inside the repositories, so no session is ever left open
by a request.

Startup never assumes the database is ready: schema initialization and the
readiness check are wrapped so a temporarily unavailable PostgreSQL simply
reports ``/ready`` as not ready instead of crashing the API.
"""

from __future__ import annotations

import datetime
import time
from collections.abc import Callable, Generator
from typing import Any, Optional

from fastapi import Request
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


def utcnow() -> datetime.datetime:
    """UTC-aware clock value used for record timestamps."""
    return datetime.datetime.now(datetime.timezone.utc)


def utcnow_naive() -> datetime.datetime:
    """UTC now without timezone info.

    Stored as the canonical ``created_at`` value. On SQLite the persisted
    string has no offset; on PostgreSQL the column is ``timestamptz`` and the
    session timezone is UTC, so naive-UTC values are interpreted correctly.
    """
    return utcnow().replace(tzinfo=None)


def is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql") or url.startswith("postgres+")


class Database:
    """Owns the SQLAlchemy engine and session factory for one app instance."""

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: int = 5,
        pool_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.url = url
        options: dict[str, Any] = {
            "echo": False,
            "future": True,
        }
        if is_postgres_url(url):
            # Survive PostgreSQL restarts under the compose stack.
            options.setdefault("pool_pre_ping", True)
            options.setdefault("pool_recycle", 1800)
            options.update({
                "pool_size": (pool_options or {}).get("pool_size", 5),
                "max_overflow": (pool_options or {}).get("max_overflow", 10),
            })
            options["connect_args"] = {"connect_timeout": connect_timeout_s}
        else:
            # SQLite is per-thread; the FastAPI test client runs in threads.
            options["connect_args"] = {"check_same_thread": False}

        self.engine: Engine = create_engine(url, **options)
        self._session_factory: sessionmaker[Session] = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, future=True,
        )

    # -- sessions ---------------------------------------------------------
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory

    def session(self) -> Session:
        return self._session_factory()

    # -- schema -----------------------------------------------------------
    def create_all(self, fail_silently: bool = False) -> bool:
        """Create tables from model metadata.

        Returns True on success. With ``fail_silently=True`` a failure returns
        False instead of raising; otherwise the underlying exception surfaces.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            return True
        except Exception:  # noqa: BLE001 - caller decides how to react
            if fail_silently:
                return False
            raise

    def ping(self) -> bool:
        """Lightweight connectivity probe: ``SELECT 1``."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False

    def dispose(self) -> None:
        try:
            self.engine.dispose()
        except Exception:  # noqa: BLE001
            pass


def build_database(
    database_url: str,
    *,
    connect_timeout_s: Optional[int] = None,
    pool_options: Optional[dict[str, Any]] = None,
) -> Database:
    return Database(
        database_url,
        connect_timeout_s=connect_timeout_s or 5,
        pool_options=pool_options,
    )


def _retry(
    attempts: int, delay_s: float, fn: Callable[..., object], *args, **kwargs
) -> tuple[bool, str]:
    """Run ``fn`` up to ``attempts`` times; return (ok, description)."""
    last_error = "unknown error"
    for _ in range(max(1, attempts)):
        try:
            fn(*args, **kwargs)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}"
            if delay_s:
                time.sleep(delay_s)
    return False, last_error


class DatabaseConnector:
    """Small startup helper for a graceful, retrying database bring-up."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def initialize(self, attempts: int = 5, delay_s: float = 1.0) -> tuple[bool, str]:
        # Do NOT pass fail_silently=True here: _retry must see the exception so
        # it can keep retrying and report failure accurately.
        ok, error = _retry(attempts, delay_s, self._database.create_all, False)
        if not ok:
            self._database.dispose()
        return ok, error


def get_db(request: Request) -> Generator[Session, None, None]:
    """FastAPI dependency: one short-lived session per request, always closed."""
    database: Database = request.app.state.database
    session = database.session()
    try:
        yield session
    finally:
        session.close()