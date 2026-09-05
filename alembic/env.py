"""Alembic environment: migrate against the configured ``DATABASE_URL``.

The effective database URL is read from the ``DATABASE_URL`` environment
variable (this is what compose, tests, and production all set). ``alembic.ini``
only provides a fallback used when the environment variable is absent.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine, pool

from app.db.models import Base

config = context.config
target_metadata = Base.metadata

# Migrations must run against the same URL the application is configured with.
_DATABASE_URL = (
    os.getenv("DATABASE_URL", "").strip()
    or config.get_main_option("sqlalchemy.url", "")
    or "sqlite:///./document_screening.db"
)


def run_migrations_offline() -> None:
    """Emit SQL to stdout (no DB connection)."""
    context.configure(
        url=_DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    connectable = create_engine(_DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()