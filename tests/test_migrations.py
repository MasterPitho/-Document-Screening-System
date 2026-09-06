"""Alembic migration correctness tests against fresh temp SQLite databases."""

from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent


def _config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("prepend_sys_path", str(REPO_ROOT))
    return cfg


def _fresh_path(tmp_path) -> str:
    return f"sqlite:///{(tmp_path / 'migration.db').as_posix()}"


def test_upgrade_head_creates_full_schema(monkeypatch, tmp_path):
    url = _fresh_path(tmp_path)
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = _config()

    command.upgrade(cfg, "head")

    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert {"users", "auth_tokens", "screenings", "screening_factors",
            "audit_logs"} <= tables

    scripts = ScriptDirectory.from_config(cfg)
    head = scripts.get_current_head()
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    assert version == head


def test_downgrade_base_removes_schema(monkeypatch, tmp_path):
    url = _fresh_path(tmp_path)
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = _config()

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    # On SQLite Alembic leaves an empty alembic_version table behind.
    engine = create_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert "screenings" not in tables
    assert "users" not in tables
    assert tables <= {"alembic_version"}


def test_migrations_are_in_sync_with_models(monkeypatch, tmp_path):
    """Upgrading to head must leave no schema drift vs Base.metadata."""
    url = _fresh_path(tmp_path)
    monkeypatch.setenv("DATABASE_URL", url)
    cfg = _config()

    command.upgrade(cfg, "head")
    command.check(cfg)
