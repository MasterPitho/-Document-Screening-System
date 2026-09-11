"""
Tests for CORS configuration and security hardening across development and production environments.
"""

import pytest
from app.config import Settings
from app.main import create_app
from app.services.face_recognition import DummyBackend, FaceDetectionResult, ModelManager
import numpy as np
from starlette.testclient import TestClient


def _dummy_manager(settings):
    dummy_face = FaceDetectionResult(
        bbox=[10, 20, 110, 140], score=0.9, landmarks=None,
        embedding=np.ones((512,), dtype=np.float32),
    )
    return ModelManager(settings, backend=DummyBackend([dummy_face]))


def test_production_forbids_wildcard_cors_with_credentials(monkeypatch, tmp_path):
    """In production, wildcard origin '*' combined with credentials must fail validation."""
    db_path = f"sqlite:///{(tmp_path / 'test_cors_prod.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_path)
    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    settings = Settings.from_env()
    with pytest.raises(ValueError, match="Wildcard origin '\\*' with credentials is not permitted in production"):
        settings.validate()


def test_cors_allows_configured_origin(monkeypatch, tmp_path):
    """Configured origins should receive proper CORS headers."""
    db_path = f"sqlite:///{(tmp_path / 'test_cors_origin.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_path)
    monkeypatch.setenv("API_ENV", "production")
    monkeypatch.setenv("CORS_ORIGINS", "https://frontend.example.com,https://app.sentinel.gov")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    settings = Settings.from_env()
    settings.validate()
    app = create_app(settings=settings, model_manager=_dummy_manager(settings))
    client = TestClient(app)

    # Allowed origin preflight
    resp = client.options(
        "/api/v1/stats",
        headers={
            "Origin": "https://frontend.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://frontend.example.com"
    assert resp.headers.get("access-control-allow-credentials") == "true"

    # Disallowed origin preflight
    bad_resp = client.options(
        "/api/v1/stats",
        headers={
            "Origin": "https://malicious.attacker.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert bad_resp.headers.get("access-control-allow-origin") is None
