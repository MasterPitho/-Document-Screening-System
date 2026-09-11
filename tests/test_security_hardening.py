"""Security hardening tests: upload constraints, path traversal, decompression bombs, and model lifecycle."""

import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.config import Settings
from app.security.image_validation import ImageValidationLimits
from app.services.face_recognition import ModelManager, DummyBackend, FaceDetectionResult
import numpy as np


def _jpeg_bytes(width=100, height=100):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(width=100, height=100):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_app()
    import uuid
    from app.api.auth import _generate_token, _hash_token
    import datetime
    from app.db.database import utcnow_naive

    user = app.state.user_repo.create(
        username=f"sec_officer_{uuid.uuid4().hex[:8]}",
        email=f"sec_{uuid.uuid4().hex[:8]}@agency.gov",
        full_name="Security Officer",
        role="officer",
        password_hash="testhash",
    )
    token = _generate_token()
    app.state.token_repo.create(
        user_id=user.id,
        token_hash=_hash_token(token),
        expires_at=utcnow_naive() + datetime.timedelta(hours=24),
    )
    return TestClient(app, headers={"Authorization": f"Bearer {token}"})


def test_path_traversal_filename_safe(client):
    """Filenames with path traversal sequences do not write to disk or cause errors."""
    traversal_filenames = [
        "../../../../etc/passwd.jpg",
        "..\\..\\..\\windows\\system32\\calc.png",
        "/var/log/syslog.webp",
        "....//....//doc.jpg",
    ]
    img = _jpeg_bytes()
    for filename in traversal_filenames:
        ext = filename.rsplit(".", 1)[-1]
        mime = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
        data = _png_bytes() if ext == "png" else img
        response = client.post(
            "/api/v1/screen",
            files={"document_image": (filename, data, mime)},
        )
        # Should process normally or validate safely, NOT 500 error or disk write
        assert response.status_code in (200, 400, 415)
        # Confirm no file was created on disk in cwd
        assert not os.path.exists(filename)


def test_filename_with_null_bytes(client):
    """Filename containing null byte does not cause internal server error."""
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc\x00traversal.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    # Fastapi/starlette might sanitize or pass through; response should not be unhandled 500
    assert response.status_code in (200, 400, 415, 422)


def test_decompression_bomb_rejected(client):
    """Images exceeding pixel dimensions are rejected with 413."""
    settings = create_app().state.settings
    validator = ImageValidationLimits(settings)

    # Construct an image exceeding max_pixels
    buf = io.BytesIO()
    w = settings.max_image_width + 10
    h = 100
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG")
    payload = buf.getvalue()

    with pytest.raises(Exception) as exc_info:
        validator.validate(payload, "image/jpeg", "Document image", "test.jpg")
    assert exc_info.value.status_code == 413


def test_spoofed_mime_type_rejected(client):
    """HTML / executable content spoofed as image/jpeg is rejected."""
    fake_jpeg = b"<html><script>alert('xss')</script></html>"
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", fake_jpeg, "image/jpeg")},
    )
    assert response.status_code in (400, 415)
    assert response.json()["error"]["code"] in ("BAD_REQUEST", "UNSUPPORTED_MEDIA_TYPE")


def test_model_manager_single_load_lifecycle():
    """ModelManager loads backend once and reuses it across requests."""
    backend = DummyBackend([FaceDetectionResult([0, 0, 10, 10], 0.95, None, np.zeros(128, dtype=np.float32))])
    manager = ModelManager(Settings.from_env(), backend=backend)

    assert manager.face_is_ready()
    assert manager.readiness()["face_recognition"] is True

    # Test singleton get_instance
    ModelManager._instance = None
    inst1 = ModelManager.get_instance(Settings.from_env())
    inst2 = ModelManager.get_instance()
    assert inst1 is inst2
    ModelManager._instance = None


def test_default_credentials_disabled_by_default(tmp_path, monkeypatch):
    """By default, predictable default credentials are NOT created in the database."""
    from app.api.auth import bootstrap_admin
    from app.db.database import Database
    from app.db.repositories import UserRepository

    db = Database(f"sqlite:///{(tmp_path / 'nobootstrap.db').as_posix()}")
    db.create_all()
    settings = Settings.from_env()

    bootstrap_admin(db, settings)
    repo = UserRepository(db)
    assert repo.get_by_username("officer") is None
    assert repo.get_by_username("LT-04") is None


def test_default_credentials_blocked_in_production(tmp_path, monkeypatch):
    """Even if DEV_BOOTSTRAP=true, production environment never creates default officer."""
    from app.api.auth import bootstrap_admin
    from app.db.database import Database
    from app.db.repositories import UserRepository

    monkeypatch.setenv("DEV_BOOTSTRAP", "true")
    monkeypatch.setenv("API_ENV", "production")
    settings = Settings.from_env()

    db = Database(f"sqlite:///{(tmp_path / 'prodbootstrap.db').as_posix()}")
    db.create_all()
    bootstrap_admin(db, settings)
    repo = UserRepository(db)
    assert repo.get_by_username("officer") is None
    assert repo.get_by_username("LT-04") is None


def test_anonymous_screen_request_returns_401():
    """Unauthenticated calls to /api/v1/screen must be rejected with 401."""
    app = create_app()
    unauth_client = TestClient(app)
    response = unauth_client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 401


def test_screening_rate_limit_exceeded_returns_429():
    """Exceeding screening rate limit per minute must return HTTP 429 with Retry-After header."""
    import uuid
    from dataclasses import replace
    from app.api.auth import _generate_token, _hash_token
    import datetime
    from app.db.database import utcnow_naive

    settings = replace(Settings.from_env(), screening_rate_limit_per_minute=2)
    app = create_app(settings=settings)
    user = app.state.user_repo.create(
        username=f"rate_officer_{uuid.uuid4().hex[:8]}",
        email=f"rate_{uuid.uuid4().hex[:8]}@agency.gov",
        full_name="Rate Officer",
        role="officer",
        password_hash="testhash",
    )
    token = _generate_token()
    app.state.token_repo.create(
        user_id=user.id,
        token_hash=_hash_token(token),
        expires_at=utcnow_naive() + datetime.timedelta(hours=24),
    )
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})

    # Request 1: success
    r1 = client.post("/api/v1/screen", files={"document_image": ("d1.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r1.status_code == 200
    # Request 2: success
    r2 = client.post("/api/v1/screen", files={"document_image": ("d2.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r2.status_code == 200
    # Request 3: rate limit exceeded (429)
    r3 = client.post("/api/v1/screen", files={"document_image": ("d3.jpg", _jpeg_bytes(), "image/jpeg")})
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_screening_concurrency_capacity_reached_returns_429():
    """When engine capacity is exhausted, immediately return HTTP 429 without queueing indefinitely."""
    import asyncio
    import uuid
    from dataclasses import replace
    from app.api.auth import _generate_token, _hash_token
    import datetime
    from app.db.database import utcnow_naive

    settings = replace(Settings.from_env(), max_concurrent_screenings=1)
    app = create_app(settings=settings)
    user = app.state.user_repo.create(
        username=f"conc_officer_{uuid.uuid4().hex[:8]}",
        email=f"conc_{uuid.uuid4().hex[:8]}@agency.gov",
        full_name="Conc Officer",
        role="officer",
        password_hash="testhash",
    )
    token = _generate_token()
    app.state.token_repo.create(
        user_id=user.id,
        token_hash=_hash_token(token),
        expires_at=utcnow_naive() + datetime.timedelta(hours=24),
    )
    client = TestClient(app, headers={"Authorization": f"Bearer {token}"})

    # Exhaust concurrency semaphore
    limiter = app.state.resource_limiter
    asyncio.run(limiter.acquire_slot())

    try:
        response = client.post("/api/v1/screen", files={"document_image": ("doc.jpg", _jpeg_bytes(), "image/jpeg")})
        assert response.status_code == 429
        assert "capacity reached" in response.json()["error"]["message"].lower()
        assert "Retry-After" in response.headers
    finally:
        limiter.release_slot()


