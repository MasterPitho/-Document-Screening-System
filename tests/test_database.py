"""Audit database, authentication, and reporting endpoint tests."""

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.api.auth import _is_expired, verify_password
from app.db.database import utcnow, utcnow_naive
from app.db.models import Screening, User
from app.main import create_app
from app.services import mrz as mrz_mod
from app.services.face_recognition import DummyBackend, ModelManager
from database import SessionLocal

skip_client = pytest.mark.skipif(
    TestClient is None, reason="FastAPI test client dependency unavailable"
)

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


class CleanTampering:
    def analyze(self, image_bytes):
        return {"status": "CLEAN", "score": 0.0, "confidence": 0.0,
                "signals": {}, "suspicious_regions": [], "explanation": []}


def _app():
    settings = create_app().state.settings
    manager = ModelManager(settings, backend=DummyBackend([]))
    app = create_app(model_manager=manager)
    app.state.tampering = CleanTampering()
    return app


def jpeg_bytes(width=800, height=500):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def _unique(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _register_and_login(client, username, password="SuperSecret123!"):
    email = f"{username}@example.com"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "full_name": "Test Officer", "password": password},
    )
    return reg, client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Registration and login
# ---------------------------------------------------------------------------

def test_token_expiry_handles_naive_and_aware_datetimes():
    import datetime

    # Naive (SQLite) expiry, as produced at login time.
    assert not _is_expired(utcnow_naive() + datetime.timedelta(hours=1))
    assert _is_expired(utcnow_naive() - datetime.timedelta(hours=1))
    assert not _is_expired(None)

    # Timezone-aware (PostgreSQL timestamptz) expiry, as read back from PG.
    aware = utcnow() + datetime.timedelta(hours=1)
    assert aware.tzinfo is not None
    assert not _is_expired(aware)
    past_aware = utcnow() - datetime.timedelta(hours=1)
    assert _is_expired(past_aware)


def test_register_and_login_flow():
    client = TestClient(_app())
    username = _unique("officer")
    reg, login = _register_and_login(client, username)
    assert reg.status_code == 201
    assert reg.json()["role"] == "officer"
    assert login.status_code == 200
    body = login.json()
    assert body["token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == username


def test_register_rejects_duplicate_username():
    client = TestClient(_app())
    username = _unique("dup")
    first, _ = _register_and_login(client, username)
    assert first.status_code == 201
    second = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"other_{username}@example.com",
              "full_name": "", "password": "SuperSecret123!"},
    )
    assert second.status_code == 409


def test_register_rejects_short_password():
    client = TestClient(_app())
    response = client.post(
        "/api/v1/auth/register",
        json={"username": _unique("weak"), "email": "weak@example.com",
              "full_name": "", "password": "short"},
    )
    assert response.status_code == 422


def test_login_rejects_bad_credentials():
    client = TestClient(_app())
    username = _unique("badlogin")
    _register_and_login(client, username)
    login = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "WrongPassword1!"},
    )
    assert login.status_code == 401


def test_me_requires_valid_token():
    client = TestClient(_app())
    username = _unique("me")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    me = client.get("/api/v1/auth/me", headers=_auth_headers(token))
    assert me.status_code == 200
    assert me.json()["username"] == username
    unauthorized = client.get("/api/v1/auth/me")
    assert unauthorized.status_code == 401


def test_logout_revokes_token():
    client = TestClient(_app())
    username = _unique("logout")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    assert client.post("/api/v1/auth/logout", headers=_auth_headers(token)).status_code == 200
    me = client.get("/api/v1/auth/me", headers=_auth_headers(token))
    assert me.status_code == 401


def test_password_is_stored_hashed_not_plaintext():
    client = TestClient(_app())
    username = _unique("pwcheck")
    _register_and_login(client, username)
    session = SessionLocal()
    try:
        user = session.execute(
            select(User).where(User.username == username)
        ).scalar_one()
        assert user is not None
        assert user.password_hash != "SuperSecret123!"
        assert "SuperSecret123!" not in user.password_hash
        assert verify_password("SuperSecret123!", user.password_hash)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Audit trail: screening results are persisted
# ---------------------------------------------------------------------------

def _make_mrz_not_detected():
    return {"detected": False, "source": "ocr", "status": "NOT_DETECTED",
            "confidence": 0.0, "module_state": "NOT_AVAILABLE"}


def _screen_request(client, token=None):
    headers = _auth_headers(token) if token else None
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
        headers=headers,
    )


def test_screening_is_persisted_for_authenticated_user(monkeypatch):
    monkeypatch.setattr(mrz_mod, "extract_mrz_from_image",
                        lambda doc, settings: _make_mrz_not_detected())
    client = TestClient(_app())
    username = _unique("audit")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    response = _screen_request(client, token)
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    detail = client.get(f"/api/v1/screenings/{request_id}", headers=_auth_headers(token))
    assert detail.status_code == 200
    body = detail.json()
    assert body["request_id"] == request_id
    assert body["mrz_source"] == "ocr"
    assert body["user_id"] is not None
    assert body["module_states"]["mrz"] == "REVIEW"


def test_screening_is_persisted_for_anonymous_user(monkeypatch):
    monkeypatch.setattr(mrz_mod, "extract_mrz_from_image",
                        lambda doc, settings: _make_mrz_not_detected())
    client = TestClient(_app())
    username = _unique("anon")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    response = _screen_request(client)
    assert response.status_code == 200
    request_id = response.json()["request_id"]
    detail = client.get(f"/api/v1/screenings/{request_id}", headers=_auth_headers(token))
    assert detail.status_code == 200
    assert detail.json()["user_id"] is None


# ---------------------------------------------------------------------------
# History and reporting endpoints are auth-protected
# ---------------------------------------------------------------------------

def test_screenings_list_requires_auth():
    client = TestClient(_app())
    assert client.get("/api/v1/screenings").status_code == 401
    assert client.get("/api/v1/report/summary").status_code == 401


def test_report_summary_rolls_up_records():
    with SessionLocal() as db:
        db.add_all([
            Screening(
                request_id=uuid.uuid4().hex, risk_score=5, risk_level="LOW_RISK",
                decision="SECONDARY_INSPECTION_REQUIRED", status_color="YELLOW",
                module_states={"mrz": "NOT_AVAILABLE", "face": "NOT_AVAILABLE",
                               "tampering": "PASS"},
                factors=[], processing_time_ms=120, document_type="UNKNOWN",
                mrz_status="NOT_AVAILABLE", face_status="NOT_AVAILABLE",
                tampering_status="PASS",
                mrz_source="ocr", user_id=None, created_at=utcnow_naive(),
            ),
            Screening(
                request_id=uuid.uuid4().hex, risk_score=10, risk_level="LOW_RISK",
                decision="CLEARED", status_color="GREEN",
                module_states={"mrz": "PASS", "face": "PASS", "tampering": "PASS"},
                factors=[], processing_time_ms=90, document_type="PASSPORT",
                mrz_status="VALID", face_status="MATCH", tampering_status="CLEAN",
                mrz_source="form", user_id=None, created_at=utcnow_naive(),
            ),
        ])
        db.commit()
    client = TestClient(_app())
    username = _unique("report")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    summary = client.get("/api/v1/report/summary", headers=_auth_headers(token))
    assert summary.status_code == 200
    body = summary.json()
    assert body["total_screenings"] >= 2
    assert body["cleared"] >= 1
    assert body["secondary_inspection"] >= 1
    assert body["by_risk_level"].get("LOW_RISK", 0) >= 2
    assert body["avg_processing_time_ms"] > 0

    listing = client.get("/api/v1/screenings", headers=_auth_headers(token))
    assert listing.status_code == 200
    assert listing.json()["total"] >= 2
    assert listing.json()["records"]


def test_screening_detail_404_for_unknown_id():
    client = TestClient(_app())
    username = _unique("missing")
    _, login = _register_and_login(client, username)
    token = login.json()["token"]
    detail = client.get(f"/api/v1/screenings/{uuid.uuid4().hex}", headers=_auth_headers(token))
    assert detail.status_code == 404
