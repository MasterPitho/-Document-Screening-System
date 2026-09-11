"""Tests for Dashboard Extensions: Applicant identity fields, trend stats,
decision updating, notifications, watchlists, CORS, and flexible auth.
"""

import io
import datetime
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db.database import Database, build_database
from app.main import create_app
from app.services.face_recognition import DummyBackend, FaceDetectionResult, ModelManager
import numpy as np


VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _jpeg(width=800, height=500):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _face():
    return FaceDetectionResult(
        bbox=[10, 20, 110, 140], score=0.9, landmarks=None,
        embedding=np.ones((512,), dtype=np.float32),
    )


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    db_path = f"sqlite:///{(tmp_path / 'test_dash.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_path)
    settings = Settings.from_env()
    manager = ModelManager(settings, backend=DummyBackend([_face()]))
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)
    return app, client


def _auth_token(client):
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": "officer_test",
        "email": "officer@example.com",
        "full_name": "Officer Test",
        "officer_id": "LT-04",
        "password": "Password123!",
    })
    assert reg_resp.status_code == 201
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "officer_test",
        "password": "Password123!",
    })
    assert login_resp.status_code == 200
    return login_resp.json()["token"]


# --------------------------------------------------------------------------- #
# 1. Applicant Identity Fields & Screening Persistence
# --------------------------------------------------------------------------- #

def test_screen_populates_applicant_identity(app_and_client):
    _, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    screening_id = body["persistence"]["screening_id"]

    # Verify via GET /api/v1/screenings/{item_id}
    detail = client.get(f"/api/v1/screenings/{screening_id}", headers=headers)
    assert detail.status_code == 200
    rec = detail.json()
    assert rec["applicant_name"] == "ANNA MARIA ERIKSSON"
    assert rec["document_number"] == "L898902C3"
    assert rec["country_code"] == "UTO"
    assert rec["notes"] is None

    # Verify via GET /api/v1/screenings list
    listing = client.get("/api/v1/screenings", headers=headers)
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1
    found = next(r for r in listing.json()["records"] if r["id"] == screening_id)
    assert found["applicant_name"] == "ANNA MARIA ERIKSSON"
    assert found["document_number"] == "L898902C3"
    assert found["country_code"] == "UTO"


def test_screen_without_mrz_stores_none_identities(app_and_client):
    _, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Blank image where OCR finds no MRZ
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("blank.jpg", _jpeg(), "image/jpeg")},
        headers=headers,
    )
    assert resp.status_code == 200
    screening_id = resp.json()["persistence"]["screening_id"]

    detail = client.get(f"/api/v1/screenings/{screening_id}", headers=headers)
    assert detail.status_code == 200
    rec = detail.json()
    assert rec["applicant_name"] is None
    assert rec["document_number"] is None
    assert rec["country_code"] is None


# --------------------------------------------------------------------------- #
# 2. Time-Series & Dashboard Trend Data
# --------------------------------------------------------------------------- #

def test_stats_trend_24h(app_and_client):
    _, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Generate a screening
    client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )

    resp = client.get("/api/v1/stats/trend?range=24h", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["range"] == "24h"
    assert data["today_total"] >= 1
    assert isinstance(data["delta_today_pct"], float)
    assert isinstance(data["avg_processing_time_ms"], float)
    assert isinstance(data["avg_time_delta_pct"], float)
    assert isinstance(data["average_risk_score"], float)
    assert isinstance(data["high_risk_events_count"], int)
    assert len(data["timeline"]) == 6
    for point in data["timeline"]:
        assert "timestamp" in point
        assert "risk_score" in point
        assert "count" in point


def test_stats_trend_7d_and_30d(app_and_client):
    _, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    resp7 = client.get("/api/v1/stats/trend?range=7d", headers=headers)
    assert resp7.status_code == 200
    assert resp7.json()["range"] == "7d"
    assert len(resp7.json()["timeline"]) == 7

    resp30 = client.get("/api/v1/stats/trend?range=30d", headers=headers)
    assert resp30.status_code == 200
    assert resp30.json()["range"] == "30d"
    assert len(resp30.json()["timeline"]) == 10


def test_stats_trend_invalid_range(app_and_client):
    _, client = app_and_client
    resp = client.get("/api/v1/stats/trend?range=100d")
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# 3. Update Screening Decision
# --------------------------------------------------------------------------- #

def test_update_screening_decision(app_and_client):
    _, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Create screening
    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("passport.jpg", _jpeg(), "image/jpeg")},
        data={"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2},
        headers=headers,
    )
    screening_id = resp.json()["persistence"]["screening_id"]

    # 1. Update to CLEARED
    patch_resp = client.patch(
        f"/api/v1/screenings/{screening_id}/decision",
        json={"decision": "CLEARED", "notes": "Approved after biometric match"},
        headers=headers,
    )
    assert patch_resp.status_code == 200
    body = patch_resp.json()
    assert body["decision"] == "CLEARED"
    assert body["status_color"] == "GREEN"
    assert body["notes"] == "Approved after biometric match"

    # 2. Update to HOLD
    patch_resp2 = client.patch(
        f"/api/v1/screenings/{screening_id}/decision",
        json={"decision": "HOLD", "notes": "Suspicious travel history"},
        headers=headers,
    )
    assert patch_resp2.status_code == 200
    body2 = patch_resp2.json()
    assert body2["decision"] == "HOLD"
    assert body2["status_color"] == "RED"
    assert body2["notes"] == "Suspicious travel history"

    # 3. Update to REVIEW
    patch_resp3 = client.patch(
        f"/api/v1/screenings/{screening_id}/decision",
        json={"decision": "REVIEW"},
        headers=headers,
    )
    assert patch_resp3.status_code == 200
    body3 = patch_resp3.json()
    assert body3["decision"] == "REVIEW"
    assert body3["status_color"] == "YELLOW"

    # 4. Invalid decision
    bad_resp = client.patch(
        f"/api/v1/screenings/{screening_id}/decision",
        json={"decision": "NON_EXISTENT_DECISION"},
        headers=headers,
    )
    assert bad_resp.status_code == 422

    # 5. Non-existent screening
    not_found = client.patch(
        "/api/v1/screenings/999999/decision",
        json={"decision": "CLEARED"},
        headers=headers,
    )
    assert not_found.status_code == 404


# --------------------------------------------------------------------------- #
# 4. Notifications & Watchlists
# --------------------------------------------------------------------------- #

def test_notifications_and_watchlists(app_and_client):
    app, client = app_and_client
    token = _auth_token(client)
    headers = {"Authorization": f"Bearer {token}"}

    # Populate a high-risk screening directly via repo
    repo = app.state.screening_repo
    repo.create(
        request_id="high-risk-test-doc-001",
        processing_time_ms=120,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.9,
        tampering_status="CLEAN",
        tampering_score=0.1,
        risk_score=85,
        risk_level="HIGH_RISK",
        decision="HIGH_RISK_REVIEW_REQUIRED",
        status_color="RED",
        module_states={"mrz": "PASS", "face": "PASS", "tampering": "PASS"},
        factor_list=[],
        mrz_source="ocr",
        document_number="A0183948",
    )

    # Notifications endpoint
    notif_resp = client.get("/api/v1/notifications?limit=5", headers=headers)
    assert notif_resp.status_code == 200
    notifs = notif_resp.json()
    assert isinstance(notifs, list)
    assert len(notifs) >= 1
    first = notifs[0]
    assert first["type"] == "HIGH_RISK_ALERT"
    assert "A0183948" in first["message"]
    assert "created_at" in first
    assert "read" in first

    # Watchlists endpoint
    watch_resp = client.get("/api/v1/watchlists", headers=headers)
    assert watch_resp.status_code == 200
    watchlists = watch_resp.json()
    assert isinstance(watchlists, list)
    assert len(watchlists) >= 1
    w = watchlists[0]
    assert w["name"] == "Tehran Ali"
    assert w["document_number"] == "V-449021"
    assert w["reason"] == "Lookout circular"
    assert w["severity"] == "HIGH"


# --------------------------------------------------------------------------- #
# 5. Flexible Authentication & CORS
# --------------------------------------------------------------------------- #

def test_flexible_auth_by_officer_id(app_and_client):
    _, client = app_and_client

    # 1. Register with officer_id
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": "patrol_officer",
        "email": "patrol@border.gov",
        "full_name": "Patrol Officer",
        "officer_id": "LT-04",
        "password": "SecurePassword123!",
    })
    assert reg_resp.status_code == 201
    assert reg_resp.json()["officer_id"] == "LT-04"

    # 2. Login using officer_id "LT-04" instead of username
    login_by_badge = client.post("/api/v1/auth/login", json={
        "username": "LT-04",
        "password": "SecurePassword123!",
    })
    assert login_by_badge.status_code == 200
    assert "token" in login_by_badge.json()
    assert login_by_badge.json()["user"]["username"] == "patrol_officer"

    # 3. Login using email
    login_by_email = client.post("/api/v1/auth/login", json={
        "username": "patrol@border.gov",
        "password": "SecurePassword123!",
    })
    assert login_by_email.status_code == 200
    assert "token" in login_by_email.json()


def test_cors_headers(app_and_client):
    _, client = app_and_client
    resp = client.options(
        "/api/v1/stats",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-credentials") == "true"


# --------------------------------------------------------------------------- #
# 6. Verification: Docs, OpenAPI, and System Endpoints
# --------------------------------------------------------------------------- #

def test_docs_and_openapi_reflect_new_routes(app_and_client):
    _, client = app_and_client

    # 1. GET /docs returns 200 HTML
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    # 2. GET /openapi.json includes new paths
    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    schema = openapi_resp.json()
    paths = schema["paths"]

    assert "/api/v1/stats/trend" in paths
    assert "get" in paths["/api/v1/stats/trend"]

    assert "/api/v1/screenings/{item_id}/decision" in paths
    assert "patch" in paths["/api/v1/screenings/{item_id}/decision"]

    assert "/api/v1/notifications" in paths
    assert "get" in paths["/api/v1/notifications"]

    assert "/api/v1/watchlists" in paths
    assert "get" in paths["/api/v1/watchlists"]


def test_health_and_ready_endpoints(app_and_client):
    _, client = app_and_client
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/ready")
    assert ready.status_code in {200, 503}
    body = ready.json()
    assert "modules" in body
    assert body["modules"]["database"] is True

