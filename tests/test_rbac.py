"""
Tests for Role-Based Access Control (RBAC) across endpoints.
Roles supported: officer, supervisor, admin.
"""

import pytest
from starlette.testclient import TestClient


def _register_and_login(client: TestClient, username: str, role: str) -> str:
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": username,
        "email": f"{username}@agency.gov",
        "full_name": f"Test {role.capitalize()}",
        "role": role,
        "password": "Password123!",
    })
    assert reg_resp.status_code == 201
    login_resp = client.post("/api/v1/auth/login", json={
        "username": username,
        "password": "Password123!",
    })
    assert login_resp.status_code == 200
    return login_resp.json()["token"]


def test_unauthenticated_user_cannot_update_decision(app_and_client):
    _, client = app_and_client
    resp = client.patch(
        "/api/v1/screenings/1/decision",
        json={"decision": "CLEARED"},
    )
    assert resp.status_code == 401


def test_officer_and_supervisor_can_update_decision(app_and_client):
    app, client = app_and_client

    # 1. Create a screening record
    repo = app.state.screening_repo
    rec = repo.create(
        request_id="rbac-test-doc-001",
        processing_time_ms=100,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.9,
        tampering_status="CLEAN",
        tampering_score=0.1,
        risk_score=50,
        risk_level="MEDIUM_RISK",
        decision="SECONDARY_INSPECTION_REQUIRED",
        status_color="YELLOW",
        module_states={"mrz": "PASS", "face": "PASS", "tampering": "PASS"},
        factor_list=[],
        mrz_source="ocr",
    )

    # 2. Officer token
    officer_token = _register_and_login(client, "officer_rbac", "officer")
    patch_resp = client.patch(
        f"/api/v1/screenings/{rec.id}/decision",
        json={"decision": "REVIEW", "notes": "Reviewed by officer"},
        headers={"Authorization": f"Bearer {officer_token}"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["decision"] == "REVIEW"

    # 3. Supervisor token
    supervisor_token = _register_and_login(client, "super_rbac", "supervisor")
    patch_resp2 = client.patch(
        f"/api/v1/screenings/{rec.id}/decision",
        json={"decision": "CLEARED", "notes": "Cleared by supervisor"},
        headers={"Authorization": f"Bearer {supervisor_token}"},
    )
    assert patch_resp2.status_code == 200
    assert patch_resp2.json()["decision"] == "CLEARED"


def test_registration_cannot_escalate_to_admin(app_and_client):
    """Attempting to register with role=admin must not grant admin role (defaults to officer)."""
    _, client = app_and_client
    resp = client.post("/api/v1/auth/register", json={
        "username": "attacker_admin",
        "email": "attacker_admin@example.com",
        "full_name": "Attacker Admin",
        "role": "admin",
        "password": "Password123!",
    })
    assert resp.status_code == 201
    user = resp.json()
    assert user["role"] == "officer"
    assert user["role"] != "admin"


def test_registration_cannot_escalate_to_supervisor(app_and_client):
    """Attempting to register with role=supervisor must not grant supervisor role (defaults to officer)."""
    _, client = app_and_client
    resp = client.post("/api/v1/auth/register", json={
        "username": "attacker_super",
        "email": "attacker_super@example.com",
        "full_name": "Attacker Supervisor",
        "role": "supervisor",
        "password": "Password123!",
    })
    assert resp.status_code == 201
    user = resp.json()
    assert user["role"] == "officer"
    assert user["role"] != "supervisor"


def test_normal_registration_creates_officer(app_and_client):
    """Normal registration without role creates officer."""
    _, client = app_and_client
    resp = client.post("/api/v1/auth/register", json={
        "username": "regular_officer",
        "email": "regular_officer@example.com",
        "full_name": "Regular Officer",
        "password": "Password123!",
    })
    assert resp.status_code == 201
    assert resp.json()["role"] == "officer"


def test_insufficient_role_is_forbidden(app_and_client):
    app, client = app_and_client
    repo = app.state.screening_repo
    rec = repo.create(
        request_id="rbac-test-doc-002",
        processing_time_ms=100,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.9,
        tampering_status="CLEAN",
        tampering_score=0.1,
        risk_score=50,
        risk_level="MEDIUM_RISK",
        decision="SECONDARY_INSPECTION_REQUIRED",
        status_color="YELLOW",
        module_states={"mrz": "PASS", "face": "PASS", "tampering": "PASS"},
        factor_list=[],
        mrz_source="ocr",
    )

    # Privileged user repo creates a viewer/guest account
    from app.api.auth import _hash_password
    app.state.user_repo.create(
        username="viewer_user",
        email="viewer@agency.gov",
        full_name="Viewer User",
        role="guest",
        password_hash=_hash_password("Password123!"),
    )
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "viewer_user",
        "password": "Password123!",
    })
    viewer_token = login_resp.json()["token"]

    resp = client.patch(
        f"/api/v1/screenings/{rec.id}/decision",
        json={"decision": "CLEARED"},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 403
    body = resp.json()
    error_msg = body.get("error", {}).get("message") or body.get("detail", "")
    assert "Forbidden" in error_msg


def test_officer_only_sees_own_screenings(app_and_client):
    """Officers must only see their own screenings; supervisors/admins see all."""
    from app.api.auth import _hash_password
    app, client = app_and_client
    repo = app.state.screening_repo
    user_repo = app.state.user_repo

    # Create Officer 1
    u1 = user_repo.create(
        username="officer_alice",
        email="alice@agency.gov",
        full_name="Officer Alice",
        role="officer",
        password_hash=_hash_password("Password123!"),
    )
    # Create Officer 2
    u2 = user_repo.create(
        username="officer_bob",
        email="bob@agency.gov",
        full_name="Officer Bob",
        role="officer",
        password_hash=_hash_password("Password123!"),
    )
    # Create Supervisor
    u_sup = user_repo.create(
        username="supervisor_carol",
        email="carol@agency.gov",
        full_name="Supervisor Carol",
        role="supervisor",
        password_hash=_hash_password("Password123!"),
    )

    # Create 1 screening for Alice and 1 for Bob
    rec_alice = repo.create(
        request_id="alice-doc-001",
        processing_time_ms=50,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.9,
        tampering_status="CLEAN",
        tampering_score=0.1,
        risk_score=10,
        risk_level="LOW_RISK",
        decision="CLEARED",
        status_color="GREEN",
        module_states={},
        factor_list=[],
        mrz_source="ocr",
        user_id=u1.id,
    )
    rec_bob = repo.create(
        request_id="bob-doc-001",
        processing_time_ms=50,
        document_type="PASSPORT",
        mrz_status="VALID",
        face_status="MATCH",
        face_similarity=0.9,
        tampering_status="CLEAN",
        tampering_score=0.1,
        risk_score=10,
        risk_level="LOW_RISK",
        decision="CLEARED",
        status_color="GREEN",
        module_states={},
        factor_list=[],
        mrz_source="ocr",
        user_id=u2.id,
    )

    # Login as Alice
    t_alice = client.post("/api/v1/auth/login", json={"username": "officer_alice", "password": "Password123!"}).json()["token"]
    resp_alice = client.get("/api/v1/screenings", headers={"Authorization": f"Bearer {t_alice}"})
    assert resp_alice.status_code == 200
    records_alice = resp_alice.json()["records"]
    assert len(records_alice) == 1
    assert records_alice[0]["id"] == rec_alice.id

    # Login as Carol (Supervisor)
    t_carol = client.post("/api/v1/auth/login", json={"username": "supervisor_carol", "password": "Password123!"}).json()["token"]
    resp_carol = client.get("/api/v1/screenings", headers={"Authorization": f"Bearer {t_carol}"})
    assert resp_carol.status_code == 200
    ids_carol = {r["id"] for r in resp_carol.json()["records"]}
    assert rec_alice.id in ids_carol
    assert rec_bob.id in ids_carol


def test_dashboard_endpoints_require_authentication(app_and_client):
    """GET /api/v1/stats/trend, /api/v1/notifications, and /api/v1/watchlists must reject anonymous callers."""
    _, client = app_and_client
    assert client.get("/api/v1/stats/trend").status_code == 401
    assert client.get("/api/v1/notifications").status_code == 401
    assert client.get("/api/v1/watchlists").status_code == 401


