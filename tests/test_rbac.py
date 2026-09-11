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

    # User registered with non-privileged role 'guest' or 'trainee'
    viewer_token = _register_and_login(client, "viewer_user", "trainee")
    resp = client.patch(
        f"/api/v1/screenings/{rec.id}/decision",
        json={"decision": "CLEARED"},
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert resp.status_code == 403
    body = resp.json()
    error_msg = body.get("error", {}).get("message") or body.get("detail", "")
    assert "Forbidden" in error_msg
