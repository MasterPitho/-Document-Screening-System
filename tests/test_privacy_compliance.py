"""
Tests for Zero-PII Persistence Compliance across Database, Schemas, and API endpoints.

Privacy Policy Guarantee:
- Uploaded images, live photos, face embeddings, raw MRZ lines, passport numbers,
  Aadhaar numbers, applicant names, and country codes must NEVER be stored in the
  Screening database, logged in application logs, or exposed in persisted records.
"""

from __future__ import annotations

import io
import pytest
from PIL import Image
from sqlalchemy import inspect
from app.db.models import Screening, ScreeningFactor, AuditLog, Base


VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501019ZE184226B<<<<<10"


def _jpeg() -> bytes:
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _auth_token(client) -> str:
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": "officer_priv",
        "email": "officer_priv@example.com",
        "full_name": "Officer Test",
        "officer_id": "LT-PRIV-01",
        "password": "Password123!",
    })
    if reg_resp.status_code == 201:
        login_resp = client.post("/api/v1/auth/login", json={
            "username": "officer_priv",
            "password": "Password123!",
        })
        return login_resp.json()["token"]
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "officer",
        "password": "Officer123!",
    })
    return login_resp.json()["token"]


def test_screening_model_has_no_pii_columns():
    """Verify that the SQLAlchemy Screening table schema contains zero PII columns."""
    column_names = {c.name for c in Screening.__table__.columns}
    
    forbidden_pii = {
        "applicant_name",
        "document_number",
        "country_code",
        "passport_number",
        "raw_mrz",
        "mrz_line1",
        "mrz_line2",
        "mrz_text",
        "face_embedding",
        "embedding",
        "image",
        "image_bytes",
        "document_image",
        "live_photo",
        "aadhaar_number",
        "pan_number",
    }
    
    present_forbidden = forbidden_pii.intersection(column_names)
    assert not present_forbidden, f"Screening table has forbidden PII columns: {present_forbidden}"


def test_screen_persists_no_pii_in_database(app_and_client):
    """End-to-end screen endpoint must not save PII in the screenings table."""
    app, client = app_and_client
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

    # Check database directly
    with app.state.database.session() as session:
        rec = session.get(Screening, screening_id)
        assert rec is not None
        # Assert no PII attributes on instance
        assert not hasattr(rec, "applicant_name")
        assert not hasattr(rec, "document_number")
        assert not hasattr(rec, "country_code")

        # Verify factor details do not leak the passport number or raw name
        for factor in rec.factor_rows:
            assert "L898902C3" not in (factor.description or "")
            assert "ANNA MARIA ERIKSSON" not in (factor.description or "")

    # Check API response
    detail = client.get(f"/api/v1/screenings/{screening_id}", headers=headers)
    assert detail.status_code == 200
    payload = detail.json()
    assert "applicant_name" not in payload
    assert "document_number" not in payload
    assert "country_code" not in payload
