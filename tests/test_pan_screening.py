"""
Tests for Indian PAN Document Analysis Strategy (PANDocumentParser).
"""

import io
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.pan import PANDocumentParser
from app.services.mrz import get_document_parser, DocumentParserRouter
from app.config import Settings


def test_pan_parser_blank_image_returns_not_detected():
    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    parser = PANDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(blank, settings)
    assert not result.detected
    assert result.status == "NOT_DETECTED"
    assert result.document_type == "PAN"


def test_pan_parser_registered_in_router():
    router = DocumentParserRouter()
    parser = router.parser_for("pan")
    assert isinstance(parser, PANDocumentParser)
    assert parser.name == "pan"
    assert parser.document_type == "PAN"


def test_pan_structure_and_entity_type_validation():
    parser = PANDocumentParser()

    # Valid individual PAN: ABCDE1234F (4th char 'P' for Individual)
    valid_pan = "ABCDE1234F"
    is_valid, entity_type = parser.validate_pan_structure("ABCPE1234F")
    assert is_valid is True
    assert entity_type == "Individual"

    # Valid company PAN: ABCDE1234F (4th char 'C' for Company)
    is_valid_c, entity_type_c = parser.validate_pan_structure("AAACC1234Z")
    assert is_valid_c is True
    assert entity_type_c == "Company"

    # Invalid 4th char 'Z'
    is_valid_bad, entity_type_bad = parser.validate_pan_structure("ABCZE1234F")
    assert is_valid_bad is False
    assert entity_type_bad == "Unknown"

    # Invalid length/format
    is_valid_len, _ = parser.validate_pan_structure("ABCD12345F")
    assert is_valid_len is False


def test_pan_parser_with_anchors_and_valid_pan(monkeypatch):
    import pytesseract
    mock_ocr = "INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number Card\nABCPE1234F\nDOB: 15/08/1985"
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: mock_ocr)

    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    parser = PANDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(blank, settings)
    assert result.detected is True
    assert result.document_type == "PAN"
    assert result.status == "VALID"
    assert result.data.get("masked_pan") == "ABCPXXXX4F"
    assert result.data.get("entity_type") == "Individual"
    # Never expose full raw PAN in result data
    assert "ABCPE1234F" not in str(result.data)


def test_pan_parser_suspicious_when_status_char_invalid(monkeypatch):
    import pytesseract
    # 4th char 'Z' is invalid entity type in Indian PAN system
    mock_ocr = "INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number Card\nABCZE1234F\nDOB: 15/08/1985"
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: mock_ocr)

    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    parser = PANDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(blank, settings)
    assert result.detected is True
    assert result.status == "SUSPICIOUS"
    assert result.raw["checks"]["status_char_valid"] is False


def test_screen_endpoint_with_pan_document_type(app_and_client, monkeypatch):
    import pytesseract
    mock_ocr = "INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number Card\nABCPE1234F\nDOB: 15/08/1985"
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: mock_ocr)

    app, client = app_and_client
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": "officer_pan",
        "email": "officer_pan@example.com",
        "full_name": "Officer PAN",
        "officer_id": "LT-PAN",
        "password": "Password123!",
    })
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "officer_pan",
        "password": "Password123!",
    })
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", blank)
    img_bytes = enc.tobytes()

    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("pancard.jpg", img_bytes, "image/jpeg")},
        data={"document_type": "pan"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document"]["type"] == "PAN"
    assert body["document"]["format"] == "PAN_CARD"
    assert body["mrz"]["status"] == "VALID"
