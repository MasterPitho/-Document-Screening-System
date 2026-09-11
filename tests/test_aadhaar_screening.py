"""
Tests for Dedicated Aadhaar Document Analysis Strategy (AadhaarDocumentParser).
"""

import io
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.aadhaar import AadhaarDocumentParser
from app.services.mrz import get_document_parser, DocumentParserRouter
from app.config import Settings


def _create_synthetic_aadhaar_image(
    anchors: list[str],
    uid_str: str = "XXXX XXXX 1234",
    with_qr_payload: str | None = None,
    width: int = 800,
    height: int = 500,
) -> bytes:
    """Create a synthetic test image with text anchors and optional QR code."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = 30
    for text in anchors:
        draw.text((40, y), text, fill=(0, 0, 0))
        y += 40

    if uid_str:
        draw.text((200, 380), uid_str, fill=(0, 0, 0))

    if with_qr_payload:
        encoder = cv2.QRCodeEncoder.create()
        mat = encoder.encode(with_qr_payload)
        scaled = cv2.resize(mat, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        qr_pil = Image.fromarray(scaled)
        img.paste(qr_pil, (width - scaled.shape[1] - 40, 100))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_aadhaar_parser_blank_image_returns_not_detected():
    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    parser = AadhaarDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(blank, settings)
    assert not result.detected
    assert result.status == "NOT_DETECTED"
    assert result.document_type == "AADHAAR"


def test_aadhaar_parser_registered_in_router():
    router = DocumentParserRouter()
    parser = router.parser_for("aadhaar")
    assert isinstance(parser, AadhaarDocumentParser)
    assert parser.name == "aadhaar"
    assert parser.document_type == "AADHAAR"


def test_aadhaar_parser_with_anchors_and_masked_uid(monkeypatch):
    import pytesseract
    mock_ocr = "Government of India\nUnique Identification Authority of India\nDOB: 01/01/1990\nGender: Male\nXXXX XXXX 9876"
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: mock_ocr)

    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    parser = AadhaarDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(blank, settings)
    assert result.document_type == "AADHAAR"
    assert "anchors_found" in result.raw
    assert len(result.raw["anchors_found"]) >= 2
    assert result.status == "VALID"
    assert result.detected is True
    assert result.data.get("masked_uid") == "XXXX-XXXX-9876"


def test_aadhaar_parser_with_qr_code():
    """Aadhaar with valid QR code passes even without OCR."""
    xml_data = '<PrintLetterBarcodeData uid="999912345678" name="Aadhaar Holder" yob="1995" gender="M" />'
    img_bytes = _create_synthetic_aadhaar_image([], with_qr_payload=xml_data)
    parser = AadhaarDocumentParser()
    settings = Settings.from_env()

    result = parser.parse(img_bytes, settings)
    assert result.detected is True
    assert result.raw["qr"]["detected"] is True
    assert result.raw["qr"]["payload_type"] == "AADHAAR_XML"
    assert result.data.get("masked_uid") == "XXXX-XXXX-5678"


def test_aadhaar_parser_verhoeff_checksum():
    parser = AadhaarDocumentParser()
    assert parser.validate_verhoeff("2363") is True
    assert parser.validate_verhoeff("2364") is False
    assert parser.validate_verhoeff("999912345678") is True
    assert parser.validate_verhoeff("999912345679") is False


def test_screen_endpoint_with_aadhaar_document_type(app_and_client, monkeypatch):
    import pytesseract
    mock_ocr = "Government of India\nUIDAI\nDOB: 12/05/1992\nMale\nXXXX XXXX 4321"
    monkeypatch.setattr(pytesseract, "image_to_string", lambda *a, **k: mock_ocr)

    app, client = app_and_client
    reg_resp = client.post("/api/v1/auth/register", json={
        "username": "officer_aadhaar",
        "email": "officer_aadhaar@example.com",
        "full_name": "Officer Aadhaar",
        "officer_id": "LT-AADHAAR",
        "password": "Password123!",
    })
    login_resp = client.post("/api/v1/auth/login", json={
        "username": "officer_aadhaar",
        "password": "Password123!",
    })
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    blank = np.full((500, 800, 3), 255, dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", blank)
    img_bytes = enc.tobytes()

    resp = client.post(
        "/api/v1/screen",
        files={"document_image": ("aadhaar.jpg", img_bytes, "image/jpeg")},
        data={"document_type": "aadhaar"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document"]["type"] == "AADHAAR"
    assert body["document"]["format"] == "AADHAAR_CARD"
    assert body["mrz"]["status"] == "VALID"
