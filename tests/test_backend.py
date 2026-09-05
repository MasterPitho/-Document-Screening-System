import io

import pytest
from PIL import Image

import main

try:
    from fastapi.testclient import TestClient
except RuntimeError:
    TestClient = None


VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
VALID_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def jpeg_bytes(width=800, height=500):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_valid_td3_mrz_and_composite_checksum():
    result = main.parse_td3_mrz(VALID_LINE1, VALID_LINE2)
    assert result["status"] == "VALID"
    assert result["checks"] == {
        "passport_number_valid": True,
        "dob_valid": True,
        "expiry_valid": True,
        "composite_valid": True,
    }


def test_mrz_checksum_failures_are_explicit():
    assert main.parse_td3_mrz(VALID_LINE1, "X" + VALID_LINE2[1:])["status"] == "INVALID"
    assert main.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:19] + "9" + VALID_LINE2[20:])["status"] == "INVALID"
    assert main.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:27] + "8" + VALID_LINE2[28:])["status"] == "INVALID"
    assert main.parse_td3_mrz(VALID_LINE1, VALID_LINE2[:-1] + "9")["status"] == "INVALID"


def test_mrz_malformed_length_and_characters():
    assert main.parse_td3_mrz(VALID_LINE1[:-1], VALID_LINE2)["status"] == "MALFORMED"
    malformed = VALID_LINE1[:2] + "!" + VALID_LINE1[3:]
    assert main.parse_td3_mrz(malformed, VALID_LINE2)["status"] == "MALFORMED"


def test_ocr_does_not_pad_wrong_length_candidates(monkeypatch):
    monkeypatch.setattr(main.pytesseract, "image_to_string", lambda *args, **kwargs: "P<UTO\nL898902C36UTO")
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False
    assert result["status"] == "NOT_DETECTED"


def test_ocr_selects_valid_td3_pair(monkeypatch):
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{VALID_LINE1}\n{VALID_LINE2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is True
    assert result["status"] == "VALID"
    assert result["data"]["checks"]["composite_valid"] is True


def test_ocr_failure_is_safe(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("tesseract unavailable")

    monkeypatch.setattr(main.pytesseract, "image_to_string", fail)
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False
    assert result["status"] == "OCR_FAILED"
    assert "tesseract unavailable" in result["reason"]


def test_ocr_low_confidence_is_not_valid(monkeypatch):
    weak_line1 = VALID_LINE1.replace("P", "V", 1)
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{weak_line1}\n{VALID_LINE2[:-1]}9",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_tampering_result_is_multi_signal():
    result = main.analyze_tampering_ela(jpeg_bytes())
    assert result["status"] in {"SUSPECTED", "NO_SIGNIFICANT_ANOMALY"}
    assert isinstance(result["signals"], list)
    assert 0 <= result["confidence"] <= 100


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_api_rejects_unsupported_type():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_api_returns_screened_schema():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SCREENED"
    assert len(body["request_id"]) == 32
    assert body["risk_assessment"]["decision"] == "SECONDARY_INSPECTION_REQUIRED"
    assert any(factor["factor"] == "MRZ_NOT_DETECTED" for factor in body["risk_assessment"]["factors"])
    assert body["face_verification"]["similarity_score"] is None or isinstance(
        body["face_verification"]["similarity_score"], float
    )
    assert "risk_assessment" in body
    assert "processing_time_ms" in body
