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


def test_configuration_validation_rejects_invalid_thresholds():
    original_face_threshold = main.FACE_SIMILARITY_THRESHOLD
    original_risk_thresholds = main.RISK_THRESHOLDS
    try:
        main.FACE_SIMILARITY_THRESHOLD = 1.1
        with pytest.raises(ValueError, match="FACE_MATCH_THRESHOLD"):
            main._validate_configuration()
        main.FACE_SIMILARITY_THRESHOLD = original_face_threshold
        main.RISK_THRESHOLDS = (70, 35)
        with pytest.raises(ValueError, match="Risk thresholds"):
            main._validate_configuration()
    finally:
        main.FACE_SIMILARITY_THRESHOLD = original_face_threshold
        main.RISK_THRESHOLDS = original_risk_thresholds


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_api_rejects_unsupported_type():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_health_endpoint():
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_api_rejects_corrupt_and_oversized_images():
    client = TestClient(main.app)
    corrupt = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", b"not-an-image", "image/jpeg")},
    )
    oversized = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", b"0" * (main.MAX_IMAGE_BYTES + 1), "image/jpeg")},
    )
    wide_buffer = io.BytesIO()
    Image.new("RGB", (main.MAX_IMAGE_WIDTH + 1, 1), "white").save(wide_buffer, format="JPEG")
    too_wide = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", wide_buffer.getvalue(), "image/jpeg")},
    )
    assert corrupt.status_code == 400
    assert oversized.status_code == 413
    assert too_wide.status_code == 413


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_skipped_face_verification_requires_review(monkeypatch):
    monkeypatch.setattr(
        main,
        "extract_and_verify_faces",
        lambda document, live: {
            "status": "SKIPPED_NO_LIVE_PHOTO",
            "match_status": "SKIPPED_NO_LIVE_PHOTO",
            "face_detected_in_document": True,
            "face_detected_in_live": None,
            "similarity_score": None,
        },
    )
    monkeypatch.setattr(
        main,
        "extract_mrz_from_image",
        lambda document: {
            "detected": True,
            "source": "ocr",
            "status": "VALID",
            "confidence": 1.0,
            "data": main.parse_td3_mrz(VALID_LINE1, VALID_LINE2),
        },
    )
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["risk_assessment"]["decision"] == "SECONDARY_INSPECTION_REQUIRED"
    assert any(factor["factor"] == "UNKNOWN_MODULE" for factor in body["risk_assessment"]["factors"])


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
    assert body["risk_assessment"]["module_statuses"]["mrz"] in {"NOT_AVAILABLE", "ERROR"}
    assert body["face_verification"]["similarity_score"] is None or isinstance(
        body["face_verification"]["similarity_score"], float
    )
    assert "risk_assessment" in body
    assert "processing_time_ms" in body
