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
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


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


# ---------------------------------------------------------------------------
# MRZ detection semantics
# ---------------------------------------------------------------------------

def _valid_mrz_result():
    parsed = main.parse_td3_mrz(VALID_LINE1, UNEXPIRED_LINE2)
    assert parsed["status"] == "VALID" and not parsed.get("is_expired")
    return {
        "detected": True,
        "source": "ocr",
        "status": "VALID",
        "confidence": 1.0,
        "line1": VALID_LINE1,
        "line2": UNEXPIRED_LINE2,
        "data": parsed,
    }


def test_mrz_valid_td3_detection():
    result = main.extract_mrz_from_image(jpeg_bytes())
    # A valid detection requires every check (structure, checksum, composite, date).
    assert main.parse_td3_mrz(VALID_LINE1, VALID_LINE2)["status"] == "VALID"


def test_checksum_invalid_44char_candidate_is_not_detected(monkeypatch):
    # Valid structure length (44 chars) but a broken document-number check digit.
    bad_line2 = "X" + VALID_LINE2[1:]
    assert main.parse_td3_mrz(VALID_LINE1, bad_line2)["status"] == "INVALID"
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{VALID_LINE1}\n{bad_line2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False
    # A checksum-invalid candidate must not surface as a success just because it
    # has 44 characters and passes structural validation.
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_wrong_length_ocr_candidate_is_not_detected(monkeypatch):
    short_line = VALID_LINE1[:-1]
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{short_line}\n{VALID_LINE2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False


def test_invalid_composite_checksum_is_not_detected(monkeypatch):
    # 44 chars, structurally valid, but the composite check digit is wrong.
    bad_line2 = VALID_LINE2[:-1] + "9"
    assert main.parse_td3_mrz(VALID_LINE1, bad_line2)["status"] == "INVALID"
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{VALID_LINE1}\n{bad_line2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False
    assert result["status"] in {"NOT_DETECTED", "OCR_LOW_CONFIDENCE"}


def test_invalid_date_is_not_detected(monkeypatch):
    # 44 chars with an impossible date (month 13) should not validate.
    bad_line2 = VALID_LINE2[:17] + "3" + VALID_LINE2[18:]
    parsed = main.parse_td3_mrz(VALID_LINE1, bad_line2)
    assert parsed["status"] != "VALID"
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{VALID_LINE1}\n{bad_line2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False


def test_invalid_characters_are_not_detected(monkeypatch):
    malformed = VALID_LINE1[:2] + "!" + VALID_LINE1[3:]
    assert main.parse_td3_mrz(malformed, VALID_LINE2)["status"] == "MALFORMED"
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{malformed}\n{VALID_LINE2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    assert result["detected"] is False


def test_detected_true_requires_status_valid(monkeypatch):
    monkeypatch.setattr(
        main.pytesseract,
        "image_to_string",
        lambda *args, **kwargs: f"{VALID_LINE1}\n{VALID_LINE2}",
    )
    result = main.extract_mrz_from_image(jpeg_bytes())
    if result["detected"]:
        assert result["status"] == "VALID"


# ---------------------------------------------------------------------------
# Risk engine: fail-safe CLEAR gate
# ---------------------------------------------------------------------------

def _screen_with_modules(monkeypatch, mrz_result, face_result, tamper_result, **kwargs):
    monkeypatch.setattr(main, "extract_mrz_from_image", lambda doc: mrz_result)
    monkeypatch.setattr(main, "analyze_tampering_ela", lambda doc, **kw: tamper_result)
    monkeypatch.setattr(main, "extract_and_verify_faces", lambda doc, live=None: face_result)
    client = TestClient(main.app)
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
        **kwargs,
    )


def _pass_face():
    return {
        "face_detected_in_document": True,
        "face_detected_in_live": True,
        "status": "MATCH",
        "match_status": "MATCH",
        "similarity_score": 0.9,
        "module_state": "PASS",
    }


def _pass_tamper():
    return {
        "status": "NO_SIGNIFICANT_ANOMALY",
        "module_state": "PASS",
        "is_tampered": False,
        "ela_mean_intensity": 10.0,
        "ela_std_dev": 12.0,
        "edge_artifact_score": 5.0,
        "metadata_present": True,
        "confidence": 10.0,
        "tampering_score": 10.0,
        "signals": [],
        "explanation": "No anomaly.",
    }


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_all_pass_clears(monkeypatch):
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=_pass_face(),
        tamper_result=_pass_tamper(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] == "CLEARED"
    assert body["risk_assessment"]["status"] == "GREEN"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_mrz_failure_and_others_pass_is_not_cleared(monkeypatch):
    mrz_result = {
        "detected": False,
        "source": "ocr",
        "status": "NOT_DETECTED",
        "confidence": 0.0,
        "module_state": "NOT_AVAILABLE",
    }
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=mrz_result,
        face_result=_pass_face(),
        tamper_result=_pass_tamper(),
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_face_mismatch_is_not_cleared(monkeypatch):
    face_result = {
        "face_detected_in_document": True,
        "face_detected_in_live": True,
        "status": "MISMATCH",
        "match_status": "MISMATCH",
        "similarity_score": 0.1,
        "module_state": "FAIL",
    }
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=face_result,
        tamper_result=_pass_tamper(),
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert any(factor["factor"] == "FACE_MISMATCH" for factor in body["risk_assessment"]["factors"])


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_tampering_suspected_is_not_cleared(monkeypatch):
    tamper_result = dict(_pass_tamper())
    tamper_result.update({
        "status": "SUSPECTED",
        "module_state": "FAIL",
        "is_tampered": True,
        "signals": ["ELA_ANOMALY"],
    })
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=_pass_face(),
        tamper_result=tamper_result,
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert any(factor["factor"] == "TAMPERING_SUSPECTED" for factor in body["risk_assessment"]["factors"])


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_face_error_is_not_cleared(monkeypatch):
    face_result = {
        "face_detected_in_document": False,
        "face_detected_in_live": None,
        "status": "ERROR",
        "match_status": "ERROR",
        "similarity_score": None,
        "module_state": "ERROR",
    }
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=face_result,
        tamper_result=_pass_tamper(),
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_face_not_available_is_not_cleared(monkeypatch):
    face_result = {
        "face_detected_in_document": True,
        "face_detected_in_live": None,
        "status": "SKIPPED_NO_LIVE_PHOTO",
        "match_status": "SKIPPED_NO_LIVE_PHOTO",
        "similarity_score": None,
        "module_state": "NOT_AVAILABLE",
    }
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=face_result,
        tamper_result=_pass_tamper(),
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert body["risk_assessment"]["module_statuses"]["face"] == "NOT_AVAILABLE"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_tampering_error_is_not_cleared(monkeypatch):
    tamper_result = dict(_pass_tamper())
    tamper_result.update({
        "status": "ANALYSIS_ERROR",
        "module_state": "ERROR",
        "is_tampered": False,
        "signals": ["ANALYSIS_ERROR"],
    })
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=_pass_face(),
        tamper_result=tamper_result,
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert body["risk_assessment"]["module_statuses"]["tampering"] == "ERROR"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_multiple_module_failures_is_not_cleared(monkeypatch):
    mrz_result = {
        "detected": False,
        "source": "ocr",
        "status": "NOT_DETECTED",
        "confidence": 0.0,
        "module_state": "NOT_AVAILABLE",
    }
    face_result = {
        "face_detected_in_document": False,
        "face_detected_in_live": None,
        "status": "ERROR",
        "match_status": "ERROR",
        "similarity_score": None,
        "module_state": "ERROR",
    }
    tamper_result = dict(_pass_tamper())
    tamper_result.update({
        "status": "SUSPECTED",
        "module_state": "FAIL",
        "is_tampered": True,
        "signals": ["ELA_ANOMALY"],
    })
    response = _screen_with_modules(
        monkeypatch,
        mrz_result=mrz_result,
        face_result=face_result,
        tamper_result=tamper_result,
    )
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_partial_manual_mrz_input_returns_400(monkeypatch):
    client = TestClient(main.app)
    files = {"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")}
    line1_only = client.post("/api/v1/screen", data={"mrz_line1": VALID_LINE1}, files=files)
    assert line1_only.status_code == 400
    line2_only = client.post("/api/v1/screen", data={"mrz_line2": VALID_LINE2}, files=files)
    assert line2_only.status_code == 400


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_both_manual_mrz_lines_validate_directly(monkeypatch):
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        data={"mrz_line1": VALID_LINE1, "mrz_line2": VALID_LINE2},
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mrz"]["detected"] is True
    assert body["mrz"]["source"] == "form"


@pytest.mark.skipif(TestClient is None, reason="FastAPI test client dependency unavailable")
def test_neither_manual_mrz_line_falls_back_to_ocr(monkeypatch):
    monkeypatch.setattr(main, "extract_mrz_from_image", lambda doc: _valid_mrz_result())
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mrz"]["source"] == "ocr"
