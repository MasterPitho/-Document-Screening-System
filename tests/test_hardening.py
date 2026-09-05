"""Hardening and regression tests for the document screening engine.

Complements tests/test_backend.py. Focus: MRZ leap-year/structure edge cases,
risk-weight fail-safety (TAMPERING_INCONCLUSIVE), tampering/face failure
handling, upload/decompression-bomb protection, high-risk decision gating, and
safe handling of unexpected internal errors.
"""

import io
import json

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

import main

try:
    from fastapi.testclient import TestClient
except RuntimeError:
    TestClient = None

skip_client = pytest.mark.skipif(
    TestClient is None, reason="FastAPI test client dependency unavailable"
)

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
VALID_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"

REQUIRED_RISK_FACTORS = {
    "TAMPERING_SUSPECTED",
    "TAMPERING_INCONCLUSIVE",
    "FACE_MISMATCH",
    "MRZ_CHECKSUM_FAILURE",
    "EXPIRED_DOCUMENT",
    "MRZ_NOT_DETECTED",
    "FACE_NOT_DETECTED",
    "UNKNOWN_MODULE",
}


def _line2(dob, expiry):
    line = list(VALID_LINE2)
    line[13:19] = dob
    line[19] = str(main.calculate_icao_checksum(dob))
    line[21:27] = expiry
    line[27] = str(main.calculate_icao_checksum(expiry))
    composite = "".join(line[0:10]) + "".join(line[13:20]) + "".join(line[21:28]) + "".join(line[28:43])
    line[43] = str(main.calculate_icao_checksum(composite))
    return "".join(line)


def jpeg_bytes(width=800, height=500):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def png_bytes(width=100, height=100):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


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
        "module_state": "PASS",
        "data": parsed,
    }


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
        "status": "CLEAN",
        "module_state": "PASS",
        "is_tampered": False,
        "ela_mean_intensity": 10.0,
        "ela_std_dev": 12.0,
        "edge_artifact_score": 5.0,
        "metadata_present": True,
        "confidence": 10.0,
        "tampering_score": 10.0,
        "signals": [],
        "indicators": [],
        "explanation": "No anomaly.",
    }


@skip_client
def _screen_with(monkeypatch, mrz_result, face_result, tamper_result, error=None):
    monkeypatch.setattr(main, "extract_mrz_from_image", lambda doc: mrz_result)
    monkeypatch.setattr(main, "extract_and_verify_faces", lambda doc, live=None: face_result)
    if error is not None:
        def _boom(document, quality=90):
            raise error
        monkeypatch.setattr(main, "analyze_tampering_ela", _boom)
    else:
        monkeypatch.setattr(main, "analyze_tampering_ela", lambda doc, **kw: tamper_result)
    client = TestClient(main.app)
    return client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )


# ---------------------------------------------------------------------------
# MRZ structure and date edge cases
# ---------------------------------------------------------------------------

def test_leap_year_dob_is_accepted():
    result = main.parse_td3_mrz(VALID_LINE1, _line2("000229", "490101"))
    assert result["status"] == "VALID"
    assert result["checks"]["dob_valid"] is True


def test_non_leap_feb_29_dob_is_rejected():
    result = main.parse_td3_mrz(VALID_LINE1, _line2("010229", "490101"))
    assert result["status"] != "VALID"


def test_invalid_sex_character_is_malformed():
    bad = VALID_LINE2[:20] + "X" + VALID_LINE2[21:]
    assert main.parse_td3_mrz(VALID_LINE1, bad)["status"] == "MALFORMED"


def test_invalid_nationality_field_is_malformed():
    bad = VALID_LINE2[:10] + "12A" + VALID_LINE2[13:]
    assert main.parse_td3_mrz(VALID_LINE1, bad)["status"] == "MALFORMED"


def test_only_one_of_nationality_sex_dob_checksum_fails_is_invalid(monkeypatch):
    # A structurally valid line with a broken DOB check digit must be INVALID,
    # never silently accepted.
    bad_dob = _line2("740812", "490101")
    bad_dob = bad_dob[:19] + "9" + bad_dob[20:]
    assert main.parse_td3_mrz(VALID_LINE1, bad_dob)["status"] == "INVALID"


# ---------------------------------------------------------------------------
# Risk engine: weights, fail-safety, and decision gating
# ---------------------------------------------------------------------------

def test_every_required_risk_factor_has_a_weight():
    assert REQUIRED_RISK_FACTORS.issubset(main.RISK_WEIGHTS.keys())


def test_risk_weights_are_non_negative():
    assert all(value >= 0 for value in main.RISK_WEIGHTS.values())


def test_maximum_possible_risk_weight_exceeds_the_ceiling():
    # Proves the score bound (max 100) actually matters with default weights.
    assert sum(main.RISK_WEIGHTS.values()) > 100


@skip_client
def test_tampering_inconclusive_returns_valid_response(monkeypatch):
    tamper = dict(_pass_tamper())
    tamper.update({
        "status": "INCONCLUSIVE",
        "module_state": "NOT_AVAILABLE",
        "is_tampered": False,
        "signals": ["EDGE_INCONSISTENCY"],
        "indicators": ["Localized edge inconsistency"],
    })
    response = _screen_with(monkeypatch, _valid_mrz_result(), _pass_face(), tamper)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert any(f["factor"] == "TAMPERING_INCONCLUSIVE" for f in body["risk_assessment"]["factors"])
    assert body["risk_assessment"]["module_statuses"]["tampering"] == "NOT_AVAILABLE"
    # INCONCLUSIVE remains distinguishable from CLEAN.
    assert body["tampering_analysis"]["status"] == "INCONCLUSIVE"


@skip_client
def test_missing_tampering_inconclusive_weight_never_500(monkeypatch):
    # Regression: a runtime RISK_WEIGHTS map without TAMPERING_INCONCLUSIVE
    # must not raise KeyError or return HTTP 500.
    missing = dict(main.RISK_WEIGHTS)
    missing.pop("TAMPERING_INCONCLUSIVE")
    monkeypatch.setattr(main, "RISK_WEIGHTS", missing)
    tamper = dict(_pass_tamper())
    tamper.update({
        "status": "INCONCLUSIVE",
        "module_state": "NOT_AVAILABLE",
        "is_tampered": False,
        "signals": ["EDGE_INCONSISTENCY"],
        "indicators": ["Localized edge inconsistency"],
    })
    response = _screen_with(monkeypatch, _valid_mrz_result(), _pass_face(), tamper)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert any(f["factor"] == "TAMPERING_INCONCLUSIVE" for f in body["risk_assessment"]["factors"])


@skip_client
def test_tampering_analysis_crash_is_safe(monkeypatch):
    response = _screen_with(
        monkeypatch,
        mrz_result=_valid_mrz_result(),
        face_result=_pass_face(),
        tamper_result={},
        error=RuntimeError("tampering exploded internally"),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["module_statuses"]["tampering"] == "ERROR"
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert "exploded" not in json.dumps(body)


@skip_client
def test_high_risk_score_forces_high_risk_review(monkeypatch):
    mrz = {
        "detected": False,
        "source": "ocr",
        "status": "INVALID",
        "confidence": 0.2,
        "module_state": "FAIL",
        "data": {
            "checks": {
                "passport_number_valid": False,
                "dob_valid": True,
                "expiry_valid": True,
                "composite_valid": True,
            },
            "is_expired": True,
        },
    }
    face = {
        "face_detected_in_document": True,
        "face_detected_in_live": True,
        "status": "MISMATCH",
        "match_status": "MISMATCH",
        "similarity_score": 0.1,
        "module_state": "FAIL",
    }
    tamper = dict(_pass_tamper())
    tamper.update({
        "status": "SUSPICIOUS",
        "module_state": "FAIL",
        "is_tampered": True,
        "signals": ["ELA_ANOMALY"],
        "indicators": ["Inconsistent JPEG compression artifacts"],
    })
    response = _screen_with(monkeypatch, mrz_result=mrz, face_result=face, tamper_result=tamper)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["score"] == 100
    assert body["risk_assessment"]["level"] == "HIGH_RISK"
    assert body["risk_assessment"]["decision"] == "HIGH_RISK_REVIEW_REQUIRED"
    assert body["risk_assessment"]["status"] == "RED"


@skip_client
def test_unexpected_internal_exception_returns_safe_500(monkeypatch):
    def boom(result):
        raise RuntimeError("boom-secret-internal-detail")

    monkeypatch.setattr(main, "_tampering_module_state", boom)
    monkeypatch.setattr(main, "extract_mrz_from_image", lambda doc: _valid_mrz_result())
    monkeypatch.setattr(main, "extract_and_verify_faces", lambda doc, live=None: _pass_face())
    monkeypatch.setattr(main, "analyze_tampering_ela", lambda doc, **kw: _pass_tamper())
    client = TestClient(main.app, raise_server_exceptions=False)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "boom-secret-internal-detail" not in json.dumps(body)


# ---------------------------------------------------------------------------
# Face verification states
# ---------------------------------------------------------------------------

class _FakeCascade:
    def __init__(self, faces):
        self._faces = list(faces)

    def empty(self):
        return False

    def detectMultiScale(self, gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)):
        return self._faces.pop(0)


def _monkeypatch_cascade(monkeypatch, faces):
    detector = _FakeCascade(list(faces))
    monkeypatch.setattr(main.cv2, "CascadeClassifier", lambda *args, **kwargs: detector)


def test_face_no_face_in_document(monkeypatch):
    _monkeypatch_cascade(monkeypatch, [np.array([])])
    result = main.extract_and_verify_faces(jpeg_bytes(), jpeg_bytes())
    assert result["status"] == "NO_FACE_IN_DOCUMENT"
    assert result["similarity_score"] is None


def test_face_multiple_faces(monkeypatch):
    _monkeypatch_cascade(monkeypatch, [np.array([(0, 0, 50, 50), (60, 60, 50, 50)])])
    result = main.extract_and_verify_faces(jpeg_bytes(), jpeg_bytes())
    assert result["status"] == "MULTIPLE_FACES"


def test_face_mismatch(monkeypatch):
    _monkeypatch_cascade(monkeypatch, [np.array([(10, 20, 100, 120)]), np.array([(50, 60, 90, 110)])])
    monkeypatch.setattr(main, "_face_embedding", lambda crop: np.array([1.0, 0.0]))
    monkeypatch.setattr(main, "_cosine_similarity", lambda first, second: 0.1)
    result = main.extract_and_verify_faces(jpeg_bytes(), jpeg_bytes())
    assert result["status"] == "MISMATCH"
    assert abs(result["similarity_score"] - 0.1) < 1e-6


def test_face_missing_live_photo(monkeypatch):
    _monkeypatch_cascade(monkeypatch, [np.array([(10, 20, 100, 120)])])
    result = main.extract_and_verify_faces(jpeg_bytes())
    assert result["status"] == "SKIPPED_NO_LIVE_PHOTO"
    assert result["face_bounding_box"] == {"x": 10, "y": 20, "w": 100, "h": 120}


def test_face_invalid_document_image():
    result = main.extract_and_verify_faces(b"not an image", jpeg_bytes())
    assert result["status"] == "INVALID_IMAGE"


# ---------------------------------------------------------------------------
# Upload hardening
# ---------------------------------------------------------------------------

def test_decompression_bomb_protection(monkeypatch):
    original_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 100
        bomb_buffer = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(bomb_buffer, format="JPEG")
        with pytest.raises(HTTPException) as excinfo:
            main._validate_image_bytes(bomb_buffer.getvalue(), "image/jpeg", "Bomb")
        assert excinfo.value.status_code == 413
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit


@skip_client
def test_valid_png_upload_passes_validation(monkeypatch):
    monkeypatch.setattr(main, "extract_mrz_from_image", lambda doc: _valid_mrz_result())
    monkeypatch.setattr(main, "extract_and_verify_faces", lambda doc, live=None: _pass_face())
    monkeypatch.setattr(main, "analyze_tampering_ela", lambda doc, **kw: _pass_tamper())
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200


@skip_client
def test_png_disguised_as_jpeg_is_rejected():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.jpg", png_bytes(), "image/jpeg")},
    )
    assert response.status_code == 415


@skip_client
def test_exe_extension_is_rejected():
    client = TestClient(main.app)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("document.exe", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 415