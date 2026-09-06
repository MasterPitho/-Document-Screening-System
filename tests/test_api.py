"""API-level tests: upload security, request IDs, readiness, screening flow."""

import io
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.security.image_validation import ImageValidationLimits
from app.services.face_recognition import (
    DummyBackend,
    FaceDetectionResult,
    ModelManager,
)
from app.services import mrz as mrz_mod

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
UNEXPIRED_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"


def _face(embedding=(1.0, 0.0), bbox=(10, 20, 110, 140), score=0.9):
    return FaceDetectionResult(
        bbox=list(bbox), score=score, landmarks=None,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


class SequenceBackend:
    def __init__(self, results) -> None:
        self._queue = [list(r) for r in results]

    def prepare(self):
        pass

    def detect_and_embed(self, image_bgr):
        if self._queue:
            return self._queue.pop(0)
        return []


class StubTampering:
    def __init__(self, result) -> None:
        self._result = result

    def analyze(self, image_bytes):
        return dict(self._result)


class StubLiveness:
    """Deterministic liveness for API tests; defaults to LIVE."""

    def __init__(self, result=None) -> None:
        self._result = result or {
            "is_live": True, "liveness_score": 0.9, "liveness_status": "LIVE",
            "method": "stub", "model_used": None, "signals": {}, "reasons": [],
            "explanation": "stubbed live result",
        }

    def analyze(self, image_bytes):
        return dict(self._result)

    def _not_checked(self, explanation):
        return {
            "is_live": False, "liveness_score": 0.0, "liveness_status": "NOT_CHECKED",
            "method": "not_checked", "model_used": None, "signals": {},
            "reasons": [], "explanation": explanation,
        }

    def readiness(self):
        return {"liveness": True, "liveness_method": "stub"}


@pytest.fixture
def client():
    return TestClient(create_app())


def _jpeg(width=800, height=500):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body.keys()) <= {"status", "service", "env"}


def test_ready_reports_not_ready_without_models(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["modules"]["face_recognition"] is False


def test_ready_reports_ready_with_loaded_model():
    manager = ModelManager(create_app().state.settings, backend=DummyBackend([_face()]))
    app = create_app(model_manager=manager)
    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# X-Request-ID
# ---------------------------------------------------------------------------

def test_request_id_is_echoed_back(client):
    response = client.get("/health", headers={"X-Request-ID": "my-correlation-id-1"})
    assert response.headers["x-request-id"] == "my-correlation-id-1"


def test_request_id_generated_when_missing(client):
    response = client.get("/health")
    assert len(response.headers["x-request-id"]) == 32


def test_request_id_overlong_is_replaced(client):
    response = client.get("/health", headers={"X-Request-ID": "z" * 200})
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32
    assert "z" * 200 != request_id


def test_request_id_is_sanitized(client):
    response = client.get("/health", headers={"X-Request-ID": "abc def/ghi<script>"})
    assert " " not in response.headers["x-request-id"]
    assert "<" not in response.headers["x-request-id"]


def test_request_id_in_screen_body_matches_header(client):
    response = client.post(
        "/api/v1/screen",
        headers={"X-Request-ID": "screening-0001"},
        files={"document_image": ("doc.jpg", _jpeg(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "screening-0001"
    assert response.headers["x-request-id"] == "screening-0001"


def test_error_response_contains_request_id(client):
    response = client.post(
        "/api/v1/screen",
        headers={"X-Request-ID": "err-0001"},
        files={"document_image": ("doc.txt", b"x", "text/plain")},
    )
    assert response.status_code == 415
    assert response.headers["x-request-id"] == "err-0001"
    assert response.json()["request_id"] == "err-0001"


# ---------------------------------------------------------------------------
# Upload security
# ---------------------------------------------------------------------------

def test_rejects_empty_file(client, jpeg_bytes):
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "BAD_REQUEST"


def test_rejects_oversized_file(client, jpeg_bytes):
    huge = b"x" * (create_app().state.settings.max_image_bytes + 1)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", huge, "image/jpeg")},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_rejects_disallowed_content_type(client, jpeg_bytes):
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.txt", jpeg_bytes(), "text/plain")},
    )
    assert response.status_code == 415


def test_rejects_disallowed_extension(client, jpeg_bytes):
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.exe", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 415


def test_rejects_declared_type_mismatch(client, png_bytes):
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", png_bytes(), "image/jpeg")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_rejects_corrupt_image(client):
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", b"not-an-image", "image/jpeg")},
    )
    assert response.status_code == 400


def test_rejects_too_wide_image(client):
    settings = create_app().state.settings
    buffer = io.BytesIO()
    Image.new("RGB", (settings.max_image_width + 1, 1), "white").save(buffer, format="JPEG")
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", buffer.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 413


def test_rejects_invalid_live_photo(client, jpeg_bytes):
    response = client.post(
        "/api/v1/screen",
        files={
            "document_image": ("doc.jpg", jpeg_bytes(), "image/jpeg"),
            "live_photo": ("live.jpg", b"corrupt-garbage", "image/jpeg"),
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "BAD_REQUEST"


def test_missing_document_field_returns_422(client):
    response = client.post("/api/v1/screen")
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["detail"], list)


# ---------------------------------------------------------------------------
# Image validation unit tests
# ---------------------------------------------------------------------------

def test_image_validation_limits_reject_decompression_bomb(jpeg_bytes, monkeypatch):
    from PIL import Image as PILImage

    original = PILImage.MAX_IMAGE_PIXELS
    try:
        PILImage.MAX_IMAGE_PIXELS = 100
        limits = ImageValidationLimits(create_app().state.settings)
        with pytest.raises(Exception) as excinfo:
            limits.validate(jpeg_bytes(), "image/jpeg", "Bomb")
        assert excinfo.value.status_code == 413
    finally:
        PILImage.MAX_IMAGE_PIXELS = original


def test_image_validation_limits_accept_valid_png(png_bytes):
    limits = ImageValidationLimits(create_app().state.settings)
    limits.validate(png_bytes(), "image/png", "Doc")
    # No exception means validation passed.


# ---------------------------------------------------------------------------
# Screening flow with injected modules
# ---------------------------------------------------------------------------

def _screen_app(face_backend=None, tampering_result=None, liveness_result=None):
    settings = create_app().state.settings
    manager = ModelManager(settings, backend=face_backend or DummyBackend([]))
    app = create_app(model_manager=manager)
    if tampering_result is not None:
        app.state.tampering = StubTampering(tampering_result)
    app.state.liveness = StubLiveness(liveness_result)
    return app


def _post_screen(app, *, live=False, data=None, headers=None):
    client = TestClient(app)
    files = {"document_image": ("doc.jpg", _jpeg(), "image/jpeg")}
    if live:
        files["live_photo"] = ("live.jpg", _jpeg(), "image/jpeg")
    if headers is None:
        headers = {"X-Request-ID": f"screen-test-{uuid.uuid4().hex[:12]}"}
    return client.post(
        "/api/v1/screen", data=data or {}, files=files,
        headers=headers,
    )


def _valid_mrz_form_data():
    return {"mrz_line1": VALID_LINE1, "mrz_line2": UNEXPIRED_LINE2}


def test_screen_clears_when_all_modules_pass():
    backend = SequenceBackend([[_face()], [_face()]])
    app = _screen_app(face_backend=backend,
                      tampering_result={"status": "CLEAN", "score": 0.0, "confidence": 0.0})
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SCREENED"
    assert body["risk_assessment"]["decision"] == "CLEARED"
    assert body["risk_assessment"]["status"] == "GREEN"
    assert body["face_verification"]["status"] == "MATCH"
    assert body["mrz"]["detected"] is True


def test_screen_face_mismatch_requires_review():
    backend = SequenceBackend([[_face(embedding=(1.0, 0.0))], [_face(embedding=(0.0, 1.0))]])
    app = _screen_app(face_backend=backend,
                      tampering_result={"status": "CLEAN", "score": 0.0})
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert any(f["factor"] == "FACE_MISMATCH" for f in body["risk_assessment"]["factors"])


def test_screen_no_live_photo_requires_review():
    backend = SequenceBackend([[_face()]])
    app = _screen_app(face_backend=backend,
                      tampering_result={"status": "CLEAN", "score": 0.0})
    response = _post_screen(app, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["face_verification"]["status"] == "SKIPPED_NO_LIVE_PHOTO"
    assert body["risk_assessment"]["decision"] == "SECONDARY_INSPECTION_REQUIRED"
    assert body["risk_assessment"]["module_statuses"]["face"] == "NOT_AVAILABLE"


def test_screen_tampering_suspicious_requires_review():
    backend = SequenceBackend([[_face()], [_face()]])
    app = _screen_app(
        face_backend=backend,
        tampering_result={"status": "SUSPICIOUS", "score": 80.0, "suspicious_regions": [],
                          "signals": {}, "explanation": ["forced"]},
    )
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] != "CLEARED"
    assert body["risk_assessment"]["module_statuses"]["tampering"] == "FAIL"
    assert any(f["factor"] == "TAMPERING_SUSPECTED" for f in body["risk_assessment"]["factors"])


def test_screen_partial_mrz_input_is_400(client, jpeg_bytes):
    response = client.post(
        "/api/v1/screen",
        data={"mrz_line1": VALID_LINE1},
        files={"document_image": ("doc.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_screen_ocr_fallback(monkeypatch):
    mrz_result = {
        "detected": True, "source": "ocr", "status": "VALID", "confidence": 1.0,
        "line1": VALID_LINE1, "line2": UNEXPIRED_LINE2,
        "data": mrz_mod.parse_td3_mrz(VALID_LINE1, UNEXPIRED_LINE2),
    }
    monkeypatch.setattr(mrz_mod, "extract_mrz_from_image", lambda doc, settings: mrz_result)
    app = _screen_app(face_backend=SequenceBackend([[_face()]]),
                      tampering_result={"status": "CLEAN", "score": 0.0})
    response = _post_screen(app)
    assert response.status_code == 200
    body = response.json()
    assert body["mrz"]["source"] == "ocr"
    assert body["mrz"]["module_state"] == "PASS"


def test_screen_response_includes_liveness():
    backend = SequenceBackend([[_face()], [_face()]])
    app = _screen_app(face_backend=backend,
                      tampering_result={"status": "CLEAN", "score": 0.0})
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["liveness"]["liveness_status"] == "LIVE"
    assert body["liveness"]["is_live"] is True
    assert body["liveness"]["module_state"] == "PASS"
    assert body["modules"]["liveness"]["liveness_status"] == "LIVE"


def test_screen_liveness_spoof_forces_high_risk():
    backend = SequenceBackend([[_face()], [_face()]])
    spoof = {
        "is_live": False, "liveness_score": 0.1, "liveness_status": "SPOOF_DETECTED",
        "method": "heuristic", "model_used": None, "signals": {}, "reasons": [],
        "explanation": "presentation attack",
    }
    app = _screen_app(
        face_backend=backend,
        tampering_result={"status": "CLEAN", "score": 0.0},
        liveness_result=spoof,
    )
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["risk_assessment"]["decision"] == "HIGH_RISK_REVIEW_REQUIRED"
    assert body["risk_assessment"]["level"] == "HIGH_RISK"
    assert body["risk_assessment"]["module_statuses"]["liveness"] == "FAIL"
    assert any(f["factor"] == "LIVENESS_FAILED"
               for f in body["risk_assessment"]["factors"])


def test_screen_liveness_not_checked_shows_fail_safe():
    backend = SequenceBackend([[_face()], [_face()]])
    not_checked = {
        "is_live": False, "liveness_score": 0.0, "liveness_status": "NOT_CHECKED",
        "method": "not_checked", "model_used": None, "signals": {}, "reasons": [],
        "explanation": "No live photo supplied.",
    }
    app = _screen_app(
        face_backend=backend,
        tampering_result={"status": "CLEAN", "score": 0.0},
        liveness_result=not_checked,
    )
    response = _post_screen(app, live=True, data=_valid_mrz_form_data())
    assert response.status_code == 200
    body = response.json()
    assert body["liveness"]["liveness_status"] == "NOT_CHECKED"
    assert body["risk_assessment"]["module_statuses"]["liveness"] == "NOT_AVAILABLE"
    assert body["risk_assessment"]["decision"] != "CLEARED"


def test_screen_invalid_document_type_returns_422():
    app = _screen_app()
    response = _post_screen(app, data={"document_type": "drivers_licence"})
    assert response.status_code == 422


def test_screen_internal_exception_returns_safe_500():
    app = _screen_app(face_backend=DummyBackend([]),
                      tampering_result={"status": "CLEAN", "score": 0.0})

    class _Boom:
        def evaluate(self, **kwargs):
            raise RuntimeError("secret-internal-detail")

    app.state.risk_engine = _Boom()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", _jpeg(), "image/jpeg")},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert "secret-internal-detail" not in str(body)
