"""Behaviour when the database is unreachable: controlled 503s, no stack
traces, no misleading SCREENED/persisted responses."""

import io
import uuid
from dataclasses import replace

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.db.database import build_database
from app.main import create_app
from app.services import mrz as mrz_mod
from app.services.face_recognition import DummyBackend, ModelManager

DEAD_URL = "postgresql+psycopg://screening:screening@127.0.0.1:1/nowhere"


class CleanTampering:
    def analyze(self, image_bytes):
        return {"status": "CLEAN", "score": 0.0, "confidence": 0.0,
                "signals": {}, "suspicious_regions": [], "explanation": []}


class SequenceBackend:
    def __init__(self, results) -> None:
        self._queue = [list(r) for r in results]

    def prepare(self):
        pass

    def detect_and_embed(self, image_bgr):
        if self._queue:
            return self._queue.pop(0)
        return []


def _face():
    import numpy as np
    from app.services.face_recognition import FaceDetectionResult
    return FaceDetectionResult(
        bbox=[10, 20, 110, 140], score=0.9, landmarks=None,
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )


def _dead_app():
    settings = replace(Settings.from_env(), db_connect_retries=1, db_retry_delay=0.0)
    dead = build_database(DEAD_URL, connect_timeout_s=1)
    manager = ModelManager(settings, backend=DummyBackend([_face()]))
    app = create_app(settings=settings, model_manager=manager, database=dead)
    app.state.tampering = CleanTampering()
    return app


def _jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 500), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_ready_reports_database_unavailable():
    client = TestClient(_dead_app())
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["modules"]["database"] is False


def test_screen_returns_503_database_unavailable(monkeypatch):
    monkeypatch.setattr(mrz_mod, "extract_mrz_from_image",
                        lambda doc, settings: {
                            "detected": False, "source": "ocr",
                            "status": "NOT_DETECTED", "confidence": 0.0,
                            "module_state": "NOT_AVAILABLE"})
    client = TestClient(_dead_app(), raise_server_exceptions=False)
    response = client.post(
        "/api/v1/screen",
        headers={"X-Request-ID": uuid.uuid4().hex},
        files={"document_image": ("doc.jpg", _jpeg(), "image/jpeg")},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "DATABASE_UNAVAILABLE"
    # No raw exception detail, no module internals, no persisted claims.
    assert "Traceback" not in response.text
    assert "OperationalError" not in response.text
    assert "psycopg" not in response.text
    assert "risk_assessment" not in body
    assert body.get("persistence") is None
    assert body["request_id"]
