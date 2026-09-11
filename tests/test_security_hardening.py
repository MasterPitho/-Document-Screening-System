"""Security hardening tests: upload constraints, path traversal, decompression bombs, and model lifecycle."""

import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.config import Settings
from app.security.image_validation import ImageValidationLimits
from app.services.face_recognition import ModelManager, DummyBackend, FaceDetectionResult
import numpy as np


def _jpeg_bytes(width=100, height=100):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="JPEG")
    return buf.getvalue()


def _png_bytes(width=100, height=100):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(create_app())


def test_path_traversal_filename_safe(client):
    """Filenames with path traversal sequences do not write to disk or cause errors."""
    traversal_filenames = [
        "../../../../etc/passwd.jpg",
        "..\\..\\..\\windows\\system32\\calc.png",
        "/var/log/syslog.webp",
        "....//....//doc.jpg",
    ]
    img = _jpeg_bytes()
    for filename in traversal_filenames:
        ext = filename.rsplit(".", 1)[-1]
        mime = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
        data = _png_bytes() if ext == "png" else img
        response = client.post(
            "/api/v1/screen",
            files={"document_image": (filename, data, mime)},
        )
        # Should process normally or validate safely, NOT 500 error or disk write
        assert response.status_code in (200, 400, 415)
        # Confirm no file was created on disk in cwd
        assert not os.path.exists(filename)


def test_filename_with_null_bytes(client):
    """Filename containing null byte does not cause internal server error."""
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc\x00traversal.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    # Fastapi/starlette might sanitize or pass through; response should not be unhandled 500
    assert response.status_code in (200, 400, 415, 422)


def test_decompression_bomb_rejected(client):
    """Images exceeding pixel dimensions are rejected with 413."""
    settings = create_app().state.settings
    validator = ImageValidationLimits(settings)

    # Construct an image exceeding max_pixels
    buf = io.BytesIO()
    w = settings.max_image_width + 10
    h = 100
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG")
    payload = buf.getvalue()

    with pytest.raises(Exception) as exc_info:
        validator.validate(payload, "image/jpeg", "Document image", "test.jpg")
    assert exc_info.value.status_code == 413


def test_spoofed_mime_type_rejected(client):
    """HTML / executable content spoofed as image/jpeg is rejected."""
    fake_jpeg = b"<html><script>alert('xss')</script></html>"
    response = client.post(
        "/api/v1/screen",
        files={"document_image": ("doc.jpg", fake_jpeg, "image/jpeg")},
    )
    assert response.status_code in (400, 415)
    assert response.json()["error"]["code"] in ("BAD_REQUEST", "UNSUPPORTED_MEDIA_TYPE")


def test_model_manager_single_load_lifecycle():
    """ModelManager loads backend once and reuses it across requests."""
    backend = DummyBackend([FaceDetectionResult([0, 0, 10, 10], 0.95, None, np.zeros(128, dtype=np.float32))])
    manager = ModelManager(Settings.from_env(), backend=backend)

    assert manager.face_is_ready()
    assert manager.readiness()["face_recognition"] is True

    # Test singleton get_instance
    ModelManager._instance = None
    inst1 = ModelManager.get_instance(Settings.from_env())
    inst2 = ModelManager.get_instance()
    assert inst1 is inst2
    ModelManager._instance = None
