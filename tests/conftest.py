"""Shared fixtures and environment setup for the screening test suite."""

import io
import os
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="sih_screening_tests_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmpdir.replace(os.sep, '/')}/test.db")


@pytest.fixture
def jpeg_bytes():
    def _make(width=800, height=500):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
        return buffer.getvalue()

    return _make


@pytest.fixture
def png_bytes():
    def _make(width=100, height=100):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    return _make


@pytest.fixture
def app_and_client(tmp_path, monkeypatch):
    import numpy as np
    from app.config import Settings
    from app.main import create_app
    from app.services.face_recognition import DummyBackend, FaceDetectionResult, ModelManager
    from starlette.testclient import TestClient

    db_path = f"sqlite:///{(tmp_path / 'test_shared.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_path)
    settings = Settings.from_env()
    dummy_face = FaceDetectionResult(
        bbox=[10, 20, 110, 140], score=0.9, landmarks=None,
        embedding=np.ones((512,), dtype=np.float32),
    )
    manager = ModelManager(settings, backend=DummyBackend([dummy_face]))
    app = create_app(settings=settings, model_manager=manager)
    client = TestClient(app)
    return app, client
