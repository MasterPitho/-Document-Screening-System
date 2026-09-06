"""Face recognition engine tests (ArcFace embeddings, injectable backend)."""

import io
from dataclasses import replace

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.face_recognition import (
    DummyBackend,
    FaceDetectionResult,
    FaceRecognitionEngine,
    ModelManager,
    int_bbox,
)


def _img_bytes(width=200, height=200):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


DOC_IMG = _img_bytes()


def _settings(**tw) -> Settings:
    return replace(Settings.from_env(), **tw)


def _face(bbox=(10, 20, 110, 140), score=0.9, embedding=(1.0, 0.0), likely=0.8):
    return FaceDetectionResult(
        bbox=list(bbox),
        score=score,
        landmarks=None,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


class SequenceBackend:
    """Returns a preconfigured result per image in document-then-live order."""

    def __init__(self, results) -> None:
        self._queue = [list(r) for r in results]
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def detect_and_embed(self, image_bgr):
        if self._queue:
            return self._queue.pop(0)
        return []


def _engine(backend=None, settings=None):
    return FaceRecognitionEngine(settings or _settings(), backend=backend)


def test_not_available_when_backend_missing():
    result = _engine().verify(b"junk", None)
    assert result["status"] == "NOT_AVAILABLE"
    assert result["matched"] is None
    assert result["similarity_score"] is None


def test_invalid_document_image():
    backend = SequenceBackend([[_face()]])
    result = _engine(backend=backend).verify(b"not-an-image", None)
    assert result["status"] == "INVALID_IMAGE"


def test_no_face_in_document():
    backend = SequenceBackend([[]])
    result = _engine(backend=backend).verify(DOC_IMG, None)
    assert result["status"] == "NO_FACE"
    assert result["similarity_score"] is None


def test_multiple_faces_in_document():
    backend = SequenceBackend([[_face((0, 0, 60, 60)), _face((10, 10, 70, 70))]])
    result = _engine(backend=backend).verify(DOC_IMG, None)
    assert result["status"] == "MULTIPLE_FACES"


def test_small_face_is_low_confidence():
    backend = SequenceBackend([[_face(bbox=(0, 0, 20, 20))]])
    result = _engine(backend=backend).verify(DOC_IMG, None)
    assert result["status"] == "LOW_CONFIDENCE"


def test_low_detection_confidence_is_low_confidence():
    backend = SequenceBackend([[_face(score=0.1)]])
    result = _engine(backend=backend).verify(DOC_IMG, None)
    assert result["status"] == "LOW_CONFIDENCE"


def test_skipped_no_live_photo_reports_bounding_box():
    backend = SequenceBackend([[_face()]])
    result = _engine(backend=backend).verify(DOC_IMG, None)
    assert result["status"] == "SKIPPED_NO_LIVE_PHOTO"
    assert result["face_bounding_box"] == int_bbox((10, 20, 110, 140))


def test_match_reports_similarity_and_bounding_box():
    doc = _face(embedding=(1.0, 0.0))
    live = _face(bbox=(50, 60, 140, 170), embedding=(1.0, 0.0))
    backend = SequenceBackend([[doc], [live]])
    result = _engine(backend=backend).verify(
        DOC_IMG,
        DOC_IMG,
    )
    assert result["status"] == "MATCH"
    assert result["matched"] is True
    assert abs(result["similarity_score"] - 1.0) < 1e-6
    assert result["face_bounding_box"] == int_bbox((10, 20, 110, 140))


def test_mismatch_never_clears():
    doc = _face(embedding=(1.0, 0.0))
    live = _face(embedding=(0.0, 1.0))
    backend = SequenceBackend([[doc], [live]])
    result = _engine(backend=backend).verify(
        DOC_IMG,
        DOC_IMG,
    )
    assert result["status"] == "MISMATCH"
    assert result["matched"] is False


def test_zero_embedding_resolves_to_mismatch_not_nan():
    doc = _face(embedding=(0.0, 0.0))
    live = _face(embedding=(0.0, 0.0))
    backend = SequenceBackend([[doc], [live]])
    result = _engine(backend=backend).verify(
        DOC_IMG,
        DOC_IMG,
    )
    assert not np.isnan(result["similarity_score"])
    assert result["status"] == "MISMATCH"


def test_live_photo_that_fails_to_decode_is_invalid_image():
    backend = SequenceBackend([[_face()]])
    result = _engine(backend=backend).verify(
        DOC_IMG, b"not-an-image")
    assert result["status"] == "INVALID_IMAGE"


def test_threshold_is_respected():
    # Identical embeddings with a threshold of 1.01 can never match.
    strict = _engine(backend=SequenceBackend([[_face()], [_face()]]),
                     settings=_settings(face_similarity_threshold=1.01))
    result = strict.verify(
        DOC_IMG,
        DOC_IMG,
    )
    assert result["status"] == "MISMATCH"
    assert result["threshold"] == 1.01

    # A permissive threshold treats the same embeddings as a match.
    permissive = _engine(backend=SequenceBackend([[_face()], [_face()]]),
                         settings=_settings(face_similarity_threshold=0.2))
    result = permissive.verify(
        DOC_IMG,
        DOC_IMG,
    )
    assert result["status"] == "MATCH"


def test_embedding_similarity_normalizes_vectors():
    engine = _engine()
    # Same direction, different magnitudes -> cosine 1.0.
    first = np.asarray([3.0, 4.0], dtype=np.float32)
    second = np.asarray([30.0, 40.0], dtype=np.float32)
    assert abs(engine._embedding_similarity(first, second) - 1.0) < 1e-6
    # Orthogonal -> 0.0 (bounded, not negative).
    assert engine._embedding_similarity(np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])) == 0.0
    # Zero vectors -> defensive 0.0, never NaN.
    assert engine._embedding_similarity(np.zeros(4), np.zeros(4)) == 0.0


def test_engine_reports_threshold_in_result():
    backend = SequenceBackend([[_face()]])
    result = _engine(backend=backend, settings=_settings(face_similarity_threshold=0.42)).verify(
        DOC_IMG, None)
    assert result["threshold"] == 0.42


class _BrokenInsightFaceBackend:
    def __init__(self, settings):
        pass

    def prepare(self):
        raise RuntimeError("model download failed offline")


def test_model_manager_init_failure_is_fail_safe(monkeypatch):
    import app.services.face_recognition as face_mod

    monkeypatch.setattr(face_mod, "_InsightFaceBackend", _BrokenInsightFaceBackend)
    manager = ModelManager(_settings())
    manager.initialize_face_backend()
    assert manager.face_is_ready() is False
    assert manager.face_engine.is_ready() is False
    readiness = manager.readiness()
    assert readiness["face_recognition"] is False
    result = manager.face_engine.verify(b"anything", None)
    assert result["status"] == "NOT_AVAILABLE"


def test_model_manager_with_injected_backend_is_ready():
    manager = ModelManager(_settings(), backend=DummyBackend([_face()]))
    assert manager.face_is_ready() is True
    assert manager.readiness()["face_recognition"] is True


def test_model_manager_without_backend_is_not_ready():
    manager = ModelManager(_settings())
    assert manager.face_is_ready() is False
    assert manager.readiness()["face_recognition"] is False
