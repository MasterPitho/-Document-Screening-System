"""
Face recognition engine using a modern embedding model (InsightFace / ArcFace).

This replaces the old prototype that compared raw grayscale pixel vectors with
Haar cascades. Detections are real face detections with landmarks, faces are
aligned by the model, and a fixed-size ArcFace embedding is produced and
compared with cosine similarity.

Privacy: embeddings are computed in memory and never logged or persisted.

The backend is wrapped behind a generic interface so it can be swapped without
changing API code. If the backend fails to load, the engine reports
NOT_AVAILABLE / ERROR instead of crashing the API and never fabricates a MATCH.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger("document_screening")


class FaceDetectionResult:
    """A single detected face with its embedding (aligned by the backend)."""

    def __init__(self, bbox: List[float], score: float, landmarks: Optional[np.ndarray],
                 embedding: Optional[np.ndarray]) -> None:
        self.bbox = bbox
        self.score = score
        self.landmarks = landmarks
        self.embedding = embedding


class FaceBackend(Protocol):
    def prepare(self) -> None: ...
    def detect_and_embed(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]: ...


class _InsightFaceBackend:
    """Backend wrapping InsightFace App (SCRFD detection + ArcFace embedding)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._app: Any = None

    def is_ready(self) -> bool:
        return self._app is not None

    def prepare(self) -> None:
        import os

        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name=self._settings.face_model_name,
            root=os.path.expanduser(self._settings.face_models_dir),
            providers=["CPUExecutionProvider"],
            download=True,
        )
        app.prepare(
            ctx_id=self._settings.face_ctx_id,
            det_size=(self._settings.face_det_size, self._settings.face_det_size),
            det_thresh=self._settings.face_min_detection_confidence,
        )
        self._app = app

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]:
        if self._app is None:
            raise RuntimeError("Face backend not initialized")
        faces = self._app.get(image_bgr)
        results: List[FaceDetectionResult] = []
        for face in faces:
            bbox = [float(v) for v in face.bbox]
            score = float(getattr(face, "det_score", 0.0))
            kps = getattr(face, "kps", None)
            landmarks = np.asarray(kps, dtype=np.float32) if kps is not None else None
            embedding: Optional[np.ndarray]
            normed = getattr(face, "normed_embedding", None)
            if normed is None:
                raw = getattr(face, "embedding", None)
                if raw is None:
                    embedding = None
                else:
                    vec = np.asarray(raw, dtype=np.float32)
                    norm = float(np.linalg.norm(vec))
                    embedding = vec / norm if norm > 0 else None
            else:
                embedding = np.asarray(normed, dtype=np.float32)
            results.append(FaceDetectionResult(
                bbox=bbox, score=score, landmarks=landmarks, embedding=embedding,
            ))
        return results


class DummyBackend(FaceBackend):
    """Injectable backend for tests. Returns preconfigured faces."""

    def __init__(self, results: Optional[List[FaceDetectionResult]] = None) -> None:
        self._results = results or []
        self._prepared = False

    def prepare(self) -> None:
        self._prepared = True

    def detect_and_embed(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]:
        return self._results


class FaceRecognitionEngine:
    """Detection + embedding + cosine comparison for exactly one face per image."""

    STATUS_EXPLANATIONS = {
        "INVALID_IMAGE": "The image could not be decoded.",
        "NO_FACE": "No face was detected in the image.",
        "MULTIPLE_FACES": "More than one face was detected; exactly one is required.",
        "LOW_CONFIDENCE": "The detected face is too small / low confidence for reliable verification.",
        "SKIPPED_NO_LIVE_PHOTO": "No live photo supplied; face verification was skipped.",
        "MATCH": "Embeddings met the configured match threshold.",
        "MISMATCH": "Embeddings did not meet the configured match threshold.",
        "ERROR": "Face recognition failed internally; secondary inspection required.",
        "NOT_AVAILABLE": "The face recognition model is not available.",
    }

    def __init__(self, settings: Settings, backend: Optional[FaceBackend] = None) -> None:
        self.settings = settings
        self.threshold = settings.face_similarity_threshold
        self.min_det_score = settings.face_min_detection_confidence
        self._backend = backend

    def is_ready(self) -> bool:
        if self._backend is None:
            return False
        if isinstance(self._backend, _InsightFaceBackend):
            return self._backend.is_ready()
        return True

    def initialize(self) -> None:
        if self._backend is not None:
            self._backend.prepare()

    @staticmethod
    def _decode_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _face_is_small(img_w: int, img_h: int, bbox: List[float]) -> bool:
        if img_w <= 0 or img_h <= 0:
            return True
        x0, y0, x1, y1 = [float(v) for v in bbox]
        face_w, face_h = x1 - x0, y1 - y0
        if face_w <= 0 or face_h <= 0:
            return True
        ratio = (face_w * face_h) / (img_w * img_h)
        return ratio < 0.02 and max(face_w, face_h) < 80

    @staticmethod
    def _embedding_similarity(first: np.ndarray, second: np.ndarray) -> float:
        dot = float(np.dot(first, second))
        n1 = float(np.linalg.norm(first))
        n2 = float(np.linalg.norm(second))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        value = dot / (n1 * n2)
        return max(0.0, min(1.0, float(value)))

    def _select_single_face(self, img_w: int, img_h: int,
                            faces: List[FaceDetectionResult]) -> tuple[Optional[FaceDetectionResult], Optional[str]]:
        if not faces:
            return None, "NO_FACE"
        if len(faces) > 1:
            return None, "MULTIPLE_FACES"
        face = faces[0]
        if self._face_is_small(img_w, img_h, face.bbox):
            return None, "LOW_CONFIDENCE"
        if face.score < self.min_det_score:
            return None, "LOW_CONFIDENCE"
        if face.embedding is None:
            return None, "ERROR"
        return face, None

    def verify(self, document_bytes: bytes, live_bytes: Optional[bytes],
               operator_name: str = "engine") -> dict[str, Any]:
        base = {
            "operator_name": operator_name,
            "similarity_score": None,
            "threshold": self.threshold,
            "matched": None,
        }
        if not self.is_ready():
            return {**base, "status": "NOT_AVAILABLE",
                    "explanation": self.STATUS_EXPLANATIONS["NOT_AVAILABLE"]}

        doc_bgr = self._decode_bgr(document_bytes)
        if doc_bgr is None:
            return {**base, "status": "INVALID_IMAGE",
                    "explanation": self.STATUS_EXPLANATIONS["INVALID_IMAGE"]}
        doc_h, doc_w = doc_bgr.shape[:2]

        try:
            doc_faces = self._select_and_detect(doc_bgr)
        except Exception:  # noqa: BLE001 - fail safe
            return {**base, "status": "ERROR", "explanation": self.STATUS_EXPLANATIONS["ERROR"]}

        doc_face, doc_state = doc_faces
        if doc_face is None:
            return {**base, "status": doc_state,
                    "explanation": self.STATUS_EXPLANATIONS[doc_state]}

        if live_bytes is None:
            return {
                **base,
                "status": "SKIPPED_NO_LIVE_PHOTO",
                "explanation": self.STATUS_EXPLANATIONS["SKIPPED_NO_LIVE_PHOTO"],
                "face_detected_in_document": True,
                "face_bounding_box": int_bbox(doc_face.bbox),
            }

        live_bgr = self._decode_bgr(live_bytes)
        if live_bgr is None:
            return {**base, "status": "INVALID_IMAGE",
                    "explanation": self.STATUS_EXPLANATIONS["INVALID_IMAGE"],
                    "face_detected_in_document": True}
        live_h, live_w = live_bgr.shape[:2]
        try:
            live_faces = self._select_and_detect(live_bgr)
        except Exception:  # noqa: BLE001
            return {**base, "status": "ERROR",
                    "explanation": self.STATUS_EXPLANATIONS["ERROR"],
                    "face_detected_in_document": True}

        live_face, live_state = live_faces
        if live_face is None:
            return {**base, "status": live_state,
                    "explanation": self.STATUS_EXPLANATIONS[live_state],
                    "face_detected_in_document": True}

        similarity = self._embedding_similarity(doc_face.embedding, live_face.embedding)
        matched = similarity >= self.threshold
        state = "MATCH" if matched else "MISMATCH"
        return {
            **base,
            "status": state,
            "explanation": self.STATUS_EXPLANATIONS[state],
            "face_detected_in_document": True,
            "face_detected_in_live": True,
            "similarity_score": round(similarity, 4),
            "matched": matched,
            "face_bounding_box": int_bbox(doc_face.bbox),
        }

    def _select_and_detect(self, image_bgr: np.ndarray):
        if self._backend is None:
            return None, "NOT_AVAILABLE"
        detected = self._backend.detect_and_embed(image_bgr)
        h, w = image_bgr.shape[:2]
        return self._select_single_face(w, h, detected)

    @staticmethod
    def _quality_signals(gray: np.ndarray) -> dict[str, float]:
        if gray.size == 0:
            return {"sharpness": 0.0, "brightness": 0.0, "contrast": 0.0}
        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness = min(1.0, laplacian / 500.0)
        brightness = float(np.mean(gray) / 255.0)
        contrast = float(np.std(gray) / 255.0)
        return {"sharpness": round(sharpness, 4), "brightness": round(brightness, 4),
                "contrast": round(contrast, 4)}


def int_bbox(bbox: List[float]) -> dict[str, int]:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    return {"x": x0, "y": y0, "w": max(0, x1 - x0), "h": max(0, y1 - y0)}


class ModelManager:
    """Loads face models once at startup and shares them across requests.

    Initialization failure is handled cleanly: readiness is reported as false
    and the engine stays in NOT_AVAILABLE rather than crashing the API.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self, settings: Settings, backend: Optional[FaceBackend] = None) -> None:
        self._settings = settings
        self.face_engine = FaceRecognitionEngine(settings, backend=backend)
        self._backends_prepared = backend is not None
        self._error: Optional[str] = None

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(settings or Settings.from_env())
        return cls._instance

    def initialize_face_backend(self) -> None:
        """Create the real InsightFace backend and load models. Called at startup."""
        try:
            backend = _InsightFaceBackend(self._settings)
            backend.prepare()
            self.face_engine._backend = backend
            self._backends_prepared = True
            self._error = None
        except Exception as exc:  # noqa: BLE001 - model loading must not crash the API
            self._backends_prepared = False
            self._error = f"{type(exc).__name__}"
            logger.warning("face_model_init_failed type=%s", type(exc).__name__)
            self.face_engine._backend = None

    def face_is_ready(self) -> bool:
        return self._backends_prepared and self.face_engine.is_ready()

    def readiness(self) -> dict[str, Any]:
        return {"face_recognition": self.face_is_ready(), "error": self._error}
