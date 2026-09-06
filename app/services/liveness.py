"""
Passive presentation attack detection (PAD) / liveness screening.

This module detects common presentation attacks on the *live* capture
(printed photos, screen recaptures) using a dual-layer strategy:

1. Primary: a lightweight ONNX anti-spoofing model wrapper (MiniFASNet /
   Silent-Face-Anti-Spoofing logits convention). Activated only when
   ``LIVENESS_MODEL_PATH`` points at an ``.onnx`` file that was baked into
   the deployment.
2. Heuristic fallback: a pure OpenCV/Numpy pipeline that analyses
   high-frequency (FFT) power distribution, Laplacian edge-variance (blur),
   luminance entropy, screen-moire periodicity, and HSV/YCrCb colour-space
   texture to flag print/web recaptures.

Privacy contract: all computation happens strictly in memory. No crops,
frames, or scores are ever written to disk or logged. The live bytes are
decoded in memory and discarded on return.

Boundaries: this is *passive* anti-spoofing, not active-challenge PAD and
not a certified security control. A ``SPOOF_DETECTED`` verdict inside the
risk engine forces ``HIGH_RISK_REVIEW_REQUIRED``; ``UNCERTAIN`` applies a
review penalty. The human officer remains the final decision maker.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from app.config import Settings

logger = logging.getLogger("document_screening")

# Liveness status vocabulary (stable contract for the API and risk engine).
# ``SKIPPED`` means the check was intentionally not run (module disabled by
# configuration); ``NOT_CHECKED`` is the fail-safe when no live capture is
# available or the capture cannot be processed.
LIVENESS_LIVE = "LIVE"
LIVENESS_SPOOF = "SPOOF_DETECTED"
LIVENESS_UNCERTAIN = "UNCERTAIN"
LIVENESS_NOT_CHECKED = "NOT_CHECKED"
LIVENESS_SKIPPED = "SKIPPED"

_METHOD_ONNX = "onnx"
_METHOD_HEURISTIC = "heuristic"
_METHOD_NOT_CHECKED = "not_checked"


@dataclass
class LivenessResult:
    """Typed heuristic presentation-attack verdict returned by ``analyze``.

    This is an *additional triage signal*, not a certified biometric PAD
    verdict: it only estimates whether the capture looks like a common
    printout/screen replay. ``score`` is a deterministic fusion in
    ``[0, 1]`` (higher = more live-like) and is always finite (NaN/inf are
    forced to the spoof-bias value). ``status`` is one of ``LIVE``,
    ``SPOOF_DETECTED``, ``UNCERTAIN``, ``NOT_CHECKED`` or ``SKIPPED``.
    ``detail`` is a human-readable explanation.
    """

    is_live: bool = False
    score: float = 0.0
    status: str = LIVENESS_NOT_CHECKED
    detail: str = ""
    method: str = _METHOD_NOT_CHECKED
    model_used: Optional[str] = None
    signals: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Legacy dict shape consumed by the risk engine, persistence and API."""
        return {
            "is_live": self.is_live,
            "liveness_score": round(float(self.score), 4),
            "liveness_status": self.status,
            "method": self.method,
            "model_used": self.model_used,
            "signals": dict(self.signals or {}),
            "reasons": list(self.reasons or []),
            "explanation": self.detail,
        }


def _decode_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class _OnnxAntiSpoofBackend:
    """Minimal ONNX Runtime wrapper for a MiniFASNet-style anti-spoofing net.

    The wrapper is deliberately generic: it resizes the (in-memory) face crop
    to ``LIVENESS_MODEL_INPUT_SIZE`` and follows the MiniFASNet /
    Silent-Face-Anti-Spoofing output convention:

    - output of width 2  -> treated as two-class logits ``[spoof, live]``
      and passed through softmax (live probability = second class);
    - anything else       -> passed through a logistic sigmoid.

    The returned value is a liveness score in ``[0, 1]`` (1 = live).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session: Any = None
        self._input_name: Optional[str] = None
        self._model_path = settings.liveness_model_path or ""

    def is_ready(self) -> bool:
        return self._session is not None

    def prepare(self) -> None:
        """Load the ONNX model. Failure raises; the caller fails safe."""
        import onnxruntime as ort

        path = self._model_path
        if not path:
            raise FileNotFoundError("LIVENESS_MODEL_PATH is not configured")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Liveness ONNX model not found: {path}")

        providers = ["CPUExecutionProvider"]
        if self._settings.liveness_model_ctx_id >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("liveness_onnx_loaded path=%s", os.path.basename(path))

    def predict(self, face_bgr: np.ndarray) -> float:
        if self._session is None:
            raise RuntimeError("Liveness ONNX backend not initialized")
        size = self._settings.liveness_model_input_size
        resized = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)
        tensor = np.ascontiguousarray(
            resized[:, :, ::-1].astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[None, ...]

        output = self._session.run(None, {self._input_name: tensor})[0]
        flat = np.asarray(output).reshape(-1)
        if flat.size == 0:
            return 0.0
        if flat.size == 2:
            exp = np.exp(flat - flat.max())
            probs = exp / exp.sum()
            return max(0.0, min(1.0, float(probs[1])))
        value = float(flat[0])
        sigmoid = 1.0 / (1.0 + math.exp(-value))
        return max(0.0, min(1.0, sigmoid))


class HeuristicLivenessAnalyzer:
    """OpenCV texture/frequency passive anti-spoofing fallback.

    Signals (each in ``[0, 1]`` where higher = more "live"):
    - ``blur``: Laplacian edge variance — printouts/screen recaptures blur
      high-frequency facial texture.
    - ``high_frequency``: FFT high-frequency power share — genuine optics
      retain fine texture; prints lose it.
    - ``entropy``: luminance histogram entropy — flat/postprocessed posters
      and posterized prints have low entropy.
    - ``moire``: periodic screen-pattern autocorrelation — screen
      recaptures exhibit repeated luminance bands (moiré).
    - ``colorfulness``: HSV/YCbCr colour-texture spread — colour prints and
      screen crops are de-saturated compared with live skin in typical cases.
    """

    def analyze(self, image_bgr: np.ndarray) -> dict[str, Any]:
        height, width = image_bgr.shape[:2]
        if height < 16 or width < 16:
            return {"score": 0.0, "reasons": ["Image too small to analyse."],
                    "signals": {}}

        # Centre crop approximates the face region without a detector.
        h0, h1 = height // 3, height // 3 * 2
        w0, w1 = width // 3, width // 3 * 2
        centre = image_bgr[h0:h1, w0:w1]
        gray = cv2.cvtColor(centre, cv2.COLOR_BGR2GRAY)

        signals: dict[str, float] = {}
        reasons: list[str] = []

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur = max(0.0, min(1.0, laplacian_var / 120.0))
        signals["blur"] = round(blur, 4)
        reasons.append(f"Laplacian blur variance {laplacian_var:.1f}")

        hf = self._fft_high_frequency(gray)
        signals["high_frequency"] = round(hf, 4)
        reasons.append(f"FFT high-frequency share {hf:.3f}")

        entropy, peakiness = self._luminance_profile(gray)
        signals["entropy"] = round(entropy, 4)
        reasons.append(f"Luminance entropy {entropy:.3f}")

        moire = self._screen_moire_strength(gray)
        signals["moire"] = round(moire, 4)
        reasons.append(f"Screen-moiré score {moire:.3f}")

        colorfulness = self._colorfulness(centre)
        signals["colorfulness"] = round(colorfulness, 4)
        reasons.append(f"Colourfulness {colorfulness:.3f}")

        # Weighted fusion. Blank/flat images collapse to a low score on every
        # signal and therefore read as SPOOF, which is the fail-safe choice
        # for a missing camera capture.
        score = (
            0.30 * blur
            + 0.25 * hf
            + 0.15 * entropy
            + 0.20 * moire
            + 0.10 * colorfulness
        )
        score = max(0.0, min(1.0, float(score)))
        return {"score": round(score, 4), "reasons": reasons, "signals": signals,
                "peakiness": round(peakiness, 4)}

    @staticmethod
    def _fft_high_frequency(gray: np.ndarray) -> float:
        dft = np.fft.fft2(gray.astype(np.float32))
        spec = np.abs(np.fft.fftshift(dft))
        rows, cols = spec.shape
        center_y, center_x = rows // 2, cols // 2
        y, x = np.indices((rows, cols))
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_r = float(distance.max()) or 1.0
        low = spec[distance <= 0.10 * max_r].mean() or 0.0
        high = spec[distance >= 0.40 * max_r].mean() or 0.0
        ratio = high / (high + low + 1e-6)
        return max(0.0, min(1.0, float(ratio / 0.30)))

    @staticmethod
    def _luminance_profile(gray: np.ndarray) -> tuple[float, float]:
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        total = float(hist.sum())
        if total <= 0:
            return 0.0, 0.0
        prob = hist / total
        nonzero = prob[prob > 0]
        entropy = -float((nonzero * np.log2(nonzero)).sum()) / 8.0
        peak = float(prob.max())
        return max(0.0, min(1.0, entropy)), peak

    @staticmethod
    def _screen_moire_strength(gray: np.ndarray) -> float:
        row_profile = gray.mean(axis=1)
        diff = np.diff(row_profile)
        if diff.size < 8:
            return 0.5
        auto = np.correlate(diff, diff, mode="full")[diff.size - 1:]
        auto = auto / (auto[0] + 1e-6)
        if auto.size < 8:
            return 0.5
        tail = auto[2:auto.size // 2]
        peak = float(tail.max()) if tail.size else 0.0
        periodicity = max(0.0, min(1.0, peak))
        # Strong periodicity => screen recapture => low "liveness".
        return 1.0 - periodicity

    @staticmethod
    def _colorfulness(bgr: np.ndarray) -> float:
        b, g, r = bgr[:, :, 0].astype(float), bgr[:, :, 1].astype(float), bgr[:, :, 2].astype(float)
        rg = r - g
        yb = 0.5 * (r + g) - b
        metric = math.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2) \
            + 0.3 * math.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
        return max(0.0, min(1.0, metric / 100.0))


class PassiveLivenessDetector:
    """Top-level PAD orchestrator: ONNX primary, OpenCV heuristic fallback.

    All analysis is in-memory; no file is ever written. ``analyze`` returns a
    ``LivenessResult`` dataclass (see ``LivenessResult.to_dict`` for the
    legacy dict consumed by the API and the risk engine): ``status`` is
    ``LIVE``/``SPOOF_DETECTED``/``UNCERTAIN``/``NOT_CHECKED``/``SKIPPED``,
    ``score`` is the clamped finite ``[0, 1]`` fusion, and ``detail`` is the
    human-readable explanation. This is a heuristic triage signal, not a
    certified biometric PAD verdict.
    """

    def __init__(
        self,
        settings: Settings,
        backend: Optional[_OnnxAntiSpoofBackend] = None,
        heuristic: Optional[HeuristicLivenessAnalyzer] = None,
    ) -> None:
        self.settings = settings
        self._backend = backend or _OnnxAntiSpoofBackend(settings)
        self._heuristic = heuristic or HeuristicLivenessAnalyzer()

    def initialize(self) -> None:
        """Load the ONNX model if configured. Failures are fail-safe, never fatal."""
        if not self.settings.liveness_model_path:
            return
        try:
            self._backend.prepare()
        except Exception:  # noqa: BLE001 - model load must not crash the API
            logger.warning("liveness_onnx_init_failed method=heuristic")

    def is_ready(self) -> bool:
        return self._backend.is_ready()

    def readiness(self) -> dict[str, Any]:
        return {
            "liveness": self.is_ready(),
            "liveness_method": _METHOD_ONNX if self.is_ready() else _METHOD_HEURISTIC,
        }

    def analyze(self, image_bytes: bytes) -> LivenessResult:
        if not self.settings.liveness_enabled:
            return self._not_checked(
                "Liveness screening is disabled by configuration.",
                status=LIVENESS_SKIPPED)
        if not image_bytes:
            return self._not_checked("No live photo supplied.")

        image_bgr = _decode_bgr(image_bytes)
        if image_bgr is None:
            return self._not_checked("The live photo could not be decoded.")

        if self.is_ready():
            try:
                score = self._backend.predict(image_bgr)
                return self._classify(score, method=_METHOD_ONNX,
                                      model_used=os.path.basename(
                                          self.settings.liveness_model_path) or None)
            except Exception:  # noqa: BLE001 - fall through to heuristic
                logger.warning("liveness_onnx_predict_failed method=heuristic")

        if self.settings.liveness_heuristic_enabled:
            try:
                analysis = self._heuristic.analyze(image_bgr)
                return self._classify(
                    float(analysis["score"]),
                    method=_METHOD_HEURISTIC,
                    signals=dict(analysis.get("signals", {})),
                    reasons=list(analysis.get("reasons", [])),
                )
            except Exception:  # noqa: BLE001 - never crash the pipeline
                return self._not_checked("Heuristic liveness analysis failed internally.")

        return self._not_checked(
            "No ONNX liveness model and heuristic fallback is disabled.")

    # -- helpers ------------------------------------------------------------
    def _classify(
        self,
        score: float,
        *,
        method: str,
        model_used: Optional[str] = None,
        signals: Optional[dict[str, float]] = None,
        reasons: Optional[list[str]] = None,
    ) -> LivenessResult:
        score = float(score)
        # NaN/inf can never be a valid liveness score; fail safe to spoof-bias.
        if not math.isfinite(score):
            score = 0.0
        score = max(0.0, min(1.0, score))
        spoof = self.settings.liveness_spoof_threshold
        uncertain = self.settings.liveness_uncertain_threshold
        if score <= spoof:
            status = LIVENESS_SPOOF
        elif score < uncertain:
            status = LIVENESS_UNCERTAIN
        else:
            status = LIVENESS_LIVE
        explanation = {
            LIVENESS_LIVE: "The live capture passed the passive liveness screen.",
            LIVENESS_SPOOF: "The live capture is a likely printout or screen recapture; "
                            "immediate high-risk review required.",
            LIVENESS_UNCERTAIN: "Liveness could not be determined confidently; "
                                "apply the configured review penalty.",
        }[status]
        return LivenessResult(
            is_live=status == LIVENESS_LIVE,
            score=score,
            status=status,
            detail=explanation,
            method=method,
            model_used=model_used,
            signals=signals or {},
            reasons=reasons or [],
        )

    @staticmethod
    def _not_checked(explanation: str, status: str = LIVENESS_NOT_CHECKED) -> LivenessResult:
        return LivenessResult(
            is_live=False,
            score=0.0,
            status=status,
            detail=explanation,
            method=_METHOD_NOT_CHECKED,
        )
