"""Passive liveness (PAD) detector tests."""

import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.liveness import (
    HeuristicLivenessAnalyzer,
    LivenessResult,
    PassiveLivenessDetector,
    _OnnxAntiSpoofBackend,
)

from app.services import risk_engine as risk_mod


def _settings() -> Settings:
    return Settings.from_env()


def _image(width=320, height=240, noise=False, seed=7):
    rng = np.random.default_rng(seed)
    if noise:
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    else:
        arr = np.full((height, width, 3), 255, np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def _detector(settings=None) -> PassiveLivenessDetector:
    return PassiveLivenessDetector(settings or _settings())


def _legacy(result: LivenessResult) -> dict:
    return result.to_dict()


def test_blank_image_is_classified_as_spoof():
    result = _detector().analyze(_image(noise=False))
    assert isinstance(result, LivenessResult)
    assert result.status == "SPOOF_DETECTED"
    assert result.is_live is False
    assert result.method == "heuristic"
    assert result.detail
    assert result.signals
    assert result.score <= _settings().liveness_spoof_threshold


def test_noisy_image_is_classified_as_live():
    result = _detector().analyze(_image(noise=True))
    assert result.status == "LIVE"
    assert result.is_live is True
    assert result.method == "heuristic"
    assert result.score >= _settings().liveness_uncertain_threshold


def test_dataclass_contract_and_legacy_dict():
    result = _detector().analyze(_image(noise=True))
    for field_name in ("is_live", "score", "status", "detail"):
        assert hasattr(result, field_name)
    legacy = _legacy(result)
    assert legacy["liveness_status"] == result.status
    assert legacy["is_live"] == result.is_live
    assert legacy["liveness_score"] == pytest.approx(result.score, abs=1e-4)
    assert legacy["explanation"] == result.detail
    assert risk_mod.liveness_module_state(legacy) == "PASS"


def test_empty_bytes_returns_not_checked():
    result = _detector().analyze(b"")
    assert result.status == "NOT_CHECKED"
    assert result.is_live is False
    assert result.method == "not_checked"
    assert result.detail
    assert risk_mod.liveness_module_state(_legacy(result)) == "NOT_AVAILABLE"


def test_corrupt_bytes_returns_not_checked():
    result = _detector().analyze(b"definitely-not-an-image")
    assert result.status == "NOT_CHECKED"
    assert result.is_live is False


def test_disabled_returns_skipped(monkeypatch):
    monkeypatch.setenv("LIVENESS_ENABLED", "false")
    settings = Settings.from_env()
    result = _detector(settings).analyze(_image(noise=True))
    assert result.status == "SKIPPED"
    assert "disabled" in result.detail.lower()
    assert risk_mod.liveness_module_state(_legacy(result)) == "NOT_AVAILABLE"


def test_heuristic_signals_are_bounded():
    analyzer = HeuristicLivenessAnalyzer()
    for image_bytes in (_image(noise=False), _image(noise=True), _image(seed=3)):
        image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        result = analyzer.analyze(image)
        assert 0.0 <= result["score"] <= 1.0
        for signal in ("blur", "high_frequency", "entropy", "moire", "colorfulness"):
            assert signal in result["signals"]
            assert 0.0 <= result["signals"][signal] <= 1.0
        assert result["reasons"]


def test_threshold_classification_boundaries():
    settings = _settings()
    det = _detector(settings)
    assert det._classify(0.0, method="heuristic").status == "SPOOF_DETECTED"
    assert det._classify(settings.liveness_spoof_threshold, method="heuristic").status == "SPOOF_DETECTED"
    mid = (settings.liveness_spoof_threshold + settings.liveness_uncertain_threshold) / 2
    assert det._classify(mid, method="heuristic").status == "UNCERTAIN"
    assert det._classify(settings.liveness_uncertain_threshold, method="heuristic").status == "LIVE"
    assert det._classify(1.0, method="heuristic").is_live is True
    assert det._classify(-0.5, method="heuristic").score == 0.0
    assert det._classify(1.5, method="heuristic").score == 1.0


def test_nan_and_inf_scores_fail_safe_to_spoof():
    det = _detector()
    for bad in (float("nan"), float("inf"), float("-inf")):
        result = det._classify(bad, method="heuristic")
        assert result.score == 0.0
        assert result.status == "SPOOF_DETECTED"
        assert result.is_live is False


def test_analyze_is_deterministic():
    det = _detector()
    image = _image(noise=True)
    first = det.analyze(image)
    second = det.analyze(image)
    assert first.status == second.status
    assert first.score == second.score
    assert first.to_dict() == second.to_dict()


def test_analyze_writes_no_files(monkeypatch):
    det = _detector()
    temp_root = tempfile.gettempdir()
    before = set(os.listdir(temp_root))

    def _forbid_write(*args, **kwargs):
        raise AssertionError("liveness must not write files to disk")

    monkeypatch.setattr("cv2.imwrite", _forbid_write)
    monkeypatch.setattr("numpy.save", _forbid_write)
    det.analyze(_image(noise=True))
    det.analyze(b"")
    det.analyze(_image(noise=False))
    after = set(os.listdir(temp_root))
    assert after == before


def test_onnx_backend_not_ready_without_model_path():
    backend = _OnnxAntiSpoofBackend(_settings())
    assert backend.is_ready() is False
    with pytest.raises(RuntimeError):
        backend.predict(np.zeros((64, 64, 3), np.uint8))


def test_onnx_backend_model_missing_when_configured(monkeypatch):
    monkeypatch.setenv("LIVENESS_MODEL_PATH", "C:\\nonexistent\\model.onnx")
    settings = Settings.from_env()
    det = _detector(settings)
    det.initialize()  # must fail safe, not raise
    assert det.is_ready() is False


def test_readiness_reflects_fallback_when_no_model():
    det = _detector()
    assert det.is_ready() is False
    readiness = det.readiness()
    assert readiness["liveness"] is False
    assert readiness["liveness_method"] == "heuristic"


def test_module_state_mapping():
    assert risk_mod.liveness_module_state({"liveness_status": "LIVE"}) == "PASS"
    assert risk_mod.liveness_module_state({"liveness_status": "SPOOF_DETECTED"}) == "FAIL"
    assert risk_mod.liveness_module_state({"liveness_status": "UNCERTAIN"}) == "REVIEW"
    assert risk_mod.liveness_module_state({"liveness_status": "NOT_CHECKED"}) == "NOT_AVAILABLE"
    assert risk_mod.liveness_module_state({"liveness_status": "SKIPPED"}) == "NOT_AVAILABLE"
    assert risk_mod.liveness_module_state({"liveness_status": "WHATEVER"}) == "NOT_AVAILABLE"
