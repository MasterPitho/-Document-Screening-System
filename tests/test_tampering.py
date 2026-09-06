"""Multi-signal tampering detector tests."""

from dataclasses import replace

from app.config import Settings
from app.services.tampering import TamperingDetector

VALID_STATUSES = {"CLEAN", "SUSPICIOUS", "INCONCLUSIVE", "ERROR"}
SIGNAL_NAMES = {"ela", "compression", "noise", "edge", "copy_move", "metadata"}


def _settings(**tw) -> Settings:
    return replace(Settings.from_env(), **tw)


def _detector(settings=None) -> TamperingDetector:
    return TamperingDetector(settings or _settings())


def test_clean_white_image_is_not_suspicious(jpeg_bytes):
    result = _detector().analyze(jpeg_bytes())
    assert result["status"] == "CLEAN"
    assert result["score"] < _settings().tampering_review_threshold or not result["signals"]


def test_result_contract(jpeg_bytes):
    result = _detector().analyze(jpeg_bytes())
    assert result["status"] in VALID_STATUSES
    assert 0.0 <= result["score"] <= 100.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert SIGNAL_NAMES.issubset(result["signals"].keys())
    assert isinstance(result["suspicious_regions"], list)
    assert isinstance(result["explanation"], list)
    assert result["signals"]["metadata"]["present"] is False  # plain PIL image


def test_corrupt_image_returns_error(jpeg_bytes):
    result = _detector().analyze(b"definitely-not-an-image")
    assert result["status"] == "ERROR"
    assert result["explanation"]


def test_metadata_absence_is_not_tampering(jpeg_bytes):
    result = _detector().analyze(jpeg_bytes())
    meta = result["signals"]["metadata"]
    assert meta["present"] is False
    assert meta["suspicious"] is False
    assert meta["score"] == 0.0
    assert "not treated as tampering" in meta.get("note", "").lower()


def test_forced_suspicious_signals_flag_suspicious(jpeg_bytes):
    class _Noisy(TamperingDetector):

        def _ela_signal(self, arr):
            return {"score": 95.0, "suspicious": True}

        def _compression_signal(self, arr):
            return {"score": 95.0, "suspicious": True}

        def _noise_signal(self, arr):
            return {"score": 95.0, "suspicious": True}

        def _edge_signal(self, arr):
            return {"score": 95.0, "suspicious": True}

        def _copy_move_signal(self, arr):
            return {"score": 95.0, "suspicious": True}

    detector = _Noisy(_settings(tampering_threshold=10.0, tampering_review_threshold=0.0))
    result = detector.analyze(jpeg_bytes())
    assert result["status"] == "SUSPICIOUS"
    assert result["score"] >= 95.0
    assert any(s in {"ela", "compression", "noise", "edge", "copy_move"} for s in result["signals"])


def test_moderate_signals_without_flags_are_inconclusive(jpeg_bytes):
    class _Mixed(TamperingDetector):

        def _ela_signal(self, arr):
            return {"score": 50.0, "suspicious": False}

        def _compression_signal(self, arr):
            return {"score": 50.0, "suspicious": False}

        def _noise_signal(self, arr):
            return {"score": 50.0, "suspicious": False}

        def _edge_signal(self, arr):
            return {"score": 50.0, "suspicious": False}

        def _copy_move_signal(self, arr):
            return {"score": 50.0, "suspicious": False}

    detector = _Mixed(_settings(tampering_threshold=70.0, tampering_review_threshold=45.0))
    result = detector.analyze(jpeg_bytes())
    assert result["status"] == "INCONCLUSIVE"


def test_single_implausible_signal_without_threshold_crossing_is_inconclusive(jpeg_bytes):
    class _OneFlag(TamperingDetector):
        def _ela_signal(self, arr):
            return {"score": 80.0, "suspicious": True}

    detector = _OneFlag(_settings(tampering_threshold=90.0, tampering_review_threshold=45.0))
    result = detector.analyze(jpeg_bytes())
    assert result["status"] == "INCONCLUSIVE"


def test_one_signal_failure_does_not_crash_analysis(monkeypatch, jpeg_bytes):
    def _boom(self, arr):
        raise RuntimeError("ela exploded")

    monkeypatch.setattr(TamperingDetector, "_ela_signal", _boom)
    result = _detector().analyze(jpeg_bytes())
    assert result["status"] in VALID_STATUSES
    assert "error" in result["signals"]["ela"]


def test_suspicious_result_reports_regions_and_reasons(jpeg_bytes):
    class _FlagEdge(TamperingDetector):
        def _edge_signal(self, arr):
            return {"score": 100.0, "suspicious": True}

    detector = _FlagEdge(_settings(tampering_threshold=10.0, tampering_review_threshold=0.0))
    result = detector.analyze(jpeg_bytes())
    assert result["status"] == "SUSPICIOUS"
    assert isinstance(result["suspicious_regions"], list)
    assert any("signal" in line or "region" in line for line in result["explanation"])
