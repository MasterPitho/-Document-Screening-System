"""Risk engine tests: weights, fail-safe gating, unknown-factor rejection."""

from dataclasses import replace

import pytest

from app.config import Settings
from app.services.risk_engine import (
    RiskEngine,
    UnknownRiskFactorError,
    face_module_state,
    mrz_module_state,
    tampering_module_state,
)

VALID_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
VALID_LINE2 = "L898902C36UTO7408122F3501014ZE184226B<<<<<16"  # unexpired


def _settings(**tw) -> Settings:
    return replace(Settings.from_env(), **tw)


def _engine(settings=None) -> RiskEngine:
    return RiskEngine(settings or _settings())


def _valid_mrz():
    from app.services.mrz import parse_td3_mrz

    return {
        "detected": True, "source": "ocr", "status": "VALID", "confidence": 1.0,
        "data": parse_td3_mrz(VALID_LINE1, VALID_LINE2),
    }


def _match_face():
    return {"status": "MATCH", "similarity_score": 0.9, "matched": True}


def _clean_tamper():
    return {"status": "CLEAN", "score": 0.0}


def _all_pass(engine=None):
    return (engine or _engine()).evaluate(
        mrz_result=_valid_mrz(), face_result=_match_face(), tamper_result=_clean_tamper(),
    )


# ---------------------------------------------------------------------------
# Module state mapping
# ---------------------------------------------------------------------------

def test_face_module_state_mapping():
    assert face_module_state({"status": "MATCH"}) == "PASS"
    assert face_module_state({"status": "MISMATCH"}) == "FAIL"
    assert face_module_state({"status": "ERROR"}) == "ERROR"
    assert face_module_state({"status": "NOT_AVAILABLE"}) == "ERROR"
    assert face_module_state({"status": "SKIPPED_NO_LIVE_PHOTO"}) == "NOT_AVAILABLE"
    assert face_module_state({"status": "NO_FACE"}) == "REVIEW"
    assert face_module_state({"status": "MISMATCH_FROTZ"}) == "NOT_AVAILABLE"


def test_mrz_module_state_mapping():
    assert mrz_module_state({"status": "VALID", "detected": True}) == "PASS"
    assert mrz_module_state({"status": "INVALID"}) == "FAIL"
    assert mrz_module_state({"status": "OCR_FAILED"}) == "ERROR"
    assert mrz_module_state({"status": "MALFORMED"}) == "REVIEW"
    assert mrz_module_state({"status": "NOT_DETECTED"}) == "REVIEW"
    assert mrz_module_state({"status": "OCR_LOW_CONFIDENCE"}) == "REVIEW"


def test_tampering_module_state_mapping():
    assert tampering_module_state({"status": "CLEAN"}) == "PASS"
    assert tampering_module_state({"status": "SUSPICIOUS"}) == "FAIL"
    assert tampering_module_state({"status": "INCONCLUSIVE"}) == "REVIEW"
    assert tampering_module_state({"status": "ERROR"}) == "ERROR"
    assert tampering_module_state({"status": "WHATEVER"}) == "NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# Fail-open regression: ERROR / NOT_AVAILABLE never yield CLEARED
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tamper", [
    {"status": "ERROR"},
    {"status": "ANALYSIS_ERROR"},
])
def test_tampering_error_is_never_cleared(tamper):
    result = _engine().evaluate(
        mrz_result=_valid_mrz(), face_result=_match_face(), tamper_result=tamper)
    assert result["decision"] != "CLEARED"
    assert result["module_statuses"]["tampering"] == "ERROR"


@pytest.mark.parametrize("face", [
    {"status": "ERROR"},
    {"status": "NOT_AVAILABLE"},
])
def test_face_error_is_never_cleared(face):
    result = _engine().evaluate(
        mrz_result=_valid_mrz(), face_result=face, tamper_result=_clean_tamper())
    assert result["decision"] != "CLEARED"
    assert result["module_statuses"]["face"] == "ERROR"


@pytest.mark.parametrize("status", ["NOT_DETECTED", "OCR_FAILED", "MALFORMED"])
def test_mrz_not_pass_is_never_cleared(status):
    mrz = {"detected": False, "status": status}
    result = _engine().evaluate(
        mrz_result=mrz, face_result=_match_face(), tamper_result=_clean_tamper())
    assert result["decision"] != "CLEARED"
    assert result["module_statuses"]["mrz"] in {"ERROR", "REVIEW", "FAIL"}


def test_no_live_photo_never_cleared():
    result = _engine().evaluate(
        mrz_result=_valid_mrz(),
        face_result={"status": "SKIPPED_NO_LIVE_PHOTO"},
        tamper_result=_clean_tamper(),
    )
    assert result["decision"] != "CLEARED"
    assert any(f["factor"] == "UNKNOWN_MODULE" for f in result["factors"])


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def test_all_pass_clears():
    result = _all_pass()
    assert result["decision"] == "CLEARED"
    assert result["level"] == "LOW_RISK"
    assert result["status"] == "GREEN"
    assert result["score"] == 0


def test_mismatch_adds_face_mismatch_factor():
    result = _engine().evaluate(
        mrz_result=_valid_mrz(),
        face_result={"status": "MISMATCH", "similarity_score": 0.1, "matched": False},
        tamper_result=_clean_tamper(),
    )
    assert any(f["factor"] == "FACE_MISMATCH" for f in result["factors"])
    assert result["decision"] != "CLEARED"


def test_tampering_suspected_weight_flags_high_risk():
    engine = _engine()  # TAMPERING_SUSPECTED weight 40
    result = engine.evaluate(
        mrz_result=_valid_mrz(), face_result=_match_face(),
        tamper_result={"status": "SUSPICIOUS"})
    factor = next(f for f in result["factors"] if f["factor"] == "TAMPERING_SUSPECTED")
    assert factor["weight"] == engine.weights["TAMPERING_SUSPECTED"]
    assert result["level"] == "MEDIUM_RISK"


def test_expired_document_adds_factor():
    mrz = dict(_valid_mrz())
    mrz["data"] = dict(mrz["data"], is_expired=True)
    result = _engine().evaluate(
        mrz_result=mrz, face_result=_match_face(), tamper_result=_clean_tamper())
    assert any(f["factor"] == "EXPIRED_DOCUMENT" for f in result["factors"])
    assert result["decision"] != "CLEARED"


def test_mrz_checksum_failure_adds_factor():
    mrz = dict(_valid_mrz())
    mrz["data"] = dict(mrz["data"], checks={
        "passport_number_valid": False,
        "dob_valid": True,
        "expiry_valid": True,
        "composite_valid": True,
    })
    result = _engine().evaluate(
        mrz_result=mrz, face_result=_match_face(), tamper_result=_clean_tamper())
    assert any(f["factor"] == "MRZ_CHECKSUM_FAILURE" for f in result["factors"])


def test_accumulated_risk_score_is_capped_at_100():
    face = {"status": "MISMATCH", "similarity_score": 0.1, "matched": False}
    tamper = {"status": "SUSPICIOUS"}
    expired_mrz = dict(_valid_mrz())
    expired_mrz["data"] = dict(expired_mrz["data"], is_expired=True,
                               checks={name: False for name in
                                       ("passport_number_valid", "dob_valid")})  # wrong len ok, just two flags
    result = _engine().evaluate(
        mrz_result=expired_mrz, face_result=face, tamper_result=tamper)
    assert result["score"] == 100
    assert result["level"] == "HIGH_RISK"
    assert result["decision"] == "HIGH_RISK_REVIEW_REQUIRED"
    assert result["status"] == "RED"


def test_unknown_factor_is_rejected_not_silently_weighted():
    engine = _engine()
    with pytest.raises(UnknownRiskFactorError):
        engine.add_factor("TOTALLY_MAKE_BELIEVE", "nope", [], [0])


def test_required_factors_all_have_weights():
    settings = _settings()
    required = {
        "TAMPERING_SUSPECTED", "TAMPERING_INCONCLUSIVE", "FACE_MISMATCH",
        "FACE_NOT_DETECTED", "FACE_LOW_CONFIDENCE", "FACE_MULTIPLE",
        "MRZ_CHECKSUM_FAILURE", "EXPIRED_DOCUMENT", "MRZ_NOT_DETECTED",
        "MRZ_LOW_CONFIDENCE", "IMAGE_QUALITY", "MODULE_ERROR", "UNKNOWN_MODULE",
    }
    assert required.issubset(settings.risk_weights)


def test_weights_are_non_negative():
    assert all(w >= 0 for w in _engine().weights.values())


def test_score_stays_within_bounds_for_every_status_combination():
    mrz_statuses = ["VALID", "INVALID", "MALFORMED", "NOT_DETECTED", "OCR_FAILED"]
    face_statuses = ["MATCH", "MISMATCH", "NO_FACE", "MULTIPLE_FACES",
                     "LOW_CONFIDENCE", "SKIPPED_NO_LIVE_PHOTO", "ERROR", "NOT_AVAILABLE"]
    tamper_statuses = ["CLEAN", "SUSPICIOUS", "INCONCLUSIVE", "ERROR"]
    engine = _engine()
    for ms in mrz_statuses:
        for fs in face_statuses:
            for ts in tamper_statuses:
                mrz = {"detected": ms == "VALID", "status": ms}
                face = {"status": fs}
                if fs == "MATCH":
                    mrz["detected"] = True
                    mrz["status"] = "VALID"
                    mrz = _valid_mrz()
                tamper = {"status": ts}
                result = engine.evaluate(mrz_result=mrz, face_result=face,
                                         tamper_result=tamper)
                assert 0 <= result["score"] <= 100
                if result["score"] >= engine.reject_threshold:
                    assert result["decision"] == "HIGH_RISK_REVIEW_REQUIRED"


def test_configuration_validation_rejects_bad_thresholds():
    with pytest.raises(ValueError, match="Risk thresholds"):
        _settings(risk_review_threshold=70, risk_reject_threshold=35).validate()


def test_configuration_validation_rejects_negative_weight():
    bad = _settings()
    bad = replace(bad, risk_weights={**bad.risk_weights, "FACE_MISMATCH": -1})
    with pytest.raises(ValueError, match="cannot be negative"):
        bad.validate()