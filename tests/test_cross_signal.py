"""
Tests for Cross-Signal Consistency Engine (CrossSignalEvaluator).
"""

import pytest
from app.services.cross_signal import CrossSignalEvaluator, CrossSignalResult
from app.services.risk_engine import RiskEngine
from app.config import Settings


def test_cross_signal_flags_document_type_mismatch():
    evaluator = CrossSignalEvaluator()
    # Requested passport, but document parser detected Aadhaar
    doc_result = {
        "detected": True,
        "document_type": "AADHAAR",
        "format": "AADHAAR_CARD",
        "status": "VALID",
    }
    res = evaluator.evaluate(
        requested_type="passport",
        doc_result=doc_result,
        tamper_result={"status": "CLEAN", "suspicious_regions": []},
        face_result={"status": "MATCH"},
    )
    assert not res.is_consistent
    factor_names = [f["factor"] for f in res.factors]
    assert "DOCUMENT_TYPE_MISMATCH" in factor_names


def test_cross_signal_flags_qr_ocr_conflict():
    evaluator = CrossSignalEvaluator()
    doc_result = {
        "detected": True,
        "document_type": "AADHAAR",
        "status": "VALID",
        "data": {
            "masked_uid": "XXXX-XXXX-1234",
        },
        "qr": {
            "detected": True,
            "readable": True,
            "payload_type": "AADHAAR_XML",
            "masked_uid": "XXXX-XXXX-9999",  # Conflicts with OCR 1234!
        },
    }
    res = evaluator.evaluate(
        requested_type="aadhaar",
        doc_result=doc_result,
        tamper_result={"status": "CLEAN", "suspicious_regions": []},
        face_result={"status": "MATCH"},
    )
    assert not res.is_consistent
    factor_names = [f["factor"] for f in res.factors]
    assert "QR_OCR_CONFLICT" in factor_names


def test_cross_signal_flags_suspicious_field_tampering():
    evaluator = CrossSignalEvaluator()
    doc_result = {
        "detected": True,
        "document_type": "PASSPORT",
        "format": "TD3",
        "status": "VALID",
    }
    # Tampering bounding box in lower 30% of image (MRZ area)
    tamper_result = {
        "status": "SUSPICIOUS",
        "suspicious_regions": [[50, 400, 700, 80]],  # y=400 in height=500 -> 80% down
    }
    res = evaluator.evaluate(
        requested_type="passport",
        doc_result=doc_result,
        tamper_result=tamper_result,
        face_result={"status": "MATCH"},
        image_shape=(500, 800),
    )
    factor_names = [f["factor"] for f in res.factors]
    assert "SUSPICIOUS_FIELD_TAMPERING" in factor_names


def test_cross_signal_incorporation_in_risk_engine():
    settings = Settings.from_env()
    risk_engine = RiskEngine(settings)

    # Clean modules
    mrz_res = {"status": "VALID", "detected": True, "document_type": "PASSPORT", "data": {"checks": {"all": True}}}
    face_res = {"status": "MATCH"}
    tamper_res = {"status": "CLEAN", "score": 0.0}
    liveness_res = {"liveness_status": "LIVE"}

    # With cross-signal conflict
    cross_res = {
        "is_consistent": False,
        "factors": [
            {"factor": "DOCUMENT_TYPE_MISMATCH", "detail": "Requested passport but document is Aadhaar."}
        ]
    }

    evaluated = risk_engine.evaluate(
        mrz_result=mrz_res,
        face_result=face_res,
        tamper_result=tamper_res,
        liveness_result=liveness_res,
        cross_signal_result=cross_res,
    )
    assert evaluated["score"] >= 30
    assert any(f["factor"] == "DOCUMENT_TYPE_MISMATCH" for f in evaluated["factors"])
