"""
Tests for QR code detection, decoding, and privacy-preserving payload classification.
"""

import cv2
import numpy as np
import pytest
from app.services.qr import QRAnalyzer, QRAnalysisResult


def _make_qr_image(text: str) -> np.ndarray:
    encoder = cv2.QRCodeEncoder.create()
    mat = encoder.encode(text)
    scaled = cv2.resize(mat, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    bordered = cv2.copyMakeBorder(scaled, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)


def test_qr_analyzer_detects_no_qr_on_blank_image():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    analyzer = QRAnalyzer()
    result = analyzer.analyze(blank)

    assert isinstance(result, QRAnalysisResult)
    assert not result.detected
    assert not result.readable
    assert result.payload_type == "NONE"
    assert result.format == "NONE"


def test_qr_analyzer_detects_generic_url():
    qr_img = _make_qr_image("https://example.gov/verify?id=12345")
    analyzer = QRAnalyzer()
    result = analyzer.analyze(qr_img)

    assert result.detected
    assert result.readable
    assert result.format == "QR_CODE"
    assert result.payload_type == "GENERIC_URL"
    assert "example.gov" in result.details.get("domain", "")
    # Must not store raw query parameters that could contain PII
    assert "12345" not in result.raw_payload_masked_summary


def test_qr_analyzer_classifies_old_aadhaar_xml():
    xml_data = '<PrintLetterBarcodeData uid="999912345678" name="John Doe" yob="1990" gender="M" />'
    qr_img = _make_qr_image(xml_data)
    analyzer = QRAnalyzer()
    result = analyzer.analyze(qr_img)

    assert result.detected
    assert result.readable
    assert result.payload_type == "AADHAAR_XML"
    # Never expose full UID or Name in result details or masked summary
    assert "999912345678" not in str(result.details)
    assert "John Doe" not in str(result.details)
    assert result.details.get("masked_uid") == "XXXX-XXXX-5678"
    assert result.details.get("has_name") is True


def test_qr_analyzer_classifies_secure_aadhaar_numeric():
    # Large numeric string characteristic of UIDAI Secure QR Code
    numeric_payload = "49823489237498237498237498273498273498273498723948723948723984723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948723948"
    qr_img = _make_qr_image(numeric_payload)
    analyzer = QRAnalyzer()
    result = analyzer.analyze(qr_img)

    assert result.detected
    assert result.readable
    assert result.payload_type == "SECURE_AADHAAR_NUMERIC"
    assert result.details.get("is_numeric") is True
    assert result.details.get("payload_length", 0) > 100
