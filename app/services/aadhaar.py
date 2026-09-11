"""
Dedicated Aadhaar Document Analysis Strategy.

Validates Indian Aadhaar identity cards using:
- UIDAI Layout Anchors (Government of India, UIDAI, DOB, Gender, etc.)
- Masked UID format validation (XXXX XXXX 1234)
- 12-digit UID Verhoeff checksum validation (without persisting raw numbers)
- UIDAI Secure QR Code / XML Barcode verification via QRAnalyzer

Privacy Guarantee:
Raw 12-digit Aadhaar numbers, cardholder names, and addresses are NEVER stored,
logged, or returned in API responses.
"""

from __future__ import annotations

import io
import re
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance

from app.config import Settings
from app.services.mrz import BaseDocumentParser, DocumentParseResult, PASSPORT_MIN_RATIO, _decode_bgr, _encode_bgr
from app.services.qr import QRAnalyzer, QRAnalysisResult


# Verhoeff algorithm multiplication table
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Verhoeff permutation table
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


class AadhaarDocumentParser(BaseDocumentParser):
    """Dedicated strategy for Indian Aadhaar cards (physical, e-Aadhaar, PVC)."""

    name = "aadhaar"
    document_type = "AADHAAR"
    format = "AADHAAR_CARD"

    AADHAAR_ANCHORS = [
        "government of india",
        "govt of india",
        "bharat sarkar",
        "unique identification authority of india",
        "uidai",
        "mera aadhaar",
        "date of birth",
        "dob",
        "year of birth",
        "yob",
        "male",
        "female",
        "transgender",
        "help@uidai.gov.in",
        "1947",
    ]

    def __init__(self) -> None:
        self._qr_analyzer = QRAnalyzer()

    def can_parse(self, image_bgr: np.ndarray) -> bool:
        if image_bgr is None or image_bgr.size == 0:
            return False
        height, width = image_bgr.shape[:2]
        if height <= 0:
            return False
        # ID-1 aspect ratio
        return (width / height) >= PASSPORT_MIN_RATIO

    @staticmethod
    def validate_verhoeff(number_str: str) -> bool:
        """Validate check digit of number using Verhoeff algorithm."""
        clean = re.sub(r"\D", "", number_str)
        if not clean:
            return False
        c = 0
        for i, item in enumerate(reversed(clean)):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(item)]]
        return c == 0

    def parse(
        self,
        image: Any,
        settings: Optional[Settings] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        line3: Optional[str] = None,
        **kwargs: Any,
    ) -> DocumentParseResult:
        if settings is None:
            settings = Settings.from_env()

        image_bytes: bytes = image if isinstance(image, (bytes, bytearray)) else _encode_bgr(image)
        image_bgr = _decode_bgr(image_bytes)

        if image_bgr is None or image_bgr.size == 0:
            return self._not_detected("Unable to decode document image.")

        # 1. QR Code Analysis
        qr_result: QRAnalysisResult = self._qr_analyzer.analyze(image_bgr)

        # 2. Text extraction via OCR
        ocr_text = self._extract_text(image_bgr)
        text_lower = ocr_text.lower()

        # 3. Anchor detection
        anchors_found = [
            anchor for anchor in self.AADHAAR_ANCHORS
            if anchor in text_lower
        ]

        # 4. UID Pattern & Verhoeff Analysis (Privacy: mask immediately!)
        ocr_masked_uid, is_masked, verhoeff_valid = self._analyze_ocr_uid(ocr_text)
        qr_masked_uid = qr_result.details.get("masked_uid") if (qr_result.detected and qr_result.readable) else None
        doc_masked_uid = ocr_masked_uid or qr_masked_uid

        # 5. Evaluate structural validity and status (never claims official government authenticity)
        detected = (len(anchors_found) >= 2) or (qr_result.detected and qr_result.payload_type in {"AADHAAR_XML", "SECURE_AADHAAR_NUMERIC"})
        if not detected and len(anchors_found) == 0 and not qr_result.detected:
            return self._not_detected("No Aadhaar anchors or QR code detected.")

        checks = {
            "anchors_detected": len(anchors_found) >= 2,
            "qr_detected": qr_result.detected,
            "qr_readable": qr_result.readable,
            "qr_aadhaar_format": qr_result.payload_type in {"AADHAAR_XML", "SECURE_AADHAAR_NUMERIC"},
            "uid_found": bool(doc_masked_uid),
            "uid_is_masked": is_masked,
            "verhoeff_valid": verhoeff_valid,
        }

        # Status rules - structural consistency only
        if verhoeff_valid is False:
            status = "SUSPICIOUS"
            confidence = 0.85
        elif checks["anchors_detected"] and (checks["qr_aadhaar_format"] or checks["uid_found"]):
            status = "VALID"
            confidence = 0.95 if checks["qr_aadhaar_format"] else 0.85
        elif checks["anchors_detected"] or checks["qr_aadhaar_format"]:
            status = "UNCERTAIN"
            confidence = 0.60
        else:
            status = "UNCERTAIN"
            confidence = 0.40

        raw_result = {
            "detected": True,
            "source": "ocr_and_qr" if (ocr_masked_uid and qr_result.detected) else ("qr" if qr_result.detected else "ocr"),
            "status": status,
            "document_type": self.document_type,
            "format": self.format,
            "confidence": confidence,
            "anchors_found": anchors_found,
            "anchors_count": len(anchors_found),
            "is_masked": is_masked,
            "checks": checks,
            "qr": {
                "detected": qr_result.detected,
                "readable": qr_result.readable,
                "payload_type": qr_result.payload_type,
                "masked_uid": qr_masked_uid,
                "masked_summary": qr_result.raw_payload_masked_summary,
            },
            "data": {
                "masked_uid": doc_masked_uid,
                "ocr_masked_uid": ocr_masked_uid,
                "qr_masked_uid": qr_masked_uid,
                "is_masked": is_masked,
            },
        }

        return DocumentParseResult(
            detected=True,
            status=status,
            document_type=self.document_type,
            format=self.format,
            confidence=confidence,
            data=raw_result["data"],
            raw=raw_result,
            error=None,
        )

    def _analyze_ocr_uid(self, ocr_text: str) -> tuple[Optional[str], bool, Optional[bool]]:
        """Search for masked or unmasked UID in OCR text, validate Verhoeff without persisting raw numbers."""
        # Check masked format in text: e.g. XXXX XXXX 1234 or XXXX-XXXX-1234
        masked_match = re.search(r"[X\*\•]{4}[\s\-][X\*\•]{4}[\s\-](\d{4})", ocr_text, re.IGNORECASE)
        if masked_match:
            return f"XXXX-XXXX-{masked_match.group(1)}", True, None

        # Check unmasked 12-digit format in text (starts with digit 2-9)
        unmasked_match = re.search(r"\b([2-9]\d{3})[\s\-](\d{4})[\s\-](\d{4})\b", ocr_text)
        if unmasked_match:
            full_digits = unmasked_match.group(1) + unmasked_match.group(2) + unmasked_match.group(3)
            verhoeff_ok = self.validate_verhoeff(full_digits)
            # IMMEDIATELY mask the number
            masked = f"XXXX-XXXX-{full_digits[-4:]}"
            return masked, False, verhoeff_ok

        return None, False, None

    def _extract_text(self, image_bgr: np.ndarray) -> str:
        """Run multi-pass OCR on document image."""
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            # Pass 1: raw grayscale
            text1 = pytesseract.image_to_string(gray, config="--psm 6")
            # Pass 2: Otsu thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text2 = pytesseract.image_to_string(thresh, config="--psm 6")
            return f"{text1}\n{text2}"
        except Exception:
            return ""

    def _not_detected(self, reason: str) -> DocumentParseResult:
        return DocumentParseResult(
            detected=False,
            status="NOT_DETECTED",
            document_type=self.document_type,
            format=self.format,
            confidence=0.0,
            data={},
            raw={
                "detected": False,
                "document_type": self.document_type,
                "format": self.format,
                "reason": reason,
            },
            error=reason,
        )
