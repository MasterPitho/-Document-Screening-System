"""
Dedicated Indian Permanent Account Number (PAN) Card Analysis Strategy.

Validates Indian Income Tax Department PAN cards using:
- PAN Card Layout Anchors ("INCOME TAX DEPARTMENT", "GOVT. OF INDIA", etc.)
- 10-Character Structure Validation (5 letters, 4 digits, 1 letter)
- 4th Character Status/Entity-Type Verification (P, C, H, F, A, T, B, L, J, G)
- Reusable QR code probe

Privacy Guarantee:
Full unmasked 10-character PAN strings and cardholder personal details are NEVER
persisted, logged, or exposed in API outputs.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]

from app.config import Settings
from app.services.mrz import BaseDocumentParser, DocumentParseResult, PASSPORT_MIN_RATIO, _decode_bgr, _encode_bgr
from app.services.qr import QRAnalyzer, QRAnalysisResult


class PANDocumentParser(BaseDocumentParser):
    """Dedicated strategy for Indian PAN cards issued by the Income Tax Department."""

    name = "pan"
    document_type = "PAN"
    format = "PAN_CARD"

    PAN_ANCHORS = [
        "income tax department",
        "govt. of india",
        "government of india",
        "permanent account number",
        "permanent account number card",
        "father's name",
        "date of birth",
        "dob",
        "signature",
    ]

    ENTITY_TYPES: dict[str, str] = {
        "P": "Individual",
        "C": "Company",
        "H": "Hindu Undivided Family",
        "F": "Firm / LLP",
        "A": "Association of Persons",
        "T": "Trust",
        "B": "Body of Individuals",
        "L": "Local Authority",
        "J": "Artificial Juridical Person",
        "G": "Government Agency",
    }

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

    def validate_pan_structure(self, pan_str: str) -> tuple[bool, str]:
        """Validate 10-character format and 4th character entity type code."""
        clean = pan_str.strip().upper()
        if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", clean):
            return False, "Unknown"

        status_char = clean[3]
        entity_type = self.ENTITY_TYPES.get(status_char)
        if entity_type is None:
            return False, "Unknown"

        return True, entity_type

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

        # 1. QR Code Analysis (Modern PVC PAN cards embed a QR code)
        qr_result: QRAnalysisResult = self._qr_analyzer.analyze(image_bgr)

        # 2. Text extraction via OCR
        ocr_text = self._extract_text(image_bgr)
        text_lower = ocr_text.lower()

        # 3. Anchor detection
        anchors_found = [
            anchor for anchor in self.PAN_ANCHORS
            if anchor in text_lower
        ]

        # 4. PAN number candidate extraction & validation
        masked_pan, entity_type, status_char_valid = self._analyze_pan_number(ocr_text)

        detected = (len(anchors_found) >= 2) or (masked_pan is not None)
        if not detected and len(anchors_found) == 0:
            return self._not_detected("No PAN anchors or structural patterns detected.")

        checks = {
            "anchors_detected": len(anchors_found) >= 2,
            "pan_number_found": bool(masked_pan),
            "status_char_valid": status_char_valid,
            "qr_detected": qr_result.detected,
            "qr_readable": qr_result.readable,
        }

        # Status rules
        if status_char_valid is False:
            status = "SUSPICIOUS"
            confidence = 0.85
        elif checks["anchors_detected"] and checks["pan_number_found"] and status_char_valid:
            status = "VALID"
            confidence = 0.90
        elif checks["anchors_detected"] or checks["pan_number_found"]:
            status = "UNCERTAIN"
            confidence = 0.55
        else:
            status = "UNCERTAIN"
            confidence = 0.35

        raw_result = {
            "detected": True,
            "source": "ocr",
            "status": status,
            "document_type": self.document_type,
            "format": self.format,
            "confidence": confidence,
            "anchors_found": anchors_found,
            "anchors_count": len(anchors_found),
            "checks": checks,
            "data": {
                "masked_pan": masked_pan,
                "entity_type": entity_type,
            },
            "qr": {
                "detected": qr_result.detected,
                "readable": qr_result.readable,
                "payload_type": qr_result.payload_type,
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

    def _analyze_pan_number(self, ocr_text: str) -> tuple[Optional[str], Optional[str], Optional[bool]]:
        """Extract PAN candidate from OCR text and mask immediately."""
        # Find 10-char candidate: 5 uppercase letters, 4 digits, 1 uppercase letter
        matches = re.findall(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", ocr_text.upper())
        for candidate in matches:
            is_valid, entity_type = self.validate_pan_structure(candidate)
            # Mask immediately: e.g. ABCPE1234F -> ABCPXXXX4F
            masked = f"{candidate[:4]}XXXX{candidate[-2:]}"
            return masked, entity_type, is_valid

        return None, None, None

    def _extract_text(self, image_bgr: np.ndarray) -> str:
        """Run multi-pass OCR on document image."""
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            text1 = pytesseract.image_to_string(gray, config="--psm 6")
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
