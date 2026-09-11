"""
QR Code Analysis Service for Document Screening.

Detects, decodes, and categorizes QR codes present on identity documents
(such as Aadhaar cards, national ID cards, visas).

Privacy rule: Raw PII (names, full identity numbers, live signatures, raw query
parameters) must NEVER be returned in QRAnalysisResult or stored in logs.
"""

from __future__ import annotations

import io
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QRAnalysisResult:
    detected: bool
    readable: bool
    format: str
    payload_type: str
    has_data: bool
    is_malformed: bool
    details: dict[str, Any] = field(default_factory=dict)
    raw_payload_masked_summary: str = ""


class QRAnalyzer:
    """Detects and categorizes QR codes from image data with privacy-preserving summaries."""

    def __init__(self) -> None:
        self._detector = cv2.QRCodeDetector()

    def analyze(self, image_input: Union[np.ndarray, Image.Image, bytes]) -> QRAnalysisResult:
        """Analyze an image for QR codes."""
        image = self._to_cv2_image(image_input)
        if image is None or image.size == 0:
            return QRAnalysisResult(
                detected=False,
                readable=False,
                format="NONE",
                payload_type="NONE",
                has_data=False,
                is_malformed=False,
                details={},
                raw_payload_masked_summary="No valid image input",
            )

        # Multi-pass detection: Grayscale -> CLAHE / Threshold fallback
        decoded_text, bbox = self._detect_and_decode(image)

        if bbox is None or len(bbox) == 0:
            return QRAnalysisResult(
                detected=False,
                readable=False,
                format="NONE",
                payload_type="NONE",
                has_data=False,
                is_malformed=False,
                details={},
                raw_payload_masked_summary="No QR code detected",
            )

        detected = True
        readable = bool(decoded_text and decoded_text.strip())

        if not readable:
            return QRAnalysisResult(
                detected=True,
                readable=False,
                format="QR_CODE",
                payload_type="UNREADABLE",
                has_data=False,
                is_malformed=True,
                details={"reason": "QR detected but payload could not be decoded"},
                raw_payload_masked_summary="Unreadable QR code",
            )

        payload_type, details, masked_summary = self._classify_payload(decoded_text)

        return QRAnalysisResult(
            detected=detected,
            readable=readable,
            format="QR_CODE",
            payload_type=payload_type,
            has_data=True,
            is_malformed=False,
            details=details,
            raw_payload_masked_summary=masked_summary,
        )

    def _detect_and_decode(self, image: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
        # Pass 1: Standard grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        decoded, bbox, _ = self._detector.detectAndDecode(gray)
        if bbox is not None and len(bbox) > 0 and decoded:
            return decoded, bbox

        # Pass 2: Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        decoded, bbox, _ = self._detector.detectAndDecode(enhanced)
        if bbox is not None and len(bbox) > 0 and decoded:
            return decoded, bbox

        # Pass 3: Otsu thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        decoded, bbox, _ = self._detector.detectAndDecode(thresh)
        if bbox is not None and len(bbox) > 0:
            return decoded, bbox

        # Return whatever bbox was found even if not decoded
        return decoded, bbox

    def _classify_payload(self, text: str) -> tuple[str, dict[str, Any], str]:
        stripped = text.strip()
        length = len(stripped)

        # 1. Old Aadhaar XML Barcode
        if "<PrintLetterBarcodeData" in stripped or (stripped.startswith("<?xml") and "Barcode" in stripped):
            details: dict[str, Any] = {
                "format_version": "Aadhaar XML v1",
                "payload_length": length,
            }
            # Masked UID extraction
            uid_match = re.search(r'uid="(\d+)"', stripped)
            if uid_match:
                raw_uid = uid_match.group(1)
                masked_uid = f"XXXX-XXXX-{raw_uid[-4:]}" if len(raw_uid) >= 4 else "XXXX"
                details["masked_uid"] = masked_uid
                details["has_uid"] = True
            else:
                details["has_uid"] = False

            details["has_name"] = bool(re.search(r'name="([^"]+)"', stripped))
            yob_match = re.search(r'yob="(\d{4})"', stripped)
            if yob_match:
                details["yob"] = yob_match.group(1)
            gender_match = re.search(r'gender="([MFTO])"', stripped, re.IGNORECASE)
            if gender_match:
                details["gender"] = gender_match.group(1).upper()

            masked_summary = f"Aadhaar XML barcode (length={length}, masked_uid={details.get('masked_uid', 'NONE')})"
            return "AADHAAR_XML", details, masked_summary

        # 2. Secure Aadhaar Numeric QR Payload (V2 Secure QR is a large numeric or base-10/hex decompressed string)
        if stripped.isdigit() and length >= 80:
            details = {
                "format_version": "Aadhaar Secure QR v2",
                "is_numeric": True,
                "payload_length": length,
                "has_secure_signature_block": length > 200,
            }
            masked_summary = f"Secure Aadhaar Numeric QR (length={length} digits, signature_block={details['has_secure_signature_block']})"
            return "SECURE_AADHAAR_NUMERIC", details, masked_summary

        # 3. Generic URL
        if stripped.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(stripped)
            domain = parsed.netloc.lower()
            details = {
                "scheme": parsed.scheme,
                "domain": domain,
                "path": parsed.path,
                "payload_length": length,
            }
            masked_summary = f"Web URL pointing to domain={domain}"
            return "GENERIC_URL", details, masked_summary

        # 4. Generic Text
        details = {
            "payload_length": length,
            "is_ascii": stripped.isascii(),
        }
        masked_summary = f"Generic text payload (length={length})"
        return "GENERIC_TEXT", details, masked_summary

    @staticmethod
    def _to_cv2_image(image_input: Union[np.ndarray, Image.Image, bytes]) -> Optional[np.ndarray]:
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(image_input, (bytes, bytearray)):
            arr = np.frombuffer(image_input, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None
