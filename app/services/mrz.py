"""
Document parsing for ICAO 9303 documents.

Architecture
------------
``BaseDocumentParser`` defines the parsing contract returned as
``DocumentParseResult``. Concrete parsers are registered behind
``DocumentParserRouter``:

- ``TD3PassportParser`` - the classic ICAO 9303 TD3 passport parser /
  Tesseract OCR candidate pipeline (unchanged, rule-based behaviour).
- ``NationalIDTD1Parser`` - structural TD1 (3x30) parser with checksum/date
  validation and a QR payload presence placeholder (ID-1 cards such as
  national IDs / Aadhaar-style documents). Format-parsing depth is a roadmap
  item, not a production claim. An alias ``NationalIDParser`` is kept for
  backward compatibility.

Aspect-ratio heuristics
-----------------------
ID-1 cards are landscape (aspect ratio >= ``PASSPORT_MIN_RATIO``), while a
passport data page is portrait; ``auto`` selection splits on that ratio. An
explicit ``document_type`` always overrides the heuristic.

Privacy: all parsing happens in memory. MRZ/ID numbers are reported only for
checksum validation; the ``QRCodeDetector`` payload is never returned.
"""

from __future__ import annotations

import datetime
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance

from app.config import Settings

# Aspect-ratio threshold used by the router for ``document_type="auto"``.
PASSPORT_MIN_RATIO = 1.45

# document_type aliases accepted by the router and the API.
PARSER_ALIASES: dict[str, str] = {
    "td3": "passport",
    "passport": "passport",
    "td1": "national_id",
    "national_id": "national_id",
    "aadhaar": "aadhaar",
    "pan": "pan",
    "unknown": "unknown",
}


def mrz_char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    elif c.isalpha():
        return ord(c.upper()) - 55
    elif c == "<":
        return 0
    return 0


def calculate_icao_checksum(data: str) -> int:
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        total += mrz_char_value(char) * weights[i % 3]
    return total % 10


def verify_mrz_field(data: str, check_digit: str) -> bool:
    if not check_digit.isdigit():
        return False
    return calculate_icao_checksum(data) == int(check_digit)


def _valid_mrz_chars(value: str) -> bool:
    return all(char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" for char in value)


def mrz_year_full(two_digit_year: int, pivot: Optional[int] = None, year_pivot: int = 50) -> int:
    pivot_value = year_pivot if pivot is None else pivot
    return 2000 + two_digit_year if two_digit_year < pivot_value else 1900 + two_digit_year


def valid_date(value: str, year_pivot: int = 50) -> bool:
    if len(value) != 6 or not value.isdigit():
        return False
    try:
        datetime.date(
            mrz_year_full(int(value[:2]), year_pivot=year_pivot),
            int(value[2:4]),
            int(value[4:6]),
        )
        return True
    except ValueError:
        return False


def _normalize_mrz_line(line: str, line_number: int) -> str:
    value = "".join(char for char in line.upper() if not char.isspace())
    if line_number != 2 or len(value) != 44:
        return value
    numeric_positions = set(range(13, 20)) | set(range(21, 28)) | {9, 43}
    replacements = {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8"}
    return "".join(replacements.get(char, char) if index in numeric_positions else char
                   for index, char in enumerate(value))


def _mrz_structure_valid(line1: str, line2: str) -> bool:
    return (
        len(line1) == 44
        and len(line2) == 44
        and _valid_mrz_chars(line1)
        and _valid_mrz_chars(line2)
        and line1[0] == "P"
        and line1[2:5].isalpha()
        and line2[10:13].isalpha()
        and line2[13:19].isdigit()
        and line2[21:27].isdigit()
        and line2[20] in {"M", "F", "<"}
        and valid_date(line2[13:19])
        and valid_date(line2[21:27])
    )


def parse_td3_mrz(line1: str, line2: str, year_pivot: int = 50) -> dict[str, Any]:
    """Parse and validate two strict ICAO TD3 passport MRZ lines."""
    line1 = _normalize_mrz_line(line1, 1)
    line2 = _normalize_mrz_line(line2, 2)

    if len(line1) != 44 or len(line2) != 44:
        return {"status": "MALFORMED", "error": "TD3 MRZ lines must each contain exactly 44 characters."}
    if not _mrz_structure_valid(line1, line2):
        return {"status": "MALFORMED", "error": "MRZ contains invalid TD3 structure or characters."}

    doc_type = line1[0:2].replace("<", "")
    issuing_country = line1[2:5].replace("<", "")
    raw_name = line1[5:44].split("<<")
    surname = raw_name[0].replace("<", " ").strip()
    given_names = raw_name[1].replace("<", " ").strip() if len(raw_name) > 1 else ""

    passport_num = line2[0:9].replace("<", "")
    passport_chk = line2[9]
    nationality = line2[10:13].replace("<", "")
    dob_raw = line2[13:19]
    dob_chk = line2[19]
    gender = line2[20].replace("<", "X")
    expiry_raw = line2[21:27]
    expiry_chk = line2[27]
    _composite_chk = line2[43]

    valid_passport = verify_mrz_field(line2[0:9], passport_chk)
    valid_dob = verify_mrz_field(dob_raw, dob_chk)
    valid_expiry = verify_mrz_field(expiry_raw, expiry_chk)
    composite_data = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    valid_composite = verify_mrz_field(composite_data, _composite_chk)

    birth_date_valid = valid_date(dob_raw, year_pivot=year_pivot)
    expiry_date_valid = valid_date(expiry_raw, year_pivot=year_pivot)
    try:
        exp_date = datetime.date(
            mrz_year_full(int(expiry_raw[0:2]), year_pivot=year_pivot),
            int(expiry_raw[2:4]),
            int(expiry_raw[4:6]),
        )
        is_expired = exp_date < datetime.date.today()
    except ValueError:
        is_expired = False

    checks = {
        "passport_number_valid": valid_passport,
        "dob_valid": valid_dob and birth_date_valid,
        "expiry_valid": valid_expiry and expiry_date_valid,
        "composite_valid": valid_composite,
    }
    status = "VALID" if all(checks.values()) else "INVALID"

    return {
        "status": status,
        "doc_type": doc_type,
        "issuing_country": issuing_country,
        "full_name": f"{given_names} {surname}".strip(),
        "passport_number": passport_num,
        "nationality": nationality,
        "date_of_birth": dob_raw,
        "gender": gender,
        "expiry_date": expiry_raw,
        "is_expired": is_expired,
        "checks": checks,
    }


def _clean_mrz_line(line: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    return "".join(char for char in line.upper() if char in allowed)


def _ocr_candidates(ocr_text: str) -> list[str]:
    candidates = []
    for raw_line in ocr_text.splitlines():
        candidate = _clean_mrz_line(raw_line)
        if len(candidate) == 44:
            candidates.append(candidate)
    return candidates


def _mrz_pair_score(line1: str, line2: str, year_pivot: int) -> tuple[float, dict[str, Any]]:
    parsed = parse_td3_mrz(line1, line2, year_pivot=year_pivot)
    if parsed.get("status") == "MALFORMED":
        return 0.0, parsed
    checks = parsed.get("checks", {})
    valid_checks = sum(bool(value) for value in checks.values()) if isinstance(checks, dict) else 0
    score = 0.4 + valid_checks / 10.0
    if parsed.get("status") == "VALID":
        score += 0.2
    return min(1.0, score), parsed


def _tc1_lines_valid(line1: str, line2: str, line3: str) -> bool:
    return (
        len(line1) == 30
        and len(line2) == 30
        and len(line3) == 30
        and _valid_mrz_chars(line1)
        and _valid_mrz_chars(line2)
        and _valid_mrz_chars(line3)
        and line1[0:2] in {"ID", "I<"}
        and line2[10:13].isalpha()
        and line2[13:19].isdigit()
        and line2[21:27].isdigit()
        and line2[20] in {"M", "F", "<"}
        and valid_date(line2[13:19])
        and valid_date(line2[21:27])
    )


def parse_td1_national_id(line1: str, line2: str, line3: str, year_pivot: int = 50) -> dict[str, Any]:
    """Structural TD1 (3x30) validation for national ID / Aadhaar-style cards."""
    if len(line1) != 30 or len(line2) != 30 or len(line3) != 30:
        return {"status": "MALFORMED",
                "error": "TD1 lines must each contain exactly 30 characters."}
    if not _tc1_lines_valid(line1, line2, line3):
        return {"status": "MALFORMED",
                "error": "TD1 structure or characters are invalid."}

    doc_number = line2[0:9].replace("<", "")
    dob_raw = line2[13:19]
    gender = line2[20].replace("<", "X")
    expiry_raw = line2[21:27]

    valid_number = verify_mrz_field(line2[0:9], line2[9])
    valid_dob = verify_mrz_field(dob_raw, line2[19])
    valid_expiry = verify_mrz_field(expiry_raw, line2[27])
    composite_data = line2[0:10] + line2[13:20] + line2[21:28] + line3[0:15]
    valid_composite = verify_mrz_field(composite_data, line3[15])

    birth_date_valid = valid_date(dob_raw, year_pivot=year_pivot)
    expiry_date_valid = valid_date(expiry_raw, year_pivot=year_pivot)
    try:
        exp_date = datetime.date(
            mrz_year_full(int(expiry_raw[0:2]), year_pivot=year_pivot),
            int(expiry_raw[2:4]),
            int(expiry_raw[4:6]),
        )
        is_expired = exp_date < datetime.date.today()
    except ValueError:
        is_expired = False

    checks = {
        "document_number_valid": valid_number,
        "dob_valid": valid_dob and birth_date_valid,
        "expiry_valid": valid_expiry and expiry_date_valid,
        "composite_valid": valid_composite,
    }
    status = "VALID" if all(checks.values()) else "INVALID"

    return {
        "status": status,
        "doc_type": line1[0:2].replace("<", ""),
        "issuing_country": line1[2:5].replace("<", ""),
        "full_name": line1[5:30].replace("<", " ").strip(),
        "document_number": doc_number,
        "nationality": line2[10:13].replace("<", ""),
        "date_of_birth": dob_raw,
        "gender": gender,
        "expiry_date": expiry_raw,
        "optional_data": line3[0:15].replace("<", " ").strip(),
        "is_expired": is_expired,
        "checks": checks,
    }


def _decode_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_bgr(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Unable to encode an ndarray for in-memory parsing.")
    return encoded.tobytes()


@dataclass
class DocumentParseResult:
    detected: bool
    status: str
    document_type: str
    format: str
    confidence: float
    data: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseDocumentParser(ABC):
    """Contract for every document parser in the system."""

    name: str = "unknown"
    document_type: str = "UNKNOWN"
    format: str = "UNKNOWN"

    @abstractmethod
    def can_parse(self, image_bgr: np.ndarray) -> bool:
        """Structural gate used by the router for ``auto`` selection."""

    @abstractmethod
    def parse(
        self,
        image: Any,
        settings: Optional[Settings] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        line3: Optional[str] = None,
        **kwargs: Any,
    ) -> DocumentParseResult:
        """Parse an in-memory document image (or pre-extracted MRZ lines).

        ``image`` accepts image bytes (historical contract) or an OpenCV BGR
        ``np.ndarray``. When ``line1``/``line2``/``line3`` are supplied they
        are validated directly instead of running OCR.
        """


class TD3PassportParser(BaseDocumentParser):
    """ICAO 9303 TD3 passport: OCR candidate pipeline + strict validation.

    Behaviour is identical to the historical single-passport implementation.
    """

    name = "passport"
    document_type = "PASSPORT"
    format = "TD3"

    def can_parse(self, image_bgr: np.ndarray) -> bool:
        if image_bgr is None or image_bgr.size == 0:
            return False
        height, width = image_bgr.shape[:2]
        if height <= 0:
            return False
        return (width / height) < PASSPORT_MIN_RATIO

    def parse(
        self,
        image: Any,
        settings: Optional[Settings] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        line3: Optional[str] = None,
        **kwargs: Any,
    ) -> DocumentParseResult:
        """Extract a structurally valid TD3 MRZ without inventing missing characters."""
        if settings is None:
            settings = Settings.from_env()
        if line1 is not None and line2 is not None:
            return self._from_lines(line1, line2, settings)
        image_bytes: bytes = image if isinstance(image, bytes) else _encode_bgr(image)
        year_pivot = settings.mrz_year_pivot
        confidence_threshold = settings.mrz_confidence_threshold
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            width, height = image.size
            mrz_crop = image.crop((0, int(height * 0.55), width, height))
            mrz_crop = mrz_crop.resize((width * 2, max(1, mrz_crop.height * 2)))
            try:
                variants = [
                    ImageEnhance.Contrast(mrz_crop).enhance(2.0),
                    np.where(np.array(mrz_crop) > 150, 255, 0).astype(np.uint8),
                    cv2.adaptiveThreshold(np.array(mrz_crop), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 11),
                ]
            finally:
                image.close()
                mrz_crop.close()
            candidates: list[str] = []
            for variant in variants:
                ocr_text = pytesseract.image_to_string(
                    variant,
                    config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                )
                candidates.extend(_ocr_candidates(ocr_text))
        except Exception:
            raw = {
                "detected": False,
                "source": "ocr",
                "status": "OCR_FAILED",
                "confidence": 0.0,
                "candidate_count": 0,
                "ocr_attempts": 9,
                "reason": "MRZ OCR unavailable; secondary inspection is recommended.",
            }
            return self._result(raw)

        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) < 2:
            raw = {
                "detected": False,
                "source": "ocr",
                "status": "NOT_DETECTED",
                "confidence": 0.0,
                "candidate_count": 0,
                "ocr_attempts": 9,
                "reason": "Unable to extract two exact-length TD3 MRZ lines.",
            }
            return self._result(raw)

        scored_pairs: list[tuple[float, str, str, dict[str, Any]]] = []
        for first_index, first in enumerate(unique_candidates):
            for second_index, second in enumerate(unique_candidates):
                if first_index != second_index:
                    score, parsed = _mrz_pair_score(first, second, year_pivot)
                    scored_pairs.append((score, first, second, parsed))
        if not scored_pairs:
            raw = {"detected": False, "source": "ocr", "status": "NOT_DETECTED", "confidence": 0.0,
                   "candidate_count": len(unique_candidates), "ocr_attempts": 9,
                   "reason": "Unable to form a TD3 MRZ candidate pair."}
            return self._result(raw)

        score, line1, line2, parsed = max(scored_pairs, key=lambda item: item[0])
        checks = parsed.get("checks")
        checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
        structurally_valid = parsed.get("status") not in {"MALFORMED", "INVALID"}
        fully_valid = parsed.get("status") == "VALID" and checks_valid

        if score < confidence_threshold:
            raw = {"detected": False, "source": "ocr", "status": "OCR_LOW_CONFIDENCE",
                   "confidence": round(score, 3), "candidate_count": len(unique_candidates),
                   "ocr_attempts": 9, "selected_score": round(score, 3),
                   "reason": "Unable to extract a structurally valid TD3 MRZ."}
            return self._result(raw)

        if not fully_valid:
            raw = {
                "detected": False,
                "source": "ocr",
                "status": "OCR_LOW_CONFIDENCE",
                "confidence": round(score, 3),
                "candidate_count": len(unique_candidates),
                "ocr_attempts": 9,
                "selected_score": round(score, 3),
                "validation": {
                    "structure_valid": structurally_valid,
                    "checksums_valid": checks_valid,
                    "dates_valid": bool(
                        isinstance(checks, dict)
                        and checks.get("dob_valid")
                        and checks.get("expiry_valid")
                    ),
                },
                "candidate_line1": line1,
                "candidate_line2": line2,
                "candidate_data": parsed,
                "reason": "A structurally plausible candidate failed TD3 checksum or date validation.",
            }
            return self._result(raw)

        raw = {
            "detected": True,
            "source": "ocr",
            "status": "VALID",
            "confidence": round(score, 3),
            "candidate_count": len(unique_candidates),
            "ocr_attempts": 9,
            "selected_score": round(score, 3),
            "validation": {
                "structure_valid": True,
                "checksums_valid": True,
                "dates_valid": True,
            },
            "line1": line1,
            "line2": line2,
            "data": parsed,
        }
        return self._result(raw)

    def _from_lines(self, line1: str, line2: str, settings: Settings) -> DocumentParseResult:
        """Validate two pre-extracted TD3 lines (form/telegram path, no OCR)."""
        parsed = parse_td3_mrz(line1, line2, year_pivot=settings.mrz_year_pivot)
        checks = parsed.get("checks")
        checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
        status = str(parsed.get("status", "MALFORMED"))
        raw = {
            "detected": bool(checks_valid),
            "source": "form",
            "status": status,
            "confidence": 1.0 if checks_valid else 0.5,
            "line1": line1,
            "line2": line2,
            "data": parsed,
            "validation": {
                "structure_valid": bool(status not in {"MALFORMED", "INVALID"}),
                "checksums_valid": checks_valid,
                "dates_valid": bool(isinstance(checks, dict)
                                    and checks.get("dob_valid")
                                    and checks.get("expiry_valid")),
            },
        }
        return self._result(raw)

    def _result(self, raw: dict[str, Any]) -> DocumentParseResult:
        result = dict(raw)
        result["format"] = self.format
        result["document_type"] = self.document_type
        return DocumentParseResult(
            detected=bool(result.get("detected", False)),
            status=str(result.get("status", "NOT_DETECTED")),
            document_type=self.document_type,
            format=self.format,
            confidence=float(result.get("confidence", 0.0)),
            data=dict(result.get("data", {})),
            raw=result,
            error=None if result.get("detected") else str(result.get("reason", "")),
        )


class NationalIDTD1Parser(BaseDocumentParser):
    """Structural TD1 (3x30) national ID parser with QR presence check.

    The TD1 structure, ICAO checksums and dates are validated like the
    passport path; nothing is ever marked ``VALID`` unless every check
    passes, so the parser never falsely reports successful validation. The
    QR-code payload is decoded (when present) purely for presence/flags;
    the payload itself is never echoed back (privacy contract).
    """

    name = "national_id"
    document_type = "NATIONAL_ID"
    format = "TD1"

    def can_parse(self, image_bgr: np.ndarray) -> bool:
        if image_bgr is None or image_bgr.size == 0:
            return False
        height, width = image_bgr.shape[:2]
        if height <= 0:
            return False
        return (width / height) >= PASSPORT_MIN_RATIO

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
        if line1 is not None and line2 is not None and line3 is not None:
            return self._from_lines(line1, line2, line3, settings)
        image_bytes: bytes = image if isinstance(image, bytes) else _encode_bgr(image)
        year_pivot = settings.mrz_year_pivot
        qr_present, qr_error = self._qr_probe(image_bytes)

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            width, height = image.size
            crop = image.crop((0, int(height * 0.55), width, height))
            crop = crop.resize((width * 2, max(1, crop.height * 2)))
            try:
                variants = [
                    ImageEnhance.Contrast(crop).enhance(2.0),
                ]
            finally:
                image.close()
                crop.close()
            lines: list[str] = []
            for variant in variants:
                ocr_text = pytesseract.image_to_string(
                    variant,
                    config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                )
                for raw_line in ocr_text.splitlines():
                    line = _clean_mrz_line(raw_line)
                    if len(line) == 30:
                        lines.append(line)
        except Exception:
            raw = {
                "detected": False,
                "source": "ocr",
                "status": "OCR_FAILED",
                "confidence": 0.0,
                "qr_present": qr_present,
                "qr_error": qr_error,
                "reason": "OCR unavailable; secondary inspection is recommended.",
            }
            return self._result(raw, settings)

        unique_lines = list(dict.fromkeys(lines))
        for first_index, first in enumerate(unique_lines):
            for second_index, second in enumerate(unique_lines):
                for third_index, third in enumerate(unique_lines):
                    if len({first_index, second_index, third_index}) != 3:
                        continue
                    if not _tc1_lines_valid(first, second, third):
                        continue
                    parsed = parse_td1_national_id(first, second, third, year_pivot=year_pivot)
                    if parsed.get("status") == "VALID" or parsed.get("checks"):
                        checks = parsed.get("checks", {})
                        checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
                        confidence = 1.0 if checks_valid else 0.5
                        raw = {
                            "detected": True,
                            "source": "ocr",
                            "status": parsed.get("status", "INVALID"),
                            "confidence": confidence,
                            "line1": first,
                            "line2": second,
                            "line3": third,
                            "qr_present": qr_present,
                            "qr_error": qr_error,
                            "validation": {
                                "structure_valid": True,
                                "checksums_valid": checks_valid,
                                "dates_valid": bool(
                                    checks.get("dob_valid") and checks.get("expiry_valid")
                                ),
                            },
                            "data": parsed,
                        }
                        return self._result(raw, settings)

        raw = {
            "detected": False,
            "source": "ocr",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "qr_present": qr_present,
            "qr_error": qr_error,
            "reason": "No valid TD1 (3x30) structure could be extracted.",
        }
        return self._result(raw, settings)

    @staticmethod
    def _qr_probe(image_bytes: bytes) -> tuple[bool, Optional[str]]:
        try:
            image_bgr = _decode_bgr(image_bytes)
            if image_bgr is None:
                return False, "Unable to decode document image for QR probe."
            detector = cv2.QRCodeDetector()
            result, _, _ = detector.detectAndDecode(image_bgr)
            if result:
                return True, None
            return False, "No QR payload decoded."
        except Exception:  # noqa: BLE001 - QR probing must be non-fatal
            return False, "QR probe failed internally."

    def _from_lines(self, line1: str, line2: str, line3: str, settings: Settings) -> DocumentParseResult:
        """Validate three pre-extracted TD1 lines (form/telegram path, no OCR)."""
        parsed = parse_td1_national_id(line1, line2, line3, year_pivot=settings.mrz_year_pivot)
        checks = parsed.get("checks")
        checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
        status = str(parsed.get("status", "MALFORMED"))
        raw = {
            "detected": bool(checks_valid),
            "source": "form",
            "status": status,
            "confidence": 1.0 if checks_valid else 0.5,
            "line1": line1,
            "line2": line2,
            "line3": line3,
            "data": parsed,
            "validation": {
                "structure_valid": bool(status not in {"MALFORMED", "INVALID"}),
                "checksums_valid": checks_valid,
                "dates_valid": bool(isinstance(checks, dict)
                                    and checks.get("dob_valid")
                                    and checks.get("expiry_valid")),
            },
        }
        return self._result(raw, settings)

    def _result(self, raw: dict[str, Any], settings: Settings) -> DocumentParseResult:
        result = dict(raw)
        result["format"] = self.format
        result["document_type"] = self.document_type
        return DocumentParseResult(
            detected=bool(result.get("detected", False)),
            status=str(result.get("status", "NOT_DETECTED")),
            document_type=self.document_type,
            format=self.format,
            confidence=float(result.get("confidence", 0.0)),
            data=dict(result.get("data", {})),
            raw=result,
            error=None if result.get("detected") else str(result.get("reason", "")),
        )


NationalIDParser = NationalIDTD1Parser  # backward-compatible alias


class UnknownDocumentParser(BaseDocumentParser):
    """Fallback strategy returned when auto-routing cannot identify document type."""

    name = "unknown"
    document_type = "UNKNOWN"
    format = "UNKNOWN"

    def can_parse(self, image_bgr: np.ndarray) -> bool:
        return True

    def parse(
        self,
        image: Any,
        settings: Optional[Settings] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        line3: Optional[str] = None,
        **kwargs: Any,
    ) -> DocumentParseResult:
        routing_info = kwargs.get("routing_info", {})
        raw = {
            "detected": False,
            "status": "NOT_DETECTED",
            "source": "auto_router",
            "confidence": 0.0,
            "reason": "Could not determine document type from visual/textual evidence; ambiguous or unidentifiable document.",
            "document_type": "UNKNOWN",
            "format": "UNKNOWN",
            "routing": routing_info,
        }
        return DocumentParseResult(
            detected=False,
            status="NOT_DETECTED",
            document_type="UNKNOWN",
            format="UNKNOWN",
            confidence=0.0,
            data={},
            raw=raw,
            error=raw["reason"],
        )


class DocumentParserRouter:
    """Selects a parser by explicit ``document_type`` or multi-signal evidence scoring."""

    def __init__(self, passport_min_ratio: float = PASSPORT_MIN_RATIO) -> None:
        self.passport_min_ratio = passport_min_ratio
        self._parsers: dict[str, BaseDocumentParser] = {}

    def _ensure_parser(self, key: str) -> BaseDocumentParser:
        if key not in self._parsers:
            if key == "passport":
                self._parsers["passport"] = TD3PassportParser()
            elif key == "national_id":
                self._parsers["national_id"] = NationalIDTD1Parser()
            elif key == "aadhaar":
                from app.services.aadhaar import AadhaarDocumentParser
                self._parsers["aadhaar"] = AadhaarDocumentParser()
            elif key == "pan":
                from app.services.pan import PANDocumentParser
                self._parsers["pan"] = PANDocumentParser()
            elif key == "unknown":
                self._parsers["unknown"] = UnknownDocumentParser()
            else:
                raise ValueError(f"Unknown parser key: {key}")
        return self._parsers[key]

    def resolve(self, document_type: str) -> str:
        normalized = (document_type or "auto").strip().lower()
        if normalized == "auto":
            return "auto"
        key = PARSER_ALIASES.get(normalized)
        if key is None:
            raise ValueError(
                "document_type must be one of: auto, "
                "td3/passport, td1/national_id, aadhaar, pan.")
        return key

    def score_evidence(
        self, image_bgr: np.ndarray
    ) -> tuple[dict[str, int], dict[str, list[str]], bool]:
        """Score visual, textual, and barcode evidence for each supported document type."""
        scores = {"passport": 0, "national_id": 0, "aadhaar": 0, "pan": 0}
        evidence: dict[str, list[str]] = {k: [] for k in scores}
        ocr_error = False

        if image_bgr is None or image_bgr.size == 0 or image_bgr.shape[0] == 0:
            return scores, evidence, False

        height, width = image_bgr.shape[:2]
        ratio = width / max(1, height)

        # 1. Aspect ratio signal
        if ratio < self.passport_min_ratio:
            scores["passport"] += 5
            evidence["passport"].append("portrait_aspect_ratio")
        else:
            scores["national_id"] += 5
            scores["aadhaar"] += 5
            scores["pan"] += 5
            evidence["national_id"].append("landscape_aspect_ratio")

        # 2. Barcode / QR signal
        try:
            from app.services.qr import QRAnalyzer
            qr_result = QRAnalyzer().scan(image_bgr)
            if qr_result and qr_result.present:
                payload = qr_result.raw_payload or ""
                payload_lower = payload.lower()
                if qr_result.is_aadhaar or "<printletterbarcodedata" in payload_lower or "<?xml" in payload_lower:
                    scores["aadhaar"] += 100
                    evidence["aadhaar"].append("aadhaar_qr_code")
                elif "income tax" in payload_lower or re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", payload):
                    scores["pan"] += 100
                    evidence["pan"].append("pan_qr_code")
                else:
                    scores["national_id"] += 15
                    scores["aadhaar"] += 15
                    scores["pan"] += 15
                    evidence["national_id"].append("generic_qr_code")
        except Exception:
            pass

        # 3. Optical Character Recognition (OCR) signal
        try:
            ocr_text = pytesseract.image_to_string(image_bgr) or ""
        except pytesseract.TesseractNotFoundError:
            ocr_text = ""
        except Exception:
            ocr_error = True
            ocr_text = ""

        if ocr_text:
            text_upper = ocr_text.upper()
            lines = [l.strip() for l in text_upper.splitlines() if l.strip()]

            # Passport checks
            for line in lines:
                if line.startswith("P<") or "P<" in line:
                    scores["passport"] += 60
                    evidence["passport"].append("td3_mrz_p_prefix")
                    break
                if len(line) >= 35 and "<<" in line:
                    scores["passport"] += 30
                    evidence["passport"].append("td3_filler_characters")
                    break
            for kw in ["PASSPORT", "REPUBLIC OF", "TRAVEL DOCUMENT"]:
                if kw in text_upper:
                    scores["passport"] += 40
                    evidence["passport"].append(f"passport_keyword_{kw}")

            # National ID checks
            for line in lines:
                if (line.startswith(("I", "A", "C")) and len(line) in (29, 30)) or (len(line) == 30 and "<<" in line):
                    scores["national_id"] += 50
                    evidence["national_id"].append("td1_mrz_prefix")
                    break
            for kw in ["IDENTITY CARD", "NATIONAL IDENTITY", "IDENTITY DOCUMENT"]:
                if kw in text_upper:
                    scores["national_id"] += 40
                    evidence["national_id"].append(f"national_id_keyword_{kw}")

            # Aadhaar checks
            aadhaar_anchors = [
                "GOVERNMENT OF INDIA", "GOVT OF INDIA", "BHARAT SARKAR",
                "UNIQUE IDENTIFICATION", "UIDAI", "MERA AADHAAR",
                "ENROLMENT NO", "HELP@UIDAI"
            ]
            for anchor in aadhaar_anchors:
                if anchor in text_upper:
                    scores["aadhaar"] += 35
                    evidence["aadhaar"].append(f"aadhaar_anchor_{anchor}")

            if re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text_upper) or re.search(r"\b\d{12}\b", text_upper):
                scores["aadhaar"] += 35
                evidence["aadhaar"].append("aadhaar_uid_pattern")

            # PAN checks
            pan_anchors = [
                "INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "PERMANENT ACCOUNT NUMBER",
                "FATHER'S NAME", "SIGNATURE"
            ]
            for anchor in pan_anchors:
                if anchor in text_upper:
                    scores["pan"] += 40
                    evidence["pan"].append(f"pan_anchor_{anchor}")

            if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text_upper):
                scores["pan"] += 50
                evidence["pan"].append("pan_pattern_match")

        return scores, evidence, ocr_error

    def parser_for(self, document_type: str = "auto",
                   image_bgr: Optional[np.ndarray] = None) -> BaseDocumentParser:
        """Resolve a parser from an explicit type or multi-signal auto routing."""
        key = self.resolve(document_type)
        if key != "auto":
            return self._ensure_parser(key)
        if image_bgr is None or image_bgr.size == 0 or image_bgr.shape[0] == 0:
            return self._ensure_parser("passport")  # backward-compatible default when no image

        scores, evidence, ocr_error = self.score_evidence(image_bgr)
        if ocr_error:
            # Fallback safely to passport when OCR runtime throws an unexpected exception
            return self._ensure_parser("passport")

        best_key, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score < 20:
            # Ambiguous or blank image: does not meet minimum evidence threshold
            return self._ensure_parser("unknown")

        return self._ensure_parser(best_key)

    def select(self, image_bytes: bytes, document_type: str = "auto") -> BaseDocumentParser:
        image_bgr = _decode_bgr(image_bytes)
        return self.parser_for(document_type, image_bgr)

    def parse(
        self,
        image_bytes: bytes,
        settings: Settings,
        document_type: str = "auto",
        **kwargs: Any,
    ) -> DocumentParseResult:
        parser = self.select(image_bytes, document_type)
        return parser.parse(image_bytes, settings, **kwargs)


_router: DocumentParserRouter = DocumentParserRouter()


def get_document_parser(
    doc_type: str = "auto",
    image: Optional[np.ndarray] = None,
) -> BaseDocumentParser:
    """Strategy factory for ``POST /api/v1/screen`` document routing.

    - ``passport`` / ``td3`` -> ``TD3PassportParser``
    - ``national_id`` / ``td1`` / ``aadhaar`` -> ``NationalIDTD1Parser``
    - ``auto`` -> aspect-ratio detection when an ``image`` is supplied,
      otherwise the classic passport parser
    - any other ``doc_type`` -> ``ValueError`` (explicit validation error)
    """
    return _router.parser_for(doc_type, image)


def extract_mrz_from_image(
    image_bytes: bytes,
    settings: Settings,
    document_type: str = "auto",
) -> dict[str, Any]:
    """Backward-compatible extraction entry point that routes by parser.

    Returns the full result dict (with ``format`` and ``document_type`` keys)
    for any parser. Keeps the historical two-argument call contract valid:
    ``extract_mrz_from_image(image_bytes, settings)`` routes automatically.
    """
    parser = get_document_parser(document_type, _decode_bgr(image_bytes))
    result = parser.parse(image_bytes, settings)
    raw = dict(result.raw)
    raw.setdefault("document_type", result.document_type)
    raw.setdefault("format", result.format)
    return raw
