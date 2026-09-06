"""
MRZ parsing + OCR extraction for ICAO 9303 TD3 passports.

Kept as the same rule-based logic (structure validation, ICAO 9303 7-3-1
checksums, pivot-year date validation) that the system has always used.
"""

from __future__ import annotations

import datetime
import io
from typing import Any, Optional

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance

from app.config import Settings


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


def extract_mrz_from_image(image_bytes: bytes, settings: Settings) -> dict[str, Any]:
    """Extract a structurally valid TD3 MRZ without inventing missing characters."""
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
            for page_mode in (6, 7, 11):
                ocr_text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {page_mode} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                )
                candidates.extend(_ocr_candidates(ocr_text))
    except Exception:
        return {
            "detected": False,
            "source": "ocr",
            "status": "OCR_FAILED",
            "confidence": 0.0,
            "candidate_count": 0,
            "ocr_attempts": 9,
            "reason": "MRZ OCR unavailable; secondary inspection is recommended.",
        }

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) < 2:
        return {
            "detected": False,
            "source": "ocr",
            "status": "NOT_DETECTED",
            "confidence": 0.0,
            "candidate_count": 0,
            "ocr_attempts": 9,
            "reason": "Unable to extract two exact-length TD3 MRZ lines.",
        }

    scored_pairs: list[tuple[float, str, str, dict[str, Any]]] = []
    for first_index, first in enumerate(unique_candidates):
        for second_index, second in enumerate(unique_candidates):
            if first_index != second_index:
                score, parsed = _mrz_pair_score(first, second, year_pivot)
                scored_pairs.append((score, first, second, parsed))
    if not scored_pairs:
        return {"detected": False, "source": "ocr", "status": "NOT_DETECTED", "confidence": 0.0,
                "candidate_count": len(unique_candidates), "ocr_attempts": 9,
                "reason": "Unable to form a TD3 MRZ candidate pair."}

    score, line1, line2, parsed = max(scored_pairs, key=lambda item: item[0])
    checks = parsed.get("checks")
    checks_valid = bool(isinstance(checks, dict) and all(checks.values()))
    structurally_valid = parsed.get("status") not in {"MALFORMED", "INVALID"}
    fully_valid = parsed.get("status") == "VALID" and checks_valid

    if score < confidence_threshold:
        return {"detected": False, "source": "ocr", "status": "OCR_LOW_CONFIDENCE",
                "confidence": round(score, 3), "candidate_count": len(unique_candidates),
                "ocr_attempts": 9, "selected_score": round(score, 3),
                "reason": "Unable to extract a structurally valid TD3 MRZ."}

    if not fully_valid:
        return {
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

    return {
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
