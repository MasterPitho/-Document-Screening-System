"""
AI-Based Fake Identity & Document Screening System

"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import io
import datetime
import logging
import os
import time
import uuid
from typing import Any, Optional
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import pytesseract  # type: ignore[import-untyped]
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger("document_screening")
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", "10000"))
MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", "10000"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MRZ_CONFIDENCE_THRESHOLD = float(os.getenv("MRZ_CONFIDENCE_THRESHOLD", "0.70"))
FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", os.getenv("FACE_SIMILARITY_THRESHOLD", "0.60")))
TAMPERING_THRESHOLD = float(os.getenv("TAMPERING_THRESHOLD", "60"))
RISK_THRESHOLDS = (
    int(os.getenv("RISK_REVIEW_THRESHOLD", os.getenv("RISK_MEDIUM_THRESHOLD", "35"))),
    int(os.getenv("RISK_REJECT_THRESHOLD", os.getenv("RISK_HIGH_THRESHOLD", "65"))),
)
RISK_WEIGHTS = {
    "TAMPERING_SUSPECTED": int(os.getenv("RISK_TAMPERING", "40")),
    "FACE_MISMATCH": int(os.getenv("RISK_FACE_MISMATCH", "35")),
    "MRZ_CHECKSUM_FAILURE": int(os.getenv("RISK_MRZ_CHECKSUM", "20")),
    "EXPIRED_DOCUMENT": int(os.getenv("RISK_EXPIRED", "25")),
    "MRZ_NOT_DETECTED": int(os.getenv("RISK_MRZ_NOT_DETECTED", "20")),
    "FACE_NOT_DETECTED": int(os.getenv("RISK_FACE_NOT_DETECTED", "20")),
    "UNKNOWN_MODULE": int(os.getenv("RISK_UNKNOWN_MODULE", "15")),
}


def _validate_configuration() -> None:
    if MAX_IMAGE_BYTES <= 0:
        raise ValueError("MAX_IMAGE_BYTES must be greater than zero")
    if MAX_IMAGE_PIXELS <= 0 or MAX_IMAGE_WIDTH <= 0 or MAX_IMAGE_HEIGHT <= 0:
        raise ValueError("Image pixel and dimension limits must be greater than zero")
    if not 0.0 <= MRZ_CONFIDENCE_THRESHOLD <= 1.0:
        raise ValueError("MRZ_CONFIDENCE_THRESHOLD must be between 0 and 1")
    if not 0.0 <= FACE_SIMILARITY_THRESHOLD <= 1.0:
        raise ValueError("FACE_MATCH_THRESHOLD must be between 0 and 1")
    if not 0.0 <= TAMPERING_THRESHOLD <= 100.0:
        raise ValueError("TAMPERING_THRESHOLD must be between 0 and 100")
    review_threshold, reject_threshold = RISK_THRESHOLDS
    if not 0 <= review_threshold < reject_threshold <= 100:
        raise ValueError("Risk thresholds must satisfy 0 <= review < reject <= 100")
    if any(weight < 0 for weight in RISK_WEIGHTS.values()):
        raise ValueError("Risk weights cannot be negative")


_validate_configuration()

app = FastAPI(
    title="AI Document Screening Engine",
    description="Border checkpoint document screening for fake passport, tampering, and face verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ORIGINS", os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    ).split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceVerificationResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    module_state: str = "NOT_AVAILABLE"
    face_detected_in_document: bool
    face_detected_in_live: Optional[bool] = None
    face_bounding_box: Optional[dict[str, int]] = None
    match_status: str
    similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MRZValidationResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    detected: bool
    source: str
    status: str = "NOT_DETECTED"
    module_state: str = "NOT_AVAILABLE"
    confidence: float = 0.0
    line1: Optional[str] = None
    line2: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TamperingResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    module_state: str = "NOT_AVAILABLE"
    ela_mean_intensity: float
    ela_std_dev: float
    edge_artifact_score: float
    metadata_present: bool
    confidence: float = Field(ge=0.0, le=100.0)
    tampering_score: float = Field(default=0.0, ge=0.0, le=100.0)
    is_tampered: bool
    signals: list[str] = Field(default_factory=list)
    explanation: str = ""


class RiskAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    status: str
    level: str
    decision: str
    factors: list[dict[str, Any]]
    reasons: list[str]
    module_statuses: dict[str, str]
    explanation: str


class ScreeningModules(BaseModel):
    tampering_analysis: TamperingResult
    face_verification: FaceVerificationResult
    mrz_validation: MRZValidationResult


class ScreenResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str
    request_id: str
    risk_assessment: RiskAssessment
    modules: ScreeningModules
    processing_time_ms: int
    document: dict[str, str]
    mrz: MRZValidationResult
    tampering_analysis: TamperingResult
    face_verification: FaceVerificationResult


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled screening error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Document screening failed unexpectedly."})


def _validate_image_bytes(image_bytes: bytes, content_type: Optional[str], label: str) -> Image.Image:
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"{label} must be JPG, PNG, or WebP.")
    if not image_bytes:
        raise HTTPException(status_code=400, detail=f"{label} is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} must be {MAX_IMAGE_BYTES // (1024 * 1024)} MB or smaller.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            actual_format = image.format
            expected_formats = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
            if actual_format != expected_formats[content_type]:
                raise HTTPException(status_code=415, detail=f"{label} content does not match its declared type.")
            if image.width > MAX_IMAGE_WIDTH or image.height > MAX_IMAGE_HEIGHT:
                raise HTTPException(status_code=413, detail=f"{label} width and height exceed safe limits.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail=f"{label} dimensions are too large.")
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            decoded.load()
            return decoded.copy()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise HTTPException(status_code=413, detail=f"{label} dimensions are unsafe.") from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"{label} is invalid or unreadable.") from error


def _tampering_error(error: Exception) -> dict[str, Any]:
    logger.warning("tampering_analysis_error type=%s", type(error).__name__)
    return {
        "status": "ANALYSIS_ERROR",
        "ela_mean_intensity": 0.0,
        "ela_std_dev": 0.0,
        "edge_artifact_score": 0.0,
        "metadata_present": False,
        "confidence": 0.0,
        "tampering_score": 0.0,
        "is_tampered": False,
        "signals": ["ANALYSIS_ERROR"],
        "explanation": "Tampering analysis was unavailable; secondary inspection is required.",
    }


def _face_error(error: Exception) -> dict[str, Any]:
    logger.warning("face_verification_error type=%s", type(error).__name__)
    return {
        "status": "ERROR",
        "match_status": "ERROR",
        "face_detected_in_document": False,
        "face_detected_in_live": None,
        "similarity_score": None,
        "error": "Face verification was unavailable.",
    }


def _mrz_module_state(result: dict[str, object]) -> str:
    if result.get("status") == "OCR_FAILED":
        return "ERROR"
    if result.get("detected") and result.get("status") == "VALID":
        return "PASS"
    if result.get("detected"):
        return "FAIL"
    return "NOT_AVAILABLE"


def _face_module_state(result: dict[str, object]) -> str:
    status = result.get("status")
    if status == "ERROR":
        return "ERROR"
    if status == "MATCH":
        return "PASS"
    if status == "MISMATCH":
        return "FAIL"
    return "NOT_AVAILABLE"


def _tampering_module_state(result: dict[str, object]) -> str:
    status = result.get("status")
    if status == "ANALYSIS_ERROR":
        return "ERROR"
    if status == "SUSPECTED":
        return "FAIL"
    return "PASS"


def _has_blocking_module_state(module_statuses: dict[str, str]) -> bool:
    """True when a required screening module did not complete successfully.

    CLEAR is only allowed if every required module reported PASS. A FAIL, ERROR,
    or NOT_AVAILABLE state must force a non-clear decision.
    """
    return any(status != "PASS" for status in module_statuses.values())

# -------------------------------------------------------------
# Module 1 & 2: ICAO Doc 9303 Checksum & Parser
# -------------------------------------------------------------
def mrz_char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    elif c.isalpha():
        return ord(c.upper()) - 55
    elif c == '<':
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


def _valid_date(value: str) -> bool:
    if len(value) != 6 or not value.isdigit():
        return False
    try:
        datetime.date(1900 + int(value[:2]), int(value[2:4]), int(value[4:6]))
        return True
    except ValueError:
        try:
            datetime.date(2000 + int(value[:2]), int(value[2:4]), int(value[4:6]))
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
        and _valid_date(line2[13:19])
        and _valid_date(line2[21:27])
    )


def parse_td3_mrz(line1: str, line2: str) -> dict[str, object]:
    """Parse and validate two strict ICAO TD3 passport MRZ lines."""
    line1 = _normalize_mrz_line(line1, 1)
    line2 = _normalize_mrz_line(line2, 2)

    if len(line1) != 44 or len(line2) != 44:
        return {"status": "MALFORMED", "error": "TD3 MRZ lines must each contain exactly 44 characters."}
    if not _mrz_structure_valid(line1, line2):
        return {"status": "MALFORMED", "error": "MRZ contains invalid TD3 structure or characters."}

    doc_type = line1[0:2].replace('<', '')
    issuing_country = line1[2:5].replace('<', '')
    raw_name = line1[5:44].split('<<')
    surname = raw_name[0].replace('<', ' ').strip()
    given_names = raw_name[1].replace('<', ' ').strip() if len(raw_name) > 1 else ""

    passport_num = line2[0:9].replace('<', '')
    passport_chk = line2[9]
    nationality = line2[10:13].replace('<', '')
    dob_raw = line2[13:19]
    dob_chk = line2[19]
    gender = line2[20].replace('<', 'X')
    expiry_raw = line2[21:27]
    expiry_chk = line2[27]
    _composite_chk = line2[43]

    # Validate Checksums
    valid_passport = verify_mrz_field(line2[0:9], passport_chk)
    valid_dob = verify_mrz_field(dob_raw, dob_chk)
    valid_expiry = verify_mrz_field(expiry_raw, expiry_chk)
    composite_data = line2[0:10] + line2[13:20] + line2[21:28] + line2[28:43]
    valid_composite = verify_mrz_field(composite_data, _composite_chk)

    birth_date_valid = _valid_date(dob_raw)
    expiry_date_valid = _valid_date(expiry_raw)
    try:
        exp_year = int(expiry_raw[0:2])
        exp_month = int(expiry_raw[2:4])
        exp_day = int(expiry_raw[4:6])
        full_year = 2000 + exp_year if exp_year < 70 else 1900 + exp_year
        exp_date = datetime.date(full_year, exp_month, exp_day)
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


def _mrz_pair_score(line1: str, line2: str) -> tuple[float, dict[str, object]]:
    parsed = parse_td3_mrz(line1, line2)
    if parsed.get("status") == "MALFORMED":
        return 0.0, parsed
    checks = parsed.get("checks", {})
    valid_checks = sum(bool(value) for value in checks.values()) if isinstance(checks, dict) else 0
    score = 0.4 + valid_checks / 10.0
    if parsed.get("status") == "VALID":
        score += 0.2
    return min(1.0, score), parsed


def extract_mrz_from_image(image_bytes: bytes) -> dict[str, object]:
    """Extract a structurally valid TD3 MRZ without inventing missing characters."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        width, height = image.size
        mrz_crop = image.crop((0, int(height * 0.55), width, height))
        mrz_crop = mrz_crop.resize((width * 2, max(1, mrz_crop.height * 2)))
        variants = [
            ImageEnhance.Contrast(mrz_crop).enhance(2.0),
            np.where(np.array(mrz_crop) > 150, 255, 0).astype(np.uint8),
            cv2.adaptiveThreshold(np.array(mrz_crop), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 11),
        ]
        candidates: list[str] = []
        for variant in variants:
            for page_mode in (6, 7, 11):
                ocr_text = pytesseract.image_to_string(
                    variant,
                    config=f"--psm {page_mode} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
                )
                candidates.extend(_ocr_candidates(ocr_text))
    except Exception as error:
        return {
            "detected": False,
            "source": "ocr",
            "status": "OCR_FAILED",
            "confidence": 0.0,
            "candidate_count": 0,
            "ocr_attempts": 9,
            "reason": f"MRZ OCR unavailable: {error}",
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

    scored_pairs: list[tuple[float, str, str, dict[str, object]]] = []
    for first_index, first in enumerate(unique_candidates):
        for second_index, second in enumerate(unique_candidates):
            if first_index != second_index:
                score, parsed = _mrz_pair_score(first, second)
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

    # A candidate below the confidence threshold is not reliable enough to report.
    if score < MRZ_CONFIDENCE_THRESHOLD:
        return {"detected": False, "source": "ocr", "status": "OCR_LOW_CONFIDENCE",
                "confidence": round(score, 3), "candidate_count": len(unique_candidates),
                "ocr_attempts": 9, "selected_score": round(score, 3),
                "reason": "Unable to extract a structurally valid TD3 MRZ."}

    # A structurally plausible candidate that fails TD3 checksum or date validation is
    # never reported as a successful detection. It is only retained as diagnostic context.
    # detected == true must mean a sufficiently reliable TD3 MRZ, not merely one that
    # "looks like" an MRZ.
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
                "dates_valid": bool(isinstance(checks, dict) and checks.get("dob_valid") and checks.get("expiry_valid")),
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

# -------------------------------------------------------------
# Module 3: Tampering Detection (Error Level Analysis & Splice)
# -------------------------------------------------------------
def analyze_tampering_ela(image_bytes: bytes, quality: int = 90) -> dict[str, Any]:
    source_image = Image.open(io.BytesIO(image_bytes))
    original = source_image.convert("RGB")

    # Recompress to JPEG at standard quality
    buffer = io.BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Calculate difference
    diff: Any = ImageChops.difference(original, resaved)
    extrema = diff.getextrema()
    extrema_values: list[int] = []
    if extrema:
        extrema_values = [int(value) for item in extrema for value in item]
    max_diff = max(extrema_values) if extrema_values else 1
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)

    diff_arr = np.array(diff)
    mean_diff = float(np.mean(diff_arr))
    std_diff = float(np.std(diff_arr))

    gray = np.array(original.convert("L"))
    edge_strength = float(np.mean(cv2.Laplacian(gray, cv2.CV_64F).var()))
    edge_artifact_score = min(100.0, edge_strength / 25.0)
    metadata_present = bool(source_image.info)
    signals: list[str] = []
    if mean_diff > 35.0 or std_diff > 45.0:
        signals.append("ELA_ANOMALY")
    if edge_artifact_score > 70.0:
        signals.append("EDGE_INCONSISTENCY")
    confidence = min(
        100.0,
        round((mean_diff / 50.0) * 55.0 + (std_diff / 70.0) * 25.0 + edge_artifact_score * 0.2, 1),
    )
    tamper_detected = confidence >= TAMPERING_THRESHOLD and bool(signals)

    return {
        "status": "SUSPECTED" if tamper_detected else "NO_SIGNIFICANT_ANOMALY",
        "ela_mean_intensity": round(mean_diff, 2),
        "ela_std_dev": round(std_diff, 2),
        "edge_artifact_score": round(edge_artifact_score, 2),
        "metadata_present": metadata_present,
        "is_tampered": tamper_detected,
        "confidence": confidence,
        "tampering_score": confidence,
        "signals": signals,
        "explanation": (
            "Potential image manipulation signals detected; secondary inspection is recommended."
            if tamper_detected
            else "No significant anomaly crossed the configured prototype thresholds."
        ),
    }

# -------------------------------------------------------------
# Module 4: Face Detection & Match (OpenCV Haar / cosine embedding)
# -------------------------------------------------------------
def _face_embedding(face_crop: Any) -> np.ndarray:
    normalized = cv2.resize(face_crop, (64, 64)).astype(np.float32) / 255.0
    normalized = cv2.equalizeHist((normalized * 255).astype(np.uint8)).astype(np.float32)
    vector = normalized.flatten()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if not first_norm or not second_norm:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def extract_and_verify_faces(doc_bytes: bytes, live_bytes: Optional[bytes] = None) -> dict[str, object]:
    # Decode doc image
    doc_nparr = np.frombuffer(doc_bytes, np.uint8)
    doc_img: Any = cv2.imdecode(doc_nparr, cv2.IMREAD_COLOR)
    if doc_img is None:
        return {
            "face_detected_in_document": False,
            "status": "INVALID_IMAGE",
            "match_status": "INVALID_IMAGE",
            "similarity_score": None
        }

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore[attr-defined]
    face_cascade: Any = cv2.CascadeClassifier(face_cascade_path)  # type: ignore[attr-defined]
    if hasattr(face_cascade, "empty") and face_cascade.empty():
        return {
            "face_detected_in_document": False,
            "status": "ERROR",
            "match_status": "ERROR",
            "similarity_score": None
        }

    doc_gray: Any = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)  # type: ignore[call-overload]
    doc_faces: Any = face_cascade.detectMultiScale(doc_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    if len(doc_faces) == 0:
        return {
            "face_detected_in_document": False,
            "status": "NO_FACE_IN_DOCUMENT",
            "match_status": "NO_FACE_IN_DOCUMENT",
            "similarity_score": None
        }
    if len(doc_faces) > 1:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": None,
            "status": "MULTIPLE_FACES",
            "match_status": "MULTIPLE_FACES",
            "similarity_score": None,
        }

    (x, y, w, h) = doc_faces[0]
    if min(w, h) < 40:
        return {
            "face_detected_in_document": True,
            "status": "LOW_CONFIDENCE",
            "match_status": "LOW_CONFIDENCE",
            "similarity_score": None,
        }
    doc_face_crop = cv2.resize(doc_gray[y:y+h, x:x+w], (150, 150))

    if not live_bytes:
        return {
            "face_detected_in_document": True,
            "face_bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "status": "SKIPPED_NO_LIVE_PHOTO",
            "match_status": "SKIPPED_NO_LIVE_PHOTO",
            "similarity_score": None
        }

    # Process live image
    live_nparr = np.frombuffer(live_bytes, np.uint8)
    live_img: Any = cv2.imdecode(live_nparr, cv2.IMREAD_COLOR)
    if live_img is None:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": False,
            "status": "INVALID_IMAGE",
            "match_status": "INVALID_IMAGE",
            "similarity_score": None
        }

    live_gray: Any = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)  # type: ignore[call-overload]
    live_faces: Any = face_cascade.detectMultiScale(live_gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    if len(live_faces) == 0:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": False,
            "status": "NO_FACE_IN_LIVE_PHOTO",
            "match_status": "NO_FACE_IN_LIVE_PHOTO",
            "similarity_score": None
        }
    if len(live_faces) > 1:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": True,
            "status": "MULTIPLE_FACES",
            "match_status": "MULTIPLE_FACES",
            "similarity_score": None,
        }

    (lx, ly, lw, lh) = live_faces[0]
    if min(lw, lh) < 40:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": True,
            "status": "LOW_CONFIDENCE",
            "match_status": "LOW_CONFIDENCE",
            "similarity_score": None,
        }
    live_face_crop = cv2.resize(live_gray[ly:ly+lh, lx:lx+lw], (150, 150))

    doc_embedding = _face_embedding(doc_face_crop)
    live_embedding = _face_embedding(live_face_crop)
    sim_score = max(0.0, min(1.0, _cosine_similarity(doc_embedding, live_embedding)))

    matched = sim_score >= FACE_SIMILARITY_THRESHOLD
    return {
        "face_detected_in_document": True,
        "face_detected_in_live": True,
        "similarity_score": round(sim_score, 3),
        "status": "MATCH" if matched else "MISMATCH",
        "match_status": "MATCH" if matched else "MISMATCH",
    }

# -------------------------------------------------------------
# Module 5: Risk Engine & Screening Endpoint
# -------------------------------------------------------------
@app.post("/api/v1/screen", response_model=ScreenResponse)
async def screen_document(
    document_image: UploadFile = File(...),
    live_photo: Optional[UploadFile] = File(None),
    mrz_line1: Optional[str] = Form(None),
    mrz_line2: Optional[str] = Form(None)
) -> dict[str, object]:
    started_at = time.perf_counter()
    request_id = uuid.uuid4().hex
    logger.info("screen_request_received request_id=%s", request_id)
    if bool(mrz_line1) != bool(mrz_line2):
        raise HTTPException(
            status_code=400,
            detail="mrz_line1 and mrz_line2 must be provided together (both or neither).",
        )
    doc_bytes = await document_image.read(MAX_IMAGE_BYTES + 1)
    live_bytes = await live_photo.read(MAX_IMAGE_BYTES + 1) if live_photo else None
    _validate_image_bytes(doc_bytes, document_image.content_type, "Document image")
    if live_photo and live_bytes is not None:
        _validate_image_bytes(live_bytes, live_photo.content_type, "Live photo")

    # 1. Tampering detection
    try:
        tamper_result = analyze_tampering_ela(doc_bytes)
    except Exception as error:
        tamper_result = _tampering_error(error)

    # 2. Face verification
    try:
        face_result = extract_and_verify_faces(doc_bytes, live_bytes)
    except Exception as error:
        face_result = _face_error(error)

    # 3. MRZ validation: use submitted lines, otherwise read the image automatically.
    mrz_result: dict[str, object]
    if mrz_line1 and mrz_line2:
        mrz_data = parse_td3_mrz(mrz_line1, mrz_line2)
        mrz_result = {
            "detected": mrz_data.get("status") == "VALID",
            "source": "form",
            "status": str(mrz_data.get("status", "MALFORMED")),
            "confidence": 1.0 if mrz_data.get("status") == "VALID" else 0.5,
            "line1": mrz_line1,
            "line2": mrz_line2,
            "data": mrz_data,
        }
    else:
        mrz_result = extract_mrz_from_image(doc_bytes)

    tamper_result["module_state"] = _tampering_module_state(tamper_result)
    face_result["module_state"] = _face_module_state(face_result)
    mrz_result["module_state"] = _mrz_module_state(mrz_result)

    # 4. Explainable prototype risk score. These weights are heuristic, not probabilities.
    risk_score = 0
    risk_factors: list[dict[str, Any]] = []

    def add_factor(name: str, detail: str) -> None:
        nonlocal risk_score
        weight = RISK_WEIGHTS[name]
        risk_score += weight
        risk_factors.append({"factor": name, "weight": weight, "detail": detail})

    if tamper_result["is_tampered"]:
        add_factor("TAMPERING_SUSPECTED", "Multiple image-forensics signals exceeded prototype thresholds.")
    elif tamper_result.get("status") == "ANALYSIS_ERROR":
        add_factor("TAMPERING_SUSPECTED", "Tampering analysis failed and requires secondary inspection.")

    face_status = face_result.get("status")
    if face_status == "MISMATCH":
        add_factor("FACE_MISMATCH", "The document and live face prototype vectors did not meet the threshold.")
    elif face_status in {"NO_FACE_IN_DOCUMENT", "NO_FACE_IN_LIVE_PHOTO", "MULTIPLE_FACES", "INVALID_IMAGE", "LOW_CONFIDENCE", "ERROR"}:
        add_factor("FACE_NOT_DETECTED", f"Face verification state: {face_status}.")
    elif face_status == "SKIPPED_NO_LIVE_PHOTO":
        add_factor("UNKNOWN_MODULE", "No live photo was supplied; face verification was skipped.")

    mrz_data = mrz_result.get("data")
    checks = mrz_data.get("checks") if isinstance(mrz_data, dict) else None
    if isinstance(checks, dict):
        for check, valid in checks.items():
            if not valid:
                add_factor("MRZ_CHECKSUM_FAILURE", f"MRZ validation failed: {check}.")
    if isinstance(mrz_data, dict) and mrz_data.get("is_expired"):
        add_factor("EXPIRED_DOCUMENT", "The parsed document expiry date is before today.")
    if not mrz_result.get("detected"):
        add_factor("MRZ_NOT_DETECTED", "No structurally valid TD3 MRZ was detected.")

    module_statuses = {
        "mrz": str(mrz_result["module_state"]),
        "face": str(face_result["module_state"]),
        "tampering": str(tamper_result["module_state"]),
    }
    # Fail-safe gate: every required module must report PASS and no blocking risk
    # factor may be present before a document can be cleared. Any FAIL, ERROR, or
    # NOT_AVAILABLE module state (or any risk factor) forbids CLEAR.
    module_clear_ok = not _has_blocking_module_state(module_statuses)

    risk_score = min(100, risk_score)
    if risk_score >= RISK_THRESHOLDS[1]:
        risk_level = "HIGH_RISK"
    elif risk_score >= RISK_THRESHOLDS[0]:
        risk_level = "MEDIUM_RISK"
    else:
        risk_level = "LOW_RISK"

    if risk_score >= RISK_THRESHOLDS[1]:
        decision = "HIGH_RISK_REVIEW_REQUIRED"
        status_color = "RED"
    elif not module_clear_ok or risk_factors:
        decision = "SECONDARY_INSPECTION_REQUIRED"
        status_color = "YELLOW"
    else:
        decision = "CLEARED"
        status_color = "GREEN"

    processing_time_ms = int((time.perf_counter() - started_at) * 1000)
    reasons = [factor["detail"] for factor in risk_factors]
    logger.info("screen_request_completed request_id=%s status=%s risk_score=%s processing_time_ms=%s",
                request_id, risk_level, risk_score, processing_time_ms)

    response: dict[str, object] = {
        "status": "SCREENED",
        "request_id": request_id,
        "risk_assessment": {
            "score": risk_score,
            "status": status_color,
            "level": risk_level,
            "decision": decision,
            "factors": risk_factors,
            "reasons": reasons,
            "module_statuses": module_statuses,
            "explanation": "Heuristic screening signals require human review; they do not prove authenticity or forgery.",
        },
        "modules": {
            "tampering_analysis": tamper_result,
            "face_verification": face_result,
            "mrz_validation": mrz_result
        },
        "processing_time_ms": processing_time_ms,
        "document": {"format": "TD3" if mrz_result.get("detected") else "UNKNOWN", "type": "PASSPORT" if mrz_result.get("detected") else "UNKNOWN"},
        "mrz": mrz_result,
        "tampering_analysis": tamper_result,
        "face_verification": face_result,
    }
    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Document Screening Engine"}
