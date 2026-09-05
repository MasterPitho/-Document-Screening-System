"""
AI-Based Fake Identity & Document Screening System

"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import io
import datetime
from typing import Any, Optional
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import pytesseract  # type: ignore[import-untyped]
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="AI Document Screening Engine",
    description="Border checkpoint document screening for fake passport, tampering, and face verification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceVerificationResult(BaseModel):
    face_detected_in_document: bool
    face_detected_in_live: Optional[bool] = None
    face_bounding_box: Optional[dict[str, int]] = None
    match_status: str
    similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class MRZValidationResult(BaseModel):
    detected: bool
    source: str
    line1: Optional[str] = None
    line2: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class TamperingResult(BaseModel):
    ela_mean_intensity: float
    ela_std_dev: float
    edge_artifact_score: float
    metadata_present: bool
    confidence: float = Field(ge=0.0, le=100.0)
    is_tampered: bool


class RiskAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    status: str
    decision: str
    factors: list[str]


class ScreeningModules(BaseModel):
    tampering_analysis: TamperingResult
    face_verification: FaceVerificationResult
    mrz_validation: MRZValidationResult


class ScreenResponse(BaseModel):
    status: str
    risk_assessment: RiskAssessment
    modules: ScreeningModules

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

def parse_td3_mrz(line1: str, line2: str) -> dict[str, object]:
    """
    Standard TD3 Passport MRZ (2 lines of 44 characters).
    """
    line1 = line1.strip().upper()
    line2 = line2.strip().upper()
    
    if len(line1) != 44 or len(line2) != 44:
        return {"error": "Invalid TD3 MRZ length. Expected 44 chars per line."}

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

    # Expiry validation
    try:
        exp_year = int(expiry_raw[0:2])
        exp_month = int(expiry_raw[2:4])
        exp_day = int(expiry_raw[4:6])
        full_year = 2000 + exp_year if exp_year < 70 else 1900 + exp_year
        exp_date = datetime.date(full_year, exp_month, exp_day)
        is_expired = exp_date < datetime.date.today()
    except Exception:
        is_expired = True

    return {
        "doc_type": doc_type,
        "issuing_country": issuing_country,
        "full_name": f"{given_names} {surname}".strip(),
        "passport_number": passport_num,
        "nationality": nationality,
        "date_of_birth": dob_raw,
        "gender": gender,
        "expiry_date": expiry_raw,
        "is_expired": is_expired,
        "checks": {
            "passport_number_valid": valid_passport,
            "dob_valid": valid_dob,
            "expiry_valid": valid_expiry,
            "composite_valid": valid_composite,
        }
    }


def _clean_mrz_line(line: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    return "".join(char for char in line.upper() if char in allowed)


def extract_mrz_from_image(image_bytes: bytes) -> dict[str, object]:
    """Read the lower passport zone and return the best two 44-character lines."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        width, height = image.size
        mrz_crop = image.crop((0, int(height * 0.55), width, height))
        mrz_crop = mrz_crop.resize((width * 2, max(1, mrz_crop.height * 2)))
        mrz_crop = ImageEnhance.Contrast(mrz_crop).enhance(2.0)
        ocr_text = pytesseract.image_to_string(
            mrz_crop,
            config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
        )
    except Exception as error:
        return {
            "detected": False,
            "source": "ocr",
            "error": f"MRZ OCR unavailable: {error}",
        }
    candidates = [_clean_mrz_line(line) for line in ocr_text.splitlines()]
    candidates = [line for line in candidates if len(line) >= 40]
    if len(candidates) < 2:
        return {
            "detected": False,
            "source": "ocr",
            "error": "Unable to detect two MRZ lines from the document image.",
        }

    line1 = min(candidates, key=lambda line: abs(len(line) - 44))[:44].ljust(44, "<")
    remaining = [line for line in candidates if line != line1]
    line2 = min(remaining, key=lambda line: abs(len(line) - 44))[:44].ljust(44, "<")
    return {
        "detected": True,
        "source": "ocr",
        "line1": line1,
        "line2": line2,
        "data": parse_td3_mrz(line1, line2),
    }

# -------------------------------------------------------------
# Module 3: Tampering Detection (Error Level Analysis & Splice)
# -------------------------------------------------------------
def analyze_tampering_ela(image_bytes: bytes, quality: int = 90) -> dict[str, float | bool]:
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
    confidence = min(
        100.0,
        round((mean_diff / 50.0) * 55.0 + (std_diff / 70.0) * 25.0 + edge_artifact_score * 0.2, 1),
    )
    tamper_detected = confidence >= 60.0

    return {
        "ela_mean_intensity": round(mean_diff, 2),
        "ela_std_dev": round(std_diff, 2),
        "edge_artifact_score": round(edge_artifact_score, 2),
        "metadata_present": metadata_present,
        "is_tampered": tamper_detected,
        "confidence": confidence,
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
            "match_status": "INVALID_DOC_IMAGE",
            "similarity_score": 0.0
        }

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore[attr-defined]
    face_cascade: Any = cv2.CascadeClassifier(face_cascade_path)  # type: ignore[attr-defined]
    if hasattr(face_cascade, "empty") and face_cascade.empty():
        return {
            "face_detected_in_document": False,
            "match_status": "FACE_CASCADE_LOAD_FAILED",
            "similarity_score": 0.0
        }

    doc_gray: Any = cv2.cvtColor(doc_img, cv2.COLOR_BGR2GRAY)  # type: ignore[call-overload]
    doc_faces: Any = face_cascade.detectMultiScale(doc_gray, scaleFactor=1.1, minNeighbors=4)

    if len(doc_faces) == 0:
        return {
            "face_detected_in_document": False,
            "match_status": "NO_FACE_IN_DOC",
            "similarity_score": 0.0
        }

    (x, y, w, h) = doc_faces[0]
    doc_face_crop = cv2.resize(doc_gray[y:y+h, x:x+w], (150, 150))

    if not live_bytes:
        return {
            "face_detected_in_document": True,
            "face_bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
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
            "match_status": "INVALID_LIVE_IMAGE",
            "similarity_score": 0.0
        }

    live_gray: Any = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)  # type: ignore[call-overload]
    live_faces: Any = face_cascade.detectMultiScale(live_gray, scaleFactor=1.1, minNeighbors=4)

    if len(live_faces) == 0:
        return {
            "face_detected_in_document": True,
            "face_detected_in_live": False,
            "match_status": "NO_FACE_IN_LIVE_STREAM",
            "similarity_score": 0.0
        }

    (lx, ly, lw, lh) = live_faces[0]
    live_face_crop = cv2.resize(live_gray[ly:ly+lh, lx:lx+lw], (150, 150))

    doc_embedding = _face_embedding(doc_face_crop)
    live_embedding = _face_embedding(live_face_crop)
    sim_score = max(0.0, min(1.0, _cosine_similarity(doc_embedding, live_embedding)))

    matched = sim_score >= 0.60
    return {
        "face_detected_in_document": True,
        "face_detected_in_live": True,
        "similarity_score": round(sim_score, 3),
        "match_status": "MATCH" if matched else "MISMATCH_ALERT"
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
    if document_image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Document must be JPG, PNG, or WebP.")
    if live_photo and live_photo.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Live photo must be JPG, PNG, or WebP.")

    doc_bytes = await document_image.read(MAX_IMAGE_BYTES + 1)
    live_bytes = await live_photo.read(MAX_IMAGE_BYTES + 1) if live_photo else None

    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Document image is empty.")
    if len(doc_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Document image must be 10 MB or smaller.")
    if live_bytes and len(live_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Live photo must be 10 MB or smaller.")

    try:
        Image.open(io.BytesIO(doc_bytes)).verify()
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="Document image is invalid or unreadable. Upload a JPG, PNG, or JPEG file.",
        ) from error

    if live_bytes:
        try:
            Image.open(io.BytesIO(live_bytes)).verify()
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail="Live photo is invalid or unreadable. Upload a JPG, PNG, or JPEG file.",
            ) from error

    # 1. Tampering detection
    tamper_result = analyze_tampering_ela(doc_bytes)

    # 2. Face verification
    face_result = extract_and_verify_faces(doc_bytes, live_bytes)

    # 3. MRZ validation: use submitted lines, otherwise read the image automatically.
    mrz_result: dict[str, object]
    if mrz_line1 and mrz_line2:
        mrz_data = parse_td3_mrz(mrz_line1, mrz_line2)
        mrz_result = {
            "detected": "error" not in mrz_data,
            "source": "form",
            "line1": mrz_line1,
            "line2": mrz_line2,
            "data": mrz_data,
        }
    else:
        mrz_result = extract_mrz_from_image(doc_bytes)

    # 4. Composite Risk Score (0 - 100)
    risk_score = 0
    risk_factors: list[str] = []

    if tamper_result["is_tampered"]:
        risk_score += 40
        risk_factors.append("Digital manipulation / splice detected via ELA")

    if face_result.get("match_status") == "MISMATCH_ALERT":
        risk_score += 35
        risk_factors.append("Live individual does not match document photo")

    mrz_data = mrz_result.get("data")
    checks = mrz_data.get("checks") if isinstance(mrz_data, dict) else None
    if isinstance(checks, dict):
        for check, valid in checks.items():
            if not valid:
                risk_score += 20
                risk_factors.append(f"Checksum failure: {check}")
    if isinstance(mrz_data, dict) and mrz_data.get("is_expired"):
        risk_score += 25
        risk_factors.append("Travel document is expired")

    risk_score = min(100, risk_score)

    if risk_score >= 65:
        decision = "DENIED_FLAGGED_FORGERY"
        status_color = "RED"
    elif risk_score >= 35:
        decision = "SECONDARY_INSPECTION_REQUIRED"
        status_color = "YELLOW"
    else:
        decision = "CLEARED"
        status_color = "GREEN"

    response: dict[str, object] = {
        "status": "Completed",
        "risk_assessment": {
            "score": risk_score,
            "status": status_color,
            "decision": decision,
            "factors": risk_factors
        },
        "modules": {
            "tampering_analysis": tamper_result,
            "face_verification": face_result,
            "mrz_validation": mrz_result
        }
    }
    return response

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Document Screening Engine"}
