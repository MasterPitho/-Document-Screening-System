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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
            "expiry_valid": valid_expiry
        }
    }

# -------------------------------------------------------------
# Module 3: Tampering Detection (Error Level Analysis & Splice)
# -------------------------------------------------------------
def analyze_tampering_ela(image_bytes: bytes, quality: int = 90) -> dict[str, float | bool]:
    original = Image.open(io.BytesIO(image_bytes)).convert('RGB')

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

    # High average ELA intensity and standard deviation indicate splice/alteration
    tamper_detected = mean_diff > 35.0 or std_diff > 45.0

    return {
        "ela_mean_intensity": round(mean_diff, 2),
        "ela_std_dev": round(std_diff, 2),
        "is_tampered": tamper_detected,
        "confidence": min(100.0, round((mean_diff / 50.0) * 100, 1))
    }

# -------------------------------------------------------------
# Module 4: Face Detection & Match (OpenCV Haar / Histogram)
# -------------------------------------------------------------
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
            "similarity_score": 1.0
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

    # Fast correlation similarity check on normalized face crops
    hist_doc = cv2.calcHist([doc_face_crop], [0], None, [64], [0, 256])
    hist_live = cv2.calcHist([live_face_crop], [0], None, [64], [0, 256])
    cv2.normalize(hist_doc, hist_doc, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_live, hist_live, 0, 1, cv2.NORM_MINMAX)

    sim_score = float(cv2.compareHist(hist_doc, hist_live, cv2.HISTCMP_CORREL))
    sim_score = max(0.0, min(1.0, sim_score))

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
@app.post("/api/v1/screen")
async def screen_document(
    document_image: UploadFile = File(...),
    live_photo: Optional[UploadFile] = File(None),
    mrz_line1: Optional[str] = Form(None),
    mrz_line2: Optional[str] = Form(None)
) -> dict[str, object]:
    doc_bytes = await document_image.read()
    live_bytes = await live_photo.read() if live_photo else None

    if not doc_bytes:
        raise HTTPException(status_code=400, detail="Document image is empty.")

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

    # 3. MRZ validation (if passed or simulated)
    mrz_result: Optional[dict[str, Any]] = None
    if mrz_line1 and mrz_line2:
        mrz_result = parse_td3_mrz(mrz_line1, mrz_line2)

    # 4. Composite Risk Score (0 - 100)
    risk_score = 0
    risk_factors: list[str] = []

    if tamper_result["is_tampered"]:
        risk_score += 40
        risk_factors.append("Digital manipulation / splice detected via ELA")

    if face_result.get("match_status") == "MISMATCH_ALERT":
        risk_score += 35
        risk_factors.append("Live individual does not match document photo")

    if isinstance(mrz_result, dict):
        checks = mrz_result.get("checks")
        if isinstance(checks, dict):
            for check, valid in checks.items():
                if not valid:
                    risk_score += 20
                    risk_factors.append(f"Checksum failure: {check}")
        if mrz_result.get("is_expired"):
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
