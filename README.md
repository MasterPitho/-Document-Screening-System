# AI-Based Fake Identity & Document Screening System

---

## Architecture Overview
This system is an automated border document verification and risk-scoring platform comprising 4 core modules:
1. **Module 1: OCR Extraction & Parser** (TD3 Passport, MRZ, and Visa field extraction).
2. **Module 2: Document Rule Validation** (ICAO Doc 9303 checksum checks: 7-3-1 weight pattern, date validity).
3. **Module 3: Tampering Detection (AI Core)** (Error Level Analysis (ELA) for digital splicing, photo swap, and text modification).
4. **Module 4: Face Verification** (Document facial portrait crop extraction and 1:1 similarity matching against live checkpoint traveler capture).

---

## Quickstart Guide

### 1. Running with Docker Compose (Recommended)
```bash
docker compose up --build
```
The API will be available at: `http://localhost:8000`
Interactive Swagger Documentation: `http://localhost:8000/docs`

---

### 2. Running Locally (Without Docker)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Automatic MRZ OCR requires the Tesseract executable. Docker installs it automatically. On Windows, install Tesseract separately and ensure `tesseract.exe` is on `PATH`.

---

## API Documentation

### `POST /api/v1/screen`
Uploads a document image and optional live photo to perform tamper check, checksum validation, and face matching.

**Form-Data Parameters:**
- `document_image` *(File, required)*: Scanned passport or identity card.
- `live_photo` *(File, optional)*: Live camera snapshot of the traveler.
- `mrz_line1` *(Text, optional)*: Line 1 of the passport MRZ zone (44 chars).
- `mrz_line2` *(Text, optional)*: Line 2 of the passport MRZ zone (44 chars).

If both MRZ fields are omitted, the API automatically crops the lower part of the document, runs OCR, and validates the detected TD3 lines. Images must be JPG, PNG, or WebP and are limited to 10 MB per upload.

**Response Schema:**
```json
{
  "status": "Completed",
  "risk_assessment": {
    "score": 15,
    "status": "GREEN",
    "decision": "CLEARED",
    "factors": []
  },
  "modules": {
    "tampering_analysis": {
      "ela_mean_intensity": 14.2,
      "ela_std_dev": 8.1,
      "edge_artifact_score": 3.2,
      "metadata_present": true,
      "is_tampered": false,
      "confidence": 28.4
    },
    "face_verification": {
      "face_detected_in_document": true,
      "face_detected_in_live": true,
      "similarity_score": 0.882,
      "match_status": "MATCH"
    },
    "mrz_validation": {
      "detected": true,
      "source": "form",
      "line1": "P<IND...",
      "line2": "M123...",
      "data": {
      "doc_type": "P",
      "issuing_country": "IND",
      "full_name": "KUMAR RAHUL",
      "passport_number": "M1234567",
      "is_expired": false,
      "checks": {
        "passport_number_valid": true,
        "dob_valid": true,
        "expiry_valid": true,
        "composite_valid": true
      }
      }
    }
  }
}
```
