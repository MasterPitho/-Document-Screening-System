# AI-Based Fake Identity and Document Screening System

This repository is an explainable Smart India Hackathon prototype for **document screening**, not a document-authenticity guarantee. It combines image validation, TD3 passport MRZ checks, optional OCR, prototype face comparison, image-forensics signals, and a heuristic risk engine. A trained human officer remains the final decision maker.

## Current Scope

- TD3 passport MRZ: two 44-character lines.
- ICAO 7-3-1 checks for document number, date of birth, expiry, and composite data.
- Expiry and impossible-date validation.
- Automatic MRZ OCR through Tesseract when form MRZ fields are omitted.
- Haar face detection plus normalized image-vector cosine similarity. This is a prototype similarity mechanism, not ArcFace, FaceNet, or production biometric recognition.
- ELA, edge-artifact, and metadata signals. ELA alone does not prove tampering.
- Explainable heuristic risk weights. They are not calibrated probabilities.

## Processing Pipeline

```text
Upload validation
    -> image decoding and dimension checks
    -> MRZ form input or OCR candidate pipeline
    -> TD3 structure and ICAO checksum validation
    -> tamper signals: ELA plus edge artifacts and metadata presence
    -> face detection, preprocessing, prototype vector, cosine similarity
    -> explainable risk score and human-review decision
```

## Run With Docker

Docker installs the Tesseract executable and all Python dependencies:

```bash
docker compose up --build
```

Open:

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

The container accepts only JPG, PNG, and WebP uploads, with a 10 MB limit per image. Docker build verification requires Docker Desktop or another Docker engine to be installed and running.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Automatic MRZ OCR also requires the **Tesseract executable**, not only the `pytesseract` Python package. Install Tesseract on Windows and put `tesseract.exe` on `PATH`. Docker already installs `tesseract-ocr`.

## API

### `GET /health`

Returns service health.

### `POST /api/v1/screen`

Multipart form fields:

- `document_image`: required JPG, PNG, or WebP document image.
- `live_photo`: optional JPG, PNG, or WebP live image.
- `mrz_line1`: optional exact TD3 line 1. If both MRZ fields are supplied, they are validated directly.
- `mrz_line2`: optional exact TD3 line 2.

Example:

```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -F "document_image=@passport.jpg" \
  -F "live_photo=@face.jpg"
```

If MRZ fields are omitted, OCR returns `NOT_DETECTED` unless it finds a structurally valid pair. The implementation never pads short OCR output into a fake MRZ.

## Response Semantics

The response contains the recommended top-level fields and a `modules` compatibility block:

```json
{
  "status": "SCREENED",
  "risk_assessment": {
    "score": 20,
    "status": "YELLOW",
    "level": "LOW_RISK",
    "decision": "SECONDARY_INSPECTION_REQUIRED",
    "factors": [
      {
        "factor": "MRZ_NOT_DETECTED",
        "weight": 20,
        "detail": "No structurally valid TD3 MRZ was detected."
      }
    ],
    "explanation": "Heuristic screening signals require human review; they do not prove authenticity or forgery."
  },
  "mrz": {
    "detected": false,
    "source": "ocr",
    "status": "NOT_DETECTED",
    "confidence": 0.0,
    "reason": "Unable to extract two exact-length TD3 MRZ lines."
  },
  "tampering_analysis": {
    "status": "NO_SIGNIFICANT_ANOMALY",
    "confidence": 12.4,
    "signals": []
  },
  "face_verification": {
    "status": "SKIPPED_NO_LIVE_PHOTO",
    "similarity_score": null
  }
}
```

Important states include `VALID`, `INVALID`, `MALFORMED`, `NOT_DETECTED`, `MATCH`, `MISMATCH`, `NO_FACE_IN_DOCUMENT`, `NO_FACE_IN_LIVE_PHOTO`, `MULTIPLE_FACES`, `INVALID_IMAGE`, and `SKIPPED_NO_LIVE_PHOTO`.

An MRZ checksum means the MRZ data is internally consistent. It does not prove that a passport is authentic. A skipped face check is not a match. An ELA or edge anomaly is a screening signal, not proof of forgery.

## Configuration

Configuration is environment-driven:

- `ALLOWED_ORIGINS`: comma-separated frontend origins. Default: `http://localhost:3000,http://localhost:5173`.
- `MAX_FILE_SIZE_MB`: default `10`.
- `MAX_IMAGE_PIXELS`: default `40000000`.
- `MRZ_CONFIDENCE_THRESHOLD`: default `0.70`.
- `FACE_SIMILARITY_THRESHOLD`: default `0.60`.
- `TAMPERING_THRESHOLD`: default `60`.
- `RISK_MEDIUM_THRESHOLD`: default `35`.
- `RISK_HIGH_THRESHOLD`: default `65`.

Risk weights are intentionally heuristic prototype values. They should be calibrated against labeled data before operational use.

## Testing

Run:

```bash
pytest -q
```

The tests cover valid and malformed TD3 MRZ input, individual and composite checksum failures, OCR failure and wrong-length candidates, tamper output shape, unsupported API uploads, and the screened response schema.

## Security and Privacy

- Uploaded images are processed in memory and are not persisted by the API.
- Declared MIME type, actual image format, byte size, and pixel dimensions are checked.
- No passport numbers, names, raw images, or biometric vectors are logged.
- CORS origins are configurable and are not wildcard-plus-credentials.
- The Docker process runs as a non-root user.

## Limitations and Future Work

This is not production-grade identity verification. TD3 OCR can fail on low-quality or unusual scans. The face module uses a lightweight normalized image vector rather than a trained embedding model. Tamper scoring is not a trained forgery classifier. The risk weights are not probabilities. Additional formats, calibrated datasets, a reviewed face model, liveness checks, stronger audit controls, and human-reviewed evaluation are required before deployment.
