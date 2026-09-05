# Document Screening Engine

An explainable **Smart India Hackathon prototype** for document screening. It validates TD3 passport MRZ data, runs image-forensics signals, exposes a prototype face-verification indicator, and computes a transparent heuristic risk score. It is **not** a guarantee of document authenticity and **not** an autonomous identity decision; a trained human officer remains the final decision maker.

## Current Scope

- TD3 passport MRZ: two 44-character lines, ICAO 9303 7-3-1 checks for document number, date of birth, expiry, and composite data, plus impossible-date and pivot-year expiry validation.
- Automatic MRZ OCR via Tesseract when the caller omits the MRZ form fields.
- Image-forensic signals (ELA, edge artifacts, metadata presence). These are heuristic indicators, not a forgery classifier.
- Prototype face detection (Haar) and normalized image-vector cosine similarity. This is not ArcFace/FaceNet and is not production biometric recognition.
- Explainable risk score: weights are heuristic prototype values, not calibrated probabilities.

## Processing Pipeline

```text
Upload validation (size, MIME, signature, dimensions)
    -> MRZ form input, or Tesseract OCR candidate pipeline
    -> TD3 structure + ICAO checksum + date validation
    -> Tamper signals: ELA + edge artifacts + metadata presence
    -> Face detection + prototype vector cosine similarity
    -> Fail-safe risk engine and human-review decision
```

## Run With Docker

Docker installs the Tesseract executable and runtime dependencies:

```bash
docker compose up --build
```

Open:

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

The container accepts only JPG, PNG, and WebP uploads with a 10 MB per-image limit by default. The Compose service loads optional values from a local `.env` file (see `.env.example`). Docker build verification requires a running Docker engine. The service runs as a non-root user.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Automatic MRZ OCR also requires the **Tesseract executable** (not just the `pytesseract` package). On Windows, install Tesseract and put `tesseract.exe` on `PATH`; the Docker image already installs `tesseract-ocr`.

## Configuration

Configuration is environment-driven; copy `.env.example` to `.env` and adjust as needed. Legacy variable names remain supported.

| Variable | Default | Notes |
| --- | --- | --- |
| `API_ENV` | `development` | Reported by `/health`; no behavioral difference. |
| `LOG_LEVEL` | `INFO` | |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Legacy alias: `ALLOWED_ORIGINS`. |
| `MAX_FILE_SIZE_MB` | `10` | Legacy alias for `MAX_IMAGE_BYTES`. |
| `MAX_IMAGE_BYTES` | `10485760` | Takes precedence over `MAX_FILE_SIZE_MB`. |
| `MAX_IMAGE_PIXELS` | `40000000` | |
| `MAX_IMAGE_WIDTH` / `MAX_IMAGE_HEIGHT` | `10000` / `10000` | |
| `ALLOWED_IMAGE_TYPES` | `image/jpeg,image/png,image/webp` | Declared MIME whitelist. |
| `ALLOWED_IMAGE_EXTENSIONS` | `jpg,jpeg,png,webp` | File-name extension whitelist. |
| `MRZ_CONFIDENCE_THRESHOLD` | `0.70` | |
| `FACE_MATCH_THRESHOLD` | `0.60` | Legacy alias: `FACE_SIMILARITY_THRESHOLD`. |
| `TAMPERING_THRESHOLD` | `60` | |
| `MRZ_YEAR_PIVOT` | `70` | Years below the pivot decode as 20xx, at/above as 19xx. |
| `RISK_REVIEW_THRESHOLD` | `35` | Legacy alias: `RISK_MEDIUM_THRESHOLD`. |
| `RISK_REJECT_THRESHOLD` | `65` | Legacy alias: `RISK_HIGH_THRESHOLD`. |
| `RISK_TAMPERING` | `40` | Weight for tampering-suspected signals. |
| `RISK_TAMPERING_INCONCLUSIVE` | `15` | Weight for inconclusive tamper signals. |
| `RISK_FACE_MISMATCH` | `35` | |
| `RISK_FACE_NOT_DETECTED` | `20` | |
| `RISK_MRZ_CHECKSUM` | `20` | |
| `RISK_EXPIRED` | `25` | |
| `RISK_MRZ_NOT_DETECTED` | `20` | |
| `RISK_UNKNOWN_MODULE` | `15` | Applied when face verification is skipped (no live photo). |

Risk weights are intentionally heuristic prototype values and should be calibrated against labeled data before any operational use.

## API

### `GET /health`

Returns `{"status": "ok", "service": "Document Screening Engine", "env": "<API_ENV>"}`. No sensitive information is exposed.

### `POST /api/v1/screen`

Multipart form fields:

- `document_image`: required JPG, PNG, or WebP document image.
- `live_photo`: optional JPG, PNG, or WebP live image.
- `mrz_line1`: optional exact TD3 line 1.
- `mrz_line2`: optional exact TD3 line 2.

Supplying only one of `mrz_line1`/`mrz_line2` returns HTTP 400; the API does not silently fall back to OCR. Supplying neither falls back to OCR.

```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -F "document_image=@passport.jpg" \
  -F "live_photo=@face.jpg"
```

A successful MRZ detection (`detected: true`, `status: VALID`) requires two exact 44-character lines passing allowed-character validation, TD3 structure validation, every ICAO 9303 check digit, and date validation. Candidates with wrong lengths are rejected; checksum-invalid or invalid-date candidates are never reported as detections.

## Response Semantics

A successful screening returns HTTP 200 with a heuristic, human-review verdict:

```json
{
  "status": "SCREENED",
  "request_id": "b7f4d1c7b6f54f9e9c2be6ccf7f1e2a1",
  "risk_assessment": {
    "score": 20,
    "status": "YELLOW",
    "level": "LOW_RISK",
    "decision": "SECONDARY_INSPECTION_REQUIRED",
    "factors": [{"factor": "MRZ_NOT_DETECTED", "weight": 20,
                 "detail": "No structurally valid TD3 MRZ was detected."}],
    "module_statuses": {"mrz": "NOT_AVAILABLE", "face": "NOT_AVAILABLE", "tampering": "PASS"},
    "explanation": "Heuristic screening signals require human review... (incl. score formula)"
  },
  "mrz": {"detected": false, "source": "ocr", "status": "NOT_DETECTED", "confidence": 0.0},
  "tampering_analysis": {"status": "CLEAN", "signals": [], "indicators": [], "confidence": 12.4},
  "face_verification": {"status": "SKIPPED_NO_LIVE_PHOTO", "similarity_score": null}
}
```

Decisions: `CLEARED` (all modules `PASS`, no risk factors), `SECONDARY_INSPECTION_REQUIRED` (any module below `PASS` or any risk factor), or `HIGH_RISK_REVIEW_REQUIRED` (score at/above the reject threshold). Fail-safe gate: none of `FAIL`, `ERROR`, or `NOT_AVAILABLE` module states can produce `CLEARED`.

Key states:

- MRZ: `VALID`, `INVALID`, `MALFORMED`, `NOT_DETECTED`, `OCR_FAILED`, `OCR_LOW_CONFIDENCE`.
- Tamper: `CLEAN`, `SUSPICIOUS`, `INCONCLUSIVE`, `ANALYSIS_ERROR` (each with `signals` and `indicators`).
- Face: `MATCH`, `MISMATCH`, `NO_FACE_IN_DOCUMENT`, `NO_FACE_IN_LIVE_PHOTO`, `MULTIPLE_FACES`, `LOW_CONFIDENCE`, `INVALID_IMAGE`, `SKIPPED_NO_LIVE_PHOTO`, `ERROR`.
- Module state: `PASS`, `FAIL`, `NOT_AVAILABLE`, `ERROR`.

## Error Format

All non-2xx responses share a structured shape and never expose stack traces, filesystem paths, or raw exception text:

```json
{
  "success": false,
  "error": {
    "code": "UNSUPPORTED_MEDIA_TYPE",
    "message": "Document image must be JPG, PNG, or WebP.",
    "detail": "Document image must be JPG, PNG, or WebP."
  }
}
```

Error codes: `BAD_REQUEST` (400), `NOT_FOUND` (404), `FILE_TOO_LARGE` (413), `UNSUPPORTED_MEDIA_TYPE` (415), `VALIDATION_ERROR` (422), `INTERNAL_ERROR` (500).

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite (43 tests) covers valid/malformed MRZ, all checksum failures, pivot-year expiry semantics, OCR failure and candidate rejection, tamper statuses including inconclusive handling, structured error bodies, upload hardening (empty files, wrong extensions, MIME spoofing, oversized images), the fail-safe decision gate, and deterministic face bounding-box reporting.

## Security and Privacy

- Uploaded images are validated by byte size, declared MIME type, actual image signature/format, and pixel dimensions, then processed in memory; the API does not persist uploads.
- `live_photo` and `document_image` files are always closed after reading.
- No passport numbers, names, raw images, or biometric vectors are logged.
- CORS origins are configurable; no wildcard-plus-credentials combination is used.
- The Docker process runs as a non-root user.

## Limitations

Not production-grade identity verification:

- TD3 OCR can fail on low-quality or unusual scans.
- Face verification uses a lightweight normalized image vector, not a trained embedding model.
- Tamper scoring is a heuristic (ELA + edge) signal, not a trained forgery classifier.
- Risk weights are not probabilities.
- The MRZ pivot-year rule is a heuristic for two-digit years; it is not an authoritative issuance-date source.
- Additional document formats, calibrated datasets, a reviewed face model, liveness checks, and human-reviewed evaluation are required before deployment.