# Document Screening Engine

An explainable **Smart India Hackathon prototype** for document screening. It validates machine-readable identity documents, runs a multi-signal image-forensics pipeline, verifies faces with **InsightFace ArcFace embeddings** (ONNX Runtime), adds **passive presentation-attack (liveness) screening**, and computes a transparent, deterministic risk score. It is **not** a guarantee of document authenticity and **not** an autonomous identity decision; a trained human officer remains the final decision maker.

## 1. Problem Statement

### 1a. Operational Pain Points

- **High-throughput inspection bottlenecks.** At border checkpoints and enrolment desks, officers physically inspect every document. Manual MRZ transcription, glance-level liveness judgement, and page-by-page tamper checks do not scale to peak-hour queues and create long dwell times.
- **Operator cognitive fatigue.** Continuous visual scrutiny degrades accuracy over a shift. Fatigue-driven misses (a wrong face, an undetected micro-tamper, an expired document) are silent, correlated failures — exactly the kind a machine should catch first.
- **Undetected micro-tampering.** High-resolution edits, photo-substitution, JPEG recompression artifacts, and print/scan recapture survive casual inspection. These leave forensic traces (Error-Level-Analysis residuals, compression/noise anomalies, moiré patterns, blur) that are invisible to the naked eye but measurable.
- **Opaque "black-box" decision models.** A model that returns a bare score without reasons cannot be audited, challenged by an officer, or defended in a border-security context. In a state handling identity decisions, an unexplainable verdict is operationally and legally unusable.

### 1b. Technical Challenge

Screen a physical identity document through **multi-modal triage** — a machine-readable zone (MRZ), digital image forensics, and biometric face matching — and fuse the signals into one explainable decision, **without persisting sensitive PII or raw biometrics**:

1. **Physical-document channel:** parse ICAO 9303 MRZ (TD3 passports today), validate structure, checksums, and dates; OCR-fallback when the zone is not typed in.
2. **Heuristic digital-forensics channel:** ELA, compression, noise, edge, copy-move, and metadata signals that localize *suspected* regions (never a forgery classifier).
3. **Biometric channel:** ArcFace face verification of the document photograph against an optional live capture, plus **passive liveness** (texture/FFT/colour forensics, ONNX anti-spoofing when a model is present) to screen for printouts and screen recaptures.
4. **Privacy constraint:** all analysis executes strictly in memory. Raw images, embeddings, MRZ text, passport numbers and names are never stored or logged.

### 1c. System Objectives

- **Assisted human-in-the-loop routing.** The engine triages automatically and routes to one of three outcomes — `CLEARED`, `SECONDARY_INSPECTION_REQUIRED`, `HIGH_RISK_REVIEW_REQUIRED` — with a human officer as the final decision maker. It never decides autonomously.
- **Transparent deterministic weighting.** Every activated risk factor has an explicit, documented weight; unknown factors raise an error rather than being silently assigned a weight. The score is a bounded, reproducible sum (0–100), not a learned probability.
- **Strict in-memory execution.** No upload, crop, frame, embedding, or MRZ string touches disk (except the privacy-preserving audit *metadata* row described below). Component singletons are created at app startup, never per request.

### 1d. Multi-Document Engine Architecture

- **TD3 Passports (ICAO 9303):** Two 44-character MRZ lines, OCR candidate filtering, 7-3-1 check digits, impossible-date & pivot-year expiry validation.
- **TD1 National Identity Cards (ICAO 9303):** Three 30-character MRZ lines, check digits, and structural layout.
- **Aadhaar Cards (UIDAI):** Dedicated layout anchor detection, Verhoeff checksum algorithm on 12-digit UID numbers, masked identity representation (`XXXX-XXXX-1234`), and embedded QR payload parsing.
- **PAN Cards (Income Tax Dept of India):** Dedicated 10-character alphanumeric regex (`[A-Z]{5}[0-9]{4}[A-Z]`), 4th character entity type validation (`P`=Individual, `C`=Company, `H`=HUF, `F`=Firm, `A`=AOP, etc.), masked identity (`ABCPXXXX4F`), and anchor recognition.
- **Router Contract:** Parser is selected either by explicit `document_type` (`auto | td3 | passport | td1 | national_id | aadhaar | pan`) or, in `auto` mode, by document aspect ratio and layout anchors. Each parser implements `parse(image_bytes, settings) -> DocumentParseResult` through `BaseDocumentParser`.

### 1e. Security Boundaries

- **This system performs 1:1 facial verification** — proving the document photograph and the live capture depict the *same person*. It does **not** perform identification against a gallery.
- **Liveness screening is passive and heuristic.** It detects common presentation attacks (printed photo, screen recapture) via ONNX anti-spoofing when weights are baked in, and falls back to OpenCV texture/frequency analysis (FFT high-frequency distribution, Laplacian blur, HSV/YCrCb colour histograms, moiré periodicity). It is **not** active-challenge anti-spoofing (no "blink now" prompt), **not** a certified PAD system — and it is **not a trained anti-spoofing model** — and can be evaded by sophisticated masks/video-replay. A `SPOOF_DETECTED` verdict forces `HIGH_RISK_REVIEW_REQUIRED`; an `UNCERTAIN` verdict applies a review penalty. The heuristic layer is an **additional triage signal only** and is not sufficient by itself to establish document or identity authenticity.
- Raw biometrics, embeddings, and liveness crops exist only in memory for the duration of a request. The audit trail stores screening metadata only.

## Current Scope

- **Multi-Document Support:** TD3 passports, TD1 ID cards, Aadhaar cards (with Verhoeff checksum & masked UID), and PAN cards (with 4th-char entity validation & masked PAN).
- **Embedded QR Code Analysis:** OpenCV-based QR detection with adaptive preprocessing, payload classification (`AADHAAR_XML`, `SECURE_AADHAAR_NUMERIC`, `GENERIC_URL`, `GENERIC_TEXT`), and privacy-safe masked summaries.
- **Cross-Signal Consistency Engine:** Validates requested document type vs detected document structure, cross-checks QR payload vs OCR metadata, and detects spatial tampering overlapping identity text / MRZ regions.
- **Multi-Signal Image Forensics:** ELA, JPEG compression, noise, edge, lightweight copy-move (duplicate-region), and metadata presence.
- **Face Verification & Liveness:** InsightFace ArcFace embeddings (`buffalo_sc` model via ONNX Runtime), passive presentation-attack detection (MiniFASNet ONNX or OpenCV texture/frequency fallback).
- **Deterministic Risk Engine:** Weighted factors with explicit, documented weights. Weights are heuristic prototype values, not calibrated probabilities.
- **Zero-PII Storage:** Fully compliant database and audit schema with no raw MRZ lines, document numbers, applicant names, biometric vectors, or image persistence.

## Processing Pipeline

```text
Upload validation (byte size, MIME, extension, signature, pixel count, dimensions)
    -> QR Code Analysis (OpenCV detector + adaptive preprocessing + payload classification)
    -> Document Parser Router (aspect ratio / layout anchors / document_type)
         -> TD3 Passport: MRZ form input or Tesseract OCR candidate pipeline
         -> TD1 National ID: 3-line MRZ structure + check digits
         -> Aadhaar Card: UIDAI layout anchors + Verhoeff checksum + masked UID
         -> PAN Card: Income Tax layout anchors + 10-char entity regex + masked PAN
    -> Image Forensics: ELA + compression + noise + edges + copy-move + metadata
    -> Face Verification: InsightFace ArcFace embeddings (doc photo vs live photo)
    -> Passive Liveness (PAD): ONNX anti-spoofing or OpenCV texture/frequency fallback
    -> Cross-Signal Consistency Engine:
         - Document type requested vs detected layout
         - QR payload vs OCR identity consistency
         - Spatial tampering overlap with identity / MRZ regions
    -> Deterministic, Explainable Risk Engine (weights, module gates, liveness gate)
    -> Role-Based Access Control (RBAC: officer, supervisor, admin)
    -> PostgreSQL-backed Zero-PII Audit Record (SQLAlchemy + Alembic migrations)
```

## Architecture and Package Layout

`app/` is the FastAPI application package. `create_app()` builds the app, its configuration, and every runtime singleton, then exposes them through `request.app.state` — components are never created at module import time (which keeps the test suite DB-independent and import-safe).

- `app/main.py` — app factory and lifespan (retrying database bring-up, admin bootstrap from env vars, face-model load), plus all HTTP routes: `GET /health`, `GET /ready`, `POST /api/v1/screen`, `POST /api/v1/auth/register|login|logout`, `GET /api/v1/auth/me`, `GET /api/v1/screenings` (list + filters), `GET /api/v1/screenings/{id_or_request_id}`, `PATCH /api/v1/screenings/{item_id}/decision` (RBAC-protected officer/supervisor override), `GET /api/v1/screenings/{id_or_request_id}/factors`, `GET /api/v1/stats`, `GET /api/v1/report/summary`.
- `app/config.py` — pydantic `Settings` loaded from the environment plus the cached `get_settings()`.
- `app/db/` — persistence layer:
  - `database.py` — the `Database` engine/session factory (`build_database`), `ping()` readiness probe, retrying `DatabaseConnector`, UTC clock helpers, and the `get_db` FastAPI dependency.
  - `models.py` — SQLAlchemy ORM models (`User`, `AuthToken`, `Screening`, `ScreeningFactor`, `AuditLog`) with indexes and strict zero-PII storage.
  - `repositories.py` — the only module that opens sessions; one short-lived session per operation (`ScreeningRepository`, `AuditLogRepository`, `UserRepository`, `AuthTokenRepository`).
- `app/api/` — `auth.py` (register/login/logout/me, `require_role` RBAC dependency, bearer-token helpers, `extract_optional_user` for anonymous screenings) and `helpers.py` (shared request utilities).
- `app/services/` — 
  - `mrz.py` (strategy-based document parsers: `BaseDocumentParser`, `TD3PassportParser`, `NationalIDParser` stub, `DocumentParserRouter`, plus backward-compatible OCR)
  - `aadhaar.py` (`AadhaarDocumentParser` with UIDAI layout anchor detection, Verhoeff checksum algorithm, and masked UID enforcement)
  - `pan.py` (`PANDocumentParser` with 10-char regex, 4th character entity type validation, and masked PAN representation)
  - `qr.py` (`QRAnalyzer` for embedded document QR detection, adaptive binarization, payload classification, and masked summaries)
  - `cross_signal.py` (`CrossSignalEvaluator` fusing cross-modal consistency between requested type, QR payload, OCR fields, and spatial tamper regions)
  - `tampering.py` (multi-signal digital image forensics: ELA, compression, noise, edge, copy-move, metadata)
  - `face_recognition.py` (ArcFace engine behind a swappable `FaceBackend` protocol, with an injectable `DummyBackend` for tests)
  - `liveness.py` (passive ONNX + OpenCV presentation-attack screening)
  - `risk_engine.py` (weighted deterministic scoring with cross-signal consistency factors).
- `app/security/image_validation.py` — upload hardening (byte size, MIME, extension, signature, pixel count, dimensions, decompression bomb safety).
- `app/models/schemas.py` — Pydantic request/response models.
- `database.py` (repo root) — a backward-compatible shim re-exporting the `app.db` models plus `engine`/`SessionLocal`/`get_db`/`init_db` so legacy tooling imports keep working.

`alembic/` holds the migration environment; the initial revision `eb993e3a880b` creates the schema and is applied automatically on container start via `alembic upgrade head`.

## Run With Docker

Docker installs the Tesseract executable, ONNX/runtime dependencies, bakes the face model into the image during build, and runs a PostgreSQL 16 service:

```bash
docker compose up --build
```

The `db` service is a `postgres:16-alpine` container reachable inside the network at `db:5432` and published on the host at `localhost:5432` (so `psql` and the example `DATABASE_URL` work locally); the API waits for its `pg_isready` healthcheck before starting. On container start the API runs `alembic upgrade head` (Alembic migrations) and then uvicorn. Container health is gated on `/ready`, which requires both the face model and a live database connection. Data persists in the `postgres-data` volume. TLS and hardened admin credentials are deployment-time concerns.

Open:

- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`

The container accepts only JPG, PNG, and WebP uploads with a 10 MB per-image limit by default. The Compose service loads optional values from a local `.env` file (see `.env.example`) and runs as a non-root user.

Building the image downloads the face model (several hundred MB); the download happens once at build time, not per request.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On first start, InsightFace downloads the `buffalo_sc` model into `FACE_MODELS_DIR` (`~/.insightface` by default). `/ready` reports `200` only after the model is loaded and the database answers `SELECT 1`; otherwise it stays `503` with the failing module listed, and the screening endpoint returns `NOT_AVAILABLE` for the face module instead of crashing.

Automatic MRZ OCR also requires the **Tesseract executable** (not just the `pytesseract` package). On Windows, install Tesseract and put `tesseract.exe` on `PATH`; the Docker image installs `tesseract-ocr`.

With a local PostgreSQL (or SQLite) the schema is applied with Alembic:

```powershell
alembic upgrade head
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`alembic downgrade base` reverts; `alembic check` verifies the migration matches the models. For a plain SQLite file the app can also create tables automatically at startup.

## Audit Database

Screening results are persisted to a privacy-preserving audit trail in the configured SQLAlchemy database (PostgreSQL recommended; SQLite is the local default). The DB layer lives in `app/db/`: `database.py` owns the engine/session factory, `models.py` defines the schema, and `repositories.py` is the only place that touches SQLAlchemy sessions (one short-lived session per operation, always committed or rolled back and closed).

Tables (managed by Alembic migrations in `alembic/`):

- `screenings` — request_id (unique), risk score/level, decision, module states, risk factors, processing time, MRZ/tamper/face statuses, MRZ source, and the authenticated operator who ran the scan.
- `screening_factors` — one normalized risk factor row per screening (name, severity HIGH/MEDIUM/LOW, weight, description).
- `audit_logs` — screening and lifecycle events with event type and request_id.
- `users` — operators and admins (PBKDF2 salted hashes, never plaintext).
- `auth_tokens` — bearer sessions (SHA-256 token hashes) with a configurable TTL.

**No raw images, biometric embeddings, MRZ text, passport numbers, or personal data are stored.** Images are processed entirely in memory.

Reusing an `X-Request-ID` for a second screening returns `CONFLICT` (409), and a single transaction ensures a screening is persisted atomically with its factors and audit event. If the database is unreachable, a completed analysis returns a controlled `503 DATABASE_UNAVAILABLE` instead of a misleading `CLEARED` record. An optional admin can be bootstrapped from environment variables (`ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD`).

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
| `FACE_MATCH_THRESHOLD` | `0.35` | Legacy alias: `FACE_SIMILARITY_THRESHOLD`. |
| `FACE_MIN_DETECTION_CONFIDENCE` | `0.50` | Face-detection score cutoff (InsightFace detector confidence). |
| `FACE_MODEL_NAME` | `buffalo_sc` | InsightFace model pack. |
| `FACE_DET_SIZE` | `640` | Detection input size. |
| `FACE_CTX_ID` | `-1` | `-1` = CPU. |
| `FACE_MODELS_DIR` | `~/.insightface` | InsightFace model cache (mounted as `/opt/insightface` in Docker). |
| `LIVENESS_ENABLED` | `true` | Master switch for passive liveness screening. |
| `LIVENESS_MODEL_PATH` | _(empty)_ | Path to a MiniFASNet/Silent-Face-style `.onnx` anti-spoofing model. Empty = heuristic fallback only. |
| `LIVENESS_HEURISTIC_ENABLED` | `true` | Allow the OpenCV texture/frequency fallback when no ONNX model is configured. |
| `LIVENESS_SPOOF_THRESHOLD` | `0.40` | Liveness score at/below which the capture is `SPOOF_DETECTED`. |
| `LIVENESS_UNCERTAIN_THRESHOLD` | `0.60` | Score below which (but above `LIVENESS_SPOOF_THRESHOLD`) is `UNCERTAIN`; at/above it is `LIVE`. |
| `LIVENESS_MODEL_INPUT_SIZE` | `160` | ONNX model input square size (e.g. 160 for MiniFASNet-style nets). |
| `LIVENESS_MODEL_CTX_ID` | `-1` | `-1` = CPU execution provider. |
| `TAMPERING_THRESHOLD` | `70` | Aggregate tamper score at/above which the image is `SUSPICIOUS`. |
| `TAMPERING_REVIEW_THRESHOLD` | `45` | Aggregate score at/above which the image is moved to `INCONCLUSIVE` review. |
| `MRZ_YEAR_PIVOT` | `50` | Two-digit years decode as 20xx when below the pivot, 19xx at/above it (00–49 → 2000–2049, 50–99 → 1950–1999). |
| `RISK_REVIEW_THRESHOLD` | `35` | Legacy alias: `RISK_MEDIUM_THRESHOLD`. |
| `RISK_REJECT_THRESHOLD` | `65` | Legacy alias: `RISK_HIGH_THRESHOLD`. |
| `RISK_TAMPERING` | `40` | Weight for tampering-suspected signals. |
| `RISK_TAMPERING_INCONCLUSIVE` | `15` | Weight for inconclusive tamper signals. |
| `RISK_FACE_MISMATCH` | `35` | |
| `RISK_FACE_NOT_DETECTED` | `20` | |
| `RISK_FACE_LOW_CONFIDENCE` | `15` | |
| `RISK_FACE_MULTIPLE` | `20` | |
| `RISK_MRZ_CHECKSUM` | `20` | |
| `RISK_EXPIRED` | `25` | |
| `RISK_MRZ_NOT_DETECTED` | `20` | |
| `RISK_MRZ_LOW_CONFIDENCE` | `10` | |
| `RISK_IMAGE_QUALITY` | `10` | Reserved for image-quality signals. |
| `RISK_MODULE_ERROR` | `25` | Applied when a module fails internally. |
| `RISK_UNKNOWN_MODULE` | `15` | Applied when face verification is skipped (no live photo). |
| `RISK_LIVENESS_FAILED` | `35` | Applied (and forces `HIGH_RISK_REVIEW_REQUIRED`) when liveness returns `SPOOF_DETECTED`. |
| `RISK_LIVENESS_UNCERTAIN` | `15` | Review penalty when liveness returns `UNCERTAIN`. |
| `RISK_DOCUMENT_TYPE_MISMATCH` | `30` | Applied when detected document structure contradicts requested document type. |
| `RISK_QR_OCR_CONFLICT` | `35` | Applied when QR payload identity metadata directly conflicts with OCR/MRZ data. |
| `RISK_SUSPICIOUS_FIELD_TAMPERING` | `25` | Applied when tampering hotspots spatially intersect with critical identity text/MRZ regions. |
| `DATABASE_URL` | `sqlite:///./document_screening.db` | Any SQLAlchemy URL; PostgreSQL recommended (`postgresql+psycopg://user:pass@host:5432/db`). |
| `DB_CONNECT_TIMEOUT` | `5` | Seconds allowed for a PostgreSQL connect (also used for the readiness probe). |
| `DB_CONNECT_RETRIES` | `5` | Startup re-attempts for `create_all` while PostgreSQL is still booting. |
| `DB_RETRY_DELAY` | `1.0` | Seconds between DB startup retries. |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `screening` | Used by `docker-compose.yml`; must match the credentials in `DATABASE_URL`. |
| `AUTH_TOKEN_TTL_HOURS` | `24` | Bearer token lifetime in hours. |
| `ADMIN_USERNAME` / `ADMIN_EMAIL` / `ADMIN_PASSWORD` | _(unset)_ | When all three are set, an admin account is created at startup. |

Risk weights are intentionally heuristic prototype values and should be calibrated against labeled data before any operational use. The engine rejects unknown risk factors at runtime rather than silently assigning them a weight.

## API

### `GET /health`

Returns `{"status": "ok", "service": "Document Screening Engine", "env": "<API_ENV>"}`. No sensitive information is exposed.

### `GET /ready`

Returns `200 {"status": "ready", "modules": {...}}` only when the face model is loaded **and** the database answers `SELECT 1`; otherwise `503` with `face_recognition`/`database` set to `false`. This is the deployment liveness/readiness probe.

### `POST /api/v1/screen`

Multipart form fields:

- `document_image`: required JPG, PNG, or WebP document image.
- `live_photo`: optional JPG, PNG, or WebP live image.
- `mrz_line1`: optional exact TD3 line 1 (passport MRZ form path).
- `mrz_line2`: optional exact TD3 line 2 (passport MRZ form path).
- `document_type`: optional parser selector: `auto` (default), `td3`/`passport`, `td1`/`national_id`, `aadhaar`, `pan`. When `auto`, the parser is chosen by document aspect ratio and layout anchor detection.

Supplying only one of `mrz_line1`/`mrz_line2` returns HTTP 400; the API does not silently fall back to OCR. Supplying neither falls back to OCR through the selected parser. Manually supplied lines are validated with the exact same TD3 structure, ICAO 9303 checksum, and date rules as OCR output; they are reported with `"source": "form"` (an explicit manual/testing input path). An INVALID/MALFORMED manual MRZ still forces secondary inspection. A `document_type` other than `auto|td3|passport|td1|national_id|aadhaar|pan` returns HTTP 422.

```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -F "document_image=@passport.jpg" \
  -F "live_photo=@face.jpg" \
  -H "X-Request-ID: my-correlation-id"
```

A successful MRZ detection (`detected: true`, `status: VALID`) requires two exact 44-character lines passing allowed-character validation, TD3 structure validation, every ICAO 9303 check digit, and date validation. Candidates with wrong lengths are rejected; checksum-invalid or invalid-date candidates are never reported as detections.

`X-Request-ID` is optional, sanitized to ≤64 alphanumeric/`-`/`_`/`.` characters, echoed in the response header, and written into the response body and audit record. Reusing the same `X-Request-ID` returns `409 CONFLICT` so correlators stay unambiguous.

On success the 200 body includes a `persistence` field: `{"status": "stored", "screening_id": <id>}`. If persist fails intermittently the endpoint returns a `503 DATABASE_UNAVAILABLE` error — the analysis itself is never reported as a recorded `CLEARED` result when it could not be stored, and it is never retried silently under the same `request_id`.

### `POST /api/v1/screen` — with authenticated operator

If the request carries a valid `Authorization: Bearer <token>` header, the screening is attributed to that operator in the audit trail. Without a header the scan still works and is recorded as anonymous. An invalid token is ignored for screening purposes (screening never fails because of a stale token).

### Operator accounts & RBAC

```bash
# Register (roles: "officer", "supervisor", "admin")
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "officer01", "email": "officer01@example.com", "full_name": "A. Officer", "password": "SuperSecret123!", "role": "officer"}'

# Login -> bearer token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "officer01", "password": "SuperSecret123!"}'
# -> {"token": "...", "token_type": "bearer", "user": {...}}
```

Endpoints:

- `POST /api/v1/auth/register` — create an operator account (201). Supports `role` assignment (`officer`, `supervisor`, `admin`).
- `POST /api/v1/auth/login` — returns a bearer token.
- `POST /api/v1/auth/logout` — revokes the current token.
- `GET /api/v1/auth/me` — current user profile.

### Screening history, reports & decision override (bearer token required)

- `GET /api/v1/screenings?limit=20&offset=0&decision=...&risk_level=...&date_from=...&date_to=...` — list audit records (newest first) with optional `decision`, `risk_level`, and naive-UTC `date_from`/`date_to` filters.
- `GET /api/v1/screenings/{id_or_request_id}` — single audit record; accepts either the integer row id or the 32-character `request_id`.
- `PATCH /api/v1/screenings/{item_id}/decision` — human-in-the-loop decision override (requires `officer`, `supervisor`, or `admin` role). Accepts `{"decision": "CLEARED" | "SECONDARY_INSPECTION_REQUIRED" | "HIGH_RISK_REVIEW_REQUIRED", "review_notes": "..."}`.
- `GET /api/v1/screenings/{id_or_request_id}/factors` — the normalized risk-factor rows (name, severity, weight, description) for one screening.
- `GET /api/v1/stats` — dashboard counts: totals, cleared, secondary inspection, high risk, MRZ failures, face mismatches, suspicious tampering, plus breakdowns by decision and risk level.
- `GET /api/v1/report/summary` — totals per risk level and decision, cleared/secondary/high-risk counts, and average processing time.

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/screenings?limit=10
```

All screening history and report endpoints return `401` without a valid token. Decision override returns `403 FORBIDDEN` if the user's role is not authorized.

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
    "module_statuses": {"mrz": "REVIEW", "face": "NOT_AVAILABLE", "tampering": "PASS"},
    "explanation": "..."
  },
  "mrz": {"detected": false, "source": "ocr", "status": "NOT_DETECTED", "confidence": 0.0,
          "module_state": "REVIEW"},
  "tampering_analysis": {"status": "CLEAN", "score": 0.0, "confidence": 0.0,
                         "signals": {"ela": {...}, "compression": {...}, "noise": {...},
                                     "edge": {...}, "copy_move": {...}, "metadata": {...}},
                         "suspicious_regions": [], "explanation": []},
  "face_verification": {"status": "SKIPPED_NO_LIVE_PHOTO", "similarity_score": null,
                        "matched": null, "module_state": "NOT_AVAILABLE"},
  "liveness": {"is_live": false, "liveness_score": 0.0, "liveness_status": "NOT_CHECKED",
               "method": "not_checked", "signals": {}, "explanation": "...",
               "module_state": "NOT_AVAILABLE"}
}
```

Decisions: `CLEARED` (all modules `PASS`, no risk factors), `SECONDARY_INSPECTION_REQUIRED` (any module below `PASS` or any risk factor), or `HIGH_RISK_REVIEW_REQUIRED` (score at/above the reject threshold, **or** liveness returns `SPOOF_DETECTED`). Fail-safe gate: none of `FAIL`, `ERROR`, or `NOT_AVAILABLE` module states can produce `CLEARED`.

Key states:

- Document/MRZ: `VALID`, `INVALID`, `MALFORMED`, `NOT_DETECTED`, `OCR_FAILED`, `OCR_LOW_CONFIDENCE` — plus document-level `format` (`TD3`/`TD1`/`UNKNOWN`) and `document_type` (`PASSPORT`/`NATIONAL_ID`/`AADHAAR`/`PAN`/`UNKNOWN`).
- Tamper: `CLEAN`, `SUSPICIOUS`, `INCONCLUSIVE`, `ERROR` — with per-signal `score`/`suspicious` entries for `ela`, `compression`, `noise`, `edge`, `copy_move`, `metadata`. A failed signal degrades to `0`/`not suspicious` within the aggregate score rather than crashing the analysis.
- Face: `MATCH`, `MISMATCH`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_CONFIDENCE`, `INVALID_IMAGE`, `SKIPPED_NO_LIVE_PHOTO`, `ERROR`, `NOT_AVAILABLE`.
- Liveness: `LIVE`, `SPOOF_DETECTED`, `UNCERTAIN`, `NOT_CHECKED`, `SKIPPED` — `is_live` is `true` only for `LIVE`; the `signals` object exposes per-signal heuristic scores and reasons when the OpenCV fallback ran.
- Module state: `PASS`, `FAIL`, `REVIEW`, `ERROR`, `NOT_AVAILABLE`.

## Error Format

All non-2xx responses share a structured shape and never expose stack traces, filesystem paths, or raw exception text:

```json
{
  "success": false,
  "error": {
    "code": "UNSUPPORTED_MEDIA_TYPE",
    "message": "Uploaded file is not an allowed image.",
    "detail": "Document image must be JPG, PNG, or WebP."
  }
}
```

Error codes: `BAD_REQUEST` (400), `NOT_FOUND` (404), `UNPROCESSABLE_ENTITY`/`VALIDATION_ERROR` (422), `FILE_TOO_LARGE` (413), `UNSUPPORTED_MEDIA_TYPE` (415), `CONFLICT` (409), `UNAUTHORIZED`/`FORBIDDEN` (401/403), `DATABASE_UNAVAILABLE` (503), `INTERNAL_ERROR` (500).

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The comprehensive test suite (**220 tests, 100% pass rate**) covers:
- **Aadhaar & PAN Parsing:** Verhoeff checksum validation, layout anchors, masked UID/PAN format validation (`tests/test_aadhaar_screening.py`, `tests/test_pan_screening.py`).
- **QR Code Analysis:** Embedded QR extraction, adaptive preprocessing, payload classification, and masked summaries (`tests/test_qr_analysis.py`).
- **Cross-Signal Consistency:** Document type mismatch, QR vs OCR conflict, spatial tampering overlap (`tests/test_cross_signal.py`).
- **Security & RBAC:** Role-based access control (`tests/test_rbac.py`), CORS origin lockdown (`tests/test_cors_security.py`), upload security, path traversal filename protection, decompression bomb prevention, and model lifecycle (`tests/test_security_hardening.py`).
- **Zero-PII Compliance:** Verification that database tables, schemas, repositories, and audit logs never persist raw names, document numbers, MRZ strings, face embeddings, or images (`tests/test_privacy_compliance.py`, `tests/test_persistence.py`).
- **Core Pipeline:** TD3/TD1 MRZ parsing, 7-3-1 checksums, impossible-date & leap-year handling, Tesseract OCR candidate filtering, multi-signal tamper forensics, InsightFace ArcFace verification, passive liveness PAD (ONNX & OpenCV fallback), deterministic risk scoring, Alembic migrations & schema synchronization (`tests/test_migrations.py`), and database resilience.

## Security and Privacy

- **Zero PII Persistence:** No raw applicant names, document numbers (passport, Aadhaar, PAN), MRZ lines, biometric embeddings, or images are stored in the database or written to disk. All biometric vectors and image buffers exist strictly in memory during request evaluation.
- Uploaded images are validated by byte size, declared MIME type, actual image signature/format, pixel count, and dimensions; no uploads touch disk.
- Passwords are stored as salted PBKDF2 hashes; bearer tokens are stored as SHA-256 hashes with an expiration TTL.
- CORS origins are explicitly whitelisted and production rejects wildcard (`*`) origins when credentials are enabled.
- Role-based access control gates manual decision overrides to authorized operators (`officer`, `supervisor`, `admin`).
- `live_photo` and `document_image` files are always closed after reading.
- The Docker process runs as a non-root user with an internal healthcheck.

## Production Readiness

**In place:** non-root Docker user, migration-driven schema (`alembic upgrade head` on container start), DB-aware `/ready` healthcheck that gates container health, a structured error format that never leaks stack traces, operator accounts with PBKDF2-hashed passwords and hashed bearer tokens, a privacy-preserving audit trail, upload hardening, and controlled degraded behavior (503 `DATABASE_UNAVAILABLE`, face `NOT_AVAILABLE`) instead of crashes.

**Honest gaps before operational deployment:**

- The API runs a single uvicorn worker with an in-process face-model cache. Scaling out requires a worker manager (e.g. `--workers N`/gunicorn) and shared read-only model artifacts.
- No TLS by default — the API is expected to sit behind a TLS-terminating reverse proxy.
- No per-operator rate limiting, account lockout, or multi-factor authentication.
- The base Docker image still carries two Debian-package HIGH CVEs (`zlib` CVE-2026-85091, `libxml2` CVE-2026-86140) that have no upstream fix yet; the Dockerfile's `apt-get upgrade` picks them up automatically on the next rebuild once Debian ships patches.
- Face-match thresholds and risk weights are heuristic prototype values; they must be calibrated against labeled data before any enforcement decision.

## Limitations

Not production-grade identity verification:

- TD3 OCR can fail on low-quality or unusual scans. TD1 / national-ID parsing is a structural stub (QR payload decoding and layout-parsing depth are roadmap items, not production claims).
- Face verification scores are not calibrated probabilities; the embedding model is a research checkpoint.
- Liveness screening is **passive and heuristic**: it detects printed-photo and screen-recapture presentations in common cases but is not certified PAD and can be evaded by sophisticated masks or video replay.
- Tamper scoring is a heuristic (ELA + compression + noise + edge + copy-move + metadata) signal set, not a trained forgery classifier.
- Risk weights are not probabilities and the risk engine is deterministic, not learned.
- The MRZ pivot-year rule is a heuristic for two-digit years; it is not an authoritative issuance-date source.
- Additional document formats, calibrated datasets, certified liveness, and human-reviewed evaluation are required before deployment.