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

### 1d. Multi-Document Expansion Roadmap

- **Phase 1 (current):** ICAO 9303 **TD3 passports** — two 44-character MRZ lines, OCR candidate filtering, 7-3-1 checksums, pivot-year date semantics. Implemented behind a strategy interface (see `app/services/mrz.py`).
- **Phase 2 (blueprint):** `NationalIDParser` stub with **TD1 (three 30-character lines)** structural validation and a **QR payload extraction placeholder** for Aadhaar-style secure QR codes — architected now, validation depth to be completed.
- **Phase 3 (blueprint):** **Aadhaar QR/XML** parsing (secure QR envelope → decrypted XML fields) and **PAN OCR layout parsing** (fixed 128×128 card line layout), both fitted into the same `DocumentParserRouter`.
- **Router contract:** the parser is selected either by an explicit `document_type` request parameter (`auto | td3 | passport | td1 | national_id | aadhaar`) or, in `auto` mode, by the document image aspect ratio (a passport data page is portrait, ratio `< 1.45`; an ID-1 card is landscape, ratio `>= 1.45`). Each parser implements `parse(image_bytes, settings) -> DocumentParseResult` through the same `BaseDocumentParser` interface, so the pipeline never changes when a new document type is added.

### 1e. Security Boundaries

- **This system performs 1:1 facial verification** — proving the document photograph and the live capture depict the *same person*. It does **not** perform identification against a gallery.
- **Liveness screening is passive and heuristic.** It detects common presentation attacks (printed photo, screen recapture) via ONNX anti-spoofing when weights are baked in, and falls back to OpenCV texture/frequency analysis (FFT high-frequency distribution, Laplacian blur, HSV/YCrCb colour histograms, moiré periodicity). It is **not** active-challenge anti-spoofing (no "blink now" prompt), **not** a certified PAD system, and can be evaded by sophisticated masks/video-replay. A `SPOOF_DETECTED` verdict forces `HIGH_RISK_REVIEW_REQUIRED`; an `UNCERTAIN` verdict applies a review penalty. This is a **triage aid, not a security guarantee**.
- Raw biometrics, embeddings, and liveness crops exist only in memory for the duration of a request. The audit trail stores screening metadata only.

## Current Scope

- TD3 passport MRZ: two 44-character lines, ICAO 9303 7-3-1 checks for document number, date of birth, expiry, and composite data, plus impossible-date and pivot-year expiry validation.
- Automatic MRZ OCR via Tesseract when the caller omits the MRZ form fields.
- Multi-signal image-forensics analysis: ELA, JPEG compression, noise, edge, lightweight copy-move (duplicate-region), and metadata presence. These are heuristic indicators, not a forgery classifier.
- Face verification with InsightFace ArcFace embeddings (`buffalo_sc` model via ONNX Runtime). Detection is SCRFD (InsightFace `FaceAnalysis`), faces are aligned, and the L2-normalized ArcFace embeddings of the document face and an optional live photo are compared by cosine similarity; **exactly one** face per image is accepted (extra/missing/low-confidence faces are reported, never guessed). The model is downloaded once at startup (or baked into the Docker image).
- Passive liveness screening (`app/services/liveness.py`): MiniFASNet-style ONNX anti-spoofing wrapper when `LIVENESS_MODEL_PATH` is configured, with an OpenCV heuristic fallback (FFT high-frequency power, Laplacian blur, colour-space histograms, moiré detection). Output is `LIVE` / `SPOOF_DETECTED` / `UNCERTAIN` / `NOT_CHECKED`, fully in memory.
- Explainable risk score: weighted factors with explicit, documented weights. Weights are heuristic prototype values, not calibrated probabilities.

## Processing Pipeline

```text
Upload validation (byte size, MIME, extension, signature, pixel count, dimensions)
    -> Document parser router (aspect ratio / document_type)
         -> TD3 passport: MRZ form input, or Tesseract OCR candidate pipeline
         -> National ID (stub): TD1 3-line structure + QR payload placeholder
    -> Tamper signals: ELA + compression + noise + edges + copy-move + metadata
    -> Face verification: InsightFace ArcFace embeddings (doc vs live photo)
    -> Passive liveness (PAD): ONNX anti-spoofing or OpenCV texture/frequency fallback
    -> Deterministic, explainable risk engine (weights, module gates, liveness gate)
    -> PostgreSQL-backed privacy-preserving audit record (SQLAlchemy + Alembic)
```

## Architecture and Package Layout

`app/` is the FastAPI application package. `create_app()` builds the app, its configuration, and every runtime singleton, then exposes them through `request.app.state` — components are never created at module import time (which keeps the test suite DB-independent and import-safe).

- `app/main.py` — app factory and lifespan (retrying database bring-up, admin bootstrap from env vars, face-model load), plus all HTTP routes: `GET /health`, `GET /ready`, `POST /api/v1/screen`, `POST /api/v1/auth/register|login|logout`, `GET /api/v1/auth/me`, `GET /api/v1/screenings` (list + filters), `GET /api/v1/screenings/{id_or_request_id}`, `GET /api/v1/screenings/{id_or_request_id}/factors`, `GET /api/v1/stats`, `GET /api/v1/report/summary`.
- `app/config.py` — pydantic `Settings` loaded from the environment plus the cached `get_settings()`.
- `app/db/` — persistence layer:
  - `database.py` — the `Database` engine/session factory (`build_database`), `ping()` readiness probe, retrying `DatabaseConnector`, UTC clock helpers, and the `get_db` FastAPI dependency.
  - `models.py` — SQLAlchemy ORM models (`User`, `AuthToken`, `Screening`, `ScreeningFactor`, `AuditLog`) with indexes.
  - `repositories.py` — the only module that opens sessions; one short-lived session per operation (`ScreeningRepository`, `AuditLogRepository`, `UserRepository`, `AuthTokenRepository`).
- `app/api/` — `auth.py` (register/login/logout/me, bearer-token helpers, `extract_optional_user` for anonymous screenings) and `helpers.py` (shared request utilities).
- `app/services/` — `mrz.py` (strategy-based document parsers: `BaseDocumentParser`, `TD3PassportParser`, `NationalIDParser` stub, `DocumentParserRouter`, plus backward-compatible OCR), `tampering.py` (multi-signal forensics), `face_recognition.py` (ArcFace engine behind a swappable `FaceBackend` protocol, with an injectable `DummyBackend` for tests), `liveness.py` (passive ONNX + OpenCV presentation-attack screening), `risk_engine.py` (weighted deterministic scoring).
- `app/security/image_validation.py` — upload hardening (byte size, MIME, extension, signature, pixel count, dimensions).
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
- `document_type`: optional parser selector: `auto` (default), `td3`/`passport`, `td1`/`national_id`/`aadhaar`. When `auto`, the parser is chosen by the document image aspect ratio (portrait passport data page vs. landscape ID-1 card).

Supplying only one of `mrz_line1`/`mrz_line2` returns HTTP 400; the API does not silently fall back to OCR. Supplying neither falls back to OCR through the selected parser. Manually supplied lines are validated with the exact same TD3 structure, ICAO 9303 checksum, and date rules as OCR output; they are reported with `"source": "form"` (an explicit manual/testing input path). An INVALID/MALFORMED manual MRZ still forces secondary inspection. A `document_type` other than `auto|td3|passport|td1|national_id|aadhaar` returns HTTP 422.

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

### Operator accounts

```bash
# Register (roles are always "officer"; admins come from ADMIN_USERNAME bootstrap)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "officer01", "email": "officer01@example.com", "full_name": "A. Officer", "password": "SuperSecret123!"}'

# Login -> bearer token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "officer01", "password": "SuperSecret123!"}'
# -> {"token": "...", "token_type": "bearer", "user": {...}}
```

Endpoints:

- `POST /api/v1/auth/register` — create an officer account (201).
- `POST /api/v1/auth/login` — returns a bearer token.
- `POST /api/v1/auth/logout` — revokes the current token.
- `GET /api/v1/auth/me` — current user profile.

### Screening history & reports (bearer token required)

- `GET /api/v1/screenings?limit=20&offset=0&decision=...&risk_level=...&date_from=...&date_to=...` — list audit records (newest first) with optional `decision`, `risk_level`, and naive-UTC `date_from`/`date_to` filters.
- `GET /api/v1/screenings/{id_or_request_id}` — single audit record; accepts either the integer row id or the 32-character `request_id`.
- `GET /api/v1/screenings/{id_or_request_id}/factors` — the normalized risk-factor rows (name, severity, weight, description) for one screening.
- `GET /api/v1/stats` — dashboard counts: totals, cleared, secondary inspection, high risk, MRZ failures, face mismatches, suspicious tampering, plus breakdowns by decision and risk level.
- `GET /api/v1/report/summary` — totals per risk level and decision, cleared/secondary/high-risk counts, and average processing time.

```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/screenings?limit=10
```

All screening history and report endpoints return `401` without a valid token.

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

- MRZ: `VALID`, `INVALID`, `MALFORMED`, `NOT_DETECTED`, `OCR_FAILED`, `OCR_LOW_CONFIDENCE` — plus document-level `format` (`TD3`/`TD1`/`UNKNOWN`) and `document_type` (`PASSPORT`/`NATIONAL_ID`/`UNKNOWN`).
- Tamper: `CLEAN`, `SUSPICIOUS`, `INCONCLUSIVE`, `ERROR` — with per-signal `score`/`suspicious` entries for `ela`, `compression`, `noise`, `edge`, `copy_move`, `metadata`. A failed signal degrades to `0`/`not suspicious` within the aggregate score rather than crashing the analysis.
- Face: `MATCH`, `MISMATCH`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_CONFIDENCE`, `INVALID_IMAGE`, `SKIPPED_NO_LIVE_PHOTO`, `ERROR`, `NOT_AVAILABLE`.
- Liveness: `LIVE`, `SPOOF_DETECTED`, `UNCERTAIN`, `NOT_CHECKED` — `is_live` is `true` only for `LIVE`; the `signals` object exposes per-signal heuristic scores and reasons when the OpenCV fallback ran.
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

The suite (159 tests) covers valid/malformed MRZ, all checksum failures, pivot-year expiry semantics, leap-year date handling, invalid sex/nationality fields, OCR failure and candidate rejection, TD1 structural parsing and checksums, parser routing by aspect ratio and explicit `document_type`, tamper statuses including INCONCLUSIVE handling, the uniform-blank-image copy-move regression, metadata-absence neutrality, passive-liveness classification (blank/spoof, noise/live, threshold boundaries, NOT_CHECKED and disabled paths, heuristic signal bounds, module-state mapping), the fail-safe and high-risk decision gates including the forced high-risk liveness gate, unknown-risk-factor rejection, ArcFace engine statuses (NO_FACE/MULTIPLE_FACES/LOW_CONFIDENCE/MATCH/MISMATCH), deterministic face bounding-box reporting, NaN/division-by-zero protection, X-Request-ID echo and sanitization, /health and /ready behavior, upload hardening (empty files, wrong extensions, MIME spoofing, oversized images, pixel-count and decompression-bomb protection), structured error bodies, safe handling of unexpected internal errors, plus the audit trail, authentication flow, protected history/report endpoints, summary rollup, repository CRUD with duplicate-request_id rejection, pagination/filters, dashboard stats, factor normalization, Alembic upgrade/downgrade/check, graceful behaviour when the database is unreachable (503, no leaks), and token-expiry comparisons with both naive (SQLite) and timezone-aware (PostgreSQL) timestamps.

## Security and Privacy

- Uploaded images are validated by byte size, declared MIME type, actual image signature/format, and pixel dimensions, then processed in memory; the API does not persist uploads.
- The audit trail stores screening risk metadata only. No passport numbers, names, MRZ text, raw images, or biometric embeddings are stored or logged.
- Passwords are stored as salted PBKDF2 hashes; bearer tokens are stored as SHA-256 hashes with a TTL.
- `live_photo` and `document_image` files are always closed after reading.
- CORS origins are configurable; no wildcard-plus-credentials combination is used.
- The Docker process runs as a non-root user with a healthcheck.

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