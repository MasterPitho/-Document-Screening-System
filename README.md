# Document Screening Engine

An explainable **Smart India Hackathon prototype** for document screening. It validates TD3 passport MRZ data, runs a multi-signal image-forensics pipeline, verifies faces with **InsightFace ArcFace embeddings** (ONNX Runtime), and computes a transparent, deterministic risk score. It is **not** a guarantee of document authenticity and **not** an autonomous identity decision; a trained human officer remains the final decision maker.

## Current Scope

- TD3 passport MRZ: two 44-character lines, ICAO 9303 7-3-1 checks for document number, date of birth, expiry, and composite data, plus impossible-date and pivot-year expiry validation.
- Automatic MRZ OCR via Tesseract when the caller omits the MRZ form fields.
- Multi-signal image-forensics analysis: ELA, JPEG compression, noise, edge, lightweight copy-move (duplicate-region), and metadata presence. These are heuristic indicators, not a forgery classifier.
- Face verification with InsightFace ArcFace embeddings (`buffalo_sc` model via ONNX Runtime). The model is downloaded once at startup (or baked into the Docker image) and compares a document face to an optional live photo.
- Explainable risk score: weighted factors with explicit, documented weights. Weights are heuristic prototype values, not calibrated probabilities.

## Processing Pipeline

```text
Upload validation (byte size, MIME, extension, signature, pixel count, dimensions)
    -> MRZ form input, or Tesseract OCR candidate pipeline
    -> TD3 structure + ICAO 9303 checksum + date validation
    -> Tamper signals: ELA + compression + noise + edges + copy-move + metadata
    -> Face verification: InsightFace ArcFace embeddings (doc vs live photo)
    -> Deterministic, explainable risk engine (weights, module gates)
    -> PostgreSQL-backed privacy-preserving audit record (SQLAlchemy + Alembic)
```

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
| `FACE_MIN_DETECTION_CONFIDENCE` | `0.50` | RetinaFace detection score cutoff. |
| `FACE_MODEL_NAME` | `buffalo_sc` | InsightFace model pack. |
| `FACE_DET_SIZE` | `640` | Detection input size. |
| `FACE_CTX_ID` | `-1` | `-1` = CPU. |
| `FACE_MODELS_DIR` | `~/.insightface` | InsightFace model cache (mounted as `/opt/insightface` in Docker). |
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
- `mrz_line1`: optional exact TD3 line 1.
- `mrz_line2`: optional exact TD3 line 2.

Supplying only one of `mrz_line1`/`mrz_line2` returns HTTP 400; the API does not silently fall back to OCR. Supplying neither falls back to OCR. Manually supplied lines are validated with the exact same TD3 structure, ICAO 9303 checksum, and date rules as OCR output; they are reported with `"source": "form"` (an explicit manual/testing input path). An INVALID/MALFORMED manual MRZ still forces secondary inspection.

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
                        "matched": null, "module_state": "NOT_AVAILABLE"}
}
```

Decisions: `CLEARED` (all modules `PASS`, no risk factors), `SECONDARY_INSPECTION_REQUIRED` (any module below `PASS` or any risk factor), or `HIGH_RISK_REVIEW_REQUIRED` (score at/above the reject threshold). Fail-safe gate: none of `FAIL`, `ERROR`, or `NOT_AVAILABLE` module states can produce `CLEARED`.

Key states:

- MRZ: `VALID`, `INVALID`, `MALFORMED`, `NOT_DETECTED`, `OCR_FAILED`, `OCR_LOW_CONFIDENCE`.
- Tamper: `CLEAN`, `SUSPICIOUS`, `INCONCLUSIVE`, `ERROR` — with per-signal `score`/`suspicious` entries for `ela`, `compression`, `noise`, `edge`, `copy_move`, `metadata`. A failed signal degrades to `0`/`not suspicious` within the aggregate score rather than crashing the analysis.
- Face: `MATCH`, `MISMATCH`, `NO_FACE`, `MULTIPLE_FACES`, `LOW_CONFIDENCE`, `INVALID_IMAGE`, `SKIPPED_NO_LIVE_PHOTO`, `ERROR`, `NOT_AVAILABLE`.
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

The suite (123 tests) covers valid/malformed MRZ, all checksum failures, pivot-year expiry semantics, leap-year date handling, invalid sex/nationality fields, OCR failure and candidate rejection, tamper statuses including INCONCLUSIVE handling, the uniform-blank-image copy-move regression, and metadata-absence neutrality; the fail-safe and high-risk decision gates, unknown-risk-factor rejection, ArcFace engine statuses (NO_FACE/MULTIPLE_FACES/LOW_CONFIDENCE/MATCH/MISMATCH), deterministic face bounding-box reporting, NaN/division-by-zero protection, X-Request-ID echo and sanitization, /health and /ready behavior, upload hardening (empty files, wrong extensions, MIME spoofing, oversized images, pixel-count and decompression-bomb protection), structured error bodies, safe handling of unexpected internal errors, plus the audit trail, authentication flow, protected history/report endpoints, summary rollup, repository CRUD with duplicate-request_id rejection, pagination/filters, dashboard stats, factor normalization, Alembic upgrade/downgrade/check, graceful behaviour when the database is unreachable (503, no leaks), and token-expiry comparisons with both naive (SQLite) and timezone-aware (PostgreSQL) timestamps.

## Security and Privacy

- Uploaded images are validated by byte size, declared MIME type, actual image signature/format, and pixel dimensions, then processed in memory; the API does not persist uploads.
- The audit trail stores screening risk metadata only. No passport numbers, names, MRZ text, raw images, or biometric embeddings are stored or logged.
- Passwords are stored as salted PBKDF2 hashes; bearer tokens are stored as SHA-256 hashes with a TTL.
- `live_photo` and `document_image` files are always closed after reading.
- CORS origins are configurable; no wildcard-plus-credentials combination is used.
- The Docker process runs as a non-root user with a healthcheck.

## Limitations

Not production-grade identity verification:

- TD3 OCR can fail on low-quality or unusual scans.
- Face verification scores are not calibrated probabilities; the embedding model is a research checkpoint and liveness detection is out of scope.
- Tamper scoring is a heuristic (ELA + compression + noise + edge + copy-move + metadata) signal set, not a trained forgery classifier.
- Risk weights are not probabilities and the risk engine is deterministic, not learned.
- The MRZ pivot-year rule is a heuristic for two-digit years; it is not an authoritative issuance-date source.
- Additional document formats, calibrated datasets, liveness checks, and human-reviewed evaluation are required before deployment.