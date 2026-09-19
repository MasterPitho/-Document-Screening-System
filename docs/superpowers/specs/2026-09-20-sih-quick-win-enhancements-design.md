# Design: SIH Quick-Win Enhancements (non-breaking)

Date: 2026-09-20
Status: Draft
Scope: backend-only; no frontend changes; no new system dependencies beyond `pymupdf`.

## Goal

Close the credibility gaps flagged in the SIH-2026 proposal (blockchain claim,
watchlist monitoring, notifications, PDF intake, exportable reports, visa
support) **without breaking any existing code path or changing existing
behaviour when features are unused**. All additions are new tables, new
endpoints, or new accepted input types. Existing tables and endpoints keep
their current contracts.

## Non-negotiables

- Existing 248 tests keep passing; new features land with their own tests
  (TDD, RED -> GREEN).
- Schema additive only. New tables must be auto-created by the existing
  `Database.create_all()` (SQLite + PostgreSQL), so existing deployments get
  the new tables on restart with no migration step.
- No changes to existing table columns or existing endpoint responses.
- Optional features must be no-ops when their config is unset (SMTP-like
  pattern): the app must boot and behave exactly as today.
- No commit / push — user explicitly forbade it this cycle.

## 1. Privacy-Preserving Hash-Chain Ledger ("blockchain")

### Purpose
Make the PDF's "Hyperledger-like hash ledger" claim real with a lightweight,
auditable, privacy-safe hash chain. Pure stdlib `hashlib`; no new deps.

### Data
New table `ledger_entries`:
- `id` (Integer PK)
- `index` (Integer, unique, monotonic)
- `prev_hash` (String(64), "" for genesis)
- `payload_hash` (String(64))
- `payload` (JSON, the redacted canonical payload)
- `screening_id` (FK screenings.id, nullable)
- `request_id` (String(64))
- `created_at` (DateTime naive UTC)

### Hashing
Canonical payload = stable JSON of a redacted screening record:
`json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)`
Doc contains **only**: `decision`, `risk_level`, `risk_score`, `document_type`, `mrz_status`,
`tampering_status`, `tampering_score`, `face_status`, `face_similarity`, `liveness_status`,
`liveness_score`, `mrz_source`, `watchlist_hit`, `request_id`, `created_at`.
NEVER raw MRZ lines, document numbers, face embeddings, or image bytes.

`payload_hash = sha256(payload_bytes).hexdigest()`
`hash = sha256((prev_hash + payload_hash).encode()).hexdigest()`
Genesis entry: `prev_hash = ""`.

### Insert point
Ledger entry is inserted **in the same DB session/transaction** as the
screening record inside `ScreeningRepository.create()` (or an explicit
`LedgerRepository.append()` called immediately after `create()` and committed
in the same transaction). Deterministic so re-running the same screening
yields the same payload hash.

### Endpoints (all require auth)
- `GET /api/v1/ledger` -> `{entries: [...]}` newest-first (limit/offset).
- `GET /api/v1/ledger/summary` -> `{total_entries, head_hash, last_index, created_at}`.
- `GET /api/v1/ledger/verify` -> recomputes the chain; returns
  `{ok: bool, broken_at_index: int|null, total_entries, verified_count}`.

### Tests
- Canonical payload determinism (same input -> same hash).
- Genesis + chain link correct.
- Tampering: edit payload of entry N -> `/verify` reports broken at N (or N+1).

## 2. Real Watchlists (replace demo data)

### Data
New table `watchlist_entries`:
- `id` (Integer PK)
- `name` (String)
- `document_number` (String, indexed)
- `reason` (Text)
- `severity` (String(12), default HIGH)
- `active` (Boolean, default True)
- `is_demo_data` (Boolean, default False)
- `created_at` (DateTime naive UTC)

On startup, if table is empty, seed the three existing demo entries
(`is_demo_data=True`, `source` preserved via `reason`/flag) so the current
frontend `GET /watchlists` keeps returning the same three entries.

### Endpoints (all require auth; admin-scoped writes optional -> keep role-agnostic for hackathon)
- `GET /api/v1/watchlists` (active/filter; falls back to all entries incl. inactive)
- `POST /api/v1/watchlists` `{name, document_number, reason, severity}`
- `PATCH /api/v1/watchlists/{id}` (update fields; `active=False` = soft delete)
- `DELETE /api/v1/watchlists/{id}` (soft delete: `active=False`)

### Screening match
During screening, derive the candidate document number from the MRZ/parser
result (`mrz_result["data"]["document_number"]`, PAN number, Aadhaar xxxx-xxxx
number). Normalise (strip spaces/dashes, uppercase, keep alnum). If it equals an
active watchlist entry document_number (also normalised):
- Add risk factor `ON_WATCHLIST` weight 40 (HIGH) via the risk engine factor list.
- Set `watchlist_hit = {"name", "reason", "severity"}` in the screening record and
  in payload for the ledger.
- Decision flow is unchanged: the increased score drives the existing
  CLEARED / REVIEW / HOLD boundaries. No hard-coded HOLD.

### Tests
- CRUD round-trip; soft delete hides from active list.
- Match adds factor + watchlist_hit; non-match leaves record unchanged.
- Normalisation equalities (e.g. "P8831042" == "p-883 1042").

## 3. Real Notifications

### Data
New table `notifications`:
- `id` (Integer PK)
- `screening_id` (FK, nullable)
- `type` (String(40), default HIGH_RISK_ALERT)
- `message` (Text)
- `channel` (String(16), default "in_app")
- `read` (Boolean, default False)
- `created_at` (DateTime naive UTC)

Record inserted whenever a screening persists with high-risk signals
(`risk_score >= 65` or `risk_level == "HIGH_RISK"` or
`decision == "HIGH_RISK_REVIEW_REQUIRED"`), same transaction as the screening.

### Endpoints
- `GET /api/v1/notifications` (existing shape kept: id, type, message, created_at, read)
- `PATCH /api/v1/notifications/{id}/read` -> marks read
- `PATCH /api/v1/notifications/read-all` (optional convenience)

Keep `ScreeningRepository.recent_notifications()` behaviour: replace with table
reads, preserving response schema so the frontend is untouched.

### Tests
- High-risk screening creates notification; normal screening does not.
- Mark-read round-trip; read-all.

## 4. PDF Document Upload

### Approach
Add `pymupdf` to `requirements.txt` (pure wheel; no poppler/system binary). In
the screen endpoint, if the uploaded file is `application/pdf` (or `.pdf`):
- Validate size <= `MAX_PDF_MB` (new config, default 20).
- Render **first page only** to an RGB JPEG at zoom ~2.0 via
  `page.get_pixmap(matrix=fitz.Matrix(2,2), colorspace=fitz.csRGB)`.
- Feed the JPEG bytes through `prepare_working_image` and the normal pipeline.
- On render failure (encrypted/zero-page PDF) return a controlled 422 with a
  clear message.
- Unknown/unsupported formats keep existing validation/422 behaviour.

The upload is decoded as bytes first (already the flow): read `UploadFile` once,
sniff content type from bytes + filename.

### Tests
- Build a tiny synthetic PDF with pymupdf (text) -> `/screen` returns SCREENED and
  a ledger entry is created.
- Reject a garbage/encrypted PDF with 422.
- Existing image path tests still pass unchanged.

## 5. CSV Export

### Endpoint
`GET /api/v1/reports/export.csv?range=24h|7d|30d|all` (auth).
Returns `text/csv` with UTF-8 BOM (`\ufeff`) so Excel reads it. Columns:
`created_at, request_id, document_type, mrz_status, face_status, tampering_status,
liveness_status, risk_score, risk_level, decision, status_color, processing_time_ms,
notes`. Overwrite-proof via `CSV` quoting. Rows from `ScreeningRepository.list()`
with date filters from `range`.

### Tests
- 200 + content-type; header row; row count matches DB; BOM present.

## 6. Visa (TD2) Parser

### Approach
Add `TD2VisaParser` in `app/services/mrz.py`:
- 2 lines x 36 chars, ICAO TD2 structure.
- Checksum validation via existing `calculate_icao_checksum`/`verify_mrz_field`.
- `document_type = "VISA"`, `format = "TD2"`.
- Register in `DocumentParserRouter._ensure_parser` + `resolve()` allowed set:
  `auto, td3/passport, td1/national_id, aadhaar, pan, visa`.
- `extract_mrz_from_image(..., document_type="visa")` path works.

No frontend change this session (backend-only).

### Tests
- Structure/checksum success + failure cases for TD2 sample strings.
- Router resolves "visa" -> TD2VisaParser.
- Screen with `document_type=visa` runs end-to-end.

## Files touched (additive)
- `requirements.txt` (+pymupdf)
- `app/config.py` (+MAX_PDF_MB, maybe watchlist/ledger toggles — only if needed)
- `app/db/models.py` (+ 3 tables)
- `app/db/repositories.py` (+ LedgerRepository, WatchlistRepository, NotificationsRepository; screening create wiring)
- `app/services/ledger.py` (new; hash/verify pure functions)
- `app/services/watchlist.py` (new; normalisation + match)
- `app/services/mrz.py` (+ TD2 parser)
- `app/services/pdf.py` (new; PDF -> JPEG)
- `app/main.py` (endpoints + screen flow integration)
- `tests/...` (new test modules per feature)

## Testing strategy
- Unit tests: hashing determinism/chain, watchlist normalisation/match, TD2
  checksums, CSV writer, PDF render on synthetic PDF.
- Integration via `app_and_client`: screen JPEG (unchanged behaviour), screen
  synthetic PDF (SCREENED + ledger entry + notification only if high-risk),
  CRUD endpoints, export.csv.
- Full suite must stay green (248 + new).