# Security & Correctness Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement comprehensive security, privacy, and correctness hardening across the document screening engine: zero PII in screening responses, public role escalation prevention, secure bootstrap credentials, mandatory authentication on screening and dashboard endpoints, lightweight rate limiting, Aadhaar structural validity semantics, robust QR ↔ OCR cross-signal verification, multi-signal evidence auto-routing, truthful risk metrics without fake confidence, PII protection in screening notes, demo watchlist isolation, and real GitHub Actions CI.

**Architecture:** Maintain the existing FastAPI, OpenCV, and SQLAlchemy architecture without unnecessary rewrites. Add lightweight in-memory resource protection, sanitize public serializers to omit raw identity PII, tighten authentication dependencies, enforce strict role assignment during registration, and upgrade evidence-based routing.

**Tech Stack:** FastAPI, Pydantic v2, OpenCV, SQLAlchemy, Alembic, Pytest.

**Spec:** Prompt requirements from User Request (# FINAL SECURITY + CORRECTNESS HARDENING).

## Global Constraints
- Zero raw identity PII in public responses: no name, document number, DOB, nationality, sex, raw MRZ, or raw OCR.
- Public registration creates only `officer` role; privileged roles cannot be acquired via `/register`.
- Default credentials prohibited in production; `DEV_BOOTSTRAP=true` required for development seed accounts.
- `POST /api/v1/screen` and dashboard endpoints require authentication. Genuinely public: `/health`, `/ready`.
- No fake model confidence: replace `confidence = risk_score / 100.0` with `risk_normalized`.
- 100% test pass rate preserved across all test suites.

---

### Task 1: Privacy-Safe Screen API Response & MRZ Serialization

**Files:**
- Modify: `app/models/schemas.py`
- Modify: `app/main.py`
- Test: `tests/test_privacy_compliance.py`

**Interfaces:**
- `PrivacySafeMRZResult`: exposes `detected`, `valid`, `document_type`, `status`, `source`, `checksum_valid`, `format_valid`, `fields_detected`, `issues`, `module_state`. Does NOT expose `line1`, `line2`, `data` (names, numbers, DOBs, nationality).
- `to_privacy_safe_mrz(raw_mrz: dict) -> dict`: serializer sanitizing internal parse results before putting them in `response["mrz"]` and `response["modules"]["mrz_validation"]`.

- [ ] **Step 1: Write failing test in `tests/test_privacy_compliance.py` asserting `/api/v1/screen` contains zero PII**
- [ ] **Step 2: Define `PrivacySafeMRZResult` schema in `app/models/schemas.py`**
- [ ] **Step 3: Implement `to_privacy_safe_mrz` in `app/main.py` and sanitize response output**
- [ ] **Step 4: Run privacy compliance tests to verify**

---

### Task 2: Fix Public Role Escalation

**Files:**
- Modify: `app/api/auth.py`
- Modify: `app/models/schemas.py`
- Test: `tests/test_rbac.py`

**Interfaces:**
- In `POST /api/v1/auth/register`: always force `role = "officer"`, ignoring client input. Privileged roles (`supervisor`, `admin`) can only be created via admin bootstrap or privileged administrative flow.

- [ ] **Step 1: Write failing tests in `tests/test_rbac.py` for registration role escalation attempts**
- [ ] **Step 2: Update `app/api/auth.py` to enforce `assigned_role = "officer"` regardless of payload**
- [ ] **Step 3: Run `tests/test_rbac.py` to verify**

---

### Task 3: Remove/Gate Predictable Default Credentials

**Files:**
- Modify: `app/config.py`
- Modify: `app/api/auth.py`
- Test: `tests/test_security_hardening.py`

**Interfaces:**
- Add `dev_bootstrap: bool` to `Settings` reading `DEV_BOOTSTRAP` (default `False`).
- In `bootstrap_admin`: only create default `officer`/`Officer123!` if `settings.dev_bootstrap is True` and `settings.api_env != "production"`. In production, only environment-provided `ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` are created.

- [ ] **Step 1: Write test verifying that in production or when `DEV_BOOTSTRAP=false`, no default officer account is created**
- [ ] **Step 2: Add `dev_bootstrap` field to `Settings` in `app/config.py`**
- [ ] **Step 3: Guard default officer creation in `bootstrap_admin` in `app/api/auth.py`**
- [ ] **Step 4: Run tests to verify**

---

### Task 4: Protect Screening & Dashboard Endpoints with Authentication

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api.py` and existing test suites

**Interfaces:**
- Enforce `Depends(get_current_user)` on `POST /api/v1/screen`, `GET /api/v1/stats/*`, `GET /api/v1/notifications`, `GET /api/v1/watchlists`.
- Keep `/health`, `/ready`, `/`, `/docs`, `/openapi.json` public.
- Update test client requests to include valid bearer tokens where authentication is required.

- [ ] **Step 1: Write test asserting anonymous `POST /api/v1/screen` returns 401 Unauthorized**
- [ ] **Step 2: Update route signatures in `app/main.py` to require `Depends(get_current_user)`**
- [ ] **Step 3: Update test helper fixtures in `tests/conftest.py` and test suites to provide auth tokens**
- [ ] **Step 4: Run tests to verify all endpoints pass with authentication**

---

### Task 5: Lightweight Concurrency & Rate Limiting (Resource Protection)

**Files:**
- Create: `app/security/rate_limiter.py`
- Modify: `app/config.py`
- Modify: `app/main.py`
- Test: `tests/test_security_hardening.py`

**Interfaces:**
- `ResourceLimiter`: In-memory sliding window rate limiter per user/IP + concurrency semaphore for screening operations.
- Emits HTTP 429 with descriptive error message and `Retry-After` header when capacity or rate limit is exhausted.

- [ ] **Step 1: Write failing test in `tests/test_security_hardening.py` for concurrency and rate limiting (HTTP 429)**
- [ ] **Step 2: Add `screening_rate_limit_per_minute` and `max_concurrent_screenings` settings to `app/config.py`**
- [ ] **Step 3: Implement `ResourceLimiter` in `app/security/rate_limiter.py`**
- [ ] **Step 4: Integrate limiter into `POST /api/v1/screen` in `app/main.py`**
- [ ] **Step 5: Run tests to verify**

---

### Task 6: Aadhaar Semantic Correction (Structural Validity, Not Official Authenticity)

**Files:**
- Modify: `app/services/aadhaar.py`
- Modify: `app/services/qr.py`
- Modify: `app/config.py`
- Modify: `README.md`
- Test: `tests/test_aadhaar_screening.py`

**Interfaces:**
- Clarify in explanations and status descriptions that QR and Verhoeff checks confirm `STRUCTURALLY_VALID` card data, not UIDAI official government authenticity without certified cryptographic signature keys.

- [ ] **Step 1: Audit and replace any claims of "AUTHENTIC" or "VERIFIED_BY_UIDAI" with "STRUCTURALLY_VALID" in `aadhaar.py` and `qr.py`**
- [ ] **Step 2: Update status descriptions and risk factor details**
- [ ] **Step 3: Run `tests/test_aadhaar_screening.py` to verify**

---

### Task 7: Robust QR ↔ OCR Cross-Signal Matching

**Files:**
- Modify: `app/services/cross_signal.py`
- Modify: `app/services/aadhaar.py`
- Test: `tests/test_cross_signal.py`

**Interfaces:**
- Standardize `ocr_masked_uid` and `qr_masked_uid` in `doc_result`.
- When matching (e.g. both last 4 digits match): NO conflict.
- When mismatching: trigger `QR_OCR_CONFLICT` (wt 35).
- Missing QR or malformed QR: no conflicting identity comparison emitted.

- [ ] **Step 1: Write explicit regression tests in `tests/test_cross_signal.py` for matching, mismatching, missing, and malformed QR**
- [ ] **Step 2: Update `CrossSignalEvaluator` and `AadhaarDocumentParser` to wire normalized masked values**
- [ ] **Step 3: Run `tests/test_cross_signal.py` to verify**

---

### Task 8: Multi-Signal Auto Document Routing & Ambiguous "UNKNOWN"

**Files:**
- Modify: `app/services/mrz.py`
- Test: `tests/test_document_router.py`

**Interfaces:**
- Enhance `DocumentParserRouter.select()` to score evidence across all 4 document types:
  - Passport: TD3 lines / `P<` prefix
  - National ID: TD1 3-line format
  - Aadhaar: UIDAI anchors or Aadhaar QR
  - PAN: Income Tax anchors or 10-char PAN regex
- If evidence is ambiguous or absent, return `UnknownDocumentParser` with `document_type="UNKNOWN"` and `status="NOT_DETECTED"`.

- [ ] **Step 1: Write failing tests in `tests/test_document_router.py` for auto routing with clear evidence vs ambiguous/blank images**
- [ ] **Step 2: Implement multi-signal evidence scoring in `DocumentParserRouter`**
- [ ] **Step 3: Run `tests/test_document_router.py` to verify**

---

### Task 9: Remove Misleading "Confidence" (Replace with `risk_normalized`)

**Files:**
- Modify: `app/services/risk_engine.py`
- Modify: `app/models/schemas.py`
- Test: `tests/test_risk_engine.py`

**Interfaces:**
- In `RiskEngine.evaluate()`: replace `"confidence": risk_score / 100.0` with `"risk_normalized": round(risk_score / 100.0, 3)`.
- Update `RiskAssessment` schema in `app/models/schemas.py`.

- [ ] **Step 1: Update schema and `risk_engine.py`**
- [ ] **Step 2: Update assertions in `tests/test_risk_engine.py`**
- [ ] **Step 3: Run tests to verify**

---

### Task 10: Protect Screening Notes from PII

**Files:**
- Modify: `app/main.py`
- Modify: `app/models/schemas.py`
- Test: `tests/test_privacy_compliance.py`

**Interfaces:**
- Validate `review_notes` in `DecisionUpdateRequest` and any notes inputs to reject raw Aadhaar (12 digits), PAN numbers, passport numbers, or card numbers.

- [ ] **Step 1: Write test attempting to patch decision with a raw Aadhaar or PAN in `review_notes`**
- [ ] **Step 2: Implement PII detection in `review_notes` validator**
- [ ] **Step 3: Run tests to verify**

---

### Task 11: Demo Watchlist Semantics & Isolation

**Files:**
- Modify: `app/main.py`
- Modify: `app/models/schemas.py`
- Test: `tests/test_dashboard_extensions.py`

**Interfaces:**
- Add `is_demo_data: bool = True` and source attribution to `WatchlistItem`.
- Mark watchlist data as synthetic demo intelligence not used for operational decisions.

- [ ] **Step 1: Update `WatchlistItem` schema with `is_demo_data`**
- [ ] **Step 2: Update `/api/v1/watchlists` endpoint**
- [ ] **Step 3: Verify with tests**

---

### Task 12: Role-Based Access on Dashboard Endpoints

**Files:**
- Modify: `app/main.py`
- Modify: `app/db/repositories.py`
- Test: `tests/test_rbac.py`

**Interfaces:**
- Officers only see their own screenings in `/api/v1/screenings`.
- Supervisors and admins see all screenings.
- Authenticate `/api/v1/notifications` and `/api/v1/stats/trend`.

- [ ] **Step 1: Add tests for officer vs supervisor/admin scoping**
- [ ] **Step 2: Implement user-scoping in `ScreeningRepository.list` and `app/main.py`**
- [ ] **Step 3: Run tests to verify**

---

### Task 13: Real GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/tests.yml`

**Interfaces:**
- Runs on push to `main` and pull requests.
- Sets up Python 3.11, installs `requirements-dev.txt`, runs `pytest -q`.

- [ ] **Step 1: Create `.github/workflows/tests.yml`**
- [ ] **Step 2: Validate YAML syntax**

---

### Task 14: Documentation & OpenAPI Synchronization

**Files:**
- Modify: `README.md`

**Interfaces:**
- Document required bearer auth on `/api/v1/screen`.
- Clarify structural validity vs UIDAI official authenticity.
- Update risk assessment format (`risk_normalized`).
- Mention CI workflow.

- [ ] **Step 1: Update `README.md` with accurate API and security details**

---

### Task 15: Full Verification & Final Security Audit

**Files:**
- All touched files

**Interfaces:**
- Full test suite run (`pytest -q`).
- `alembic check` verification.
- `git diff` inspection for accidental secrets, PII, or insecure defaults.

- [ ] **Step 1: Run full test suite (`pytest -q`) and verify 100% pass**
- [ ] **Step 2: Verify `alembic check`**
- [ ] **Step 3: Review `git diff` for leaks or regressions**
