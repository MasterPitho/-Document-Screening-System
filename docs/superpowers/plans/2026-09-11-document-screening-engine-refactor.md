# Comprehensive Document Screening Engine Audit & Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the SIH 2026 Document Screening Engine into a production-grade, privacy-first, multi-document (TD3, TD1, Aadhaar, PAN) screening backend with cross-signal consistency, strict non-PII persistence, environment-based CORS security, and comprehensive verification.

**Architecture:** Strategy-based document parser routing behind a unified FastAPI pipeline; standalone in-memory QR analysis service; cross-signal consistency evaluator; multi-signal image forensics and ArcFace biometric matching; deterministic weighted risk engine; privacy-safe PostgreSQL/SQLite audit logging without PII.

**Tech Stack:** Python 3.13+, FastAPI, OpenCV, Pillow, PyTesseract, InsightFace/ArcFace, ONNX Runtime, SQLAlchemy 2.0, Alembic, Pytest.

**Spec:** SIH 2026 AI-Based Fake Identity & Document Screening System Architecture & Privacy Hardening Specification.

## Global Constraints

- **Strict Privacy**: Uploaded images, live photos, face embeddings, raw MRZ lines, passport numbers, Aadhaar numbers, and applicant names must NEVER be persisted in the database, logged in logs, or exposed in error traces.
- **Explainability**: Every risk verdict is a deterministic weighted sum of explicit factors. No black-box guesses or LLM-generated identity truth.
- **Fail-Safe**: Technical failures or missing signals must yield `SECONDARY_INSPECTION_REQUIRED` or `HIGH_RISK_REVIEW_REQUIRED`, never false `CLEARED`.
- **Backward Compatibility**: Existing public API routes (`/api/v1/screen`, `/api/v1/screenings`, `/api/v1/stats`, `/ready`, `/health`) maintain contract integrity.
- **Zero Schema Drift**: Every database change has a corresponding Alembic migration verified against `Base.metadata`.

---

## Tasks

### Task 1: Phase 1 Code Audit & Documentation

**Files:**
- Create: `docs/superpowers/specs/2026-09-11-code-audit-findings.md`
- Inspect: `app/main.py`, `app/config.py`, `app/db/models.py`, `app/services/mrz.py`, `app/services/tampering.py`, `app/services/face_recognition.py`, `app/services/liveness.py`, `app/services/risk_engine.py`

- [ ] **Step 1: Document all audit findings into checklist categories A–F**
- [ ] **Step 2: Verify existing source code lines against README claims**

---

### Task 2: Phase 2 Critical Privacy Fix (Remove PII from Database & Schemas)

**Files:**
- Modify: `app/db/models.py`
- Create: `alembic/versions/d8b2e4f6a1c3_remove_pii_columns.py`
- Modify: `app/db/repositories.py`
- Modify: `app/models/schemas.py`
- Modify: `app/main.py`
- Test: `tests/test_privacy_compliance.py`
- Modify: `tests/test_dashboard_extensions.py`

**Interfaces:**
- Removes `applicant_name`, `document_number`, `country_code` from `Screening` model and `ScreeningRecordOut` schema.
- Retains operational `notes` on `Screening`.

- [ ] **Step 1: Write privacy compliance test verifying no PII columns exist on `Screening` and no PII is saved**
- [ ] **Step 2: Update `Screening` model in `app/db/models.py` to remove `applicant_name`, `document_number`, `country_code`**
- [ ] **Step 3: Create Alembic migration `d8b2e4f6a1c3_remove_pii_columns.py` to drop the columns and index**
- [ ] **Step 4: Update `ScreeningRepository` in `app/db/repositories.py` to remove PII arguments**
- [ ] **Step 5: Update `ScreeningRecordOut` in `app/models/schemas.py`**
- [ ] **Step 6: Update `POST /api/v1/screen` and `_as_screening_out` in `app/main.py`**
- [ ] **Step 7: Update `tests/test_dashboard_extensions.py` and run tests to verify green**

---

### Task 3: Phase 3 Production CORS Security Hardening

**Files:**
- Modify: `app/config.py`
- Modify: `app/main.py`
- Test: `tests/test_cors_security.py`

- [ ] **Step 1: Write test verifying that production environment forbids wildcard `*` with credentials**
- [ ] **Step 2: Update `Settings` and `main.py` to configure CORS strictly based on `api_env` and `cors_origins`**
- [ ] **Step 3: Run pytest to verify all CORS tests pass**

---

### Task 4: Phase 7 Reusable QR Code Analysis Service

**Files:**
- Create: `app/services/qr.py`
- Test: `tests/test_qr_analysis.py`

**Interfaces:**
- Produces `QRAnalysisResult` with `detected: bool`, `readable: bool`, `format: str`, `payload_type: str`, `has_data: bool`, `is_malformed: bool`, `details: dict[str, Any]` (NO raw PII).

- [ ] **Step 1: Write failing test for QR code detection, decoding, and structural validation**
- [ ] **Step 2: Implement `QRAnalyzer` in `app/services/qr.py` using OpenCV QRCodeDetector and payload structure analysis**
- [ ] **Step 3: Run tests to verify QR detection and privacy compliance**

---

### Task 5: Phase 5 Dedicated Aadhaar Document Analysis Strategy

**Files:**
- Create: `app/services/aadhaar.py`
- Modify: `app/services/mrz.py` (DocumentParserRouter integration)
- Modify: `app/main.py` (allowed document types)
- Test: `tests/test_aadhaar_screening.py`

**Interfaces:**
- `AadhaarDocumentParser(BaseDocumentParser)`:
  - Detects Aadhaar text anchors ("Government of India", "UIDAI", "DOB", "Male/Female", etc.).
  - Extracts and decodes Aadhaar QR code via `QRAnalyzer`.
  - Validates masked UID format (`XXXX XXXX 1234` or 12-digit format).
  - Flags tampering or layout inconsistencies without storing raw numbers or PII.

- [ ] **Step 1: Write failing test for Aadhaar document parsing with QR and anchor validation**
- [ ] **Step 2: Implement `AadhaarDocumentParser` in `app/services/aadhaar.py`**
- [ ] **Step 3: Register `AadhaarDocumentParser` in `DocumentParserRouter`**
- [ ] **Step 4: Run tests to verify Aadhaar screening**

---

### Task 6: Phase 6 PAN Document Analysis Strategy

**Files:**
- Create: `app/services/pan.py`
- Modify: `app/services/mrz.py` (DocumentParserRouter integration)
- Modify: `app/main.py` (add `"pan"` to allowed types)
- Test: `tests/test_pan_screening.py`

**Interfaces:**
- `PANDocumentParser(BaseDocumentParser)`:
  - Detects PAN layout anchors ("INCOME TAX DEPARTMENT", "GOVT. OF INDIA", "Permanent Account Number").
  - Evaluates 10-char PAN structure: `[A-Z]{5}[0-9]{4}[A-Z]`.
  - Validates 4th character status code (`P`, `C`, `H`, `F`, `A`, `T`, `B`, `L`, `J`, `G`).
  - Evaluates image quality, sharpness, and field position.
  - Returns `VALID`, `SUSPICIOUS`, `MALFORMED`, or `UNCERTAIN`.

- [ ] **Step 1: Write failing test for PAN card validation**
- [ ] **Step 2: Implement `PANDocumentParser` in `app/services/pan.py`**
- [ ] **Step 3: Register `PANDocumentParser` in `DocumentParserRouter`**
- [ ] **Step 4: Run tests to verify PAN screening**

---

### Task 7: Phase 4 Enhanced TD1 ID-Card Parsing

**Files:**
- Modify: `app/services/mrz.py` (`NationalIDTD1Parser`)
- Test: `tests/test_document_router.py`

- [ ] **Step 1: Write tests for edge-case TD1 parsing, multi-pass candidate extraction, and UNCERTAIN handling**
- [ ] **Step 2: Improve `NationalIDTD1Parser` candidate extraction and error reporting**
- [ ] **Step 3: Run pytest to verify all TD1 tests pass**

---

### Task 8: Phase 12 Cross-Signal Consistency Engine

**Files:**
- Create: `app/services/cross_signal.py`
- Modify: `app/services/risk_engine.py`
- Modify: `app/main.py`
- Test: `tests/test_cross_signal.py`

**Interfaces:**
- `CrossSignalEvaluator.evaluate(...)`:
  - Validates document type requested vs document structure detected.
  - Checks expiration date against current date.
  - Cross-checks QR payload metadata vs OCR/MRZ metadata.
  - Checks if tampering suspicious regions overlap with identity/document text regions.
  - Emits explainable risk factors: `DOCUMENT_TYPE_MISMATCH`, `DOCUMENT_EXPIRED`, `QR_OCR_CONFLICT`, `SUSPICIOUS_FIELD_TAMPERING`.

- [ ] **Step 1: Write failing test for cross-signal consistency checks**
- [ ] **Step 2: Implement `CrossSignalEvaluator` in `app/services/cross_signal.py`**
- [ ] **Step 3: Wire cross-signal factors into `RiskEngine` in `app/services/risk_engine.py`**
- [ ] **Step 4: Run tests to verify cross-signal scoring**

---

### Task 9: Phase 14 Role-Based Access Control (RBAC)

**Files:**
- Modify: `app/api/auth.py`
- Modify: `app/main.py`
- Test: `tests/test_rbac.py`

**Interfaces:**
- Adds `require_role(*roles: str)` dependency.
- Enforces permissions: `admin` for admin actions, `supervisor` / `officer` for screening updates.

- [ ] **Step 1: Write failing test for RBAC authorization checks**
- [ ] **Step 2: Implement `require_role` dependency in `app/api/auth.py`**
- [ ] **Step 3: Apply `require_role` to administrative and decision endpoints in `app/main.py`**
- [ ] **Step 4: Run tests to verify RBAC**

---

### Task 10: Phase 15 Privacy-Safe Audit Logging Verification

**Files:**
- Modify: `app/db/repositories.py`
- Test: `tests/test_persistence.py`

- [ ] **Step 1: Verify that `AuditLogRepository` and `ScreeningRepository` never log raw PII**
- [ ] **Step 2: Add assertions verifying audit log messages contain only non-PII metadata**
- [ ] **Step 3: Run pytest**

---

### Task 11: Phase 17 Security Hardening & Phase 18 Performance Verification

**Files:**
- Modify: `app/security/image_validation.py`
- Modify: `app/main.py`
- Test: `tests/test_security_hardening.py`

- [ ] **Step 1: Write tests for path traversal, decompression bombs, oversized uploads, and malformed input**
- [ ] **Step 2: Harden upload verification and exception handling**
- [ ] **Step 3: Verify single-load model management**

---

### Task 12: Phase 19 Deployment & Phase 20 Documentation Synchronization

**Files:**
- Modify: `Dockerfile`
- Modify: `README.md`
- Test: Full test suite verification

- [ ] **Step 1: Run full test suite (`pytest`) and verify 100% pass rate**
- [ ] **Step 2: Run Alembic migration check (`alembic check`) to verify zero schema drift**
- [ ] **Step 3: Update `README.md` to reflect real source code architecture, privacy policy, and supported document types**
- [ ] **Step 4: Generate final verification and completion report**
