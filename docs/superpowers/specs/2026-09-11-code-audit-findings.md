# SIH 2026 Document Screening Engine — Code Audit Findings

**Date**: 2026-09-11
**Auditor**: Antigravity Agent
**Repository**: `MasterPitho/-Document-Screening-System`

---

## Executive Summary

The code audit systematically examined all modules across the codebase to compare existing implementations against the architecture claims and requirements. 

### Key Findings Matrix

| Component | Status | Finding | Action Required |
|---|---|---|---|
| **Privacy / PII Persistence** | **FAILED** | `Screening` DB table and schema store `applicant_name`, `document_number`, and `country_code` directly. Contradicts zero-PII persistence architecture. | Drop PII columns via Alembic migration. Update schema, repository, and screening endpoint to store privacy-safe audit metadata only. |
| **CORS Configuration** | **INSECURE IN PROD** | `app/main.py` uses `allow_origins=["*"]` with `allow_credentials=True`. In browsers, `*` with credentials violates CORS specs and allows CSRF/token leaks. | Make CORS configurable via `Settings` and restrict credentials + wildcard in production. |
| **Aadhaar Support** | **STUB / INCORRECT** | `PARSER_ALIASES["aadhaar"] = "national_id"` assumed Aadhaar has TD1 MRZ. Aadhaar cards do NOT have TD1 MRZ; they have QR codes and UIDAI text layouts. | Implement dedicated `AadhaarDocumentParser` with QR analysis and masked UID format checking. |
| **PAN Card Support** | **MISSING** | No PAN card parser exists in `app/services/mrz.py` or anywhere in the backend. | Implement `PANDocumentParser` enforcing 10-char regex (`[A-Z]{5}[0-9]{4}[A-Z]`) and 4th status char validation. |
| **TD1 National ID** | **BASIC** | Basic 3x30 regex parsing exists in `NationalIDTD1Parser`, but lacks error resilience, multi-candidate fallback, and field validation. | Enhance TD1 parser with conservative status mapping and OCR noise cleanup. |
| **Cross-Signal Consistency** | **MISSING** | Tampering, MRZ, and OCR results are combined strictly via weighted arithmetic; no logic checks whether extracted document type matches requested type, or if dates are expired, or if QR matches OCR. | Implement `CrossSignalEvaluator` and inject factor penalties into `RiskEngine`. |
| **Role-Based Access Control** | **PARTIAL** | JWT auth exists with roles (`admin`, `officer`, `supervisor`), but endpoints only check `get_current_user`, not specific roles. | Implement `require_role` dependency and enforce on admin/decision endpoints. |
| **Image Forensics** | **OPERATIONAL** | ELA, Laplacian blur, noise variance, clone detection, and copy-move operate deterministically. | Retain existing algorithms; feed findings to cross-signal evaluator. |
| **Face Recognition** | **OPERATIONAL** | InsightFace ArcFace cosine similarity with fallback distance scoring. | Retain; ensure embeddings are never persisted or logged. |
| **Database Migrations** | **SYNCED** | Currently at revision `c7a1f2b3c4d5` with 0 schema drift. | Add migration `d8b2e4f6a1c3` to drop PII columns safely. |
