"""
Cross-Signal Consistency Engine.

Performs deterministic cross-validation across all modalities:
1. Requested document type vs structure detected
2. Cross-modality QR vs visual OCR consistency
3. Spatial tampering overlap with identity and MRZ fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class CrossSignalResult:
    is_consistent: bool
    conflicts: list[str]
    factors: list[dict[str, Any]] = field(default_factory=list)


class CrossSignalEvaluator:
    """Evaluates multi-modal screening outputs for cross-signal contradictions."""

    DOCUMENT_TYPE_GROUPS: dict[str, set[str]] = {
        "passport": {"PASSPORT", "TD3"},
        "td3": {"PASSPORT", "TD3"},
        "national_id": {"NATIONAL_ID", "TD1"},
        "td1": {"NATIONAL_ID", "TD1"},
        "aadhaar": {"AADHAAR", "AADHAAR_CARD"},
        "pan": {"PAN", "PAN_CARD"},
    }

    def evaluate(
        self,
        requested_type: str,
        doc_result: dict[str, Any],
        tamper_result: dict[str, Any],
        face_result: dict[str, Any],
        image_shape: tuple[int, int] = (500, 800),
    ) -> CrossSignalResult:
        conflicts: list[str] = []
        factors: list[dict[str, Any]] = []

        # 1. Document Type Consistency
        norm_req = (requested_type or "auto").strip().lower()
        detected_type = str(doc_result.get("document_type") or doc_result.get("format") or "UNKNOWN").upper()
        detected_doc = doc_result.get("detected", False)

        if norm_req != "auto" and detected_doc and detected_type != "UNKNOWN":
            expected_set = self.DOCUMENT_TYPE_GROUPS.get(norm_req, set())
            if expected_set and detected_type not in expected_set:
                msg = f"Requested {norm_req} but document structure matches {detected_type}."
                conflicts.append(msg)
                factors.append({
                    "factor": "DOCUMENT_TYPE_MISMATCH",
                    "detail": msg,
                })

        # 2. QR vs OCR Consistency
        qr_info = doc_result.get("qr", {}) if isinstance(doc_result.get("qr"), dict) else {}
        doc_data = doc_result.get("data", {}) if isinstance(doc_result.get("data"), dict) else {}
        ocr_masked_uid = doc_data.get("ocr_masked_uid") or doc_data.get("masked_uid")
        qr_masked_uid = (
            doc_data.get("qr_masked_uid")
            or qr_info.get("masked_uid")
            or (qr_info.get("masked_summary", {}).get("masked_uid") if isinstance(qr_info.get("masked_summary"), dict) else None)
            or (qr_info.get("details", {}).get("masked_uid") if isinstance(qr_info.get("details"), dict) else None)
        )

        import re

        def _normalize_masked(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            digits = re.sub(r"\D", "", str(val))
            if len(digits) >= 4:
                return digits[-4:]
            return str(val).strip().upper()

        norm_ocr = _normalize_masked(ocr_masked_uid)
        norm_qr = _normalize_masked(qr_masked_uid)
        qr_readable = qr_info.get("readable", True)

        if norm_ocr and norm_qr and qr_readable:
            if norm_ocr != norm_qr:
                msg = "QR identity metadata conflicts with visual document text."
                conflicts.append(msg)
                factors.append({
                    "factor": "QR_OCR_CONFLICT",
                    "detail": msg,
                })

        # 3. Spatial Tampering Overlap with Critical Fields
        suspicious_regions = tamper_result.get("suspicious_regions") or []
        height, width = image_shape[:2] if len(image_shape) >= 2 else (500, 800)

        for region in suspicious_regions:
            if isinstance(region, dict):
                rx = float(region.get("x", 0.0))
                ry = float(region.get("y", 0.0))
                rw = float(region.get("w", 0.0))
                rh = float(region.get("h", 0.0))
            else:
                if len(region) < 4:
                    continue
                rx, ry, rw, rh = region[:4]
            # Check if region is in the bottom MRZ / identity area (y >= 0.65 * height)
            if (ry + rh) >= (0.65 * height) and rw > (0.3 * width):
                msg = "Tampering detected in document machine-readable / identity text region."
                if msg not in conflicts:
                    conflicts.append(msg)
                    factors.append({
                        "factor": "SUSPICIOUS_FIELD_TAMPERING",
                        "detail": msg,
                    })
                break

        is_consistent = len(conflicts) == 0
        return CrossSignalResult(
            is_consistent=is_consistent,
            conflicts=conflicts,
            factors=factors,
        )
