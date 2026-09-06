"""
Deterministic, explainable risk engine.

Consumes MRZ validation, face recognition, tampering analysis, OCR confidence,
and module errors and produces a single human-review decision.

Guarantees:
- No technical failure (ERROR / NOT_AVAILABLE) can ever yield CLEARED.
- Every risk factor has an explicitly defined weight; an unknown factor raises
  UnknownRiskFactorError instead of being silently assigned an arbitrary weight.
- The score is a deterministic sum of configured weights, bounded 0-100.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import Settings


class UnknownRiskFactorError(ValueError):
    """Raised when a factor not present in the configured risk weights is used.

    This is the "no silent fallback" guarantee: an unrecognized factor is never
    assigned an arbitrary weight; it is a configuration/implementation error.
    """


def face_module_state(result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status == "ERROR" or status == "NOT_AVAILABLE":
        return "ERROR"
    if status == "MATCH":
        return "PASS"
    if status == "MISMATCH":
        return "FAIL"
    if status == "SKIPPED_NO_LIVE_PHOTO":
        return "NOT_AVAILABLE"
    if status in {"NO_FACE", "MULTIPLE_FACES", "LOW_CONFIDENCE", "INVALID_IMAGE"}:
        return "REVIEW"
    return "NOT_AVAILABLE"


def mrz_module_state(result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status == "OCR_FAILED":
        return "ERROR"
    if status == "VALID" and result.get("detected"):
        return "PASS"
    if status == "INVALID":
        return "FAIL"
    if status in {"MALFORMED", "NOT_DETECTED", "OCR_LOW_CONFIDENCE"}:
        return "REVIEW"
    return "NOT_AVAILABLE"


def tampering_module_state(result: Dict[str, Any]) -> str:
    status = result.get("status")
    if status == "ERROR" or status == "ANALYSIS_ERROR":
        return "ERROR"
    if status == "SUSPICIOUS":
        return "FAIL"
    if status == "INCONCLUSIVE":
        return "REVIEW"
    if status == "CLEAN":
        return "PASS"
    return "NOT_AVAILABLE"


def liveness_module_state(result: Dict[str, Any]) -> str:
    """Map a passive-liveness result to a module state.

    ``NOT_CHECKED`` is deliberately ``NOT_AVAILABLE``: a capture that was not
    assessed can never contribute to a ``CLEARED`` decision (consistent with
    the face module treating a missing live photo as uncertain).
    """
    status = result.get("liveness_status") or result.get("status")
    if status == "LIVE":
        return "PASS"
    if status == "SPOOF_DETECTED":
        return "FAIL"
    if status == "UNCERTAIN":
        return "REVIEW"
    return "NOT_AVAILABLE"


class RiskEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.weights = dict(settings.risk_weights)
        self.review_threshold, self.reject_threshold = (
            settings.risk_review_threshold,
            settings.risk_reject_threshold,
        )

    def add_factor(self, name: str, detail: str,
                   factors: List[Dict[str, Any]], score: List[int]) -> None:
        if name not in self.weights:
            raise UnknownRiskFactorError(f"Risk factor {name!r} has no configured weight.")
        weight = self.weights[name]
        factors.append({"factor": name, "weight": weight, "detail": detail})
        score[0] += weight

    def evaluate(
        self,
        *,
        mrz_result: Dict[str, Any],
        face_result: Dict[str, Any],
        tamper_result: Dict[str, Any],
        image_quality: float = 1.0,
        liveness_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        score = [0]
        factors: List[Dict[str, Any]] = []
        forced_high_risk = False

        # ---- Tampering ----
        tamper_status = tamper_result.get("status")
        if tamper_status == "SUSPICIOUS":
            self.add_factor("TAMPERING_SUSPECTED",
                            "Multi-signal tampering analysis flagged possible manipulation.", factors, score)
        elif tamper_status in {"ERROR", "ANALYSIS_ERROR"}:
            self.add_factor(
                "MODULE_ERROR", "Tampering analysis failed and requires secondary inspection.",
                factors, score)
        elif tamper_status == "INCONCLUSIVE":
            self.add_factor("TAMPERING_INCONCLUSIVE",
                            "Tampering analysis was inconclusive; secondary inspection recommended.", factors, score)

        # ---- Face ----
        face_status = face_result.get("status")
        if face_status == "MISMATCH":
            self.add_factor(
                "FACE_MISMATCH", "Document and live face embeddings did not meet the threshold.",
                factors, score)
        elif face_status == "MATCH":
            pass
        elif face_status == "SKIPPED_NO_LIVE_PHOTO":
            self.add_factor("UNKNOWN_MODULE", "No live photo supplied; face verification skipped.", factors, score)
        elif face_status == "NOT_AVAILABLE" or face_status == "ERROR":
            self.add_factor("MODULE_ERROR", f"Face recognition state: {face_status}.", factors, score)
        elif face_status == "NO_FACE":
            self.add_factor("FACE_NOT_DETECTED", "No face detected in the required image.", factors, score)
        elif face_status == "MULTIPLE_FACES":
            self.add_factor("FACE_MULTIPLE", "Multiple faces detected; exactly one required.", factors, score)
        elif face_status == "LOW_CONFIDENCE":
            self.add_factor("FACE_LOW_CONFIDENCE", "Face detection confidence or size too low.", factors, score)
        elif face_status == "INVALID_IMAGE":
            self.add_factor("FACE_NOT_DETECTED", "Face image could not be decoded.", factors, score)

        # ---- MRZ ----
        mrz_status = mrz_result.get("status")
        mrz_data = mrz_result.get("data")
        checks = mrz_data.get("checks") if isinstance(mrz_data, dict) else None
        if isinstance(checks, dict):
            for check, valid in checks.items():
                if not valid:
                    self.add_factor("MRZ_CHECKSUM_FAILURE", f"MRZ validation failed: {check}.", factors, score)
        if isinstance(mrz_data, dict) and mrz_data.get("is_expired"):
            self.add_factor("EXPIRED_DOCUMENT", "The parsed document expiry date is before today.", factors, score)
        if mrz_status == "OCR_FAILED":
            self.add_factor("MODULE_ERROR", "MRZ OCR failed; secondary inspection required.", factors, score)
        elif mrz_status == "NOT_DETECTED":
            self.add_factor("MRZ_NOT_DETECTED", "No structurally valid TD3 MRZ detected.", factors, score)
        elif mrz_status == "OCR_LOW_CONFIDENCE":
            self.add_factor("MRZ_LOW_CONFIDENCE", "MRZ extraction below confidence threshold.", factors, score)

        # ---- Image quality ----
        if image_quality < 0.05:
            self.add_factor("IMAGE_QUALITY", "Image quality is too poor for reliable analysis.", factors, score)

        # ---- Passive liveness (PAD) ----
        if liveness_result is not None:
            liveness_status = liveness_result.get("liveness_status")
            if liveness_status == "SPOOF_DETECTED":
                self.add_factor(
                    "LIVENESS_FAILED",
                    "Passive liveness flagged a presentation attack; "
                    "immediate high-risk review required.",
                    factors, score)
                forced_high_risk = True
            elif liveness_status == "UNCERTAIN":
                self.add_factor(
                    "LIVENESS_UNCERTAIN",
                    "Liveness check was inconclusive; review penalty applied.",
                    factors, score)

        module_statuses = {
            "mrz": mrz_module_state(mrz_result),
            "face": face_module_state(face_result),
            "tampering": tampering_module_state(tamper_result),
        }
        if liveness_result is not None:
            module_statuses["liveness"] = liveness_module_state(liveness_result)

        # Fail-safe gate: any module not reporting PASS (or face having no live
        # photo) forbids CLEARED.
        any_not_pass = any(state != "PASS" for state in module_statuses.values())

        risk_score = max(0, min(100, score[0]))
        if forced_high_risk or risk_score >= self.reject_threshold:
            risk_level = "HIGH_RISK"
            decision = "HIGH_RISK_REVIEW_REQUIRED"
            status_color = "RED"
        elif any_not_pass or factors:
            decision = "SECONDARY_INSPECTION_REQUIRED"
            if risk_score >= self.review_threshold:
                risk_level = "MEDIUM_RISK" if risk_score < self.reject_threshold else "HIGH_RISK"
            else:
                risk_level = "LOW_RISK"
            status_color = "YELLOW"
        else:
            risk_level = "LOW_RISK"
            decision = "CLEARED"
            status_color = "GREEN"

        reasons = [f["detail"] for f in factors]
        return {
            "score": risk_score,
            "level": risk_level,
            "decision": decision,
            "status": status_color,
            "factors": factors,
            "reasons": reasons,
            "module_statuses": module_statuses,
            "confidence": round(max(0.0, min(1.0, risk_score / 100.0)), 3),
            "explanation": (
                "Deterministic weighted sum of activated risk factors, bounded 0-100. "
                "All factors are explainable; unknown factors are treated as errors. "
                f"Review threshold {self.review_threshold}, reject threshold {self.reject_threshold}. "
                "Heuristic screening never proves authenticity and always requires human review."
            ),
        }
