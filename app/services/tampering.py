"""
Multi-signal document tampering analysis engine.

Combines independent heuristic signals (ELA, JPEG compression inconsistency,
local noise inconsistency, edge/text sharpness inconsistency, lightweight
copy-move duplicate-region detection, and metadata presence). None of these is
a reliable forgery classifier on its own; together they produce an explainable
CLEAN / SUSPICIOUS / INCONCLUSIVE verdict that always requires human review
before any operational decision.

Explicitly NOT a guaranteed forgery detector.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from app.config import Settings


class TamperingResult:
    def __init__(self, result: Dict[str, Any]) -> None:
        self.result = result

    def as_dict(self) -> Dict[str, Any]:
        return self.result


class TamperingDetector:
    def __init__(self, settings: Settings) -> None:
        self.threshold = settings.tampering_threshold
        self.review_threshold = settings.tampering_review_threshold

    @staticmethod
    def _load_rgb(image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.load()
                rgb = img.convert("RGB")
                arr = np.array(rgb)
                return arr
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Individual signals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ela_signal(arr: np.ndarray) -> Dict[str, Any]:
        """Error Level Analysis: recompress to JPEG and measure local error."""
        try:
            image = Image.fromarray(arr)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=90)
            buffer.seek(0)
            with Image.open(buffer) as resaved:
                resaved.load()
                diff = ImageChops.difference(Image.fromarray(arr), resaved)
            extrema_value = max(
                (int(v) for item in diff.getextrema() for v in item),
                default=1,
            )
            if extrema_value == 0:
                extrema_value = 1
            scaled = ImageEnhance.Brightness(diff).enhance(255.0 / extrema_value)
            diff_arr = np.array(scaled)
            scaled.close()
            diff.close()
            mean = float(np.mean(diff_arr))
            std = float(np.std(diff_arr))
            score = min(100.0, (mean / 50.0) * 60.0 + (std / 70.0) * 40.0)
            return {"score": round(score, 2), "mean": round(mean, 2),
                    "std": round(std, 2), "suspicious": score > 45.0}
        except Exception:  # noqa: BLE001
            return {"score": 0.0, "mean": 0.0, "std": 0.0, "suspicious": False,
                    "error": "ELA analysis failed"}

    @staticmethod
    def _compression_signal(arr: np.ndarray) -> Dict[str, Any]:
        """Detect unusual recompression: compare quality-5 and quality-95 resaves."""
        try:
            image = Image.fromarray(arr)
            low_buf = io.BytesIO()
            image.save(low_buf, "JPEG", quality=5)
            low_buf.seek(0)
            with Image.open(low_buf) as low_img:
                low_arr = np.array(low_img.convert("RGB"))
            mismatch = float(np.mean(np.abs(arr.astype(np.float32) - low_arr.astype(np.float32))))
            # A very low mismatch suggests the image was already low-quality JPEG
            # (heavily compressed). A large mismatch suggests recent recompression.
            score = min(100.0, (mismatch / 40.0) * 100.0)
            return {"score": round(score, 2), "mismatch": round(mismatch, 2),
                    "suspicious": score > 60.0}
        except Exception:  # noqa: BLE001
            return {"score": 0.0, "mismatch": 0.0, "suspicious": False,
                    "error": "Compression analysis failed"}

    @staticmethod
    def _noise_signal(arr: np.ndarray) -> Dict[str, Any]:
        """Analyze local noise / residual differences across regions."""
        try:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            residual = cv2.GaussianBlur(gray, (0, 0), 2.0) - gray
            h, w = residual.shape
            blocks = []
            step = max(16, w // 8)
            for y in range(0, max(1, h - step), step):
                for x in range(0, max(1, w - step), step):
                    block = residual[y:y + step, x:x + step]
                    if block.size:
                        blocks.append(float(np.std(block)))
            if not blocks:
                return {"score": 0.0, "suspicious": False, "blocks": 0}
            std = float(np.std(blocks))
            score = min(100.0, (std / 15.0) * 100.0)
            return {"score": round(score, 2), "noise_variance": round(std, 2),
                    "blocks": len(blocks), "suspicious": score > 55.0}
        except Exception:  # noqa: BLE001
            return {"score": 0.0, "suspicious": False, "blocks": 0,
                    "error": "Noise analysis failed"}

    @staticmethod
    def _edge_signal(arr: np.ndarray) -> Dict[str, Any]:
        """Detect suspicious differences in edge sharpness across regions."""
        try:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            h, w = gray.shape
            step = max(16, w // 8)
            blocks = []
            for y in range(0, max(1, h - step), step):
                for x in range(0, max(1, w - step), step):
                    block = laplacian[y:y + step, x:x + step]
                    if block.size:
                        blocks.append(float(np.var(block)))
            if not blocks:
                return {"score": 0.0, "suspicious": False, "blocks": 0}
            # High variance across region edge strengths hints at localized edits.
            score = min(100.0, (float(np.std(blocks)) / 60.0) * 100.0)
            mean_edge = float(np.mean(blocks))
            suspicious = score > 60.0 or mean_edge > 300.0
            return {"score": round(score, 2), "mean_edge": round(mean_edge, 2),
                    "blocks": len(blocks), "suspicious": suspicious}
        except Exception:  # noqa: BLE001
            return {"score": 0.0, "suspicious": False, "blocks": 0,
                    "error": "Edge analysis failed"}

    @staticmethod
    def _copy_move_signal(arr: np.ndarray) -> Dict[str, Any]:
        """Lightweight duplicate-region (copy-move edge) detector.

        Uniform (low-texture) patches are skipped: identical blank regions are
        ordinary in documents and are not evidence of duplication.
        """
        try:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            scale = 0.25
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small = small[:, :, None] if small.ndim == 2 else small
            h, w = small.shape
            block = 32
            if h < block * 2 or w < block * 2:
                return {"score": 0.0, "suspicious": False, "blocks_compared": 0}
            xs = range(0, w - block, block // 2)
            ys = range(0, h - block, block // 2)
            patches: Dict[tuple[int, int], np.ndarray] = {}
            for y in ys:
                for x in xs:
                    patch = small[y:y + block, x:x + block]
                    if patch.size != block * block:
                        continue
                    if float(patch.std()) < 6.0:
                        continue
                    patches[(x, y)] = patch
            keys = list(patches.keys())
            duplicates = 0
            compared = 0
            for i in range(min(len(keys), 400)):
                for j in range(i + 1, min(len(keys), i + 40)):
                    compared += 1
                    a = patches[keys[i]].astype(np.float32)
                    b = patches[keys[j]].astype(np.float32)
                    if float(np.mean(np.abs(a - b))) < 2.0:
                        duplicates += 1
            if not compared:
                return {"score": 0.0, "suspicious": False, "blocks_compared": 0}
            score = min(100.0, (duplicates / compared) * 100.0)
            return {"score": round(score, 2), "duplicates": duplicates,
                    "blocks_compared": compared, "suspicious": score > 35.0}
        except Exception:  # noqa: BLE001
            return {"score": 0.0, "suspicious": False, "blocks_compared": 0,
                    "error": "Copy-move analysis failed"}

    @staticmethod
    def _metadata_signal(image_bytes: bytes) -> Dict[str, Any]:
        """Inspect EXIF / metadata presence. Absence is NOT treated as tampering."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                info = dict(img.info or {})
            # Pillow auto-populates JFIF/DPI headers on any JPEG; those are not
            # authorship metadata, so only user-authored metadata counts here.
            auto_fields = ("jfif", "dpi", "progression", "adobe", "icc_profile")
            significant = {k: v for k, v in info.items()
                           if k not in auto_fields and not k.startswith("jfif")}
            has_metadata = bool(significant)
            return {"present": has_metadata, "score": 0.0, "suspicious": False,
                    "note": "Metadata absence is not treated as tampering."}
        except Exception:  # noqa: BLE001
            return {"present": False, "score": 0.0, "suspicious": False,
                    "error": "Metadata analysis failed"}

    # ------------------------------------------------------------------ #
    # Aggregation
    # ------------------------------------------------------------------ #
    def _extract_regions(self, arr: np.ndarray) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        h, w = gray.shape
        step = max(16, w // 8)
        edge_values = {}
        for y in range(0, max(1, h - step), step):
            for x in range(0, max(1, w - step), step):
                block = laplacian[y:y + step, x:x + step]
                if block.size:
                    edge_values[(x, y)] = float(np.var(block))
        if edge_values:
            values = list(edge_values.values())
            mean = float(np.mean(values))
            std = float(np.std(values))
            threshold = mean + 1.5 * std
            for (x, y), value in edge_values.items():
                if value > threshold:
                    regions.append({"x": int(x), "y": int(y),
                                    "w": step, "h": step,
                                    "edge_var": round(value, 2)})
        return regions[:10]

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        operator = "multi-signal tampering engine"
        arr = self._load_rgb(image_bytes)
        if arr is None:
            return {
                "status": "ERROR",
                "score": 0.0,
                "confidence": 0.0,
                "signals": {
                    "ela": {"error": "unreadable"},
                    "compression": {"error": "unreadable"},
                    "noise": {"error": "unreadable"},
                    "edge": {"error": "unreadable"},
                    "copy_move": {"error": "unreadable"},
                    "metadata": {"present": False},
                },
                "suspicious_regions": [],
                "explanation": ["The image could not be decoded for tampering analysis."],
                "operator_name": operator,
            }

        signals = {}
        for name, compute in (
            ("ela", lambda: self._ela_signal(arr)),
            ("compression", lambda: self._compression_signal(arr)),
            ("noise", lambda: self._noise_signal(arr)),
            ("edge", lambda: self._edge_signal(arr)),
            ("copy_move", lambda: self._copy_move_signal(arr)),
            ("metadata", lambda: self._metadata_signal(image_bytes)),
        ):
            try:
                signals[name] = compute()
            except Exception:  # noqa: BLE001 - one failing signal must not break the rest
                signals[name] = {
                    "score": 0.0, "suspicious": False,
                    "error": f"{name} analysis failed",
                }

        explanation: List[str] = []
        suspicious_signals: List[str] = []
        for name, signal in signals.items():
            if signal.get("suspicious"):
                suspicious_signals.append(name)
                explanation.append(f"{name.upper()} signal flagged (score {signal.get('score')}).")

        # Composite score: blend each signal's score (metadata excluded).
        scored = [signals[k].get("score", 0.0)
                  for k in ("ela", "compression", "noise", "edge", "copy_move")]
        composite = float(np.mean(scored)) if scored else 0.0

        regions = self._extract_regions(arr)
        if regions:
            explanation.append(f"{len(regions)} region(s) flagged for edge irregularity; local review recommended.")

        if composite >= self.threshold and suspicious_signals:
            status = "SUSPICIOUS"
        elif composite >= self.review_threshold or suspicious_signals:
            status = "INCONCLUSIVE"
        else:
            status = "CLEAN"

        confidence = round(max(0.0, min(1.0, composite / 100.0)), 3)
        return {
            "status": status,
            "score": round(composite, 2),
            "confidence": confidence,
            "signals": signals,
            "suspicious_regions": regions,
            "explanation": explanation if explanation else [
                "No tampering signals crossed the configured review thresholds."
            ],
            "operator_name": operator,
            "analysis_type": "multi-signal heuristic",
        }
