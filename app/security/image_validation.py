"""
Upload security: validate image bytes before any expensive processing.

Never trust the client. We check, in order: empty body, byte size, declared
MIME type, file extension, real image signature + format match, pixel
dimensions, and decompression-bomb protection. All validation happens in
memory; no uploads are persisted.
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import HTTPException
from PIL import Image

from app.config import Settings


class ImageValidationLimits:
    def __init__(self, settings: Settings) -> None:
        self.max_bytes = settings.max_image_bytes
        self.max_pixels = settings.max_image_pixels
        self.max_width = settings.max_image_width
        self.max_height = settings.max_image_height
        self.allowed_types = settings.allowed_image_types
        self.allowed_extensions = settings.allowed_image_extensions

    _FORMAT_BY_MIME = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }

    def validate(self, image_bytes: bytes, content_type: Optional[str], label: str,
                 filename: Optional[str] = None) -> None:
        """Validate an uploaded image. Raises HTTPException on failure."""
        if not image_bytes:
            raise HTTPException(status_code=400, detail=f"{label} is empty.")
        if len(image_bytes) > self.max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{label} must be {self.max_bytes // (1024 * 1024)} MB or smaller.",
            )
        if content_type not in self.allowed_types:
            raise HTTPException(status_code=415, detail=f"{label} must be JPG, PNG, or WebP.")
        if filename:
            extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if extension not in self.allowed_extensions:
                raise HTTPException(status_code=415, detail=f"{label} must be a JPG, PNG, or WebP file.")

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                actual_format = image.format
                if actual_format is None:
                    raise HTTPException(status_code=400, detail=f"{label} is invalid or unreadable.")
                expected_formats = self._FORMAT_BY_MIME.get(content_type or "", "")
                if actual_format != expected_formats:
                    raise HTTPException(
                        status_code=415,
                        detail=f"{label} content does not match its declared type.",
                    )
                if image.width > self.max_width or image.height > self.max_height:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{label} width and height exceed safe limits.",
                    )
                if image.width * image.height > self.max_pixels:
                    raise HTTPException(status_code=413, detail=f"{label} dimensions are too large.")
                image.verify()
            # Re-open and fully decode to confirm the file is not truncated.
            with Image.open(io.BytesIO(image_bytes)) as decoded:
                decoded.load()
        except HTTPException:
            raise
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise HTTPException(status_code=413, detail=f"{label} dimensions are unsafe.") from error
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"{label} is invalid or unreadable.") from error
