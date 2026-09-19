"""PDF upload support: detect PDF uploads and render the first page to a JPEG
so the existing image pipeline can screen scanned PDF documents."""

from __future__ import annotations

from typing import Optional


def is_pdf_content_type(content_type: Optional[str] = None,
                        filename: Optional[str] = None) -> bool:
    ct = (content_type or "").lower()
    name = (filename or "").lower()
    return ct == "application/pdf" or name.endswith(".pdf") or name.endswith(".PDF")


def render_first_page(pdf_bytes: bytes, *, max_dimension: int = 4096,
                      jpeg_quality: int = 80) -> Optional[bytes]:
    """Render page 1 as a JPEG at bounded resolution; None on any failure."""
    if not pdf_bytes:
        return None
    try:
        import fitz  # pymupdf
    except Exception:  # noqa: BLE001
        return None
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if document.page_count < 1:
                return None
            page = document.load_page(0)
            rect = page.rect
            max_side = max(rect.width, rect.height) or 1.0
            zoom = min(max_dimension / max_side, 2.0)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
        finally:
            document.close()
    except Exception:  # noqa: BLE001 - invalid/encrypted PDFs must never 500
        return None