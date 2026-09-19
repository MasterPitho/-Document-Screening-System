"""Unit tests for the in-memory working-image preparation used by screening.

The helper must (a) pass through images that already fit the working size
without re-encoding, (b) downscale oversized images to the working size, and
(c) never materialise a full-size RGB buffer when downscaling -- doing so
spikes memory on top of the resident InsightFace/ONNX models and was the root
cause of worker OOM kills (frontend "Engine unreachable") on the 512 MB tier.
"""

from __future__ import annotations

import io

from PIL import Image

from app.main import WORKING_IMAGE_MAX_SIDE, prepare_working_image


def _jpeg_bytes(width: int, height: int, color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_in_limit_image_returns_original_bytes() -> None:
    original = _jpeg_bytes(800, 500)
    prepared = prepare_working_image(original, "document")

    assert prepared == original
    assert max(Image.open(io.BytesIO(prepared)).size) <= WORKING_IMAGE_MAX_SIDE


def test_oversized_image_is_downscaled_to_working_side() -> None:
    prepared = prepare_working_image(_jpeg_bytes(4000, 3000), "document")

    with Image.open(io.BytesIO(prepared)) as img:
        width, height = img.size
    assert max(width, height) <= WORKING_IMAGE_MAX_SIDE
    assert img.format == "JPEG"


def test_oversized_image_never_converted_to_rgb_at_native_size(monkeypatch) -> None:
    convert_calls: list[tuple[int, int]] = []

    original_convert = Image.Image.convert

    def tracked_convert(self: Image.Image, *args, **kwargs) -> Image.Image:
        convert_calls.append(self.size)
        return original_convert(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "convert", tracked_convert)

    prepare_working_image(_jpeg_bytes(4000, 3000), "document")

    assert convert_calls, "RGB conversion should still happen for oversized images"
    for size in convert_calls:
        assert max(size) <= WORKING_IMAGE_MAX_SIDE, (
            "Image was converted to RGB at native resolution before downscaling; "
            "this spikes memory (root cause of OOM worker kills)."
        )


def test_large_image_fallback_returns_original_when_decode_fails(monkeypatch) -> None:
    original = _jpeg_bytes(800, 500)

    def broken_open(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Image, "open", broken_open)

    assert prepare_working_image(original, "document") == original