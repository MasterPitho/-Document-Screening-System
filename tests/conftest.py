"""Shared fixtures and environment setup for the screening test suite."""

import io
import os
import tempfile

import pytest
from PIL import Image

_tmpdir = tempfile.mkdtemp(prefix="sih_screening_tests_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmpdir.replace(os.sep, '/')}/test.db")


@pytest.fixture
def jpeg_bytes():
    def _make(width=800, height=500):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
        return buffer.getvalue()

    return _make


@pytest.fixture
def png_bytes():
    def _make(width=100, height=100):
        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    return _make
