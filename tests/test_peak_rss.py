"""Unit tests for the peak-RSS diagnostic helper used in stage timings."""

from __future__ import annotations

import sys

from app.main import peak_rss_mb


def test_peak_rss_reports_value_on_posix_or_none_elsewhere() -> None:
    value = peak_rss_mb()
    if sys.platform.startswith("win"):
        assert value is None
    else:
        assert value is not None
        assert value > 0