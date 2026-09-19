"""Unit tests for the peak-RSS diagnostic helper used in stage timings."""

from __future__ import annotations

import sys

from app.main import (
    cgroup_memory_mb,
    cgroup_oom_kills,
    peak_child_rss_mb,
    peak_rss_mb,
)


def test_peak_rss_reports_value_on_posix_or_none_elsewhere() -> None:
    value = peak_rss_mb()
    if sys.platform.startswith("win"):
        assert value is None
    else:
        assert value is not None
        assert value > 0


def test_peak_child_rss_reports_value_on_posix_or_none_elsewhere() -> None:
    value = peak_child_rss_mb()
    if sys.platform.startswith("win"):
        assert value is None
    else:
        assert value is not None
        assert value >= 0


def test_cgroup_memory_reports_value_inside_cgroup_or_none_elsewhere() -> None:
    value = cgroup_memory_mb()
    if sys.platform.startswith("win"):
        assert value is None
    else:
        assert value is None or value > 0


def test_cgroup_oom_kills_reports_counter_or_none_when_unavailable() -> None:
    value = cgroup_oom_kills()
    if sys.platform.startswith("win"):
        assert value is None
    else:
        assert value is None or value >= 0