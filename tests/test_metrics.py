"""Tests for analysis metrics."""

from src.analysis.metrics import (
    mean_attendance,
    overcrowding_rate,
    std_attendance,
    variance_from_threshold,
)


def test_mean_attendance() -> None:
    assert mean_attendance([50, 60, 70]) == 60.0


def test_std_attendance() -> None:
    result = std_attendance([50, 60, 70])
    assert abs(result - 8.165) < 0.01


def test_variance_from_threshold() -> None:
    result = variance_from_threshold([58, 60, 62], threshold=60)
    expected = (4 + 0 + 4) / 3
    assert abs(result - expected) < 1e-9


def test_overcrowding_rate_excludes_exact_threshold() -> None:
    """Exact threshold (A_t == L) is NOT overcrowded under weak-threshold convention."""
    assert overcrowding_rate([59, 60, 61], 60) == 1 / 3


def test_overcrowding_rate_all_below() -> None:
    assert overcrowding_rate([50, 55, 59], 60) == 0.0


def test_overcrowding_rate_all_above() -> None:
    assert overcrowding_rate([61, 70, 80], 60) == 1.0
