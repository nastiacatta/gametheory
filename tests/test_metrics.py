"""Tests for analysis metrics."""

import math

from src.analysis.metrics import (
    attendance_autocorr_1,
    compute_all_metrics,
    mad_from_threshold,
    mean_attendance,
    overcrowding_rate,
    std_attendance,
    switch_rate,
    variance_from_threshold,
)
from src.game.payoff import build_stage_outcome


def test_mean_attendance() -> None:
    assert mean_attendance([50, 60, 70]) == 60.0


def test_std_attendance() -> None:
    result = std_attendance([50, 60, 70])
    assert abs(result - 8.165) < 0.01


def test_variance_from_threshold() -> None:
    result = variance_from_threshold([58, 60, 62], threshold=60)
    expected = (4 + 0 + 4) / 3
    assert abs(result - expected) < 1e-9


def test_overcrowding_rate_includes_exact_threshold() -> None:
    """Exact threshold (A_t == L) IS overcrowded under strict-threshold convention."""
    assert overcrowding_rate([59, 60, 61], 60) == 2 / 3


def test_overcrowding_rate_all_below() -> None:
    assert overcrowding_rate([50, 55, 59], 60) == 0.0


def test_overcrowding_rate_all_above() -> None:
    assert overcrowding_rate([61, 70, 80], 60) == 1.0


def test_overcrowding_rate_includes_threshold_exact() -> None:
    """Under strict-threshold: A=60, 61, 60 are overcrowded (A >= L), only A=59 is not."""
    assert overcrowding_rate([60, 61, 59, 60], 60) == 0.75


# -----------------------------------------------------------------------------
# Boundary tests for threshold consistency between StageOutcome and metrics
# -----------------------------------------------------------------------------


def test_stage_outcome_overcrowded_at_threshold() -> None:
    """StageOutcome.overcrowded is True when attendance == threshold (strict convention)."""
    actions = [1, 1, 1, 0]
    out = build_stage_outcome(actions, threshold=3)
    assert out.attendance == 3
    assert out.overcrowded is True


def test_stage_outcome_overcrowded_above_threshold() -> None:
    """StageOutcome.overcrowded is True when attendance > threshold."""
    actions = [1, 1, 1, 1]
    out = build_stage_outcome(actions, threshold=3)
    assert out.attendance == 4
    assert out.overcrowded is True


def test_overcrowding_rate_uses_greater_than_or_equal_threshold() -> None:
    """overcrowding_rate counts rounds with A_t >= L (strict-threshold convention)."""
    history = [2, 3, 4, 3]
    rate = overcrowding_rate(history, threshold=3)
    assert rate == 0.75


# -----------------------------------------------------------------------------
# Tests for mad_from_threshold
# -----------------------------------------------------------------------------


def test_mad_from_threshold_basic() -> None:
    """MAD_L = mean of |A_t - L|."""
    result = mad_from_threshold([58, 60, 62], threshold=60)
    expected = (2 + 0 + 2) / 3
    assert abs(result - expected) < 1e-9


def test_mad_from_threshold_all_at_threshold() -> None:
    """MAD is 0 when all attendance equals threshold."""
    result = mad_from_threshold([60, 60, 60], threshold=60)
    assert result == 0.0


def test_mad_from_threshold_asymmetric() -> None:
    """MAD handles asymmetric deviations correctly."""
    result = mad_from_threshold([50, 60, 70], threshold=60)
    expected = (10 + 0 + 10) / 3
    assert abs(result - expected) < 1e-9


# -----------------------------------------------------------------------------
# Tests for attendance_autocorr_1
# -----------------------------------------------------------------------------


def test_attendance_autocorr_1_perfect_positive() -> None:
    """Perfect positive autocorrelation."""
    history = [1, 2, 3, 4, 5]
    result = attendance_autocorr_1(history)
    assert abs(result - 1.0) < 0.01


def test_attendance_autocorr_1_perfect_negative() -> None:
    """Perfect negative autocorrelation (alternating pattern)."""
    history = [0, 100, 0, 100, 0, 100]
    result = attendance_autocorr_1(history)
    assert abs(result - (-1.0)) < 0.01


def test_attendance_autocorr_1_zero_variance() -> None:
    """Constant series has zero autocorrelation (by convention)."""
    history = [60, 60, 60, 60]
    result = attendance_autocorr_1(history)
    assert result == 0.0


def test_attendance_autocorr_1_short_history() -> None:
    """History too short returns NaN."""
    history = [60]
    result = attendance_autocorr_1(history)
    assert math.isnan(result)


# -----------------------------------------------------------------------------
# Tests for switch_rate
# -----------------------------------------------------------------------------


def test_switch_rate_no_switches() -> None:
    """No switches when all agents use same predictor."""
    histories = [[0, 0, 0], [1, 1, 1]]
    result = switch_rate(histories)
    assert result == 0.0


def test_switch_rate_all_switches() -> None:
    """Every agent switches every round."""
    histories = [[0, 1, 0], [1, 0, 1]]
    result = switch_rate(histories)
    expected = 4 / (2 * 2)
    assert abs(result - expected) < 1e-9


def test_switch_rate_partial_switches() -> None:
    """Partial switch rate calculation."""
    histories = [[0, 0, 1], [0, 1, 1]]
    result = switch_rate(histories)
    expected = 2 / (2 * 2)
    assert abs(result - expected) < 1e-9


def test_switch_rate_empty_list() -> None:
    """Empty predictor histories returns NaN."""
    result = switch_rate([])
    assert math.isnan(result)


def test_switch_rate_single_round() -> None:
    """Single round (no possible switches) returns 0."""
    histories = [[0], [1]]
    result = switch_rate(histories)
    assert result == 0.0


# -----------------------------------------------------------------------------
# Tests for compute_all_metrics
# -----------------------------------------------------------------------------


def test_compute_all_metrics_basic() -> None:
    """compute_all_metrics returns expected keys."""
    attendance = [55, 60, 65, 60, 55]
    payoffs = [10, 20, -5, 15, 10]
    threshold = 60
    
    result = compute_all_metrics(attendance, payoffs, threshold)
    
    expected_keys = {
        "n_rounds", "mean_attendance", "std_attendance",
        "variance_from_threshold", "mad_from_threshold", "overcrowding_rate",
        "mean_cumulative_payoff", "std_cumulative_payoff",
        "min_cumulative_payoff", "max_cumulative_payoff",
        "mean_payoff_per_round", "attendance_autocorr_1"
    }
    assert expected_keys <= set(result.keys())


def test_compute_all_metrics_with_predictor_histories() -> None:
    """compute_all_metrics includes switch_rate when predictor_histories provided."""
    attendance = [55, 60, 65]
    payoffs = [10, 20, -5]
    threshold = 60
    predictor_histories = [[0, 0, 1], [1, 1, 0]]
    
    result = compute_all_metrics(
        attendance, payoffs, threshold,
        predictor_histories=predictor_histories
    )
    
    assert "switch_rate" in result
    assert result["switch_rate"] == 0.5


def test_compute_all_metrics_values_consistent() -> None:
    """Values computed by compute_all_metrics match individual functions."""
    attendance = [58, 60, 62, 61, 59]
    payoffs = [5, 10, -5, 0, 5]
    threshold = 60
    
    result = compute_all_metrics(attendance, payoffs, threshold)
    
    assert result["mean_attendance"] == mean_attendance(attendance)
    assert result["variance_from_threshold"] == variance_from_threshold(attendance, threshold)
    assert result["mad_from_threshold"] == mad_from_threshold(attendance, threshold)
    assert result["overcrowding_rate"] == overcrowding_rate(attendance, threshold)
