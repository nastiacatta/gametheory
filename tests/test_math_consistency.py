"""
Tests for mathematical consistency across the codebase.

Verifies:
1. Overcrowding uses A_t >= L (strict-threshold convention)
2. Total payoff identity: sum_i u_i = A if A < L, else -A
3. Theoretical benchmark formulas match expected values
"""

from __future__ import annotations

import pytest

from src.analysis.benchmarks import expected_iid_threshold_mse
from src.analysis.metrics import overcrowding_rate
from src.game.payoff import build_stage_outcome


def test_overcrowding_rate_is_strict_threshold() -> None:
    """Overcrowding requires A_t >= L."""
    assert overcrowding_rate([59, 60, 61], 60) == 2 / 3


def test_stage_total_payoff_identity_below_threshold() -> None:
    """When A < L, total payoff equals attendance."""
    stage = build_stage_outcome([1, 1, 0, 0, 0], threshold=3)
    assert stage.attendance == 2
    assert sum(stage.payoffs) == stage.attendance


def test_stage_total_payoff_identity_above_threshold() -> None:
    """When A > L, total payoff equals -attendance."""
    stage = build_stage_outcome([1, 1, 1, 1, 0], threshold=3)
    assert stage.attendance == 4
    assert sum(stage.payoffs) == -stage.attendance


def test_stage_total_payoff_identity_at_threshold() -> None:
    """At exactly A = L, attendees lose under strict threshold."""
    stage = build_stage_outcome([1, 1, 1, 0, 0], threshold=3)
    assert stage.attendance == 3
    assert stage.overcrowded
    assert sum(stage.payoffs) == -stage.attendance


def test_iid_threshold_mse_formula_when_mean_matches_threshold() -> None:
    """When p = L/n, the MSE simplifies to np(1-p)."""
    n_players = 101
    threshold = 60
    p = threshold / n_players
    theory = expected_iid_threshold_mse(n_players, p, threshold)
    expected = n_players * p * (1.0 - p)
    assert abs(theory - expected) < 1e-12


def test_iid_threshold_mse_for_default_config() -> None:
    """For n=101, L=60, p=60/101: E[(A-L)^2] ≈ 24.356."""
    n_players = 101
    threshold = 60
    p = threshold / n_players
    theory = expected_iid_threshold_mse(n_players, p, threshold)
    assert abs(theory - 24.356435643564356) < 1e-6


def test_overcrowding_all_below() -> None:
    """No overcrowding when all attendances are strictly below threshold."""
    assert overcrowding_rate([50, 55, 59], 60) == 0.0


def test_overcrowding_all_above() -> None:
    """100% overcrowding when all attendances exceed threshold."""
    assert overcrowding_rate([61, 70, 80], 60) == 1.0


@pytest.mark.parametrize(
    "actions,threshold,expected_overcrowded",
    [
        ([1, 1, 1], 3, True),
        ([1, 1, 1], 2, True),
        ([1, 1, 0], 2, True),
        ([1, 1, 0], 1, True),
        ([0, 0, 0], 1, False),
    ],
)
def test_overcrowded_flag_consistency(
    actions: list, threshold: int, expected_overcrowded: bool
) -> None:
    """Verify overcrowded flag matches A >= L."""
    stage = build_stage_outcome(actions, threshold)
    assert stage.overcrowded == expected_overcrowded
