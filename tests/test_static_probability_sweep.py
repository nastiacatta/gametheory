"""
Tests for the static probability sweep experiment.

These tests verify the vectorised Monte Carlo simulation produces correct
boundary behaviour and reproducible results.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.run_static_probability_sweep import (
    StaticSweepConfig,
    run_probability_sweep,
    simulate_static_mixed_profile,
)


class TestSimulateStaticMixedProfile:
    """Tests for the core simulation function."""

    def test_p_zero_attendance_near_zero(self) -> None:
        """When p=0, no one attends: mean attendance should be ~0."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.0,
            n_players=101,
            threshold=60,
            n_samples=10_000,
            rng=rng,
        )
        assert result["mean_attendance"] == pytest.approx(0.0, abs=0.01)
        assert result["mean_n_positive"] == pytest.approx(0.0, abs=0.01)
        assert result["mean_n_negative"] == pytest.approx(0.0, abs=0.01)
        assert result["mean_n_zero"] == pytest.approx(101.0, abs=0.01)
        assert result["overcrowding_rate"] == pytest.approx(0.0, abs=0.001)

    def test_p_one_attendance_near_n_players(self) -> None:
        """When p=1, everyone attends: mean attendance should be ~n_players."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=1.0,
            n_players=101,
            threshold=60,
            n_samples=10_000,
            rng=rng,
        )
        assert result["mean_attendance"] == pytest.approx(101.0, abs=0.01)
        assert result["mean_n_positive"] == pytest.approx(0.0, abs=0.01)
        assert result["mean_n_negative"] == pytest.approx(101.0, abs=0.01)
        assert result["mean_n_zero"] == pytest.approx(0.0, abs=0.01)
        assert result["overcrowding_rate"] == pytest.approx(1.0, abs=0.001)

    def test_mean_attendance_approximately_n_times_p(self) -> None:
        """Mean attendance should be approximately n * p."""
        rng = np.random.default_rng(42)
        n_players = 101

        for p in [0.2, 0.4, 0.6, 0.8]:
            result = simulate_static_mixed_profile(
                p=p,
                n_players=n_players,
                threshold=60,
                n_samples=10_000,
                rng=rng,
            )
            expected = n_players * p
            assert result["mean_attendance"] == pytest.approx(expected, rel=0.05)

    def test_overcrowding_rate_low_for_small_p(self) -> None:
        """For p well below L/n, overcrowding should be rare."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.3,
            n_players=101,
            threshold=60,
            n_samples=10_000,
            rng=rng,
        )
        assert result["overcrowding_rate"] < 0.01

    def test_overcrowding_rate_high_for_large_p(self) -> None:
        """For p well above L/n, overcrowding should be common."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.9,
            n_players=101,
            threshold=60,
            n_samples=10_000,
            rng=rng,
        )
        assert result["overcrowding_rate"] > 0.99

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Results should be identical when using the same seed."""
        kwargs = dict(p=0.5, n_players=101, threshold=60, n_samples=1_000)

        rng1 = np.random.default_rng(123)
        result1 = simulate_static_mixed_profile(**kwargs, rng=rng1)

        rng2 = np.random.default_rng(123)
        result2 = simulate_static_mixed_profile(**kwargs, rng=rng2)

        assert result1 == result2

    def test_different_seeds_give_different_results(self) -> None:
        """Results should differ with different seeds (sanity check)."""
        kwargs = dict(p=0.5, n_players=101, threshold=60, n_samples=1_000)

        rng1 = np.random.default_rng(111)
        result1 = simulate_static_mixed_profile(**kwargs, rng=rng1)

        rng2 = np.random.default_rng(999)
        result2 = simulate_static_mixed_profile(**kwargs, rng=rng2)

        assert result1["mean_attendance"] != result2["mean_attendance"]

    def test_payoff_bucket_decomposition(self) -> None:
        """mean_n_positive + mean_n_negative + mean_n_zero should equal n_players."""
        rng = np.random.default_rng(42)
        n_players = 101

        result = simulate_static_mixed_profile(
            p=0.6,
            n_players=n_players,
            threshold=60,
            n_samples=10_000,
            rng=rng,
        )
        total = (
            result["mean_n_positive"]
            + result["mean_n_negative"]
            + result["mean_n_zero"]
        )
        assert total == pytest.approx(n_players, abs=0.01)


class TestRunProbabilitySweep:
    """Tests for the full sweep function."""

    def test_output_shape(self) -> None:
        """Output DataFrame should have correct number of rows and columns."""
        config = StaticSweepConfig(
            n_players=21,
            threshold=10,
            n_samples=100,
            n_grid_points=11,
            seed=42,
        )
        df = run_probability_sweep(config)

        assert len(df) == 11
        expected_cols = {
            "p",
            "mean_attendance",
            "std_attendance",
            "mean_payoff_per_player",
            "overcrowding_rate",
            "mean_fraction_going",
            "mean_n_positive",
            "mean_n_negative",
            "mean_n_zero",
        }
        assert set(df.columns) == expected_cols

    def test_p_column_values(self) -> None:
        """The p column should span [0, 1] with correct endpoints."""
        config = StaticSweepConfig(
            n_players=21,
            threshold=10,
            n_samples=100,
            n_grid_points=11,
            seed=42,
        )
        df = run_probability_sweep(config)

        assert df["p"].iloc[0] == pytest.approx(0.0)
        assert df["p"].iloc[-1] == pytest.approx(1.0)

    def test_sweep_reproducibility(self) -> None:
        """Full sweep should be reproducible with the same seed."""
        config = StaticSweepConfig(
            n_players=21,
            threshold=10,
            n_samples=500,
            n_grid_points=11,
            seed=42,
        )
        df1 = run_probability_sweep(config)
        df2 = run_probability_sweep(config)

        assert df1["mean_attendance"].tolist() == df2["mean_attendance"].tolist()
        assert df1["mean_payoff_per_player"].tolist() == df2["mean_payoff_per_player"].tolist()

    def test_attendance_monotonic_in_p(self) -> None:
        """Mean attendance should be monotonically increasing in p."""
        config = StaticSweepConfig(
            n_players=51,
            threshold=25,
            n_samples=5_000,
            n_grid_points=21,
            seed=42,
        )
        df = run_probability_sweep(config)

        attendance = df["mean_attendance"].to_numpy()
        assert np.all(np.diff(attendance) >= -0.5)

    def test_overcrowding_rate_monotonic_in_p(self) -> None:
        """Overcrowding rate should be monotonically increasing in p."""
        config = StaticSweepConfig(
            n_players=51,
            threshold=25,
            n_samples=5_000,
            n_grid_points=21,
            seed=42,
        )
        df = run_probability_sweep(config)

        overcrowding = df["overcrowding_rate"].to_numpy()
        assert np.all(np.diff(overcrowding) >= -0.05)

    def test_payoff_zero_near_capacity_matching(self) -> None:
        """Payoff should be approximately zero near p = L/n (capacity-matching)."""
        config = StaticSweepConfig(
            n_players=101,
            threshold=60,
            n_samples=10_000,
            n_grid_points=201,
            seed=42,
        )
        df = run_probability_sweep(config)

        p_cap = config.threshold / config.n_players
        near_cap = df[(df["p"] - p_cap).abs() < 0.01]
        payoff_near_cap = near_cap["mean_payoff_per_player"].mean()

        assert abs(payoff_near_cap) < 0.1

    def test_best_payoff_below_capacity_matching(self) -> None:
        """Best p for payoff should be below L/n (fewer go => less overcrowding)."""
        config = StaticSweepConfig(
            n_players=101,
            threshold=60,
            n_samples=10_000,
            n_grid_points=201,
            seed=42,
        )
        df = run_probability_sweep(config)

        best_idx = df["mean_payoff_per_player"].idxmax()
        best_p = df.loc[best_idx, "p"]
        p_cap = config.threshold / config.n_players

        assert best_p < p_cap
        assert best_p > 0.3
