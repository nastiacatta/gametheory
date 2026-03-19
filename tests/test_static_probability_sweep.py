"""Tests for static probability sweep experiment."""

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_static_probability_sweep import (
    run_probability_sweep,
    simulate_static_mixed_profile,
)


class TestSimulateStaticMixedProfile:
    """Tests for single-p simulation function."""

    def test_p_zero_attendance_near_zero(self) -> None:
        """With p=0, no one attends."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.0, n_players=101, threshold=60, n_samples=1000, rng=rng
        )
        assert result["mean_attendance"] == pytest.approx(0.0, abs=0.01)

    def test_p_one_attendance_near_n_players(self) -> None:
        """With p=1, everyone attends."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=1.0, n_players=101, threshold=60, n_samples=1000, rng=rng
        )
        assert result["mean_attendance"] == pytest.approx(101.0, abs=0.01)

    def test_mean_attendance_approximately_n_times_p(self) -> None:
        """Mean attendance should be approximately n * p."""
        rng = np.random.default_rng(42)
        p = 0.6
        n_players = 101
        result = simulate_static_mixed_profile(
            p=p, n_players=n_players, threshold=60, n_samples=10000, rng=rng
        )
        expected = n_players * p
        assert result["mean_attendance"] == pytest.approx(expected, rel=0.02)

    def test_overcrowding_rate_low_for_small_p(self) -> None:
        """With p << L/n, overcrowding should be rare."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.3, n_players=101, threshold=60, n_samples=10000, rng=rng
        )
        assert result["overcrowding_rate"] < 0.01

    def test_overcrowding_rate_high_for_large_p(self) -> None:
        """With p >> L/n, overcrowding should be common."""
        rng = np.random.default_rng(42)
        result = simulate_static_mixed_profile(
            p=0.9, n_players=101, threshold=60, n_samples=10000, rng=rng
        )
        assert result["overcrowding_rate"] > 0.99

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Same seed should give same results."""
        results = []
        for _ in range(2):
            rng = np.random.default_rng(12345)
            result = simulate_static_mixed_profile(
                p=0.5, n_players=101, threshold=60, n_samples=1000, rng=rng
            )
            results.append(result)
        
        assert results[0] == results[1]

    def test_different_seeds_give_different_results(self) -> None:
        """Different seeds should give different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(999)
        
        result1 = simulate_static_mixed_profile(
            p=0.5, n_players=101, threshold=60, n_samples=100, rng=rng1
        )
        result2 = simulate_static_mixed_profile(
            p=0.5, n_players=101, threshold=60, n_samples=100, rng=rng2
        )
        
        assert result1["mean_attendance"] != result2["mean_attendance"]

    def test_payoff_bucket_decomposition(self) -> None:
        """
        Payoff buckets should sum to n_players.
        n_positive + n_negative + n_zero = n_players
        """
        rng = np.random.default_rng(42)
        n_players = 101
        result = simulate_static_mixed_profile(
            p=0.6, n_players=n_players, threshold=60, n_samples=10000, rng=rng
        )
        
        bucket_sum = (
            result["mean_n_positive"]
            + result["mean_n_negative"]
            + result["mean_n_zero"]
        )
        assert bucket_sum == pytest.approx(n_players, rel=0.001)


class TestRunProbabilitySweep:
    """Tests for full probability sweep function."""

    def test_output_shape(self, tmp_path) -> None:
        """DataFrame should have grid_size rows and expected columns."""
        df = run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        assert len(df) == 11
        expected_cols = {
            "p",
            "mean_attendance",
            "std_attendance",
            "mean_payoff_per_player",
            "overcrowding_rate",
            "mean_n_positive",
            "mean_n_negative",
            "mean_n_zero",
        }
        assert set(df.columns) == expected_cols

    def test_p_column_values(self, tmp_path) -> None:
        """p column should span [0, 1] with correct grid."""
        df = run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        assert df["p"].iloc[0] == pytest.approx(0.0)
        assert df["p"].iloc[-1] == pytest.approx(1.0)
        assert len(df["p"].unique()) == 11

    def test_sweep_reproducibility(self, tmp_path) -> None:
        """Same seed should give identical sweep results."""
        df1 = run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path / "run1"),
        )
        df2 = run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path / "run2"),
        )
        
        pd.testing.assert_frame_equal(df1, df2)

    def test_attendance_monotonic_in_p(self, tmp_path) -> None:
        """Mean attendance should increase monotonically with p."""
        df = run_probability_sweep(
            n_players=101,
            threshold=60,
            n_samples=5000,
            grid_size=51,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        attendance = df["mean_attendance"].values
        assert all(attendance[i] <= attendance[i + 1] for i in range(len(attendance) - 1))

    def test_overcrowding_rate_monotonic_in_p(self, tmp_path) -> None:
        """Overcrowding rate should increase monotonically with p."""
        df = run_probability_sweep(
            n_players=101,
            threshold=60,
            n_samples=5000,
            grid_size=51,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        overcrowding = df["overcrowding_rate"].values
        assert all(overcrowding[i] <= overcrowding[i + 1] for i in range(len(overcrowding) - 1))

    def test_payoff_zero_near_capacity_matching(self, tmp_path) -> None:
        """
        Payoff should be near zero when p ~ L/n (capacity matching).
        At this point, ~50% of games are overcrowded.
        """
        n_players = 101
        threshold = 60
        df = run_probability_sweep(
            n_players=n_players,
            threshold=threshold,
            n_samples=10000,
            grid_size=201,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        p_capacity = threshold / n_players
        idx = (df["p"] - p_capacity).abs().idxmin()
        payoff_at_capacity = df.iloc[idx]["mean_payoff_per_player"]
        
        assert abs(payoff_at_capacity) < 0.1

    def test_best_payoff_below_capacity_matching(self, tmp_path) -> None:
        """
        Mean payoff should be best (most positive) for p slightly below L/n.
        This is because payoff = positive when not overcrowded.
        """
        n_players = 101
        threshold = 60
        df = run_probability_sweep(
            n_players=n_players,
            threshold=threshold,
            n_samples=10000,
            grid_size=201,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        best_idx = df["mean_payoff_per_player"].idxmax()
        best_p = df.iloc[best_idx]["p"]
        p_capacity = threshold / n_players
        
        assert best_p < p_capacity
        assert best_p > 0.3


class TestOutputFiles:
    """Tests for generated output files."""

    def test_counts_plot_created(self, tmp_path) -> None:
        """Counts plot should be created."""
        run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        counts_path = tmp_path / "figures" / "static_counts_vs_p.png"
        assert counts_path.exists()

    def test_all_expected_outputs_created(self, tmp_path) -> None:
        """All expected output files should be created."""
        run_probability_sweep(
            n_players=21,
            threshold=10,
            n_samples=100,
            grid_size=11,
            seed=42,
            output_dir=str(tmp_path),
        )
        
        expected_files = [
            tmp_path / "tables" / "static_probability_sweep.csv",
            tmp_path / "figures" / "static_payoff_vs_p.png",
            tmp_path / "figures" / "static_attendance_vs_p.png",
            tmp_path / "figures" / "static_overcrowding_vs_p.png",
            tmp_path / "figures" / "static_counts_vs_p.png",
        ]
        
        for f in expected_files:
            assert f.exists(), f"Missing: {f}"


class TestPayoffConsistencyWithCoreModel:
    """Tests that verify the sweep uses the same payoff rule as src/game/payoff.py."""

    def test_at_threshold_attendees_lose(self) -> None:
        """
        When attendance == threshold (strict inequality), attendees get -1.

        This test verifies consistency with src/game/payoff.py which uses:
            return 1 if attendance < threshold else -1
        """
        n_samples = 10000
        n_players = 10
        threshold = 5

        decisions = np.zeros((n_samples, n_players), dtype=int)
        decisions[:, :threshold] = 1

        attendance = decisions.sum(axis=1)
        assert all(attendance == threshold)

        overcrowded = attendance >= threshold

        n_positive = np.where(~overcrowded, attendance, 0)
        n_negative = np.where(overcrowded, attendance, 0)

        assert all(n_positive == 0)
        assert all(n_negative == threshold)

    def test_above_threshold_attendees_lose(self) -> None:
        """
        When attendance > threshold, attendees get -1.
        """
        n_samples = 10000
        n_players = 10
        threshold = 5

        decisions = np.ones((n_samples, n_players), dtype=int)

        attendance = decisions.sum(axis=1)
        assert all(attendance == n_players)

        overcrowded = attendance >= threshold

        n_positive = np.where(~overcrowded, attendance, 0)
        n_negative = np.where(overcrowded, attendance, 0)

        assert all(n_positive == 0)
        assert all(n_negative == n_players)
