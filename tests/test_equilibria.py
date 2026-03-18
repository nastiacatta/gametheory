"""Tests for static equilibrium analysis."""

import math

import pytest

from src.analysis.equilibria import (
    count_pure_ne,
    solve_symmetric_mixed_p_star,
    compute_expected_attendance_under_mixed,
    static_equilibrium_summary,
)


class TestCountPureNE:
    """Tests for pure-strategy Nash equilibrium counting."""

    def test_count_pure_ne_default_params(self) -> None:
        """NE count for n=101, L=60 should be C(101, 60)."""
        result = count_pure_ne(101, 60)
        assert result == math.comb(101, 60)

    def test_count_pure_ne_small_game(self) -> None:
        """NE count for n=5, L=2 should be C(5, 2) = 10."""
        assert count_pure_ne(5, 2) == 10

    def test_count_pure_ne_boundary_all_attend(self) -> None:
        """NE count for L=n should be 1 (all attend)."""
        assert count_pure_ne(10, 10) == 1

    def test_count_pure_ne_boundary_none_attend(self) -> None:
        """NE count for L=0 should be 1 (none attend)."""
        assert count_pure_ne(10, 0) == 1

    def test_count_pure_ne_invalid_threshold(self) -> None:
        """Should raise for threshold > n."""
        with pytest.raises(ValueError, match="threshold must be in"):
            count_pure_ne(10, 15)

    def test_count_pure_ne_negative_threshold(self) -> None:
        """Should raise for negative threshold."""
        with pytest.raises(ValueError, match="threshold must be in"):
            count_pure_ne(10, -1)


class TestSolveSymmetricMixedPStar:
    """Tests for symmetric mixed Nash equilibrium solver."""

    def test_p_star_in_unit_interval(self) -> None:
        """p* should be in (0, 1) for interior equilibrium."""
        p_star = solve_symmetric_mixed_p_star(101, 60)
        assert 0.0 < p_star < 1.0

    def test_p_star_small_game(self) -> None:
        """p* for small game should be in valid range."""
        p_star = solve_symmetric_mixed_p_star(11, 5)
        assert 0.0 < p_star < 1.0

    def test_p_star_convergence(self) -> None:
        """Check that solution satisfies equilibrium condition approximately."""
        from scipy import stats
        
        n, L = 101, 60
        p_star = solve_symmetric_mixed_p_star(n, L)
        
        cdf_val = stats.binom.cdf(L - 1, n - 1, p_star)
        assert abs(cdf_val - 0.5) < 1e-8

    def test_p_star_invalid_small_n(self) -> None:
        """Should raise for n < 2."""
        with pytest.raises(ValueError, match="n_players must be at least 2"):
            solve_symmetric_mixed_p_star(1, 0)

    def test_p_star_boundary_threshold(self) -> None:
        """Should raise for boundary thresholds (no interior equilibrium)."""
        with pytest.raises(ValueError, match="threshold must be in"):
            solve_symmetric_mixed_p_star(10, 0)
        with pytest.raises(ValueError, match="threshold must be in"):
            solve_symmetric_mixed_p_star(10, 10)


class TestComputeExpectedAttendance:
    """Tests for expected attendance computation."""

    def test_expected_attendance_formula(self) -> None:
        """E[A] = n * p*."""
        n = 101
        p_star = 0.6
        expected = compute_expected_attendance_under_mixed(n, p_star)
        assert expected == pytest.approx(60.6)

    def test_expected_attendance_bounds(self) -> None:
        """Expected attendance should be in [0, n]."""
        n = 101
        p_star = solve_symmetric_mixed_p_star(n, 60)
        expected = compute_expected_attendance_under_mixed(n, p_star)
        assert 0 <= expected <= n


class TestStaticEquilibriumSummary:
    """Tests for the summary function."""

    def test_summary_contains_all_keys(self) -> None:
        """Summary should contain all expected keys."""
        summary = static_equilibrium_summary(101, 60)
        expected_keys = {
            "n_players",
            "threshold",
            "pure_ne_count",
            "mixed_p_star",
            "expected_attendance_under_mixed",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_values_consistent(self) -> None:
        """Summary values should be internally consistent."""
        summary = static_equilibrium_summary(101, 60)
        assert summary["pure_ne_count"] == math.comb(101, 60)
        assert summary["expected_attendance_under_mixed"] == pytest.approx(
            101 * summary["mixed_p_star"]
        )
