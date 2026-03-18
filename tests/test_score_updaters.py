"""Tests for score updater policies (cumulative and recency)."""

import pytest

from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater


class TestCumulativeScoreUpdater:
    """Tests for non-recency (cumulative) score updating."""

    def test_cumulative_score_update_zero_error(self) -> None:
        """Score unchanged when prediction equals realised attendance."""
        updater = CumulativeScoreUpdater()
        new_score = updater.update(old_score=-10.0, prediction=60.0, realised_attendance=60)
        assert new_score == -10.0

    def test_cumulative_score_update_positive_error(self) -> None:
        """Score decreases by absolute error when prediction is too high."""
        updater = CumulativeScoreUpdater()
        new_score = updater.update(old_score=-10.0, prediction=65.0, realised_attendance=60)
        assert new_score == -15.0

    def test_cumulative_score_update_negative_error(self) -> None:
        """Score decreases by absolute error when prediction is too low."""
        updater = CumulativeScoreUpdater()
        new_score = updater.update(old_score=-10.0, prediction=55.0, realised_attendance=60)
        assert new_score == -15.0

    def test_cumulative_from_zero(self) -> None:
        """Starting from zero, score equals negative cumulative error."""
        updater = CumulativeScoreUpdater()
        new_score = updater.update(old_score=0.0, prediction=57.0, realised_attendance=60)
        assert new_score == -3.0


class TestRecencyScoreUpdater:
    """Tests for recency-weighted score updating with exponential decay."""

    def test_recency_score_update_with_decay(self) -> None:
        """Score decays and then error is subtracted."""
        updater = RecencyScoreUpdater(lambda_decay=0.9)
        new_score = updater.update(old_score=-10.0, prediction=57.0, realised_attendance=60)
        assert new_score == pytest.approx(0.9 * (-10.0) - 3.0)
        assert new_score == -12.0

    def test_recency_score_update_zero_error(self) -> None:
        """Score only decays when prediction is perfect."""
        updater = RecencyScoreUpdater(lambda_decay=0.8)
        new_score = updater.update(old_score=-10.0, prediction=60.0, realised_attendance=60)
        assert new_score == pytest.approx(0.8 * (-10.0))
        assert new_score == -8.0

    def test_recency_from_zero(self) -> None:
        """Starting from zero, decay has no effect; score is just -error."""
        updater = RecencyScoreUpdater(lambda_decay=0.9)
        new_score = updater.update(old_score=0.0, prediction=57.0, realised_attendance=60)
        assert new_score == -3.0

    def test_recency_lambda_one_equals_cumulative(self) -> None:
        """With lambda=1, recency is equivalent to cumulative."""
        recency = RecencyScoreUpdater(lambda_decay=1.0)
        cumulative = CumulativeScoreUpdater()

        old = -15.0
        pred = 65.0
        real = 60

        assert recency.update(old, pred, real) == cumulative.update(old, pred, real)

    def test_recency_lambda_validation_zero(self) -> None:
        """Should raise for lambda_decay = 0."""
        with pytest.raises(ValueError, match="lambda_decay must lie in"):
            RecencyScoreUpdater(lambda_decay=0.0)

    def test_recency_lambda_validation_negative(self) -> None:
        """Should raise for negative lambda_decay."""
        with pytest.raises(ValueError, match="lambda_decay must lie in"):
            RecencyScoreUpdater(lambda_decay=-0.5)

    def test_recency_lambda_validation_above_one(self) -> None:
        """Should raise for lambda_decay > 1."""
        with pytest.raises(ValueError, match="lambda_decay must lie in"):
            RecencyScoreUpdater(lambda_decay=1.5)

    def test_recency_forgetting_effect(self) -> None:
        """Lower lambda causes faster forgetting of old score."""
        updater_fast = RecencyScoreUpdater(lambda_decay=0.5)
        updater_slow = RecencyScoreUpdater(lambda_decay=0.95)

        old = -100.0
        pred = 60.0
        real = 60

        new_fast = updater_fast.update(old, pred, real)
        new_slow = updater_slow.update(old, pred, real)

        assert new_fast == -50.0
        assert new_slow == -95.0
        assert abs(new_fast) < abs(new_slow)
