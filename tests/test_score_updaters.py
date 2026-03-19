"""Tests for score updater policies (virtual-payoff cumulative and recency)."""

from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater


def test_non_recency_virtual_payoff_attend_and_win():
    """Attend when not overcrowded -> +1."""
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=55.0,           # implies attend
        realised_attendance=58,    # not overcrowded (A < L)
        threshold=60,
    )
    assert new_score == 1.0


def test_non_recency_virtual_payoff_attend_and_lose():
    """Attend when overcrowded -> -1."""
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=55.0,           # implies attend
        realised_attendance=70,    # overcrowded (A >= L)
        threshold=60,
    )
    assert new_score == -1.0


def test_non_recency_virtual_payoff_stay_home_is_zero():
    """Stay home always yields 0."""
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=75.0,           # implies stay home
        realised_attendance=58,    # not overcrowded
        threshold=60,
    )
    assert new_score == 0.0


def test_non_recency_stay_home_crowded_is_zero():
    """Stay home yields 0 even when it was the right call."""
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=75.0,           # implies stay home
        realised_attendance=70,    # overcrowded
        threshold=60,
    )
    assert new_score == 0.0


def test_recency_virtual_payoff():
    """Recency: decay old score, then add virtual payoff."""
    updater = RecencyScoreUpdater(lambda_decay=0.9)
    new_score = updater.update(
        old_score=2.0,
        prediction=55.0,           # implies attend
        realised_attendance=70,    # overcrowded -> -1
        threshold=60,
    )
    assert new_score == 0.8        # 0.9*2 + (-1)


def test_recency_stay_home_is_zero():
    """Recency: stay home yields 0, only decay applies."""
    updater = RecencyScoreUpdater(lambda_decay=0.5)
    new_score = updater.update(
        old_score=4.0,
        prediction=75.0,           # implies stay home -> 0
        realised_attendance=70,
        threshold=60,
    )
    assert new_score == 2.0        # 0.5*4 + 0
