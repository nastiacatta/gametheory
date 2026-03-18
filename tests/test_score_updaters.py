"""Tests for score updater policies (virtual-payoff cumulative and recency)."""

from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater


def test_non_recency_virtual_payoff_attend_and_win():
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=55.0,           # implies attend
        realised_attendance=58,    # not overcrowded
        threshold=60,
    )
    assert new_score == 1.0


def test_non_recency_virtual_payoff_attend_and_lose():
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=55.0,           # implies attend
        realised_attendance=70,    # overcrowded
        threshold=60,
    )
    assert new_score == -1.0


def test_non_recency_virtual_payoff_stay_home_is_zero():
    updater = CumulativeScoreUpdater()
    new_score = updater.update(
        old_score=0.0,
        prediction=75.0,           # implies stay home
        realised_attendance=58,
        threshold=60,
    )
    assert new_score == 0.0


def test_recency_virtual_payoff():
    updater = RecencyScoreUpdater(lambda_decay=0.9)
    new_score = updater.update(
        old_score=2.0,
        prediction=55.0,           # implies attend
        realised_attendance=70,    # overcrowded
        threshold=60,
    )
    assert new_score == 0.8        # 0.9*2 + (-1)
