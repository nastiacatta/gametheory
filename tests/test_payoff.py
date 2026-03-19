import pytest

from src.game.payoff import (
    attendance_from_actions,
    build_stage_outcome,
    payoff_for_action,
    payoffs_for_actions,
    validate_action,
)


def test_validate_action_rejects_invalid() -> None:
    validate_action(0)
    validate_action(1)
    with pytest.raises(ValueError, match="Action must be 0"):
        validate_action(2)
    with pytest.raises(ValueError, match="Action must be 0"):
        validate_action(-1)


def test_attendance_from_actions_validates() -> None:
    assert attendance_from_actions([1, 0, 1]) == 2
    with pytest.raises(ValueError, match="Action must be 0"):
        attendance_from_actions([1, 2])


def test_payoff_below_threshold_attenders_win() -> None:
    assert payoff_for_action(1, attendance=59, threshold=60) == 1
    assert payoff_for_action(0, attendance=59, threshold=60) == 0


def test_payoff_at_threshold_attenders_lose() -> None:
    assert payoff_for_action(1, attendance=60, threshold=60) == -1
    assert payoff_for_action(0, attendance=60, threshold=60) == 0


def test_all_stay_home_zero_payoff() -> None:
    assert payoffs_for_actions([0, 0, 0], threshold=1) == [0, 0, 0]


def test_all_attend_over_capacity() -> None:
    assert payoffs_for_actions([1, 1, 1], threshold=2) == [-1, -1, -1]


def test_payoff_above_threshold_attenders_lose() -> None:
    assert payoff_for_action(1, attendance=61, threshold=60) == -1
    assert payoff_for_action(0, attendance=61, threshold=60) == 0


def test_stage_outcome_at_threshold_overcrowded() -> None:
    outcome = build_stage_outcome([1] * 60 + [0], threshold=60)
    assert outcome.attendance == 60
    assert outcome.overcrowded is True


def test_stage_outcome_above_threshold_overcrowded() -> None:
    outcome = build_stage_outcome([1] * 61, threshold=60)
    assert outcome.attendance == 61
    assert outcome.overcrowded is True
