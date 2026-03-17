from src.game.payoff import payoff_for_action, payoffs_for_actions


def test_payoff_at_threshold() -> None:
    assert payoff_for_action(1, attendance=60, threshold=60) == 1
    assert payoff_for_action(0, attendance=60, threshold=60) == -1


def test_payoff_above_threshold() -> None:
    assert payoff_for_action(1, attendance=61, threshold=60) == -1
    assert payoff_for_action(0, attendance=61, threshold=60) == 1


def test_all_stay_home() -> None:
    assert payoffs_for_actions([0, 0, 0], threshold=1) == [-1, -1, -1]


def test_all_attend_over_capacity() -> None:
    assert payoffs_for_actions([1, 1, 1], threshold=2) == [-1, -1, -1]
