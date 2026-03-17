from src.game.payoff import payoff_for_action, payoffs_for_actions


def test_payoff_at_threshold_attenders_win() -> None:
    assert payoff_for_action(action=1, attendance=60, threshold=60) == 1
    assert payoff_for_action(action=0, attendance=60, threshold=60) == -1


def test_payoff_above_threshold_stayers_win() -> None:
    assert payoff_for_action(action=1, attendance=61, threshold=60) == -1
    assert payoff_for_action(action=0, attendance=61, threshold=60) == 1


def test_payoff_zero_attendance() -> None:
    assert payoffs_for_actions([0, 0, 0], threshold=1) == [-1, -1, -1]


def test_payoff_full_attendance() -> None:
    assert payoffs_for_actions([1, 1, 1], threshold=2) == [-1, -1, -1]
