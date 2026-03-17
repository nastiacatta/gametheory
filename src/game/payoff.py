from __future__ import annotations

from typing import Iterable, List


def payoff_for_action(action: int, attendance: int, threshold: int) -> int:
    """
    Compute a single player's payoff.

    action:
        1 means attend
        0 means stay home
    """
    if action not in (0, 1):
        raise ValueError("action must be 0 or 1.")

    if attendance <= threshold:
        return 1 if action == 1 else -1
    return 1 if action == 0 else -1


def payoffs_for_actions(actions: Iterable[int], threshold: int) -> List[int]:
    """Compute payoffs for a full action profile."""
    action_list = list(actions)
    attendance = sum(action_list)
    return [
        payoff_for_action(action=a, attendance=attendance, threshold=threshold)
        for a in action_list
    ]
