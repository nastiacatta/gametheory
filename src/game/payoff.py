from __future__ import annotations

from typing import Iterable, List


def validate_action(action: int) -> None:
    if action not in (0, 1):
        raise ValueError("Action must be 0 (stay home) or 1 (attend).")


def attendance_from_actions(actions: Iterable[int]) -> int:
    action_list = list(actions)
    for action in action_list:
        validate_action(action)
    return sum(action_list)


def payoff_for_action(action: int, attendance: int, threshold: int) -> int:
    """
    Threshold formulation used in the coursework brief:

    - if attendance <= threshold, attenders are happy
    - if attendance > threshold, stay-home players are happy
    """
    validate_action(action)

    if attendance <= threshold:
        return 1 if action == 1 else -1
    return 1 if action == 0 else -1


def payoffs_for_actions(actions: Iterable[int], threshold: int) -> List[int]:
    action_list = list(actions)
    attendance = attendance_from_actions(action_list)
    return [
        payoff_for_action(action=action, attendance=attendance, threshold=threshold)
        for action in action_list
    ]
