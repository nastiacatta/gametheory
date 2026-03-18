""" Stage-game payoff rules for the El Farol / threshold minority game.

Weak-threshold convention:
u_i(a) = +1 if a_i = 1 and A <= L
         -1 if a_i = 1 and A > L
          0 if a_i = 0
where A = sum_i a_i is total attendance and L is the threshold (capacity).
"""

from __future__ import annotations
from dataclasses import dataclass
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
    validate_action(action)
    if action == 0:
        return 0
    return 1 if attendance <= threshold else -1


def payoffs_for_actions(actions: Iterable[int], threshold: int) -> List[int]:
    action_list = list(actions)
    attendance = attendance_from_actions(action_list)
    return [
        payoff_for_action(action=a, attendance=attendance, threshold=threshold)
        for a in action_list
    ]


@dataclass(frozen=True)
class StageOutcome:
    actions: List[int]
    attendance: int
    payoffs: List[int]
    overcrowded: bool


def build_stage_outcome(actions: Iterable[int], threshold: int) -> StageOutcome:
    action_list = list(actions)
    attendance = attendance_from_actions(action_list)
    payoffs = payoffs_for_actions(action_list, threshold)
    return StageOutcome(
        actions=action_list,
        attendance=attendance,
        payoffs=payoffs,
        overcrowded=attendance > threshold,
    )
