"""
Stage-game payoff rules for the El Farol / threshold minority game.

Lecture-consistent formulation (Lecture 15):

    u_i(a) =
        +1  if a_i = 1 and A <= L   (attend, not overcrowded)
        -1  if a_i = 1 and A >  L   (attend, overcrowded)
         0  if a_i = 0              (stay home)

where A = sum_i a_i is total attendance and L is the threshold (capacity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


def validate_action(action: int) -> None:
    """Raise ValueError if action is not 0 (stay home) or 1 (attend)."""
    if action not in (0, 1):
        raise ValueError("Action must be 0 (stay home) or 1 (attend).")


def attendance_from_actions(actions: Iterable[int]) -> int:
    """Sum of actions after validating each; returns total attendance A."""
    action_list = list(actions)
    for action in action_list:
        validate_action(action)
    return sum(action_list)


def payoff_for_action(action: int, attendance: int, threshold: int) -> int:
    """
    Per-player payoff under the lecture-consistent convention.

    - action = 1, A <= L  =>  +1  (attend, not overcrowded)
    - action = 1, A >  L  =>  -1  (attend, overcrowded)
    - action = 0           =>   0  (stay home, neutral)

    Raises:
        ValueError: if action is not 0 or 1.
    """
    validate_action(action)

    if action == 0:
        return 0
    return 1 if attendance <= threshold else -1


def payoffs_for_actions(actions: Iterable[int], threshold: int) -> List[int]:
    """Payoffs for a full action profile; order matches input actions."""
    action_list = list(actions)
    attendance = attendance_from_actions(action_list)
    return [
        payoff_for_action(action=a, attendance=attendance, threshold=threshold)
        for a in action_list
    ]


@dataclass(frozen=True)
class StageOutcome:
    """Outcome of a single stage game (one round)."""

    actions: List[int]
    attendance: int
    payoffs: List[int]
    overcrowded: bool


def build_stage_outcome(actions: Iterable[int], threshold: int) -> StageOutcome:
    """Build a StageOutcome from an action profile under the threshold payoff."""
    action_list = list(actions)
    attendance = attendance_from_actions(action_list)
    payoffs = payoffs_for_actions(action_list, threshold)
    return StageOutcome(
        actions=action_list,
        attendance=attendance,
        payoffs=payoffs,
        overcrowded=attendance > threshold,
    )
