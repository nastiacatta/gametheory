"""
Base agent interface for the El Farol threshold game.

Agents observe the attendance history and may adapt between rounds using
Arthur-inspired inductive heuristics. Each agent receives a round context
containing the shared state they may condition on.
"""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

__all__ = ["RoundContext", "BaseAgent"]


@dataclass(frozen=True)
class RoundContext:
    """
    Immutable per-round context passed to agents.

    Attributes:
        n_players: total number of players in the game.
        threshold: attendance threshold L.
        attendance_history: realised attendance history before the current round.
        round_index: 0-based round index for repeated play; None for single-shot.
    """

    n_players: int
    threshold: int
    attendance_history: tuple[int, ...]
    round_index: int | None = None


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(
        self,
        context: RoundContext,
        rng: np.random.Generator,
    ) -> int:
        """
        Return 1 for attend, 0 for stay home.
        """
        raise NotImplementedError

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        """
        Default: no learning or adaptation.
        """
        _ = context, action, realised_attendance, payoff
        return None

    def reset(self) -> None:
        """Reset any internal state before a new repeated-game run."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return serialisable state for reporting/exports."""
        return {"agent_type": self.__class__.__name__}

    def name(self) -> str:
        return self.__class__.__name__
