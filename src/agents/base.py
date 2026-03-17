"""
Base agent interface for the minority game.

RoundContext is passed each round so that future inductive agents can use
history; update() is a no-op for non-learning agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["RoundContext", "BaseAgent"]


@dataclass(frozen=True)
class RoundContext:
    """Immutable context for one round: index, game size, threshold, past attendances."""

    round_index: int
    n_players: int
    threshold: int
    attendance_history: Tuple[int, ...]


class BaseAgent(ABC):
    """
    Base class for agents in the minority game.

    The repeated-game hook is included now so that the same agent interface
    can be reused later for inductive strategies.
    """

    @abstractmethod
    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
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
        This is intentional for the first two coursework blocks.
        """
        return None

    def name(self) -> str:
        return self.__class__.__name__
