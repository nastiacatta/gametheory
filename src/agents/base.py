"""
Base agent interface for the minority game.

The coursework brief motivates repeated play where agents observe the
attendance history and may adapt between rounds (Arthur-style inductive
predictors). For that reason, the core interface is history-aware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

__all__ = ["BaseAgent"]


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(
        self,
        history: Sequence[int],
        threshold: int,
        rng: np.random.Generator,
    ) -> int:
        """
        Return 1 for attend, 0 for stay home.
        """
        raise NotImplementedError

    def update(
        self,
        history_before: Sequence[int],
        realised_attendance: int,
        realised_payoff: int,
    ) -> None:
        """
        Default: no learning or adaptation.
        """
        return None

    def reset(self) -> None:
        """Reset any internal state before a new repeated-game run."""
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return serialisable state for reporting/exports."""
        return {"agent_type": self.__class__.__name__}

    def name(self) -> str:
        return self.__class__.__name__
