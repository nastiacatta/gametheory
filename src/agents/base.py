from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseAgent(ABC):
    """Base class for single-shot minority-game agents."""

    @abstractmethod
    def choose_action(self, rng: np.random.Generator) -> int:
        """
        Return 1 for attend and 0 for stay home.
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__
