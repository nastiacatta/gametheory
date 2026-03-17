from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    """Attend with probability p."""

    def __init__(self, p_attend: float = 0.5) -> None:
        if not (0.0 <= p_attend <= 1.0):
            raise ValueError("p_attend must be between 0 and 1.")
        self.p_attend = p_attend

    def choose_action(self, rng: np.random.Generator) -> int:
        return int(rng.random() < self.p_attend)
