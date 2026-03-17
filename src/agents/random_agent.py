"""
Mixed-strategy agent: attend with fixed probability p_attend (Bernoulli over {0, 1}).
"""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent, RoundContext


class RandomAgent(BaseAgent):
    """
    Mixed-strategy agent: attend with probability p_attend.
    Ignores context; uses only the provided RNG for reproducibility.
    """

    def __init__(self, p_attend: float = 0.5) -> None:
        if not (0.0 <= p_attend <= 1.0):
            raise ValueError("p_attend must be in [0, 1].")
        self.p_attend = p_attend

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        _ = context
        return int(rng.random() < self.p_attend)
