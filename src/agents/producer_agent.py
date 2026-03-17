"""
Producer (non-adaptive) agent for heterogeneous population experiments.

Models agents whose attendance decision is independent of past attendance
dynamics — analogous to 'producers' in the ecology framing of markets.
Each round the agent forms a noisy prediction of attendance and attends
if that prediction falls at or below the threshold.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent


class ProducerAgent(BaseAgent):
    """
    Non-adaptive noisy-threshold agent.

    Args:
        base_prediction: centre of the attendance estimate.
        noise_std: standard deviation of Gaussian noise added each round.
    """

    def __init__(self, base_prediction: float, noise_std: float = 5.0) -> None:
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        self.base_prediction = base_prediction
        self.noise_std = noise_std

    def choose_action(self, history, threshold: int, rng: np.random.Generator) -> int:
        _ = history
        prediction = self.base_prediction + rng.normal(0.0, self.noise_std)
        return int(prediction <= threshold)
