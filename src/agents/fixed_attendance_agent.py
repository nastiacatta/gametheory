"""
Pure-strategy agent: attend iff predicted_attendance <= threshold (uses context.threshold).
"""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent


class FixedAttendanceAgent(BaseAgent):
    """
    Pure-strategy agent: attend if predicted_attendance <= threshold.
    Deterministic; does not use the RNG.
    """

    def __init__(self, predicted_attendance: int) -> None:
        if predicted_attendance < 0:
            raise ValueError("predicted_attendance must be non-negative.")
        self.predicted_attendance = predicted_attendance

    def choose_action(self, history, threshold: int, rng: np.random.Generator) -> int:
        _ = history, rng
        return int(self.predicted_attendance <= threshold)
