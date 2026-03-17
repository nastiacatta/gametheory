from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent


class FixedAttendanceAgent(BaseAgent):
    """
    Attend if a fixed predicted attendance is weakly below the threshold.
    """

    def __init__(self, predicted_attendance: int, threshold: int) -> None:
        self.predicted_attendance = predicted_attendance
        self.threshold = threshold

    def choose_action(self, rng: np.random.Generator) -> int:
        _ = rng
        return int(self.predicted_attendance <= self.threshold)
