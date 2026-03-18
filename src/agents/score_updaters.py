"""
Score update policies for inductive predictor agents.

Two policies:
  - CumulativeScoreUpdater (non-recency):
        s_j(t+1) = s_j(t) - |forecast_j(t) - A_t|

  - RecencyScoreUpdater (recency):
        s_j(t+1) = lambda * s_j(t) - |forecast_j(t) - A_t|

Both use absolute forecast error as the loss function. The only
difference is whether past performance is exponentially discounted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScoreUpdater(Protocol):
    """Protocol for pluggable score-update rules."""

    def update(self, old_score: float, prediction: float, realised_attendance: int) -> float:
        """Return updated predictor score."""
        ...


@dataclass(frozen=True)
class CumulativeScoreUpdater:
    """
    Non-recency score update.

    s_j(t+1) = s_j(t) - |forecast_j(t) - A_t|
    """

    def update(self, old_score: float, prediction: float, realised_attendance: int) -> float:
        return old_score - abs(prediction - realised_attendance)


@dataclass(frozen=True)
class RecencyScoreUpdater:
    """
    Recency-weighted score update with exponential forgetting.

    s_j(t+1) = lambda_decay * s_j(t) - |forecast_j(t) - A_t|
    """

    lambda_decay: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_decay <= 1.0):
            raise ValueError("lambda_decay must lie in (0, 1].")

    def update(self, old_score: float, prediction: float, realised_attendance: int) -> float:
        return self.lambda_decay * old_score - abs(prediction - realised_attendance)
