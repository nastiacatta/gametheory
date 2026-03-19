"""
Score update policies for inductive predictor agents.

Weak-threshold virtual payoff convention:
- attend and A <= L  -> +1
- attend and A > L   -> -1
- stay home          ->  0

For a predictor forecast \hat A_j(t), its implied action is:
- attend if \hat A_j(t) <= L
- stay home otherwise

Non-recency:
    s_j(t+1) = s_j(t) + \tilde u_j(t)

Recency:
    s_j(t+1) = lambda * s_j(t) + \tilde u_j(t)

where \tilde u_j(t) is the virtual payoff the predictor would have earned
under the realised attendance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScoreUpdater(Protocol):
    def update(
        self,
        old_score: float,
        prediction: float,
        realised_attendance: int,
        threshold: int,
    ) -> float:
        """Return updated predictor score."""


def _virtual_payoff_from_prediction(
    prediction: float,
    realised_attendance: int,
    threshold: int,
) -> float:
    """
    Weak-threshold virtual payoff.

    Staying home always yields 0.
    Attending yields +1 if A <= L, -1 if A > L.
    """
    implied_action = int(prediction <= threshold)

    if implied_action == 0:
        return 0.0

    return 1.0 if realised_attendance <= threshold else -1.0


@dataclass(frozen=True)
class CumulativeScoreUpdater:
    """
    Non-recency virtual-payoff update.

    s_j(t+1) = s_j(t) + \tilde u_j(t)
    """

    def update(
        self,
        old_score: float,
        prediction: float,
        realised_attendance: int,
        threshold: int,
    ) -> float:
        virtual_payoff = _virtual_payoff_from_prediction(
            prediction=prediction,
            realised_attendance=realised_attendance,
            threshold=threshold,
        )
        return old_score + virtual_payoff


@dataclass(frozen=True)
class RecencyScoreUpdater:
    """
    Recency-weighted virtual-payoff update.

    s_j(t+1) = lambda_decay * s_j(t) + \tilde u_j(t)
    """

    lambda_decay: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_decay <= 1.0):
            raise ValueError("lambda_decay must lie in (0, 1].")

    def update(
        self,
        old_score: float,
        prediction: float,
        realised_attendance: int,
        threshold: int,
    ) -> float:
        virtual_payoff = _virtual_payoff_from_prediction(
            prediction=prediction,
            realised_attendance=realised_attendance,
            threshold=threshold,
        )
        return self.lambda_decay * old_score + virtual_payoff
