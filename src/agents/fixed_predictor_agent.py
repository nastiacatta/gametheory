"""
Fixed-predictor agent: uses one predictor throughout the entire repeated game.

This agent is assigned a single predictor at construction and never switches.
It serves as a non-adaptive baseline for comparison with inductive agents
that track predictor performance and switch to the currently best one.

The decision rule is: attend iff predicted_attendance <= threshold.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor


class FixedPredictorAgent(BaseAgent):
    """
    Non-adaptive agent that uses the same predictor in every round.

    Unlike BestPredictorAgent, this agent:
    - holds exactly one predictor (not a bank)
    - never updates scores or switches predictors
    - provides a baseline for measuring the value of predictor adaptation
    """

    def __init__(self, predictor_name: str, predictor_fn: Predictor) -> None:
        """Initialize a fixed-predictor agent.

        Args:
            predictor_name: Human-readable name for reporting/snapshots.
            predictor_fn: Callable (history, n_players, threshold) -> float
                returning predicted attendance for the next round.
        """
        self.predictor_name = predictor_name
        self.predictor_fn = predictor_fn

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        """Predict attendance and attend iff predicted_attendance <= threshold.

        Args:
            context: Current game state including attendance history.
            rng: Shared RNG (unused by this deterministic agent).

        Returns:
            1 if attending, 0 if staying home.
        """
        _ = rng
        predicted_attendance = self.predictor_fn(
            context.attendance_history,
            context.n_players,
            context.threshold,
        )
        return int(predicted_attendance <= context.threshold)

    def snapshot(self) -> dict[str, Any]:
        """Return serialisable state for reporting/exports."""
        return {
            "agent_type": self.__class__.__name__,
            "predictor_name": self.predictor_name,
        }

    def name(self) -> str:
        return f"FixedPredictorAgent({self.predictor_name})"
