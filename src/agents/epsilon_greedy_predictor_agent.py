"""
Epsilon-greedy predictor-selection agent.

With probability epsilon, selects a random predictor (exploration).
With probability 1 - epsilon, selects the predictor with the highest score (exploitation).
Ties in best scores are broken randomly.

Agent attends iff the chosen predictor forecasts attendance < threshold.

This is a simple exploration-exploitation balance that is easy to explain:
    j_i(t) = uniform random predictor       with probability epsilon
           = argmax_j s_{ij}(t)             with probability 1 - epsilon
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class EpsilonGreedyPredictorAgent(BaseAgent):
    """
    Epsilon-greedy predictor-selection agent.
    Balances exploration (random predictor) and exploitation (best predictor).
    """

    def __init__(
        self,
        predictors: list[tuple[str, Predictor]] | None = None,
        epsilon: float = 0.1,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1].")

        self.predictor_names: list[str] = [name for name, _ in predictors]
        self.predictors: list[Predictor] = [fn for _, fn in predictors]
        self.scores: list[float] = [0.0] * len(self.predictors)
        self._last_predictions: list[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: list[int] = []
        self.epsilon: float = epsilon

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        predictions = [
            p(context.attendance_history, context.n_players, context.threshold)
            for p in self.predictors
        ]
        self._last_predictions = predictions

        if rng.random() < self.epsilon:
            idx = int(rng.integers(len(self.predictors)))
        else:
            scores_arr = np.array(self.scores)
            best_value = scores_arr.max()
            best_candidates = np.flatnonzero(scores_arr == best_value)
            idx = int(rng.choice(best_candidates))

        self._active_idx = idx
        self.predictor_history.append(idx)
        return int(predictions[idx] < context.threshold)

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        _ = context, action, payoff
        for j, pred in enumerate(self._last_predictions):
            self.scores[j] -= abs(pred - realised_attendance)

    def reset(self) -> None:
        """Reset scores and history for a new game."""
        self.scores = [0.0] * len(self.predictors)
        self._last_predictions = [0.0] * len(self.predictors)
        self._active_idx = 0
        self.predictor_history = []

    @property
    def active_predictor_name(self) -> str:
        return self.predictor_names[self._active_idx]

    def snapshot(self) -> dict[str, Any]:
        """Return agent state for exports."""
        return {
            "agent_type": self.__class__.__name__,
            "epsilon": self.epsilon,
            "predictor_names": list(self.predictor_names),
            "scores": list(self.scores),
            "active_predictor": self.active_predictor_name,
        }
