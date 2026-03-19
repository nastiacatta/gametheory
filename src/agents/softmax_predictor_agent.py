"""
Temperature-based predictor-selection agent with virtual-payoff scoring.

Same predictor bank and scoring as BestPredictorAgent, but the active
predictor is chosen stochastically via a Boltzmann / softmax distribution:

    P(j | t) = exp(beta * score_j) / sum_k exp(beta * score_k)

beta = 0  =>  uniform random choice  (pure exploration)
beta -> inf  =>  hard argmax         (pure exploitation)

Agent attends iff the chosen predictor forecasts attendance < threshold.

Scores are updated using virtual payoff: each predictor is scored by the
payoff it would have earned under the realised attendance, whether or not
it was selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class SoftmaxPredictorAgent(BaseAgent):
    """
    Inductive agent: softmax (Boltzmann) selection over predictor scores.
    """

    def __init__(
        self,
        predictors: list[tuple[str, Predictor]] | None = None,
        beta: float = 1.0,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        if beta < 0.0:
            raise ValueError("beta (inverse temperature) must be non-negative.")
        self.predictor_names: list[str] = [name for name, _ in predictors]
        self.predictors: list[Predictor] = [fn for _, fn in predictors]
        self.scores: list[float] = [0.0] * len(self.predictors)
        self.beta: float = beta
        self._last_predictions: list[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: list[int] = []

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        predictions = [
            p(context.attendance_history, context.n_players, context.threshold) for p in self.predictors
        ]
        self._last_predictions = predictions

        scores_arr = np.array(self.scores)
        shifted = self.beta * (scores_arr - scores_arr.max())
        weights = np.exp(shifted)
        probs = weights / weights.sum()

        chosen_idx = int(rng.choice(len(self.predictors), p=probs))
        self._active_idx = chosen_idx
        self.predictor_history.append(chosen_idx)

        return int(predictions[chosen_idx] < context.threshold)

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        _ = action, payoff
        overcrowded = realised_attendance >= context.threshold
        for j, pred in enumerate(self._last_predictions):
            implied_action = int(pred < context.threshold)
            hypothetical_payoff = (
                1 if (implied_action == 1 and not overcrowded)
                or (implied_action == 0 and overcrowded)
                else -1
            )
            self.scores[j] += hypothetical_payoff

    def reset(self) -> None:
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
            "beta": self.beta,
            "predictor_names": list(self.predictor_names),
            "scores": list(self.scores),
            "active_predictor": self.active_predictor_name,
        }
