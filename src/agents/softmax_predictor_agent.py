"""
Temperature-based (softmax) predictor-selection agent.

Same predictor bank and scoring as BestPredictorAgent, but the active
predictor is chosen stochastically via a Boltzmann / softmax distribution:

    P(j | t) = exp(beta * score_j) / sum_k exp(beta * score_k)

beta = 0  =>  uniform random choice  (pure exploration)
beta -> inf  =>  hard argmax         (pure exploitation)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class SoftmaxPredictorAgent(BaseAgent):
    """
    Inductive agent: softmax (Boltzmann) selection over predictor scores.
    """

    def __init__(
        self,
        predictors: Optional[List[Tuple[str, Predictor]]] = None,
        beta: float = 1.0,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        if beta < 0.0:
            raise ValueError("beta (inverse temperature) must be non-negative.")
        self.predictor_names: List[str] = [name for name, _ in predictors]
        self.predictors: List[Predictor] = [fn for _, fn in predictors]
        self.scores: List[float] = [0.0] * len(self.predictors)
        self.beta: float = beta
        self._last_predictions: List[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: List[int] = []

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

        return int(predictions[chosen_idx] <= context.threshold)

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

    @property
    def active_predictor_name(self) -> str:
        return self.predictor_names[self._active_idx]
