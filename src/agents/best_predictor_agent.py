"""
Predictor-score best-predictor agent (hard argmax).

Each agent holds a bank of attendance predictors and a cumulative accuracy
score for each.  Every round it uses the predictor with the highest score
to forecast attendance, then attends iff the forecast <= threshold.

After the realised attendance is observed, all predictor scores are updated:
    score_j  <-  score_j  -  |forecast_j - A_t|

Inspired by Arthur's predictor-based adaptation; not an exact replication.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class BestPredictorAgent(BaseAgent):
    """
    Arthur-inspired predictor-selection agent: hard-argmax over predictor scores.
    Ties are broken randomly (uniform selection among best candidates).
    """

    def __init__(
        self,
        predictors: Optional[List[Tuple[str, Predictor]]] = None,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        self.predictor_names: List[str] = [name for name, _ in predictors]
        self.predictors: List[Predictor] = [fn for _, fn in predictors]
        self.scores: List[float] = [0.0] * len(self.predictors)
        self._last_predictions: List[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: List[int] = []

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        predictions = [
            p(context.attendance_history, context.n_players, context.threshold) for p in self.predictors
        ]
        self._last_predictions = predictions

        scores_arr = np.array(self.scores)
        best_value = scores_arr.max()
        best_candidates = np.flatnonzero(scores_arr == best_value)
        best_idx = int(rng.choice(best_candidates))

        self._active_idx = best_idx
        self.predictor_history.append(best_idx)

        return int(predictions[best_idx] <= context.threshold)

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
        self.scores = [0.0] * len(self.predictors)
        self._last_predictions = [0.0] * len(self.predictors)
        self._active_idx = 0
        self.predictor_history = []

    @property
    def active_predictor_name(self) -> str:
        return self.predictor_names[self._active_idx]
