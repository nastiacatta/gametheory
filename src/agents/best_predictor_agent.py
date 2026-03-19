"""
Predictor-score best-predictor agent (hard argmax) with cumulative absolute-error scoring.

Each agent holds a bank of attendance predictors and a cumulative score
for each. Every round it uses the predictor with the highest score to
forecast attendance, then attends iff the forecast <= threshold.

After the realised attendance is observed, all predictor scores are updated
using cumulative absolute forecast error:
    score_j  <-  score_j  -  |forecast_j - A_t|

This answers the question: which predictor forecasts attendance best?

Note: This uses forecast-error scoring, which differs from virtual-payoff scoring
used by VirtualPayoffPredictorAgent, SoftmaxPredictorAgent, and others. Forecast-error
scoring rewards accurate predictions regardless of whether they would have led to
profitable actions, while virtual-payoff scoring rewards predictions that would
have resulted in positive payoffs.

See also: EpsilonGreedyPredictorAgent (also uses forecast-error scoring)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class BestPredictorAgent(BaseAgent):
    """
    Arthur-inspired predictor-selection agent: hard-argmax over predictor scores.
    Uses cumulative absolute-error scoring.
    Ties are broken randomly (uniform selection among best candidates).
    """

    def __init__(
        self,
        predictors: list[tuple[str, Predictor]] | None = None,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        self.predictor_names: list[str] = [name for name, _ in predictors]
        self.predictors: list[Predictor] = [fn for _, fn in predictors]
        self.scores: list[float] = [0.0] * len(self.predictors)
        self._last_predictions: list[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: list[int] = []

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

    def snapshot(self) -> dict[str, Any]:
        """Return agent state for exports."""
        return {
            "agent_type": self.__class__.__name__,
            "predictor_names": list(self.predictor_names),
            "scores": list(self.scores),
            "active_predictor": self.active_predictor_name,
        }
