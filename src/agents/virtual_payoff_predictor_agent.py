"""
Arthur-inspired predictor-selection agent with payoff-aligned virtual scores.

This agent keeps Arthur's "monitor all predictors, act on the currently best one"
structure, but replaces forecast-error scoring with virtual payoff scoring.
Each predictor is evaluated by the payoff it would have earned under the
realised attendance, whether or not it was selected.

This is closer to the Minority Game notion of virtual scores, where strategies
are scored by whether they would have won, even if they were not actually played.

Virtual score update:
    s_j <- s_j + u(a_j_pred, A_t)

where
    a_j_pred = 1{ forecast_j <= threshold }

and, under the standard El Farol payoff convention,
    u(a, A) = +1  if a=1 and A<=L  (attended and not overcrowded)
            = +1  if a=0 and A>L   (stayed home and was overcrowded)
            = -1  otherwise

This differs from BestPredictorAgent which uses forecast-error scoring:
    s_j <- s_j - |forecast_j - A_t|

The two answer different questions:
- BestPredictorAgent: which predictor forecasts attendance best?
- VirtualPayoffPredictorAgent: which predictor would have earned the best payoff?
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class VirtualPayoffPredictorAgent(BaseAgent):
    """
    Arthur-inspired predictor-selection agent with payoff-aligned virtual scores.

    Each round:
    - every predictor generates an attendance forecast
    - the currently highest-scoring predictor is selected (hard argmax)
    - the agent attends iff that forecast <= threshold

    After realised attendance is observed, ALL monitored predictors are updated
    by the payoff they would have earned, even if they were not selected.
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
            p(context.attendance_history, context.n_players, context.threshold)
            for p in self.predictors
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
        _ = action, payoff

        overcrowded = realised_attendance > context.threshold

        for j, pred in enumerate(self._last_predictions):
            implied_action = int(pred <= context.threshold)

            hypothetical_payoff = (
                1
                if (implied_action == 1 and not overcrowded)
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
        return {
            "agent_type": self.__class__.__name__,
            "predictor_names": list(self.predictor_names),
            "scores": list(self.scores),
            "active_predictor_name": self.active_predictor_name,
        }
