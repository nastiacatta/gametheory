"""
Arthur-inspired predictor-selection agent with virtual-payoff scoring.

This agent keeps Arthur's "monitor all predictors, act on the currently best one"
structure, using virtual payoff scoring where each predictor is evaluated by the
payoff it would have earned under the realised attendance, whether or not selected.

Virtual payoff follows the strict-threshold game payoff:
    u(a, A) = +1  if a=1 and A<L   (attended, not overcrowded)
            = -1  if a=1 and A>=L  (attended, overcrowded)
            =  0  if a=0           (stayed home)

This matches the game's own payoff function where the outside option is
always neutral. Unlike forecast-error scoring (BestPredictorAgent), this
rewards predictions that would have led to profitable actions rather than
accurate attendance forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class VirtualPayoffPredictorAgent(BaseAgent):
    """
    Arthur-inspired predictor-selection agent with payoff-aligned virtual scores.

    Each round:
    - every predictor generates an attendance forecast
    - the currently highest-scoring predictor is selected (hard argmax)
    - the agent attends iff that forecast < threshold

    After realised attendance is observed, ALL monitored predictors are updated
    by the payoff they would have earned, even if they were not selected.
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

        return int(predictions[best_idx] < context.threshold)

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

            if implied_action == 0:
                hypothetical_payoff = 0
            else:
                hypothetical_payoff = 1 if not overcrowded else -1

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
            "active_predictor": self.active_predictor_name,
        }
