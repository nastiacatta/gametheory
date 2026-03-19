"""
Arthur-inspired predictor-selection agent with symmetric binary virtual payoff scoring.

This agent keeps Arthur's "monitor all predictors, act on the currently best one"
structure, using virtual payoff scoring where each predictor is evaluated by the
payoff it would have earned under the realised attendance, whether or not selected.

This uses the "Symmetric Binary" virtual payoff convention, which is closer to
Minority Game scoring where correctly predicting the minority wins:

    Symmetric Binary Convention (this agent, SoftmaxPredictorAgent,
    RecencyWeightedPredictorAgent, TurnoverPredictorAgent):
        u(a, A) = +1  if a=1 and A<=L  (attended and not overcrowded)
                = +1  if a=0 and A>L   (stayed home and was overcrowded)
                = -1  otherwise

    This rewards both correct attendance AND correct abstention.

Compare with the El Farol / Weak-Threshold Convention (InductivePredictorAgent):
        u(a, A) = +1  if a=1 and A<=L
                = -1  if a=1 and A>L
                =  0  if a=0  (stay home is always neutral)

    The El Farol convention treats staying home as a risk-free neutral option.

Also compare with Forecast-Error Scoring (BestPredictorAgent, EpsilonGreedyPredictorAgent):
    s_j <- s_j - |forecast_j - A_t|

    This directly measures prediction accuracy rather than implied payoff.

The different scoring rules answer different questions:
- Forecast-error: which predictor forecasts attendance most accurately?
- El Farol virtual payoff: which predictor makes the best attendance decisions?
- Symmetric virtual payoff: which predictor best predicts the winning action?
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
    - the agent attends iff that forecast <= threshold

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
            "active_predictor": self.active_predictor_name,
        }
