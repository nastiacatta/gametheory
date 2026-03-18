"""
Unified inductive predictor agent with pluggable score updating.

Each agent holds a bank of attendance predictors and a score for each.
Every round it uses the predictor with the highest score to forecast
attendance, then attends iff the forecast <= threshold.

After the realised attendance is observed, all predictor scores are updated
via the plugged-in ScoreUpdater using virtual payoff under the weak-threshold
El Farol convention.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library
from src.agents.score_updaters import ScoreUpdater, CumulativeScoreUpdater


class InductivePredictorAgent(BaseAgent):
    """
    Unified inductive agent: hard-argmax selection, pluggable score update.
    Selection: highest-scoring predictor wins (ties broken randomly).
    Score update: delegated to a ScoreUpdater instance.
    """

    def __init__(
        self,
        predictors: Optional[List[Tuple[str, Predictor]]] = None,
        score_updater: Optional[ScoreUpdater] = None,
    ) -> None:
        if predictors is None:
            predictors = default_predictor_library()
        if score_updater is None:
            score_updater = CumulativeScoreUpdater()

        self.predictor_names: List[str] = [name for name, _ in predictors]
        self.predictors: List[Predictor] = [fn for _, fn in predictors]
        self.score_updater: ScoreUpdater = score_updater
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

        for j, pred in enumerate(self._last_predictions):
            self.scores[j] = self.score_updater.update(
                old_score=self.scores[j],
                prediction=pred,
                realised_attendance=realised_attendance,
                threshold=context.threshold,
            )

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
            "score_updater": type(self.score_updater).__name__,
            "active_predictor": self.active_predictor_name,
            "scores": list(self.scores),
        }
