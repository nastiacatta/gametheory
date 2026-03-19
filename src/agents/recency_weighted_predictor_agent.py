"""
Recency-weighted predictor-selection agent with exponential forgetting and virtual-payoff scoring.

Each agent holds a bank of attendance predictors with scores that decay
over time, making the agent more responsive to recent performance.

Score update rule (virtual payoff with decay):
    s_{ij}(t+1) = lambda * s_{ij}(t) + u(implied_action_j, A_t)

where lambda in (0, 1] is the decay factor, and u = +1 if the implied
action would have won, -1 otherwise.

Lower lambda means faster forgetting of past performance and quicker
adaptation to regime changes.

Selection: argmax or softmax over decayed scores.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class RecencyWeightedPredictorAgent(BaseAgent):
    """
    Predictor-selection agent with exponential score decay (recency weighting).
    
    More responsive to recent forecast accuracy than BestPredictorAgent,
    which uses cumulative scores without decay.
    """

    def __init__(
        self,
        predictors: list[tuple[str, Predictor]] | None = None,
        lambda_decay: float = 0.95,
        selection: str = "argmax",
        beta: float = 1.0,
    ) -> None:
        """Initialize recency-weighted predictor agent.
        
        Args:
            predictors: List of (name, callable) predictor pairs.
            lambda_decay: Score decay factor in (0, 1]. Lower = faster forgetting.
            selection: "argmax" for hard selection, "softmax" for stochastic.
            beta: Inverse temperature for softmax selection (ignored if argmax).
        """
        if predictors is None:
            predictors = default_predictor_library()
        if not (0.0 < lambda_decay <= 1.0):
            raise ValueError("lambda_decay must be in (0, 1]")
        if selection not in ("argmax", "softmax"):
            raise ValueError("selection must be 'argmax' or 'softmax'")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        
        self.predictor_names: list[str] = [name for name, _ in predictors]
        self.predictors: list[Predictor] = [fn for _, fn in predictors]
        self.lambda_decay: float = lambda_decay
        self.selection: str = selection
        self.beta: float = beta
        
        self.scores: list[float] = [0.0] * len(self.predictors)
        self._last_predictions: list[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self.predictor_history: list[int] = []
        self.score_history: list[list[float]] = []

    def reset(self) -> None:
        """Reset scores and history for a new game."""
        self.scores = [0.0] * len(self.predictors)
        self._last_predictions = [0.0] * len(self.predictors)
        self._active_idx = 0
        self.predictor_history = []
        self.score_history = []

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        predictions = [
            p(context.attendance_history, context.n_players, context.threshold)
            for p in self.predictors
        ]
        self._last_predictions = predictions

        scores_arr = np.array(self.scores)
        
        if self.selection == "argmax":
            best_value = scores_arr.max()
            best_candidates = np.flatnonzero(scores_arr == best_value)
            chosen_idx = int(rng.choice(best_candidates))
        else:
            shifted = self.beta * (scores_arr - scores_arr.max())
            weights = np.exp(shifted)
            probs = weights / weights.sum()
            chosen_idx = int(rng.choice(len(self.predictors), p=probs))

        self._active_idx = chosen_idx
        self.predictor_history.append(chosen_idx)
        self.score_history.append(list(self.scores))

        return int(predictions[chosen_idx] <= context.threshold)

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        """Apply exponential decay and update scores with virtual payoffs."""
        _ = action, payoff
        overcrowded = realised_attendance > context.threshold
        for j, pred in enumerate(self._last_predictions):
            decayed = self.lambda_decay * self.scores[j]
            implied_action = int(pred <= context.threshold)
            hypothetical_payoff = (
                1 if (implied_action == 1 and not overcrowded)
                or (implied_action == 0 and overcrowded)
                else -1
            )
            self.scores[j] = decayed + hypothetical_payoff

    @property
    def active_predictor_name(self) -> str:
        return self.predictor_names[self._active_idx]

    def snapshot(self) -> dict[str, object]:
        """Return agent state for exports."""
        return {
            "agent_type": self.__class__.__name__,
            "lambda_decay": self.lambda_decay,
            "selection": self.selection,
            "beta": self.beta,
            "active_predictor": self.active_predictor_name,
            "scores": list(self.scores),
        }
