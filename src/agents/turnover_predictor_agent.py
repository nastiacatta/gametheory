"""
Turnover predictor agent with hypothesis replacement and virtual-payoff scoring.

This agent implements a closer approximation to Arthur's (1994) description
of inductive reasoning:
  - Agents hold k predictors (hypotheses) and track their performance
  - After each round, scores are updated with exponential decay
  - If the active predictor fails for `patience` consecutive rounds,
    the worst-scoring predictor is discarded and replaced with a new one
    sampled from the master library

Score update (virtual payoff with decay):
    s_{ij}(t+1) = lambda * s_{ij}(t) + u(implied_action_j, A_t)

where u = +1 if the implied action would have won, -1 otherwise.

Turnover rule:
    If active predictor would have earned -1 payoff for `patience`
    consecutive rounds, replace the worst-scoring predictor with a
    fresh sample from the master library.

This creates a population of predictors that evolves over time,
discarding poor performers and exploring new hypotheses.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import Predictor, default_predictor_library


class TurnoverPredictorAgent(BaseAgent):
    """
    Arthur-style predictor agent with hypothesis turnover.
    
    Tracks predictor performance with decay, and replaces poorly-performing
    predictors with new samples from the master library when the active
    predictor fails repeatedly.
    """

    def __init__(
        self,
        predictors: list[tuple[str, Predictor]] | None = None,
        lambda_decay: float = 0.95,
        patience: int = 10,
        error_threshold: float = 5.0,
        master_library: list[tuple[str, Predictor]] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize turnover predictor agent.
        
        Args:
            predictors: Initial predictor bank (name, callable pairs).
            lambda_decay: Score decay factor in (0, 1].
            patience: Consecutive failures before triggering replacement.
            error_threshold: Prediction error threshold for counting as failure.
            master_library: Full library to sample replacements from.
            seed: Random seed for reproducible predictor replacement.
        """
        if predictors is None:
            predictors = default_predictor_library()[:6]
        if master_library is None:
            master_library = default_predictor_library()
        if not (0.0 < lambda_decay <= 1.0):
            raise ValueError("lambda_decay must be in (0, 1]")
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if error_threshold < 0:
            raise ValueError("error_threshold must be non-negative")
        
        self.predictor_names: list[str] = [name for name, _ in predictors]
        self.predictors: list[Predictor] = [fn for _, fn in predictors]
        self.lambda_decay: float = lambda_decay
        self.patience: int = patience
        self.error_threshold: float = error_threshold
        self.master_library: list[tuple[str, Predictor]] = master_library
        self._seed: int | None = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        
        self.scores: list[float] = [0.0] * len(self.predictors)
        self._last_predictions: list[float] = [0.0] * len(self.predictors)
        self._active_idx: int = 0
        self._consecutive_failures: int = 0
        self._replacements_count: int = 0
        
        self.predictor_history: list[int] = []
        self.replacement_events: list[int] = []

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.scores = [0.0] * len(self.predictors)
        self._last_predictions = [0.0] * len(self.predictors)
        self._active_idx = 0
        self._consecutive_failures = 0
        self._replacements_count = 0
        self.predictor_history = []
        self.replacement_events = []
        self._rng = np.random.default_rng(self._seed)

    def _get_unused_predictor(self, rng: np.random.Generator) -> tuple[str, Predictor] | None:
        """Sample a predictor from master library not currently in use."""
        current_names: set[str] = set(self.predictor_names)
        available = [
            (name, fn) for name, fn in self.master_library
            if name not in current_names
        ]
        if not available:
            return None
        idx = rng.integers(0, len(available))
        return available[idx]

    def _replace_worst_predictor(self, rng: np.random.Generator) -> bool:
        """Replace the worst-scoring predictor with a new one."""
        new_predictor = self._get_unused_predictor(rng)
        if new_predictor is None:
            return False
        
        scores_arr = np.array(self.scores)
        worst_idx = int(np.argmin(scores_arr))
        
        new_name, new_fn = new_predictor
        self.predictor_names[worst_idx] = new_name
        self.predictors[worst_idx] = new_fn
        self.scores[worst_idx] = 0.0
        
        self._replacements_count += 1
        return True

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        predictions = [
            p(context.attendance_history, context.n_players, context.threshold)
            for p in self.predictors
        ]
        self._last_predictions = predictions

        scores_arr = np.array(self.scores)
        best_value = scores_arr.max()
        best_candidates = np.flatnonzero(scores_arr == best_value)
        chosen_idx = int(rng.choice(best_candidates))

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
        """Update scores with decay and virtual payoffs, potentially replace worst predictor."""
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
        
        active_pred = self._last_predictions[self._active_idx]
        active_implied = int(active_pred <= context.threshold)
        active_payoff = (
            1 if (active_implied == 1 and not overcrowded)
            or (active_implied == 0 and overcrowded)
            else -1
        )
        
        if active_payoff == -1:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        
        if self._consecutive_failures >= self.patience:
            round_idx = len(self.predictor_history) - 1
            if self._replace_worst_predictor(self._rng):
                self.replacement_events.append(round_idx)
            self._consecutive_failures = 0

    @property
    def active_predictor_name(self) -> str:
        return self.predictor_names[self._active_idx]

    @property
    def n_replacements(self) -> int:
        return self._replacements_count

    def snapshot(self) -> dict[str, object]:
        """Return agent state for exports."""
        return {
            "agent_type": self.__class__.__name__,
            "lambda_decay": self.lambda_decay,
            "patience": self.patience,
            "error_threshold": self.error_threshold,
            "active_predictor": self.active_predictor_name,
            "n_replacements": self.n_replacements,
            "current_predictors": list(self.predictor_names),
            "scores": list(self.scores),
        }
