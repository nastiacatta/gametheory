"""
Hybrid variant: Nash-initialised fixed-predictor agent.

Uses the symmetric mixed-strategy Nash equilibrium probability to decide
attendance when the attendance history is too short for the assigned predictor,
then falls back to the fixed predictor for all subsequent rounds.

This is NOT part of the core "Repeated Fixed Strategy" baseline.  It is a
separate robustness check labelled distinctly so that the fixed-strategy
definition stays clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base import RoundContext
from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.predictors import Predictor
from src.analysis.equilibria import solve_symmetric_mixed_p_star


class NashInitialisedFixedPredictorAgent(FixedPredictorAgent):
    """
    Fixed-predictor agent that plays the symmetric mixed Nash equilibrium
    when attendance history is empty, then switches permanently to its
    assigned predictor once history is available.

    The full contingent rule is chosen ex ante and never updated, so this
    is still a fixed *policy* in the game-theoretic sense, but it is not a
    pure fixed-predictor strategy because the decision rule changes with
    the history length.
    """

    def __init__(
        self,
        predictor_name: str,
        predictor_fn: Predictor,
        p_star: float | None = None,
        n_players: int | None = None,
        threshold: int | None = None,
    ) -> None:
        super().__init__(predictor_name=predictor_name, predictor_fn=predictor_fn)
        if p_star is not None:
            self._p_star = p_star
        elif n_players is not None and threshold is not None:
            self._p_star = solve_symmetric_mixed_p_star(n_players, threshold)
        else:
            raise ValueError(
                "Supply either p_star or both n_players and threshold."
            )

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        if not context.attendance_history:
            return int(rng.random() < self._p_star)
        return super().choose_action(context, rng)

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        base["p_star"] = self._p_star
        return base

    def name(self) -> str:
        return f"NashInitFixedPredictor({self.predictor_name})"
