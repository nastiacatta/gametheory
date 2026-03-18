"""
Single-shot (static) minority game: one round, no learning.

Returns per-player actions and payoffs for a given (n, L) and agent list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.game.payoff import build_stage_outcome


@dataclass(frozen=True)
class StaticGameResult:
    """Result of one static round: actions, attendance, payoffs, and derived lists."""

    actions: List[int]
    attendance: int
    payoffs: List[int]
    winners: List[int]
    losers: List[int]
    attendance_rate: float
    overcrowded: bool


class StaticMinorityGame:
    """
    Single-shot threshold minority game.
    All randomness is via the provided seed; agents receive RoundContext and shared RNG.
    """

    def __init__(
        self,
        n_players: int,
        threshold: int,
        agents: List[BaseAgent],
        seed: int = 42,
    ) -> None:
        if n_players <= 0:
            raise ValueError("n_players must be positive.")
        if n_players % 2 == 0:
            raise ValueError("n_players should be odd for this coursework setup.")
        if len(agents) != n_players:
            raise ValueError("Number of agents must equal n_players.")
        if not (0 <= threshold <= n_players):
            raise ValueError("threshold must be between 0 and n_players.")

        self.n_players = n_players
        self.threshold = threshold
        self.agents = agents
        self.rng = np.random.default_rng(seed)

    def play(self, history: Sequence[int] | None = None) -> StaticGameResult:
        history_tuple = tuple([] if history is None else list(history))
        context = RoundContext(
            n_players=self.n_players,
            threshold=self.threshold,
            attendance_history=history_tuple,
            round_index=None,
        )
        actions = [
            agent.choose_action(context=context, rng=self.rng)
            for agent in self.agents
        ]
        stage = build_stage_outcome(actions, self.threshold)

        winners = [i for i, payoff in enumerate(stage.payoffs) if payoff > 0]
        losers = [i for i, payoff in enumerate(stage.payoffs) if payoff < 0]

        return StaticGameResult(
            actions=stage.actions,
            attendance=stage.attendance,
            payoffs=stage.payoffs,
            winners=winners,
            losers=losers,
            attendance_rate=stage.attendance / self.n_players,
            overcrowded=stage.overcrowded,
        )
