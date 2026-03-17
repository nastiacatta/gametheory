"""
Single-shot (static) minority game: one round, no learning.

Returns per-player actions and payoffs for a given (n, L) and agent list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.game.payoff import attendance_from_actions, payoffs_for_actions


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

    def play(self) -> StaticGameResult:
        context = RoundContext(
            round_index=0,
            n_players=self.n_players,
            threshold=self.threshold,
            attendance_history=(),
        )

        actions = [agent.choose_action(context, self.rng) for agent in self.agents]
        attendance = attendance_from_actions(actions)
        payoffs = payoffs_for_actions(actions, self.threshold)

        winners = [i for i, payoff in enumerate(payoffs) if payoff > 0]
        losers = [i for i, payoff in enumerate(payoffs) if payoff < 0]

        return StaticGameResult(
            actions=actions,
            attendance=attendance,
            payoffs=payoffs,
            winners=winners,
            losers=losers,
            attendance_rate=attendance / self.n_players,
            overcrowded=attendance > self.threshold,
        )
