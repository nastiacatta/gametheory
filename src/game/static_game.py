from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.agents.base import BaseAgent
from src.game.payoff import payoffs_for_actions


@dataclass(frozen=True)
class StaticGameResult:
    actions: List[int]
    attendance: int
    payoffs: List[int]
    winners: List[int]
    losers: List[int]
    attendance_rate: float
    overcrowded: bool


class StaticMinorityGame:
    """Single-shot El Farol threshold game."""

    def __init__(
        self,
        n_players: int,
        threshold: int,
        agents: List[BaseAgent],
        seed: int = 42,
    ) -> None:
        if n_players <= 0:
            raise ValueError("n_players must be positive.")
        if len(agents) != n_players:
            raise ValueError("Number of agents must equal n_players.")
        if not (0 <= threshold <= n_players):
            raise ValueError("threshold must lie between 0 and n_players.")

        self.n_players = n_players
        self.threshold = threshold
        self.agents = agents
        self.rng = np.random.default_rng(seed)

    def play(self) -> StaticGameResult:
        actions = [agent.choose_action(self.rng) for agent in self.agents]
        attendance = sum(actions)
        payoffs = payoffs_for_actions(actions, self.threshold)

        winners = [i for i, p in enumerate(payoffs) if p > 0]
        losers = [i for i, p in enumerate(payoffs) if p < 0]

        return StaticGameResult(
            actions=actions,
            attendance=attendance,
            payoffs=payoffs,
            winners=winners,
            losers=losers,
            attendance_rate=attendance / self.n_players,
            overcrowded=attendance > self.threshold,
        )
