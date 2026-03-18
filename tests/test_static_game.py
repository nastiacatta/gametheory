from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.game.static_game import StaticMinorityGame


class DeterministicAgent(BaseAgent):
    def __init__(self, action: int) -> None:
        self.action = action

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        _ = context, rng
        return self.action


def test_static_game_one_round() -> None:
    agents = [
        DeterministicAgent(1),
        DeterministicAgent(1),
        DeterministicAgent(0),
        DeterministicAgent(0),
        DeterministicAgent(1),
    ]
    game = StaticMinorityGame(n_players=5, threshold=2, agents=agents, seed=1)
    result = game.play()

    assert result.actions == [1, 1, 0, 0, 1]
    assert result.attendance == 3
    assert result.overcrowded is True
    # Lecture payoff: attend + overcrowded => -1; stay home => 0
    assert result.payoffs == [-1, -1, 0, 0, -1]
    assert result.winners == []
    assert result.losers == [0, 1, 4]


def test_static_game_accepts_even_n() -> None:
    agents = [DeterministicAgent(0) for _ in range(4)]
    game = StaticMinorityGame(n_players=4, threshold=2, agents=agents, seed=1)
    result = game.play()
    assert result.attendance == 0


def test_static_game_at_threshold_is_overcrowded() -> None:
    agents = [
        DeterministicAgent(1),
        DeterministicAgent(1),
        DeterministicAgent(0),
    ]
    game = StaticMinorityGame(n_players=3, threshold=2, agents=agents, seed=1)
    result = game.play()

    assert result.attendance == 2
    assert result.overcrowded is True
    assert result.payoffs == [-1, -1, 0]
    assert result.winners == []
    assert result.losers == [0, 1]
