from __future__ import annotations

import numpy as np

from src.agents.base import BaseAgent, RoundContext
from src.analysis.metrics import compute_all_metrics
from src.game.repeated_game import RepeatedMinorityGame


class DeterministicAgent(BaseAgent):
    def __init__(self, action: int) -> None:
        self.action = action

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        _ = context, rng
        return self.action


def test_repeated_game_cumulative_payoffs_at_threshold() -> None:
    agents = [DeterministicAgent(1), DeterministicAgent(1), DeterministicAgent(0)]
    game = RepeatedMinorityGame(
        n_players=3,
        threshold=2,
        n_rounds=4,
        agents=agents,
        seed=1,
    )
    result = game.play()

    assert result.attendance_history == [2, 2, 2, 2]
    assert result.cumulative_payoffs == [4, 4, 0]

    summary = result.summary()
    assert summary["n_rounds"] == 4.0
    assert summary["mean_attendance"] == 2.0
    assert summary["overcrowding_rate"] == 0.0


def test_repeated_game_accepts_even_n() -> None:
    agents = [DeterministicAgent(0) for _ in range(4)]
    game = RepeatedMinorityGame(n_players=4, threshold=2, n_rounds=2, agents=agents, seed=1)
    result = game.play()
    assert result.attendance_history == [0, 0]


def test_repeated_game_summary_matches_compute_all_metrics() -> None:
    agents = [DeterministicAgent(1), DeterministicAgent(1), DeterministicAgent(0)]
    game = RepeatedMinorityGame(
        n_players=3,
        threshold=2,
        n_rounds=4,
        agents=agents,
        seed=1,
    )
    result = game.play()
    assert result.summary() == compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        threshold=2,
    )
