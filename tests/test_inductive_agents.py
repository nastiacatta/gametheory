"""Tests for BestPredictorAgent and SoftmaxPredictorAgent."""

from __future__ import annotations

import numpy as np

from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent


def test_best_predictor_agent_chooses_action() -> None:
    agent = BestPredictorAgent(n_players=101)
    rng = np.random.default_rng(42)
    action = agent.choose_action(history=(), threshold=60, rng=rng)
    assert action in (0, 1)


def test_best_predictor_agent_updates_scores() -> None:
    agent = BestPredictorAgent(n_players=101)
    rng = np.random.default_rng(42)
    agent.choose_action(history=(55, 58), threshold=60, rng=rng)
    before = agent.scores.copy()
    agent.update(history_before=(55, 58), realised_attendance=60, realised_payoff=1)
    assert agent.scores != before


def test_best_predictor_agent_predictor_history() -> None:
    agent = BestPredictorAgent(n_players=101)
    rng = np.random.default_rng(42)
    agent.choose_action(history=(), threshold=60, rng=rng)
    assert len(agent.predictor_history) == 1


def test_softmax_predictor_agent_chooses_action() -> None:
    agent = SoftmaxPredictorAgent(n_players=101, beta=1.0)
    rng = np.random.default_rng(42)
    action = agent.choose_action(history=(), threshold=60, rng=rng)
    assert action in (0, 1)


def test_softmax_predictor_agent_rejects_negative_beta() -> None:
    import pytest
    with pytest.raises(ValueError, match="beta"):
        SoftmaxPredictorAgent(n_players=101, beta=-0.1)
