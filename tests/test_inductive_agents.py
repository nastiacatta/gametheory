"""Tests for BestPredictorAgent and SoftmaxPredictorAgent."""

from __future__ import annotations

import numpy as np

from src.agents.base import RoundContext
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.predictors import default_predictor_library
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent


def test_best_predictor_agent_chooses_action() -> None:
    agent = BestPredictorAgent()
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    action = agent.choose_action(ctx, rng)
    assert action in (0, 1)


def test_best_predictor_agent_updates_scores() -> None:
    agent = BestPredictorAgent()
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=(55, 58))
    rng = np.random.default_rng(42)
    agent.choose_action(ctx, rng)
    before = agent.scores.copy()
    agent.update(ctx, action=1, realised_attendance=60, payoff=1)
    assert agent.scores != before


def test_best_predictor_agent_predictor_history() -> None:
    agent = BestPredictorAgent()
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    agent.choose_action(ctx, rng)
    assert len(agent.predictor_history) == 1


def test_softmax_predictor_agent_chooses_action() -> None:
    agent = SoftmaxPredictorAgent(beta=1.0)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    action = agent.choose_action(ctx, rng)
    assert action in (0, 1)


def test_softmax_predictor_agent_rejects_negative_beta() -> None:
    import pytest
    with pytest.raises(ValueError, match="beta"):
        SoftmaxPredictorAgent(beta=-0.1)
