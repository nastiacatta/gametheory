"""Tests for inductive (predictor-based) agents."""

import numpy as np

from src.agents.base import RoundContext
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent


def test_best_predictor_agent_attends_at_threshold() -> None:
    """Agent should attend when prediction equals threshold (weak-threshold convention)."""
    predictors = [("always_threshold", lambda history, n, L: float(L))]
    agent = BestPredictorAgent(predictors=predictors)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    assert agent.choose_action(ctx, rng) == 1


def test_softmax_predictor_agent_attends_at_threshold() -> None:
    """Agent should attend when prediction equals threshold (weak-threshold convention)."""
    predictors = [("always_threshold", lambda history, n, L: float(L))]
    agent = SoftmaxPredictorAgent(predictors=predictors, beta=1.0)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    assert agent.choose_action(ctx, rng) == 1


def test_best_predictor_agent_stays_home_above_threshold() -> None:
    """Agent should stay home when prediction exceeds threshold."""
    predictors = [("above_threshold", lambda history, n, L: float(L + 1))]
    agent = BestPredictorAgent(predictors=predictors)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    assert agent.choose_action(ctx, rng) == 0


def test_softmax_predictor_agent_stays_home_above_threshold() -> None:
    """Agent should stay home when prediction exceeds threshold."""
    predictors = [("above_threshold", lambda history, n, L: float(L + 1))]
    agent = SoftmaxPredictorAgent(predictors=predictors, beta=1.0)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    assert agent.choose_action(ctx, rng) == 0


def test_best_predictor_agent_attends_below_threshold() -> None:
    """Agent should attend when prediction is below threshold."""
    predictors = [("below_threshold", lambda history, n, L: float(L - 1))]
    agent = BestPredictorAgent(predictors=predictors)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)
    assert agent.choose_action(ctx, rng) == 1


def test_best_predictor_agent_update_scores() -> None:
    """Predictor scores should be updated based on prediction error."""
    predictors = [
        ("good", lambda history, n, L: 60.0),
        ("bad", lambda history, n, L: 80.0),
    ]
    agent = BestPredictorAgent(predictors=predictors)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)

    agent.choose_action(ctx, rng)
    agent.update(ctx, action=1, realised_attendance=60, payoff=1)

    assert agent.scores[0] == 0.0
    assert agent.scores[1] == -20.0


def test_softmax_predictor_agent_with_multiple_predictors() -> None:
    """Softmax agent should select from multiple predictors stochastically."""
    predictors = [
        ("pred_a", lambda history, n, L: float(L - 5)),
        ("pred_b", lambda history, n, L: float(L)),
        ("pred_c", lambda history, n, L: float(L + 5)),
    ]
    agent = SoftmaxPredictorAgent(predictors=predictors, beta=0.0)
    ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
    rng = np.random.default_rng(42)

    actions = [agent.choose_action(ctx, rng) for _ in range(100)]
    assert 0 in actions and 1 in actions
