"""Tests for advanced inductive agents (recency-weighted)."""

import numpy as np
import pytest

from src.agents.base import RoundContext
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent


class TestRecencyWeightedPredictorAgent:
    """Tests for recency-weighted predictor agent with exponential forgetting."""

    def test_recency_agent_attends_at_threshold(self) -> None:
        """Agent should attend when prediction equals threshold."""
        predictors = [("always_threshold", lambda h, n, L: float(L))]
        agent = RecencyWeightedPredictorAgent(predictors=predictors, lambda_decay=0.9)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 1

    def test_recency_agent_stays_home_above_threshold(self) -> None:
        """Agent should stay home when prediction exceeds threshold."""
        predictors = [("above_threshold", lambda h, n, L: float(L + 5))]
        agent = RecencyWeightedPredictorAgent(predictors=predictors, lambda_decay=0.9)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 0

    def test_recency_scores_decay(self) -> None:
        """Scores should decay by lambda_decay factor each round."""
        predictors = [
            ("good", lambda h, n, L: 60.0),
            ("bad", lambda h, n, L: 80.0),
        ]
        lambda_decay = 0.8
        agent = RecencyWeightedPredictorAgent(predictors=predictors, lambda_decay=lambda_decay)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        agent.choose_action(ctx, rng)
        agent.update(ctx, action=1, realised_attendance=60, payoff=1)
        
        assert agent.scores[0] == pytest.approx(0.8 * 0.0 - 0.0)
        assert agent.scores[1] == pytest.approx(0.8 * 0.0 - 20.0)
        
        agent.choose_action(ctx, rng)
        agent.update(ctx, action=1, realised_attendance=60, payoff=1)
        
        assert agent.scores[0] == pytest.approx(0.8 * 0.0 - 0.0)
        assert agent.scores[1] == pytest.approx(0.8 * (-20.0) - 20.0)

    def test_recency_forgetting_effect(self) -> None:
        """Old errors should be forgotten over time with low lambda."""
        predictors = [
            ("pred_a", lambda h, n, L: float(L)),
            ("pred_b", lambda h, n, L: float(L)),
        ]
        agent = RecencyWeightedPredictorAgent(predictors=predictors, lambda_decay=0.5)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        agent.scores = [-100.0, 0.0]
        
        for _ in range(10):
            agent.choose_action(ctx, rng)
            agent.update(ctx, action=1, realised_attendance=60, payoff=1)
        
        assert abs(agent.scores[0] - agent.scores[1]) < 1.0

    def test_recency_softmax_selection(self) -> None:
        """Softmax selection should select predictors stochastically."""
        predictors = [
            ("pred_a", lambda h, n, L: float(L - 5)),
            ("pred_b", lambda h, n, L: float(L)),
            ("pred_c", lambda h, n, L: float(L + 5)),
        ]
        agent = RecencyWeightedPredictorAgent(
            predictors=predictors,
            lambda_decay=0.95,
            selection="softmax",
            beta=0.1,
        )
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        choices = [agent.choose_action(ctx, rng) for _ in range(100)]
        assert 0 in choices and 1 in choices

    def test_recency_invalid_lambda(self) -> None:
        """Should raise for invalid lambda_decay."""
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            RecencyWeightedPredictorAgent(lambda_decay=0.0)
        with pytest.raises(ValueError, match="lambda_decay must be in"):
            RecencyWeightedPredictorAgent(lambda_decay=1.5)


class TestReproducibility:
    """Tests for reproducibility under fixed seeds."""

    def test_recency_agent_reproducible(self) -> None:
        """Same seed should produce same results."""
        predictors = [
            ("a", lambda h, n, L: float(L - 5)),
            ("b", lambda h, n, L: float(L + 5)),
        ]
        
        results = []
        for _ in range(2):
            agent = RecencyWeightedPredictorAgent(
                predictors=predictors,
                lambda_decay=0.9,
                selection="softmax",
                beta=1.0,
            )
            ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
            rng = np.random.default_rng(12345)
            actions = [agent.choose_action(ctx, rng) for _ in range(50)]
            results.append(actions)
        
        assert results[0] == results[1]
