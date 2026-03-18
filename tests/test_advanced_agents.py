"""Tests for advanced inductive agents (recency-weighted and turnover)."""

import numpy as np
import pytest

from src.agents.base import RoundContext
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent
from src.agents.turnover_predictor_agent import TurnoverPredictorAgent


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


class TestTurnoverPredictorAgent:
    """Tests for turnover predictor agent with hypothesis replacement."""

    def test_turnover_agent_attends_at_threshold(self) -> None:
        """Agent should attend when prediction equals threshold."""
        predictors = [("always_threshold", lambda h, n, L: float(L))]
        master_lib = [("always_threshold", lambda h, n, L: float(L))]
        agent = TurnoverPredictorAgent(
            predictors=predictors,
            patience=5,
            master_library=master_lib,
        )
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 1

    def test_turnover_replaces_after_patience(self) -> None:
        """Agent should replace worst predictor after patience failures."""
        bad_predictor = ("always_wrong", lambda h, n, L: float(L + 20))
        good_predictor = ("always_right", lambda h, n, L: float(L))
        replacement = ("replacement", lambda h, n, L: float(L))
        
        agent = TurnoverPredictorAgent(
            predictors=[bad_predictor, good_predictor],
            patience=3,
            error_threshold=5.0,
            lambda_decay=0.95,
            master_library=[replacement],
        )
        
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        agent.scores = [100.0, -100.0]
        
        for i in range(5):
            agent.choose_action(ctx, rng)
            agent.update(
                context=ctx,
                action=0,
                realised_attendance=60,
                payoff=-1,
                rng=rng,
            )
        
        assert agent.n_replacements >= 1

    def test_turnover_no_replacement_if_good(self) -> None:
        """Agent should not replace predictors if active one is performing well."""
        good_predictor = ("good", lambda h, n, L: float(L))
        agent = TurnoverPredictorAgent(
            predictors=[good_predictor],
            patience=3,
            error_threshold=5.0,
            master_library=[("other", lambda h, n, L: float(L + 10))],
        )
        
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            agent.choose_action(ctx, rng)
            agent.update(
                context=ctx,
                action=1,
                realised_attendance=62,
                payoff=1,
                rng=rng,
            )
        
        assert agent.n_replacements == 0

    def test_turnover_scores_decay(self) -> None:
        """Scores should decay by lambda_decay factor."""
        predictors = [("pred", lambda h, n, L: 60.0)]
        lambda_decay = 0.9
        agent = TurnoverPredictorAgent(
            predictors=predictors,
            lambda_decay=lambda_decay,
            patience=100,
            master_library=[],
        )
        
        agent.scores = [10.0]
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        agent.choose_action(ctx, rng)
        agent.update(ctx, action=1, realised_attendance=60, payoff=1, rng=rng)
        
        assert agent.scores[0] == pytest.approx(0.9 * 10.0 - 0.0)

    def test_turnover_invalid_patience(self) -> None:
        """Should raise for invalid patience."""
        with pytest.raises(ValueError, match="patience must be at least 1"):
            TurnoverPredictorAgent(patience=0)

    def test_turnover_replacement_events_tracked(self) -> None:
        """Replacement events should be tracked."""
        bad = ("bad", lambda h, n, L: float(L + 50))
        replacement = ("replacement", lambda h, n, L: float(L))
        
        agent = TurnoverPredictorAgent(
            predictors=[bad],
            patience=2,
            error_threshold=5.0,
            master_library=[replacement],
        )
        
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        
        for _ in range(5):
            agent.choose_action(ctx, rng)
            agent.update(ctx, action=0, realised_attendance=60, payoff=-1, rng=rng)
        
        assert len(agent.replacement_events) > 0


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

    def test_turnover_agent_reproducible(self) -> None:
        """Same seed should produce same replacement pattern."""
        predictors = [("pred", lambda h, n, L: float(L + 20))]
        master = [("rep", lambda h, n, L: float(L))]
        
        results = []
        for _ in range(2):
            agent = TurnoverPredictorAgent(
                predictors=predictors,
                patience=3,
                master_library=master,
            )
            ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
            rng = np.random.default_rng(12345)
            
            for _ in range(10):
                agent.choose_action(ctx, rng)
                agent.update(ctx, action=0, realised_attendance=60, payoff=-1, rng=rng)
            
            results.append(agent.replacement_events.copy())
        
        assert results[0] == results[1]
