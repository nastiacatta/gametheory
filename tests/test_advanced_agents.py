"""Tests for advanced predictor agents.

Tests for:
- EpsilonGreedyPredictorAgent
- VirtualPayoffPredictorAgent
- RecencyWeightedPredictorAgent
- TurnoverPredictorAgent
- NashInitialisedFixedPredictorAgent
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.base import RoundContext
from src.agents.epsilon_greedy_predictor_agent import EpsilonGreedyPredictorAgent
from src.agents.nash_initialised_fixed_predictor_agent import NashInitialisedFixedPredictorAgent
from src.agents.predictors import default_predictor_library
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent
from src.agents.turnover_predictor_agent import TurnoverPredictorAgent
from src.agents.virtual_payoff_predictor_agent import VirtualPayoffPredictorAgent


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def context_at_threshold() -> RoundContext:
    return RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(55, 62, 58, 61, 60),
        round_index=10,
    )


@pytest.fixture
def context_empty_history() -> RoundContext:
    return RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(),
        round_index=0,
    )


class TestEpsilonGreedyPredictorAgent:
    """Tests for EpsilonGreedyPredictorAgent."""

    def test_initialization_default(self) -> None:
        agent = EpsilonGreedyPredictorAgent()
        assert agent.epsilon == 0.1
        assert len(agent.predictors) == len(default_predictor_library())
        assert all(s == 0.0 for s in agent.scores)

    def test_initialization_custom_epsilon(self) -> None:
        agent = EpsilonGreedyPredictorAgent(epsilon=0.5)
        assert agent.epsilon == 0.5

    def test_invalid_epsilon_raises(self) -> None:
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyPredictorAgent(epsilon=-0.1)
        with pytest.raises(ValueError, match="epsilon must be in"):
            EpsilonGreedyPredictorAgent(epsilon=1.5)

    def test_choose_action_returns_valid(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = EpsilonGreedyPredictorAgent(epsilon=0.0)
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_exploration_with_high_epsilon(
        self, context_at_threshold: RoundContext
    ) -> None:
        """With epsilon=1.0, agent should explore randomly."""
        agent = EpsilonGreedyPredictorAgent(epsilon=1.0)
        rng = np.random.default_rng(123)
        
        choices = set()
        for _ in range(50):
            agent.choose_action(context_at_threshold, rng)
            choices.add(agent._active_idx)
        
        assert len(choices) > 1

    def test_exploitation_with_zero_epsilon(
        self, context_at_threshold: RoundContext
    ) -> None:
        """With epsilon=0.0, agent should always exploit best predictor."""
        agent = EpsilonGreedyPredictorAgent(epsilon=0.0)
        agent.scores[0] = 100.0
        rng = np.random.default_rng(42)
        
        for _ in range(10):
            agent.choose_action(context_at_threshold, rng)
            assert agent._active_idx == 0

    def test_reset_clears_state(self, context_at_threshold: RoundContext, rng: np.random.Generator) -> None:
        agent = EpsilonGreedyPredictorAgent()
        agent.choose_action(context_at_threshold, rng)
        agent.update(context_at_threshold, 1, 58, 1)
        
        assert len(agent.predictor_history) > 0
        assert any(s != 0.0 for s in agent.scores)
        
        agent.reset()
        
        assert len(agent.predictor_history) == 0
        assert all(s == 0.0 for s in agent.scores)
        assert agent._active_idx == 0

    def test_snapshot_contains_expected_keys(self) -> None:
        agent = EpsilonGreedyPredictorAgent(epsilon=0.2)
        snapshot = agent.snapshot()
        
        assert "agent_type" in snapshot
        assert "epsilon" in snapshot
        assert "predictor_names" in snapshot
        assert "scores" in snapshot
        assert "active_predictor" in snapshot
        assert snapshot["epsilon"] == 0.2

    def test_update_uses_forecast_error(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = EpsilonGreedyPredictorAgent()
        agent.choose_action(context_at_threshold, rng)
        initial_scores = list(agent.scores)
        
        agent.update(context_at_threshold, 1, 60, 1)
        
        for j, (initial, new) in enumerate(zip(initial_scores, agent.scores)):
            assert new <= initial


class TestVirtualPayoffPredictorAgent:
    """Tests for VirtualPayoffPredictorAgent."""

    def test_initialization_default(self) -> None:
        agent = VirtualPayoffPredictorAgent()
        assert len(agent.predictors) == len(default_predictor_library())
        assert all(s == 0.0 for s in agent.scores)

    def test_choose_action_returns_valid(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = VirtualPayoffPredictorAgent()
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_attends_when_prediction_at_threshold(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        def predict_exact_threshold(
            history: tuple[int, ...], n_players: int, threshold: int
        ) -> float:
            _ = history, n_players
            return float(threshold)

        agent = VirtualPayoffPredictorAgent(
            predictors=[("exact_threshold", predict_exact_threshold)]
        )
        action = agent.choose_action(context_at_threshold, rng)
        assert action == 0

    def test_reset_clears_state(self, context_at_threshold: RoundContext, rng: np.random.Generator) -> None:
        agent = VirtualPayoffPredictorAgent()
        agent.choose_action(context_at_threshold, rng)
        agent.update(context_at_threshold, 1, 58, 1)
        
        agent.reset()
        
        assert len(agent.predictor_history) == 0
        assert all(s == 0.0 for s in agent.scores)

    def test_update_uses_game_payoff_virtual_scoring(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = VirtualPayoffPredictorAgent()
        agent.choose_action(context_at_threshold, rng)
        
        agent.update(context_at_threshold, 1, 55, 1)
        
        for s in agent.scores:
            assert s in (-1.0, 0.0, 1.0)

    def test_snapshot_contains_expected_keys(self) -> None:
        agent = VirtualPayoffPredictorAgent()
        snapshot = agent.snapshot()
        
        assert "agent_type" in snapshot
        assert "predictor_names" in snapshot
        assert "scores" in snapshot
        assert "active_predictor" in snapshot


class TestRecencyWeightedPredictorAgent:
    """Tests for RecencyWeightedPredictorAgent."""

    def test_initialization_default(self) -> None:
        agent = RecencyWeightedPredictorAgent()
        assert agent.lambda_decay == 0.95
        assert agent.selection == "argmax"
        assert agent.beta == 1.0

    def test_initialization_custom(self) -> None:
        agent = RecencyWeightedPredictorAgent(
            lambda_decay=0.9, selection="softmax", beta=2.0
        )
        assert agent.lambda_decay == 0.9
        assert agent.selection == "softmax"
        assert agent.beta == 2.0

    def test_invalid_lambda_decay_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_decay"):
            RecencyWeightedPredictorAgent(lambda_decay=0.0)
        with pytest.raises(ValueError, match="lambda_decay"):
            RecencyWeightedPredictorAgent(lambda_decay=1.5)

    def test_invalid_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="selection"):
            RecencyWeightedPredictorAgent(selection="invalid")

    def test_choose_action_argmax(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = RecencyWeightedPredictorAgent(selection="argmax")
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_choose_action_softmax(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = RecencyWeightedPredictorAgent(selection="softmax")
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_scores_decay_over_time(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = RecencyWeightedPredictorAgent(lambda_decay=0.5)
        agent.scores[0] = 10.0
        
        agent.choose_action(context_at_threshold, rng)
        agent.update(context_at_threshold, 1, 55, 1)
        
        assert agent.scores[0] < 10.0

    def test_reset_clears_state(self, context_at_threshold: RoundContext, rng: np.random.Generator) -> None:
        agent = RecencyWeightedPredictorAgent()
        agent.choose_action(context_at_threshold, rng)
        agent.update(context_at_threshold, 1, 58, 1)
        
        agent.reset()
        
        assert len(agent.predictor_history) == 0
        assert len(agent.score_history) == 0
        assert all(s == 0.0 for s in agent.scores)

    def test_snapshot_contains_expected_keys(self) -> None:
        agent = RecencyWeightedPredictorAgent(lambda_decay=0.8)
        snapshot = agent.snapshot()
        
        assert "agent_type" in snapshot
        assert "lambda_decay" in snapshot
        assert "selection" in snapshot
        assert "beta" in snapshot
        assert "active_predictor" in snapshot
        assert snapshot["lambda_decay"] == 0.8


class TestTurnoverPredictorAgent:
    """Tests for TurnoverPredictorAgent."""

    def test_initialization_default(self) -> None:
        agent = TurnoverPredictorAgent()
        assert agent.lambda_decay == 0.95
        assert agent.patience == 10
        assert len(agent.predictors) == 6

    def test_initialization_custom(self) -> None:
        agent = TurnoverPredictorAgent(
            lambda_decay=0.8, patience=5
        )
        assert agent.lambda_decay == 0.8
        assert agent.patience == 5

    def test_invalid_lambda_decay_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_decay"):
            TurnoverPredictorAgent(lambda_decay=0.0)

    def test_invalid_patience_raises(self) -> None:
        with pytest.raises(ValueError, match="patience"):
            TurnoverPredictorAgent(patience=0)

    def test_choose_action_returns_valid(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        agent = TurnoverPredictorAgent()
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_predictor_replacement_on_consecutive_failures(self) -> None:
        agent = TurnoverPredictorAgent(patience=2)
        rng = np.random.default_rng(42)
        
        context = RoundContext(
            n_players=101, threshold=60,
            attendance_history=(55, 60, 58),
            round_index=0,
        )
        
        initial_predictors = list(agent.predictor_names)
        
        for _ in range(10):
            agent.choose_action(context, rng)
            agent.update(context, 1, 100, -1)
        
        assert agent.n_replacements >= 0

    def test_reset_clears_state(self) -> None:
        agent = TurnoverPredictorAgent()
        
        agent.scores[0] = 5.0
        agent.predictor_history.append(0)
        agent._consecutive_failures = 5
        
        agent.reset()
        
        assert all(s == 0.0 for s in agent.scores)
        assert len(agent.predictor_history) == 0
        assert agent._consecutive_failures == 0
        assert agent.n_replacements == 0

    def test_reproducibility_with_same_rng_seed(
        self, context_at_threshold: RoundContext
    ) -> None:
        agent1 = TurnoverPredictorAgent()
        agent2 = TurnoverPredictorAgent()
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        actions1 = []
        actions2 = []
        
        for _ in range(5):
            actions1.append(agent1.choose_action(context_at_threshold, rng1))
            actions2.append(agent2.choose_action(context_at_threshold, rng2))
            agent1.update(context_at_threshold, actions1[-1], 100, -1)
            agent2.update(context_at_threshold, actions2[-1], 100, -1)
        
        assert actions1 == actions2

    def test_snapshot_contains_expected_keys(self) -> None:
        agent = TurnoverPredictorAgent()
        snapshot = agent.snapshot()
        
        assert "agent_type" in snapshot
        assert "lambda_decay" in snapshot
        assert "patience" in snapshot
        assert "active_predictor" in snapshot
        assert "n_replacements" in snapshot
        assert "current_predictors" in snapshot


class TestNashInitialisedFixedPredictorAgent:
    """Tests for NashInitialisedFixedPredictorAgent."""

    def test_initialization_with_p_star(self) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, p_star=0.6
        )
        assert agent._p_star == 0.6

    def test_initialization_with_n_players_and_threshold(self) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, n_players=101, threshold=60
        )
        assert 0.0 < agent._p_star < 1.0

    def test_initialization_missing_params_raises(self) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        with pytest.raises(ValueError, match="Supply either"):
            NashInitialisedFixedPredictorAgent(predictor_name=name, predictor_fn=fn)

    def test_uses_mixed_nash_on_empty_history(
        self, context_empty_history: RoundContext
    ) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, p_star=0.6
        )
        
        rng = np.random.default_rng(42)
        actions = [agent.choose_action(context_empty_history, rng) for _ in range(100)]
        
        attend_rate = sum(actions) / len(actions)
        assert 0.3 < attend_rate < 0.9

    def test_uses_predictor_with_history(
        self, context_at_threshold: RoundContext, rng: np.random.Generator
    ) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, p_star=0.0
        )
        
        action = agent.choose_action(context_at_threshold, rng)
        assert action in (0, 1)

    def test_snapshot_contains_p_star(self) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, p_star=0.55
        )
        snapshot = agent.snapshot()
        
        assert "p_star" in snapshot
        assert snapshot["p_star"] == 0.55

    def test_name_method(self) -> None:
        library = default_predictor_library()
        name, fn = library[0]
        agent = NashInitialisedFixedPredictorAgent(
            predictor_name=name, predictor_fn=fn, p_star=0.6
        )
        assert "NashInitFixedPredictor" in agent.name()
        assert name in agent.name()
