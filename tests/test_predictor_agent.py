"""Tests for unified InductivePredictorAgent."""

import numpy as np
import pytest

from src.agents.base import RoundContext
from src.agents.predictor_agent import InductivePredictorAgent
from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater


def p_low(history, n_players, threshold):
    """Always predicts 55 (below threshold 60)."""
    return 55.0


def p_high(history, n_players, threshold):
    """Always predicts 70 (above threshold 60)."""
    return 70.0


def p_threshold(history, n_players, threshold):
    """Always predicts exactly at threshold."""
    return float(threshold)


class TestInductivePredictorAgentBasics:
    """Basic functionality tests."""

    def test_agent_attends_at_threshold(self) -> None:
        """Agent should attend when prediction equals threshold."""
        predictors = [("at_threshold", p_threshold)]
        agent = InductivePredictorAgent(predictors=predictors)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 1

    def test_agent_attends_below_threshold(self) -> None:
        """Agent should attend when prediction is below threshold."""
        predictors = [("below", p_low)]
        agent = InductivePredictorAgent(predictors=predictors)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 1

    def test_agent_stays_home_above_threshold(self) -> None:
        """Agent should stay home when prediction exceeds threshold."""
        predictors = [("above", p_high)]
        agent = InductivePredictorAgent(predictors=predictors)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)
        assert agent.choose_action(ctx, rng) == 0


class TestInductivePredictorAgentSelection:
    """Tests for predictor selection logic."""

    def test_chooses_highest_score_predictor(self) -> None:
        """Agent should select predictor with highest score."""
        predictors = [("low", p_low), ("high", p_high)]
        agent = InductivePredictorAgent(predictors=predictors)
        agent.scores = [-1.0, -5.0]

        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)

        action = agent.choose_action(ctx, rng)

        assert agent._active_idx == 0
        assert action == 1

    def test_breaks_ties_randomly(self) -> None:
        """With equal scores, selection should be random."""
        predictors = [("low", p_low), ("high", p_high)]
        agent = InductivePredictorAgent(predictors=predictors)
        agent.scores = [0.0, 0.0]

        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())

        selections = []
        for seed in range(100):
            agent.scores = [0.0, 0.0]
            rng = np.random.default_rng(seed)
            agent.choose_action(ctx, rng)
            selections.append(agent._active_idx)

        assert 0 in selections and 1 in selections

    def test_tracks_predictor_history(self) -> None:
        """Predictor history should track selections over time."""
        predictors = [("pred", p_low)]
        agent = InductivePredictorAgent(predictors=predictors)
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)

        for _ in range(5):
            agent.choose_action(ctx, rng)

        assert len(agent.predictor_history) == 5
        assert all(idx == 0 for idx in agent.predictor_history)


class TestInductivePredictorAgentUpdates:
    """Tests for score update mechanics."""

    def test_updates_all_scores_non_recency(self) -> None:
        """All predictor scores should be updated after each round (virtual payoff)."""
        predictors = [("low", p_low), ("high", p_high)]
        agent = InductivePredictorAgent(
            predictors=predictors,
            score_updater=CumulativeScoreUpdater(),
        )

        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)

        agent.choose_action(ctx, rng)
        agent.update(ctx, action=1, realised_attendance=62, payoff=-1)

        # A=62 > L=60 means overcrowded
        # p_low predicts 55 -> attend; overcrowded -> -1
        # p_high predicts 70 -> stay home -> 0
        assert agent.scores[0] == pytest.approx(-1.0)
        assert agent.scores[1] == pytest.approx(0.0)

    def test_updates_all_scores_recency(self) -> None:
        """Recency updater should decay then add virtual payoff."""
        predictors = [("low", p_low), ("high", p_high)]
        agent = InductivePredictorAgent(
            predictors=predictors,
            score_updater=RecencyScoreUpdater(lambda_decay=0.5),
        )
        agent.scores = [-4.0, -6.0]

        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)

        agent.choose_action(ctx, rng)
        agent.update(ctx, action=1, realised_attendance=62, payoff=-1)

        # A=62 > L=60 means overcrowded
        # p_low: attend, overcrowded -> -1
        # p_high: stay home -> 0
        assert agent.scores[0] == pytest.approx(0.5 * (-4.0) + (-1.0))
        assert agent.scores[1] == pytest.approx(0.5 * (-6.0) + 0.0)


class TestInductivePredictorAgentReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self) -> None:
        """reset() should clear all mutable state."""
        agent = InductivePredictorAgent()
        agent.scores[0] = -10.0
        agent._last_predictions[0] = 55.0
        agent._active_idx = 3
        agent.predictor_history = [0, 1, 2]

        agent.reset()

        assert all(s == 0.0 for s in agent.scores)
        assert all(p == 0.0 for p in agent._last_predictions)
        assert agent._active_idx == 0
        assert agent.predictor_history == []


class TestInductivePredictorAgentIntegration:
    """Integration tests for the unified agent."""

    def test_non_recency_mode_runs(self) -> None:
        """Non-recency mode should run without errors."""
        agent = InductivePredictorAgent(score_updater=CumulativeScoreUpdater())
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=(58, 62, 55))
        rng = np.random.default_rng(42)

        for t in range(10):
            action = agent.choose_action(ctx, rng)
            assert action in (0, 1)
            agent.update(ctx, action=action, realised_attendance=60, payoff=0)

        assert len(agent.predictor_history) == 10

    def test_recency_mode_runs(self) -> None:
        """Recency mode should run without errors."""
        agent = InductivePredictorAgent(score_updater=RecencyScoreUpdater(lambda_decay=0.95))
        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=(58, 62, 55))
        rng = np.random.default_rng(42)

        for t in range(10):
            action = agent.choose_action(ctx, rng)
            assert action in (0, 1)
            agent.update(ctx, action=action, realised_attendance=60, payoff=0)

        assert len(agent.predictor_history) == 10

    def test_both_modes_same_bank_different_behavior(self) -> None:
        """Given same bank, recency should forget faster than non-recency."""
        predictors = [("low", p_low), ("high", p_high)]

        agent_non = InductivePredictorAgent(
            predictors=predictors,
            score_updater=CumulativeScoreUpdater(),
        )
        agent_rec = InductivePredictorAgent(
            predictors=predictors,
            score_updater=RecencyScoreUpdater(lambda_decay=0.5),
        )

        agent_non.scores = [-100.0, -100.0]
        agent_rec.scores = [-100.0, -100.0]

        ctx = RoundContext(round_index=0, n_players=101, threshold=60, attendance_history=())
        rng = np.random.default_rng(42)

        for _ in range(20):
            agent_non.choose_action(ctx, rng)
            agent_non.update(ctx, action=1, realised_attendance=60, payoff=0)

            agent_rec.choose_action(ctx, rng)
            agent_rec.update(ctx, action=1, realised_attendance=60, payoff=0)

        assert abs(agent_rec.scores[0]) < abs(agent_non.scores[0])
