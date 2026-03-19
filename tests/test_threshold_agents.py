"""Regression tests for threshold boundary behaviour in predictor-based agents.

These tests verify that agents correctly stay home when predicted_attendance == threshold
(the < boundary), matching the strict threshold convention and the payoff engine.
"""

from __future__ import annotations

import numpy as np

from src.agents.base import RoundContext
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent


def constant_predictor(value: float):
    def predictor(history, n_players, threshold):
        return value
    return predictor


def test_fixed_attendance_agent_stays_home_at_threshold() -> None:
    agent = FixedAttendanceAgent(predicted_attendance=60)
    context = RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(),
        round_index=0,
    )
    rng = np.random.default_rng(1)
    assert agent.choose_action(context, rng) == 0


def test_best_predictor_agent_stays_home_at_threshold() -> None:
    predictors = [("eq_threshold", constant_predictor(60.0))]
    agent = BestPredictorAgent(predictors=predictors)
    context = RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(),
        round_index=0,
    )
    rng = np.random.default_rng(1)
    assert agent.choose_action(context, rng) == 0


def test_softmax_predictor_agent_stays_home_at_threshold() -> None:
    predictors = [("eq_threshold", constant_predictor(60.0))]
    agent = SoftmaxPredictorAgent(predictors=predictors, beta=1.0)
    context = RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(),
        round_index=0,
    )
    rng = np.random.default_rng(1)
    assert agent.choose_action(context, rng) == 0


def test_threshold_agents_stay_home_above_threshold() -> None:
    context = RoundContext(
        n_players=101,
        threshold=60,
        attendance_history=(),
        round_index=0,
    )
    rng = np.random.default_rng(1)

    fixed = FixedAttendanceAgent(predicted_attendance=61)
    assert fixed.choose_action(context, rng) == 0

    best = BestPredictorAgent(predictors=[("above", constant_predictor(61.0))])
    assert best.choose_action(context, rng) == 0

    softmax = SoftmaxPredictorAgent(
        predictors=[("above", constant_predictor(61.0))],
        beta=1.0,
    )
    assert softmax.choose_action(context, rng) == 0
