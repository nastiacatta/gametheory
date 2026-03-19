"""
Tests for FixedPredictorAgent and the Repeated Fixed Strategy baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.base import RoundContext
from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.predictors import last_value, mirror, make_rolling_mean, default_predictor_library
from src.experiments.populations import build_fixed_predictor_population
from src.experiments.run_repeated_fixed_strategy import (
    count_predictor_assignments,
    compute_predictor_payoffs,
)


class TestFixedPredictorAgent:
    """Tests for the FixedPredictorAgent class."""

    def test_attend_when_prediction_below_threshold(self) -> None:
        """Agent attends when predicted attendance < threshold."""
        agent = FixedPredictorAgent("last_value", last_value)
        rng = np.random.default_rng(42)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(50,),
            round_index=1,
        )
        action = agent.choose_action(context, rng)
        assert action == 1

    def test_stay_when_prediction_above_threshold(self) -> None:
        """Agent stays home when predicted attendance >= threshold."""
        agent = FixedPredictorAgent("last_value", last_value)
        rng = np.random.default_rng(42)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(70,),
            round_index=1,
        )
        action = agent.choose_action(context, rng)
        assert action == 0

    def test_stay_when_prediction_equals_threshold(self) -> None:
        """Agent stays home when predicted attendance == threshold (strict threshold)."""
        agent = FixedPredictorAgent("last_value", last_value)
        rng = np.random.default_rng(42)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(60,),
            round_index=1,
        )
        action = agent.choose_action(context, rng)
        assert action == 0

    def test_fallback_to_threshold_on_empty_history(self) -> None:
        """With empty history, last_value returns threshold, so agent stays home (strict)."""
        agent = FixedPredictorAgent("last_value", last_value)
        rng = np.random.default_rng(42)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(),
            round_index=0,
        )
        action = agent.choose_action(context, rng)
        assert action == 0

    def test_mirror_predictor_contrarian_behaviour(self) -> None:
        """Mirror predictor: if last was 30, predicts 101-30=71 > 60, so stay."""
        agent = FixedPredictorAgent("mirror", mirror)
        rng = np.random.default_rng(42)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(30,),
            round_index=1,
        )
        action = agent.choose_action(context, rng)
        assert action == 0

    def test_snapshot_contains_predictor_name(self) -> None:
        """Snapshot should include predictor_name for reporting."""
        agent = FixedPredictorAgent("rolling_mean_4", make_rolling_mean(4))
        snap = agent.snapshot()

        assert snap["agent_type"] == "FixedPredictorAgent"
        assert snap["predictor_name"] == "rolling_mean_4"

    def test_name_method(self) -> None:
        """Name method should return descriptive string."""
        agent = FixedPredictorAgent("last_value", last_value)
        assert "FixedPredictorAgent" in agent.name()
        assert "last_value" in agent.name()

    def test_deterministic_given_same_context(self) -> None:
        """Agent produces same action for same context (deterministic)."""
        agent = FixedPredictorAgent("last_value", last_value)
        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(999)

        context = RoundContext(
            n_players=101,
            threshold=60,
            attendance_history=(55,),
            round_index=5,
        )

        action1 = agent.choose_action(context, rng1)
        action2 = agent.choose_action(context, rng2)
        assert action1 == action2


class TestBuildFixedPredictorPopulation:
    """Tests for the population builder function."""

    def test_returns_correct_number_of_agents(self) -> None:
        """Should return exactly n_players agents."""
        agents = build_fixed_predictor_population(101, seed=42)
        assert len(agents) == 101

    def test_all_agents_are_fixed_predictor_type(self) -> None:
        """All returned agents should be FixedPredictorAgent."""
        agents = build_fixed_predictor_population(50, seed=42)

        for agent in agents:
            assert isinstance(agent, FixedPredictorAgent)

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed should produce same predictor assignments."""
        agents1 = build_fixed_predictor_population(50, seed=42)
        names1 = [a.predictor_name for a in agents1]

        agents2 = build_fixed_predictor_population(50, seed=42)
        names2 = [a.predictor_name for a in agents2]

        assert names1 == names2

    def test_different_seeds_give_different_assignments(self) -> None:
        """Different seeds should (almost certainly) give different assignments."""
        agents1 = build_fixed_predictor_population(100, seed=42)
        names1 = [a.predictor_name for a in agents1]

        agents2 = build_fixed_predictor_population(100, seed=999)
        names2 = [a.predictor_name for a in agents2]

        assert names1 != names2

    def test_uses_predictors_from_library(self) -> None:
        """All assigned predictor names should be from the library."""
        library = default_predictor_library()
        library_names = {name for name, _ in library}

        agents = build_fixed_predictor_population(200, seed=42)
        for agent in agents:
            assert agent.predictor_name in library_names

    def test_cover_all_predictors_true(self) -> None:
        """With cover_all_predictors=True, every predictor should appear."""
        library = default_predictor_library()
        library_names = {name for name, _ in library}

        agents = build_fixed_predictor_population(
            101, seed=42, cover_all_predictors=True
        )
        assigned_names = {a.predictor_name for a in agents}

        assert assigned_names == library_names


class TestCountPredictorAssignments:
    """Tests for the assignment counting function."""

    def test_counts_sum_to_total_agents(self) -> None:
        """Sum of counts should equal number of agents."""
        agents = build_fixed_predictor_population(101, seed=42)

        counts = count_predictor_assignments(agents)
        assert sum(counts.values()) == 101

    def test_returns_dict_of_strings_to_ints(self) -> None:
        """Counts should be a dict mapping predictor names to counts."""
        agents = build_fixed_predictor_population(50, seed=42)

        counts = count_predictor_assignments(agents)
        assert isinstance(counts, dict)
        for name, count in counts.items():
            assert isinstance(name, str)
            assert isinstance(count, int)
            assert count >= 1


class TestComputePredictorPayoffs:
    """Tests for the predictor payoff aggregation function."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """DataFrame should have predictor_name, n_users, mean_payoff, std_payoff."""
        agents = build_fixed_predictor_population(50, seed=42)
        cumulative_payoffs = list(range(50))

        df = compute_predictor_payoffs(agents, cumulative_payoffs)
        expected_cols = {"predictor_name", "n_users", "mean_payoff", "std_payoff"}
        assert set(df.columns) == expected_cols

    def test_n_users_sums_to_total(self) -> None:
        """Sum of n_users should equal total agents."""
        agents = build_fixed_predictor_population(101, seed=42)
        cumulative_payoffs = [0] * 101

        df = compute_predictor_payoffs(agents, cumulative_payoffs)
        assert df["n_users"].sum() == 101

    def test_mean_payoff_computation(self) -> None:
        """Mean payoff should be correctly computed per predictor."""
        agents = [
            FixedPredictorAgent("A", last_value),
            FixedPredictorAgent("A", last_value),
            FixedPredictorAgent("B", mirror),
        ]
        cumulative_payoffs = [10, 20, 30]

        df = compute_predictor_payoffs(agents, cumulative_payoffs)
        df_a = df[df["predictor_name"] == "A"]
        df_b = df[df["predictor_name"] == "B"]

        assert df_a["mean_payoff"].iloc[0] == pytest.approx(15.0)
        assert df_a["n_users"].iloc[0] == 2
        assert df_b["mean_payoff"].iloc[0] == pytest.approx(30.0)
        assert df_b["n_users"].iloc[0] == 1
