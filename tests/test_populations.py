"""Tests for population builders."""

from __future__ import annotations

import pytest

from src.agents.predictor_agent import InductivePredictorAgent
from src.agents.producer_agent import ProducerAgent
from src.agents.random_agent import RandomAgent
from src.agents.predictors import default_predictor_library
from src.experiments.populations import (
    build_heterogeneous,
    build_homogeneous_non_recency,
    build_homogeneous_recency,
    build_producer_speculator,
)


def test_build_homogeneous_non_recency_returns_n_agents() -> None:
    agents = build_homogeneous_non_recency(101, predictors_per_agent=6, seed=42)
    assert len(agents) == 101


def test_build_homogeneous_recency_returns_n_agents() -> None:
    agents = build_homogeneous_recency(101, lambda_decay=0.95, predictors_per_agent=6, seed=42)
    assert len(agents) == 101


def test_homogeneous_non_recency_uses_sampled_banks() -> None:
    """With n_players=5, predictors_per_agent=6, every agent has exactly 6 predictor names
    and not all agents have identical predictor_names."""
    agents = build_homogeneous_non_recency(n_players=5, predictors_per_agent=6, seed=42)
    assert len(agents) == 5
    for agent in agents:
        assert isinstance(agent, InductivePredictorAgent)
        assert len(agent.predictor_names) == 6
    all_names = [tuple(a.predictor_names) for a in agents]
    assert len(set(all_names)) > 1


def test_homogeneous_recency_uses_sampled_banks() -> None:
    """Same checks for recency mode, plus lambda preserved."""
    agents = build_homogeneous_recency(
        n_players=5, lambda_decay=0.9, predictors_per_agent=6, seed=42
    )
    assert len(agents) == 5
    for agent in agents:
        assert isinstance(agent, InductivePredictorAgent)
        assert len(agent.predictor_names) == 6
    all_names = [tuple(a.predictor_names) for a in agents]
    assert len(set(all_names)) > 1


def test_build_heterogeneous_returns_correct_total_length() -> None:
    n_players = 101
    agents = build_heterogeneous(
        n_players,
        p_inductive=0.8,
        p_random=0.2,
        lambda_decay=None,
        predictors_per_agent=6,
        seed=42,
    )
    assert len(agents) == n_players


def test_build_heterogeneous_counts_preserved() -> None:
    """Type counts still match requested proportions."""
    n_players = 100
    agents = build_heterogeneous(
        n_players,
        p_inductive=0.8,
        p_random=0.2,
        lambda_decay=None,
        predictors_per_agent=6,
        seed=42,
    )
    n_inductive = sum(isinstance(a, InductivePredictorAgent) for a in agents)
    n_random = sum(isinstance(a, RandomAgent) for a in agents)
    assert n_inductive == 80
    assert n_random == 20
    assert n_inductive + n_random == n_players


def test_build_producer_speculator_rejects_too_many_producers() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        build_producer_speculator(
            n_players=101, n_producers=102,
            predictors_per_agent=6, seed=42, threshold=60
        )


def test_build_heterogeneous_uses_requested_shares() -> None:
    agents = build_heterogeneous(
        101,
        p_inductive=1.0,
        p_random=0.0,
        lambda_decay=None,
        predictors_per_agent=6,
        seed=42,
    )
    assert len(agents) == 101
    assert sum(isinstance(a, RandomAgent) for a in agents) == 0
    assert sum(isinstance(a, InductivePredictorAgent) for a in agents) == 101


def test_build_heterogeneous_rejects_invalid_share_sum() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        build_heterogeneous(
            101, p_inductive=0.6, p_random=0.6,
            predictors_per_agent=6, seed=42
        )


def test_build_producer_speculator_defaults_to_threshold() -> None:
    agents = build_producer_speculator(
        n_players=101,
        n_producers=10,
        lambda_decay=None,
        predictors_per_agent=6,
        seed=42,
        threshold=60,
    )
    producers = [a for a in agents if isinstance(a, ProducerAgent)]
    assert len(producers) == 10
    assert all(a.base_prediction == 60.0 for a in producers)


def test_build_producer_speculator_counts_preserved() -> None:
    """Producer/speculator counts unchanged, speculators get sampled banks."""
    n_players = 101
    n_producers = 30
    agents = build_producer_speculator(
        n_players=n_players,
        n_producers=n_producers,
        lambda_decay=None,
        predictors_per_agent=6,
        seed=42,
        threshold=60,
    )
    producers = [a for a in agents if isinstance(a, ProducerAgent)]
    speculators = [a for a in agents if isinstance(a, InductivePredictorAgent)]
    assert len(producers) == n_producers
    assert len(speculators) == n_players - n_producers
    for spec in speculators:
        assert len(spec.predictor_names) == 6
    all_names = [tuple(s.predictor_names) for s in speculators]
    assert len(set(all_names)) > 1


def test_experiment_runners_import_and_main_smoke(monkeypatch, tmp_path) -> None:
    """
    Smoke-test that the experiment-layer entrypoints still parse args and run.
    We keep runs tiny and redirect outputs to a temp directory.
    """
    from src.experiments import run_repeated_baselines, run_inductive

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_repeated_baselines",
            "--n_players",
            "101",
            "--threshold",
            "60",
            "--n_rounds",
            "2",
            "--seed",
            "1",
            "--output_dir",
            str(tmp_path / "baselines"),
        ],
    )
    run_repeated_baselines.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_inductive",
            "--mode",
            "non_recency",
            "--n_players",
            "101",
            "--threshold",
            "60",
            "--n_rounds",
            "2",
            "--seed",
            "1",
            "--output_dir",
            str(tmp_path / "inductive_non_recency"),
        ],
    )
    run_inductive.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_inductive",
            "--mode",
            "recency",
            "--lambda_decay",
            "0.95",
            "--n_players",
            "101",
            "--threshold",
            "60",
            "--n_rounds",
            "2",
            "--seed",
            "1",
            "--output_dir",
            str(tmp_path / "inductive_recency"),
        ],
    )
    run_inductive.main()
