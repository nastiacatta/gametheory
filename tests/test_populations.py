from __future__ import annotations

import pytest

from src.experiments.populations import (
    build_heterogeneous,
    build_homogeneous_best_predictor,
    build_homogeneous_softmax,
    build_producer_speculator,
)


def test_build_homogeneous_best_predictor_returns_n_agents() -> None:
    agents = build_homogeneous_best_predictor(101)
    assert len(agents) == 101


def test_build_homogeneous_softmax_returns_n_agents() -> None:
    agents = build_homogeneous_softmax(101, beta=1.0)
    assert len(agents) == 101


def test_build_heterogeneous_returns_correct_total_length() -> None:
    n_players = 101
    agents = build_heterogeneous(
        n_players,
        p_best=0.3,
        p_softmax=0.5,
        p_random=0.2,
        beta=1.0,
    )
    assert len(agents) == n_players


def test_build_producer_speculator_rejects_too_many_producers() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        build_producer_speculator(n_players=101, n_producers=102, speculator_type="best")


def test_experiment_runners_import_and_main_smoke(monkeypatch, tmp_path) -> None:
    """
    Smoke-test that the experiment-layer entrypoints still parse args and run.
    We keep runs tiny and redirect outputs to a temp directory.
    """

    from src.experiments import run_repeated_baselines, run_inductive, run_heterogeneous

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
            "best",
            "--n_players",
            "101",
            "--threshold",
            "60",
            "--n_rounds",
            "2",
            "--seed",
            "1",
            "--output_dir",
            str(tmp_path / "inductive_best"),
        ],
    )
    run_inductive.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_heterogeneous",
            "--mode",
            "mix",
            "--n_players",
            "101",
            "--threshold",
            "60",
            "--n_rounds",
            "2",
            "--p_best",
            "0.5",
            "--p_softmax",
            "0.5",
            "--seed",
            "1",
            "--output_dir",
            str(tmp_path / "heterogeneous_mix"),
        ],
    )
    run_heterogeneous.main()

