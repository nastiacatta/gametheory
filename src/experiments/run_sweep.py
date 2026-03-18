"""
Multi-seed parameter sweep for the El Farol threshold game.

Runs experiments over:
  - seeds (0..49 by default)
  - n_rounds (200, 1000)
  - modes: best, softmax (multiple beta), recency, turnover, heterogeneous

Outputs:
  - sweep_results.csv: one row per run
  - sweep_summary.csv: aggregated statistics by mode

Metrics computed:
  - σ²_L = (1/T) Σ (A_t - L)² (variance from threshold)
  - MAD_L = (1/T) Σ |A_t - L| (mean absolute deviation from threshold)
  - Overcrowding rate
  - Mean cumulative payoff
  - Payoff dispersion (std of cumulative payoffs)
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_heterogeneous,
    build_homogeneous_best_predictor,
    build_homogeneous_recency,
    build_homogeneous_softmax,
    build_homogeneous_turnover,
    build_producer_speculator,
)
from src.game.repeated_game import RepeatedMinorityGame


def run_single_experiment(
    n_players: int,
    threshold: int,
    n_rounds: int,
    seed: int,
    mode: str,
    mode_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run a single experiment and return metrics plus metadata."""
    config = RepeatedGameConfig(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        seed=seed,
    )

    predictors_per_agent = mode_params.get("predictors_per_agent", 6)
    beta = mode_params.get("beta", 1.0)
    lambda_decay = mode_params.get("lambda_decay", 0.95)

    if mode == "best":
        agents = build_homogeneous_best_predictor(
            config.n_players,
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
        )
    elif mode == "softmax":
        agents = build_homogeneous_softmax(
            config.n_players,
            beta=beta,
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
        )
    elif mode == "recency":
        agents = build_homogeneous_recency(
            config.n_players,
            lambda_decay=lambda_decay,
            selection=mode_params.get("selection", "argmax"),
            beta=beta,
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
        )
    elif mode == "turnover":
        agents = build_homogeneous_turnover(
            config.n_players,
            lambda_decay=lambda_decay,
            patience=mode_params.get("patience", 10),
            error_threshold=mode_params.get("error_threshold", 5.0),
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
        )
    elif mode == "heterogeneous_mix":
        agents = build_heterogeneous(
            config.n_players,
            p_best=mode_params.get("p_best", 0.4),
            p_softmax=mode_params.get("p_softmax", 0.4),
            p_random=mode_params.get("p_random", 0.2),
            beta=beta,
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
        )
    elif mode == "producer_speculator":
        agents = build_producer_speculator(
            config.n_players,
            n_producers=mode_params.get("n_producers", 50),
            speculator_type=mode_params.get("speculator_type", "best"),
            beta=beta,
            predictors_per_agent=predictors_per_agent,
            seed=config.seed,
            threshold=config.threshold,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    predictor_histories = [
        getattr(a, "predictor_history", []) for a in agents
    ]
    predictor_histories = [h for h in predictor_histories if h]

    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        config.threshold,
        predictor_histories=predictor_histories if predictor_histories else None,
    )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed parameter sweep")
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--n_seeds", type=int, default=50, help="Number of seeds (0 to n_seeds-1)")
    parser.add_argument("--output_dir", type=str, default="outputs/sweep")
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    args = parser.parse_args()

    seeds = list(range(args.n_seeds))
    n_rounds_list = args.n_rounds if isinstance(args.n_rounds, list) else [args.n_rounds]
    
    mode_configs: List[tuple[str, Dict[str, Any]]] = [
        ("best", {}),
        ("softmax", {"beta": 0.25}),
        ("softmax", {"beta": 1.0}),
        ("softmax", {"beta": 3.0}),
        ("recency", {"lambda_decay": 0.9}),
        ("recency", {"lambda_decay": 0.95}),
        ("turnover", {"patience": 10, "lambda_decay": 0.95}),
        ("turnover", {"patience": 15, "lambda_decay": 0.95}),
        ("heterogeneous_mix", {"p_best": 0.4, "p_softmax": 0.4, "p_random": 0.2}),
        ("producer_speculator", {"n_producers": 50}),
    ]

    results: List[Dict] = []
    total_runs = len(seeds) * len(n_rounds_list) * len(mode_configs)
    run_count = 0

    print(f"Starting sweep: {total_runs} total runs")
    print(f"  Seeds: 0-{args.n_seeds - 1}")
    print(f"  Rounds: {n_rounds_list}")
    print(f"  Modes: {len(mode_configs)}")

    for seed, n_rounds, (mode, params) in itertools.product(seeds, n_rounds_list, mode_configs):
        run_count += 1
        if run_count % 100 == 0 or run_count == 1:
            print(f"Running {run_count}/{total_runs}...")

        params_with_defaults = {
            "predictors_per_agent": args.predictors_per_agent,
            **params,
        }

        try:
            metrics = run_single_experiment(
                n_players=args.n_players,
                threshold=args.threshold,
                n_rounds=n_rounds,
                seed=seed,
                mode=mode,
                mode_params=params_with_defaults,
            )

            mode_label = mode
            if mode == "softmax":
                mode_label = f"softmax_beta{params.get('beta', 1.0)}"
            elif mode == "recency":
                mode_label = f"recency_lambda{params.get('lambda_decay', 0.95)}"
            elif mode == "turnover":
                mode_label = f"turnover_p{params.get('patience', 10)}"

            row = {
                "seed": seed,
                "n_rounds": n_rounds,
                "mode": mode,
                "mode_label": mode_label,
                **params_with_defaults,
                **metrics,
            }
            results.append(row)
        except Exception as e:
            print(f"Error in run {run_count}: {e}")
            continue

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)

    print(f"\nSweep complete: {len(results)} runs")
    print(f"Results saved to: {csv_path.resolve()}")

    summary_df = df.groupby(["mode_label", "n_rounds"]).agg({
        "variance_from_threshold": ["mean", "std"],
        "mad_from_threshold": ["mean", "std"],
        "overcrowding_rate": ["mean", "std"],
        "mean_cumulative_payoff": ["mean", "std"],
        "std_cumulative_payoff": ["mean", "std"],
    }).round(4)
    
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path.resolve()}")

    print("\n=== SUMMARY BY MODE (n_rounds=200) ===")
    short_summary = df[df["n_rounds"] == 200].groupby("mode_label").agg({
        "variance_from_threshold": ["mean", "std"],
        "overcrowding_rate": ["mean", "std"],
        "mean_cumulative_payoff": ["mean", "std"],
    }).round(4)
    print(short_summary.to_string())


if __name__ == "__main__":
    main()
