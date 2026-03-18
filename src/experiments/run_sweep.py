"""
Batch runner for parameter sweeps over inductive agent experiments.

Runs multiple seeds and beta values for BestPredictorAgent and SoftmaxPredictorAgent
populations, collecting metrics into a single summary CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.experiments.populations import (
    build_homogeneous_best_predictor,
    build_homogeneous_softmax,
)
from src.game.repeated_game import RepeatedMinorityGame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run parameter sweep over inductive agent experiments"
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--output_csv", type=str, default="outputs/sweep/summary.csv")
    args = parser.parse_args()

    rows = []

    for seed in args.seeds:
        agents = build_homogeneous_best_predictor(
            args.n_players, predictors_per_agent=3, seed=seed
        )
        game = RepeatedMinorityGame(
            n_players=args.n_players,
            threshold=args.threshold,
            n_rounds=args.n_rounds,
            agents=agents,
            seed=seed,
        )
        result = game.play()
        predictor_histories = [a.predictor_history for a in agents]
        metrics = compute_all_metrics(
            result.attendance_history,
            result.cumulative_payoffs,
            args.threshold,
            predictor_histories=predictor_histories,
        )
        rows.append({"mode": "best", "beta": None, "seed": seed, **metrics})

        for beta in args.betas:
            agents = build_homogeneous_softmax(
                args.n_players,
                beta=beta,
                predictors_per_agent=3,
                seed=seed,
            )
            game = RepeatedMinorityGame(
                n_players=args.n_players,
                threshold=args.threshold,
                n_rounds=args.n_rounds,
                agents=agents,
                seed=seed,
            )
            result = game.play()
            predictor_histories = [a.predictor_history for a in agents]
            metrics = compute_all_metrics(
                result.attendance_history,
                result.cumulative_payoffs,
                args.threshold,
                predictor_histories=predictor_histories,
            )
            rows.append({"mode": "softmax", "beta": beta, "seed": seed, **metrics})

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved sweep results to: {out.resolve()}")


if __name__ == "__main__":
    main()
