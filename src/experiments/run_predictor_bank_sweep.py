"""
Sweep over predictors_per_agent to test whether 6 is a good default.

For a given mode (non_recency or recency), runs experiments with different
bank sizes and collects key metrics to identify optimal bank size.

Output:
  - {mode}_predictor_bank_sweep.csv: one row per bank size with metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_homogeneous_non_recency,
    build_homogeneous_recency,
)
from src.game.repeated_game import RepeatedMinorityGame


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep predictors_per_agent")
    parser.add_argument("--mode", choices=["non_recency", "recency"], required=True)
    parser.add_argument("--grid", type=int, nargs="+", default=[2, 4, 6, 8, 12, 16])
    parser.add_argument("--lambda_decay", type=float, default=0.95)
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/predictor_bank_sweep")
    args = parser.parse_args()

    rows = []

    print(f"Sweeping predictors_per_agent for mode={args.mode}")
    print(f"Grid: {args.grid}")
    print()

    for k in args.grid:
        print(f"  Running k={k}...", end=" ", flush=True)

        config = RepeatedGameConfig(
            n_players=args.n_players,
            threshold=args.threshold,
            n_rounds=args.n_rounds,
            seed=args.seed,
        )

        if args.mode == "non_recency":
            agents = build_homogeneous_non_recency(
                config.n_players,
                predictors_per_agent=k,
                seed=config.seed,
            )
        else:
            agents = build_homogeneous_recency(
                config.n_players,
                lambda_decay=args.lambda_decay,
                predictors_per_agent=k,
                seed=config.seed,
            )

        game = RepeatedMinorityGame(
            n_players=config.n_players,
            threshold=config.threshold,
            n_rounds=config.n_rounds,
            agents=agents,
            seed=config.seed,
        )
        result = game.play()

        predictor_histories = [getattr(a, "predictor_history", []) for a in agents]
        use_histories = predictor_histories if (predictor_histories and predictor_histories[0]) else None

        metrics = compute_all_metrics(
            result.attendance_history,
            result.cumulative_payoffs,
            config.threshold,
            predictor_histories=use_histories,
        )
        metrics["mode"] = args.mode
        metrics["predictors_per_agent"] = k
        rows.append(metrics)

        print(f"MAD={metrics['mad_from_threshold']:.2f}, payoff={metrics['mean_cumulative_payoff']:.2f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    output_file = out / f"{args.mode}_predictor_bank_sweep.csv"
    df.to_csv(output_file, index=False)

    print()
    print(f"Saved to: {output_file.resolve()}")
    print()
    print("=== SUMMARY ===")
    print(df[["predictors_per_agent", "mad_from_threshold", "mean_cumulative_payoff", "switch_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
