"""
Paired comparison: non_recency vs recency on identical predictor banks.

Runs both modes with the exact same predictor assignments across multiple seeds
to ensure any difference in outcomes is due to the scoring rule, not bank-assignment effects.

Outputs:
  - paired_differences.csv: per-seed comparison
  - summary_comparison.csv: aggregate statistics
  - recency_comparison.png: paired difference plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import plot_recency_comparison
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_homogeneous_non_recency,
    build_homogeneous_recency,
    sample_predictor_banks,
)
from src.game.repeated_game import RepeatedMinorityGame


def run_one_pair(
    n_players: int,
    threshold: int,
    n_rounds: int,
    predictors_per_agent: int,
    lambda_decay: float,
    seed: int,
) -> dict:
    """Run one matched pair (non_recency vs recency) and return metrics."""
    banks = sample_predictor_banks(
        n_players=n_players,
        predictors_per_agent=predictors_per_agent,
        seed=seed,
    )

    # Non-recency
    agents_nr = build_homogeneous_non_recency(
        n_players,
        predictors_per_agent=predictors_per_agent,
        seed=seed,
        predictor_banks=banks,
    )
    game_nr = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=agents_nr,
        seed=seed,
    )
    result_nr = game_nr.play()
    metrics_nr = compute_all_metrics(
        result_nr.attendance_history,
        result_nr.cumulative_payoffs,
        threshold,
    )

    # Recency
    agents_r = build_homogeneous_recency(
        n_players,
        lambda_decay=lambda_decay,
        predictors_per_agent=predictors_per_agent,
        seed=seed,
        predictor_banks=banks,
    )
    game_r = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=agents_r,
        seed=seed,
    )
    result_r = game_r.play()
    metrics_r = compute_all_metrics(
        result_r.attendance_history,
        result_r.cumulative_payoffs,
        threshold,
    )

    return {
        "seed": seed,
        "non_recency_mean_payoff": metrics_nr["mean_cumulative_payoff"],
        "recency_mean_payoff": metrics_r["mean_cumulative_payoff"],
        "non_recency_overcrowding_rate": metrics_nr["overcrowding_rate"],
        "recency_overcrowding_rate": metrics_r["overcrowding_rate"],
        "non_recency_mad_from_threshold": metrics_nr["mad_from_threshold"],
        "recency_mad_from_threshold": metrics_r["mad_from_threshold"],
        "delta_mean_payoff": metrics_r["mean_cumulative_payoff"] - metrics_nr["mean_cumulative_payoff"],
        "delta_overcrowding_rate": metrics_r["overcrowding_rate"] - metrics_nr["overcrowding_rate"],
        "delta_mad_from_threshold": metrics_r["mad_from_threshold"] - metrics_nr["mad_from_threshold"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison: non_recency vs recency (multiple seeds)"
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--lambda_decay", type=float, default=0.95)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/inductive_paired")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.n_seeds} matched pairs (non_recency vs recency)...")
    rows = []
    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.n_seeds)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Seed {i + 1}/{args.n_seeds}...")
        rows.append(
            run_one_pair(
                n_players=args.n_players,
                threshold=args.threshold,
                n_rounds=args.n_rounds,
                predictors_per_agent=args.predictors_per_agent,
                lambda_decay=args.lambda_decay,
                seed=seed,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(out / "paired_differences.csv", index=False)

    summary = pd.DataFrame([{
        "n_seeds": len(df),
        "n_players": args.n_players,
        "threshold": args.threshold,
        "n_rounds": args.n_rounds,
        "lambda_decay": args.lambda_decay,
        "mean_non_recency_payoff": df["non_recency_mean_payoff"].mean(),
        "mean_recency_payoff": df["recency_mean_payoff"].mean(),
        "mean_delta_mean_payoff": df["delta_mean_payoff"].mean(),
        "std_delta_mean_payoff": df["delta_mean_payoff"].std(),
        "mean_delta_overcrowding_rate": df["delta_overcrowding_rate"].mean(),
        "std_delta_overcrowding_rate": df["delta_overcrowding_rate"].std(),
        "mean_delta_mad_from_threshold": df["delta_mad_from_threshold"].mean(),
        "std_delta_mad_from_threshold": df["delta_mad_from_threshold"].std(),
        "share_recency_better_payoff": (df["delta_mean_payoff"] > 0).mean(),
        "share_recency_lower_overcrowding": (df["delta_overcrowding_rate"] < 0).mean(),
        "share_recency_lower_deviation": (df["delta_mad_from_threshold"] < 0).mean(),
    }])
    summary.to_csv(out / "summary_comparison.csv", index=False)

    plot_recency_comparison(df, output_path=out / "recency_comparison.png")

    print()
    print(f"Wrote outputs to: {out.resolve()}")
    print()
    print("=== SUMMARY ===")
    print(f"  Seeds: {args.n_seeds}, Rounds: {args.n_rounds}, λ: {args.lambda_decay}")
    print()
    print(f"  Mean non_recency payoff: {summary['mean_non_recency_payoff'].iloc[0]:.2f}")
    print(f"  Mean recency payoff:     {summary['mean_recency_payoff'].iloc[0]:.2f}")
    print(f"  Mean Δ payoff:           {summary['mean_delta_mean_payoff'].iloc[0]:+.2f} (recency - non_recency)")
    print(f"  Mean Δ overcrowd:        {summary['mean_delta_overcrowding_rate'].iloc[0]:+.4f}")
    print(f"  Mean Δ MAD:              {summary['mean_delta_mad_from_threshold'].iloc[0]:+.2f}")
    print()
    print(f"  Recency better payoff:      {summary['share_recency_better_payoff'].iloc[0]*100:.1f}% of seeds")
    print(f"  Recency lower overcrowding: {summary['share_recency_lower_overcrowding'].iloc[0]*100:.1f}% of seeds")
    print(f"  Recency lower MAD:          {summary['share_recency_lower_deviation'].iloc[0]*100:.1f}% of seeds")


if __name__ == "__main__":
    main()
