"""
Scoring-rule comparison: absolute-error vs virtual-payoff predictor scoring.

Runs many matched pairs of repeated games across different seeds.
For each seed:
  - same pre-sampled predictor banks
  - same bootstrap history
  - same (n, L, T, seed)
  - different agent objects (one with absolute-error scoring, one with virtual-payoff)
  - different game objects

Any difference in outcomes is attributable to the scoring rule alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import plot_paired_scoring_differences, plot_scoring_rule_comparison
from src.experiments.populations import (
    sample_predictor_banks,
    build_best_predictor_from_banks,
    build_virtual_payoff_from_banks,
)
from src.game.repeated_game import RepeatedMinorityGame


def _bootstrap_history(
    n_players: int,
    threshold: int,
    length: int,
    seed: int,
) -> list[int]:
    """Generate a short synthetic attendance history for predictor warm-up."""
    if length <= 0:
        return []
    rng = np.random.default_rng(seed + 10_000)
    p = threshold / n_players
    return [int(rng.binomial(n_players, p)) for _ in range(length)]


def _run_one_pair(
    n_players: int,
    threshold: int,
    n_rounds: int,
    predictors_per_agent: int,
    bootstrap_history_len: int,
    seed: int,
) -> dict:
    """Run one matched pair of games and return metrics."""
    banks = sample_predictor_banks(
        n_players=n_players,
        predictors_per_agent=predictors_per_agent,
        seed=seed,
    )

    init_history = _bootstrap_history(
        n_players=n_players,
        threshold=threshold,
        length=bootstrap_history_len,
        seed=seed,
    )

    abs_agents = build_best_predictor_from_banks(banks)
    virt_agents = build_virtual_payoff_from_banks(banks)

    abs_game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=abs_agents,
        seed=seed,
        initial_attendance_history=init_history,
    )
    virt_game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=virt_agents,
        seed=seed,
        initial_attendance_history=init_history,
    )

    abs_result = abs_game.play()
    virt_result = virt_game.play()

    abs_metrics = compute_all_metrics(
        abs_result.attendance_history,
        abs_result.cumulative_payoffs,
        threshold,
    )
    virt_metrics = compute_all_metrics(
        virt_result.attendance_history,
        virt_result.cumulative_payoffs,
        threshold,
    )

    return {
        "seed": seed,
        "abs_mean_payoff": abs_metrics["mean_cumulative_payoff"],
        "virt_mean_payoff": virt_metrics["mean_cumulative_payoff"],
        "abs_overcrowding_rate": abs_metrics["overcrowding_rate"],
        "virt_overcrowding_rate": virt_metrics["overcrowding_rate"],
        "abs_mad_from_threshold": abs_metrics["mad_from_threshold"],
        "virt_mad_from_threshold": virt_metrics["mad_from_threshold"],
        "delta_mean_payoff": virt_metrics["mean_cumulative_payoff"] - abs_metrics["mean_cumulative_payoff"],
        "delta_overcrowding_rate": virt_metrics["overcrowding_rate"] - abs_metrics["overcrowding_rate"],
        "delta_mad_from_threshold": virt_metrics["mad_from_threshold"] - abs_metrics["mad_from_threshold"],
    }


def _run_single_pair_with_details(
    n_players: int,
    threshold: int,
    n_rounds: int,
    predictors_per_agent: int,
    bootstrap_history_len: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Run one pair and save detailed outputs including the four-panel comparison plot."""
    banks = sample_predictor_banks(
        n_players=n_players,
        predictors_per_agent=predictors_per_agent,
        seed=seed,
    )

    init_history = _bootstrap_history(
        n_players=n_players,
        threshold=threshold,
        length=bootstrap_history_len,
        seed=seed,
    )

    abs_agents = build_best_predictor_from_banks(banks)
    virt_agents = build_virtual_payoff_from_banks(banks)

    abs_game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=abs_agents,
        seed=seed,
        initial_attendance_history=init_history,
    )
    virt_game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=virt_agents,
        seed=seed,
        initial_attendance_history=init_history,
    )

    abs_result = abs_game.play()
    virt_result = virt_game.play()

    abs_histories = [a.predictor_history for a in abs_agents]
    virt_histories = [a.predictor_history for a in virt_agents]
    predictor_names = abs_agents[0].predictor_names if abs_agents else []

    plot_scoring_rule_comparison(
        attendance_abs=abs_result.attendance_history,
        attendance_virtual=virt_result.attendance_history,
        predictor_histories_abs=abs_histories,
        predictor_histories_virtual=virt_histories,
        predictor_names=predictor_names,
        threshold=threshold,
        output_path=output_dir / "single_seed_comparison.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scoring-rule comparison: absolute-error vs virtual-payoff (many seeds)",
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--bootstrap_history_len", type=int, default=8)
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/scoring_rule_comparison",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Running {args.n_seeds} matched pairs...")
    rows = []
    for i, seed in enumerate(range(args.seed_start, args.seed_start + args.n_seeds)):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Seed {i + 1}/{args.n_seeds}...")
        rows.append(
            _run_one_pair(
                n_players=args.n_players,
                threshold=args.threshold,
                n_rounds=args.n_rounds,
                predictors_per_agent=args.predictors_per_agent,
                bootstrap_history_len=args.bootstrap_history_len,
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
        "mean_delta_mean_payoff": df["delta_mean_payoff"].mean(),
        "std_delta_mean_payoff": df["delta_mean_payoff"].std(),
        "mean_delta_overcrowding_rate": df["delta_overcrowding_rate"].mean(),
        "std_delta_overcrowding_rate": df["delta_overcrowding_rate"].std(),
        "mean_delta_mad_from_threshold": df["delta_mad_from_threshold"].mean(),
        "std_delta_mad_from_threshold": df["delta_mad_from_threshold"].std(),
        "share_virtual_better_payoff": (df["delta_mean_payoff"] > 0).mean(),
        "share_virtual_lower_overcrowding": (df["delta_overcrowding_rate"] < 0).mean(),
        "share_virtual_lower_deviation": (df["delta_mad_from_threshold"] < 0).mean(),
    }])
    summary.to_csv(out / "comparison_summary.csv", index=False)

    plot_paired_scoring_differences(df, output_path=out / "paired_difference_plot.png")

    _run_single_pair_with_details(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        predictors_per_agent=args.predictors_per_agent,
        bootstrap_history_len=args.bootstrap_history_len,
        seed=args.seed_start,
        output_dir=out,
    )

    print()
    print(f"Wrote outputs to: {out.resolve()}")
    print()
    print("=== SUMMARY ===")
    print(f"  Seeds: {args.n_seeds}")
    print(f"  Mean Δ payoff:      {summary['mean_delta_mean_payoff'].iloc[0]:+.3f} (virtual - abs)")
    print(f"  Mean Δ overcrowd:   {summary['mean_delta_overcrowding_rate'].iloc[0]:+.4f}")
    print(f"  Mean Δ MAD:         {summary['mean_delta_mad_from_threshold'].iloc[0]:+.3f}")
    print()
    print(f"  Virtual better payoff:      {summary['share_virtual_better_payoff'].iloc[0]*100:.1f}% of seeds")
    print(f"  Virtual lower overcrowding: {summary['share_virtual_lower_overcrowding'].iloc[0]*100:.1f}% of seeds")
    print(f"  Virtual lower MAD:          {summary['share_virtual_lower_deviation'].iloc[0]*100:.1f}% of seeds")


if __name__ == "__main__":
    main()
