"""
Paired comparison: non_recency vs recency on identical predictor banks.

Runs both modes with the exact same predictor assignments to ensure any
difference in outcomes is due to the scoring rule, not bank-assignment effects.

Outputs:
  - summary_comparison.csv: two-row comparison of key metrics
  - rounds_comparison.csv: round-level data with a mode column
  - players_comparison.csv: player-level data with a mode column
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
    sample_predictor_banks,
)
from src.game.repeated_game import RepeatedMinorityGame


def run_one(
    mode: str,
    config: RepeatedGameConfig,
    predictor_banks,
    lambda_decay: float,
):
    """Run one mode (non_recency or recency) with given predictor banks."""
    if mode == "non_recency":
        agents = build_homogeneous_non_recency(
            config.n_players,
            predictors_per_agent=len(predictor_banks[0]),
            seed=config.seed,
            predictor_banks=predictor_banks,
        )
    else:
        agents = build_homogeneous_recency(
            config.n_players,
            lambda_decay=lambda_decay,
            predictors_per_agent=len(predictor_banks[0]),
            seed=config.seed,
            predictor_banks=predictor_banks,
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
    metrics["mode"] = mode

    rounds = result.rounds_dataframe().copy()
    rounds["mode"] = mode

    players = result.players_dataframe().copy()
    players["mode"] = mode
    players["agent_type"] = [type(a).__name__ for a in agents]

    return metrics, rounds, players


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison: non_recency vs recency on identical predictor banks"
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--lambda_decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/inductive_paired")
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    predictor_banks = sample_predictor_banks(
        n_players=config.n_players,
        predictors_per_agent=args.predictors_per_agent,
        seed=config.seed,
    )

    records = []
    rounds_frames = []
    player_frames = []

    for mode in ["non_recency", "recency"]:
        metrics, rounds, players = run_one(
            mode=mode,
            config=config,
            predictor_banks=predictor_banks,
            lambda_decay=args.lambda_decay,
        )
        records.append(metrics)
        rounds_frames.append(rounds)
        player_frames.append(players)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(out / "summary_comparison.csv", index=False)

    pd.concat(rounds_frames, ignore_index=True).to_csv(out / "rounds_comparison.csv", index=False)
    pd.concat(player_frames, ignore_index=True).to_csv(out / "players_comparison.csv", index=False)

    print(f"Paired comparison: {out.resolve()}")
    print()
    print("=== SUMMARY ===")
    for _, row in summary_df.iterrows():
        mode = row["mode"]
        print(f"\n{mode}:")
        print(f"  mean_attendance = {row['mean_attendance']:.2f}")
        print(f"  mad_from_threshold = {row['mad_from_threshold']:.2f}")
        print(f"  overcrowding_rate = {row['overcrowding_rate']:.3f}")
        print(f"  mean_cumulative_payoff = {row['mean_cumulative_payoff']:.2f}")
        print(f"  switch_rate = {row['switch_rate']:.4f}")


if __name__ == "__main__":
    main()
