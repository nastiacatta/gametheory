"""
Run repeated-game baselines (non-adaptive agents, no inductive strategies).

Supports three baseline types via --mode:
  - fixed_random: half random agents, half fixed-attendance agents (original baseline)
  - fixed_predictor: homogeneous fixed predictor, same predictor for all players
  - heterogeneous_fixed_predictor: each player gets one predictor at t=0, never switches

These provide fair comparators for learned inductive strategies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import (
    plot_attendance_over_time,
    plot_cumulative_average_attendance,
    plot_payoff_histogram,
)
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_homogeneous_fixed_predictor,
    build_heterogeneous_fixed_predictor,
)
from src.game.repeated_game import RepeatedMinorityGame


def _build_fixed_random_agents(n_players: int) -> list:
    """Baseline: half random, half fixed-attendance."""
    agents = []
    split = n_players // 2

    for _ in range(split):
        agents.append(RandomAgent(p_attend=0.55))

    for _ in range(n_players - split):
        agents.append(FixedAttendanceAgent(predicted_attendance=58))

    return agents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "fixed_random",
            "fixed_predictor",
            "heterogeneous_fixed_predictor",
        ],
        default="fixed_random",
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/repeated_baselines")

    parser.add_argument(
        "--predictor_name",
        type=str,
        default="last_value",
        help="Predictor for homogeneous fixed-predictor mode",
    )

    parser.add_argument(
        "--cover_all_predictors",
        action="store_true",
        help="Ensure every predictor appears at least once when possible",
    )

    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    if args.mode == "fixed_random":
        agents = _build_fixed_random_agents(config.n_players)
    elif args.mode == "fixed_predictor":
        agents = build_homogeneous_fixed_predictor(
            config.n_players,
            predictor_name=args.predictor_name,
        )
    else:
        agents = build_heterogeneous_fixed_predictor(
            config.n_players,
            seed=config.seed,
            cover_all_predictors=args.cover_all_predictors,
        )

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )

    result = game.play()

    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        config.threshold,
    )

    out = Path(args.output_dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    result.rounds_dataframe().to_csv(out / "rounds.csv", index=False)
    result.players_dataframe().to_csv(out / "players.csv", index=False)

    import pandas as pd
    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False)

    plot_attendance_over_time(
        result.attendance_history,
        config.threshold,
        out / "attendance.png",
    )
    plot_cumulative_average_attendance(
        result.attendance_history,
        config.threshold,
        out / "cum_avg_attendance.png",
    )
    plot_payoff_histogram(
        result.cumulative_payoffs,
        out / "payoff_hist.png",
    )

    print(f"Repeated baseline ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
