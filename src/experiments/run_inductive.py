"""
Run repeated-game experiments with inductive strategies (non_recency, recency).

Virtual-payoff scoring under weak-threshold convention:
  - non_recency: s_j(t+1) = s_j(t) + \tilde u_j(t)
  - recency:     s_j(t+1) = lambda * s_j(t) + \tilde u_j(t)

Both use the same predictor bank, same action rule (hard argmax), same
repeated-game engine. The only difference is whether old predictor performance
is exponentially forgotten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import (
    plot_attendance_over_time,
    plot_attendance_deviation_over_time,
    plot_cumulative_average_attendance,
    plot_rolling_variance_from_threshold,
    plot_threshold_distance_histogram,
    plot_attendance_histogram,
    plot_payoff_histogram,
    plot_ranked_final_payoffs,
    plot_predictor_share_over_time,
    plot_predictor_share_heatmap,
)
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_homogeneous_non_recency,
    build_homogeneous_recency,
)
from src.game.repeated_game import RepeatedMinorityGame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inductive repeated game with virtual-payoff scoring",
    )
    parser.add_argument(
        "--mode",
        choices=["non_recency", "recency"],
        default="recency",
        help="Score update mode (default: recency)",
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument(
        "--lambda_decay",
        type=float,
        default=0.95,
        help="Virtual-payoff score decay for recency mode (ignored for non_recency)",
    )
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/inductive")
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    if args.mode == "non_recency":
        agents = build_homogeneous_non_recency(
            config.n_players,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    else:
        agents = build_homogeneous_recency(
            config.n_players,
            lambda_decay=args.lambda_decay,
            predictors_per_agent=args.predictors_per_agent,
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

    predictor_histories = [
        getattr(a, "predictor_history", []) for a in agents
    ]
    predictor_names = (
        agents[0].predictor_names if agents and hasattr(agents[0], "predictor_names") else []
    )
    use_histories = predictor_histories if (predictor_histories and predictor_histories[0]) else None

    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        config.threshold,
        predictor_histories=use_histories,
    )

    out = Path(args.output_dir) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    result.rounds_dataframe().to_csv(out / "rounds.csv", index=False)

    player_df = result.players_dataframe().copy()
    player_df["agent_type"] = [type(a).__name__ for a in agents]
    player_df.to_csv(out / "players.csv", index=False)

    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False)

    plot_attendance_over_time(
        result.attendance_history,
        config.threshold,
        out / "attendance.png",
    )

    plot_attendance_deviation_over_time(
        result.attendance_history,
        config.threshold,
        out / "attendance_deviation.png",
    )

    plot_cumulative_average_attendance(
        result.attendance_history,
        config.threshold,
        out / "cum_avg_attendance.png",
    )

    plot_rolling_variance_from_threshold(
        result.attendance_history,
        config.threshold,
        window=max(20, config.n_rounds // 10),
        output_path=out / "rolling_variance.png",
    )

    plot_threshold_distance_histogram(
        result.attendance_history,
        config.threshold,
        out / "attendance_deviation_hist.png",
    )

    plot_attendance_histogram(
        result.attendance_history,
        config.threshold,
        out / "attendance_hist.png",
    )

    plot_payoff_histogram(
        result.cumulative_payoffs,
        out / "payoff_hist.png",
    )

    plot_ranked_final_payoffs(
        result.cumulative_payoffs,
        out / "ranked_final_payoffs.png",
    )

    if use_histories:
        plot_predictor_share_over_time(
            predictor_histories,
            predictor_names,
            out / "predictor_share.png",
        )
        plot_predictor_share_heatmap(
            predictor_histories,
            predictor_names,
            out / "predictor_share_heatmap.png",
        )

    print(f"Inductive ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
