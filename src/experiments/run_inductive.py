"""
Run repeated-game experiments with inductive strategies (best-predictor, softmax).
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
)
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_homogeneous_best_predictor,
    build_homogeneous_softmax,
)
from src.game.repeated_game import RepeatedMinorityGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["best", "softmax"], required=True)
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--beta", type=float, default=1.0, help="Softmax inverse temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/inductive")
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    if args.mode == "best":
        agents = build_homogeneous_best_predictor(
            config.n_players, predictors_per_agent=3, seed=config.seed
        )
    else:
        agents = build_homogeneous_softmax(
            config.n_players, beta=args.beta, predictors_per_agent=3, seed=config.seed
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

    out = Path(args.output_dir)
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
        window=max(10, config.n_rounds // 10),
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

    print(f"Inductive ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
