"""
Run repeated-game experiments with heterogeneous populations (mix, producer/speculator).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import (
    plot_attendance_over_time,
    plot_cumulative_average_attendance,
    plot_payoff_histogram,
)
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_heterogeneous,
    build_producer_speculator,
)
from src.game.repeated_game import RepeatedMinorityGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mix", "producer_speculator"], required=True)
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--p_best", type=float, default=0.5, help="Share of best-predictor (mix mode)")
    parser.add_argument("--p_softmax", type=float, default=0.5, help="Share of softmax (mix mode)")
    parser.add_argument("--p_random", type=float, default=0.0, help="Share of random agents (mix mode)")
    parser.add_argument("--n_producers", type=int, default=50, help="Producer count (producer_speculator mode)")
    parser.add_argument("--speculator_type", choices=["best", "softmax"], default="best")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--producer_base_prediction", type=float, default=None, help="Producer base prediction (defaults to threshold)")
    parser.add_argument("--producer_noise_std", type=float, default=5.0, help="Producer noise std")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/heterogeneous")
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    if args.mode == "mix":
        agents = build_heterogeneous(
            config.n_players,
            p_best=args.p_best,
            p_softmax=args.p_softmax,
            p_random=args.p_random,
            beta=args.beta,
            predictors_per_agent=3,
            seed=config.seed,
        )
    else:
        agents = build_producer_speculator(
            config.n_players,
            n_producers=args.n_producers,
            speculator_type=args.speculator_type,
            beta=args.beta,
            predictors_per_agent=3,
            seed=config.seed,
            producer_base_prediction=args.producer_base_prediction,
            producer_noise_std=args.producer_noise_std,
            threshold=config.threshold,
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
    predictor_histories = [
        h for h in predictor_histories if h
    ]

    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        config.threshold,
        predictor_histories=predictor_histories if predictor_histories else None,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.rounds_dataframe().to_csv(out / "rounds.csv", index=False)
    result.players_dataframe().to_csv(out / "players.csv", index=False)

    import pandas as pd
    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False)

    plot_attendance_over_time(result.attendance_history, config.threshold, out / "attendance.png")
    plot_cumulative_average_attendance(result.attendance_history, config.threshold, out / "cum_avg_attendance.png")
    plot_payoff_histogram(result.cumulative_payoffs, out / "payoff_hist.png")

    print(f"Heterogeneous ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
