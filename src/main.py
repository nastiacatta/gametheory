"""
Unified CLI for the El Farol threshold minority game.

Subcommands:
    static      - Single-shot game
    repeated    - Repeated game with basic populations
    inductive   - Repeated game with inductive agents (best, softmax, recency, turnover)
    heterogeneous - Repeated game with mixed populations
    sweep       - Parameter sweep over multiple seeds and modes

Run from project root: python -m src.main <subcommand> [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from src.agents.base import BaseAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
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
)
from src.config import RepeatedGameConfig, StaticGameConfig
from src.experiments.populations import (
    build_heterogeneous,
    build_homogeneous_best_predictor,
    build_homogeneous_recency,
    build_homogeneous_softmax,
    build_homogeneous_turnover,
    build_producer_speculator,
)
from src.game.repeated_game import RepeatedMinorityGame
from src.game.static_game import StaticMinorityGame


def build_basic_agents(args: argparse.Namespace) -> List[BaseAgent]:
    """Build agent population for basic static/repeated modes."""
    n_players = args.n_players

    if args.population == "random":
        return [RandomAgent(p_attend=args.p_attend) for _ in range(n_players)]

    if args.population == "fixed":
        return [
            FixedAttendanceAgent(predicted_attendance=args.predicted_attendance)
            for _ in range(n_players)
        ]

    if args.population == "mixed":
        split = n_players // 2
        agents: List[BaseAgent] = []
        agents.extend(RandomAgent(p_attend=args.p_attend) for _ in range(split))
        agents.extend(
            FixedAttendanceAgent(predicted_attendance=args.predicted_attendance)
            for _ in range(n_players - split)
        )
        return agents

    raise ValueError(f"Unknown population: {args.population}")


def run_static(args: argparse.Namespace) -> None:
    """Run single-shot game."""
    config = StaticGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        seed=args.seed,
    )
    agents = build_basic_agents(args)

    game = StaticMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    print("=== STATIC GAME ===")
    print(f"n_players={config.n_players}")
    print(f"threshold={config.threshold}")
    print(f"attendance={result.attendance}")
    print(f"attendance_rate={result.attendance_rate:.3f}")
    print(f"overcrowded={result.overcrowded}")
    print(f"number_of_winners={len(result.winners)}")
    print(f"number_of_losers={len(result.losers)}")
    print(f"first_10_actions={result.actions[:10]}")
    print(f"first_10_payoffs={result.payoffs[:10]}")


def run_repeated(args: argparse.Namespace) -> None:
    """Run repeated game with basic populations."""
    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    agents = build_basic_agents(args)

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    summary = result.summary()
    print("=== REPEATED GAME ===")
    for key, value in summary.items():
        print(f"{key}={value}")

    output_dir = Path(args.output_dir)
    result.save_outputs(output_dir)
    print(f"Saved CSV and figure outputs to: {output_dir.resolve()}")


def run_inductive(args: argparse.Namespace) -> None:
    """Run repeated game with inductive agents."""
    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    if args.mode == "best":
        agents = build_homogeneous_best_predictor(
            config.n_players,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    elif args.mode == "softmax":
        agents = build_homogeneous_softmax(
            config.n_players,
            beta=args.beta,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    elif args.mode == "recency":
        agents = build_homogeneous_recency(
            config.n_players,
            lambda_decay=args.lambda_decay,
            selection=args.selection,
            beta=args.beta,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    elif args.mode == "turnover":
        agents = build_homogeneous_turnover(
            config.n_players,
            lambda_decay=args.lambda_decay,
            patience=args.patience,
            error_threshold=args.error_threshold,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown inductive mode: {args.mode}")

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
    
    player_df = result.players_dataframe().copy()
    player_df["agent_type"] = [type(a).__name__ for a in agents]
    player_df.to_csv(out / "players.csv", index=False)
    
    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False)

    plot_attendance_over_time(result.attendance_history, config.threshold, out / "attendance.png")
    plot_attendance_deviation_over_time(result.attendance_history, config.threshold, out / "attendance_deviation.png")
    plot_cumulative_average_attendance(result.attendance_history, config.threshold, out / "cum_avg_attendance.png")
    plot_rolling_variance_from_threshold(
        result.attendance_history, config.threshold,
        window=max(10, config.n_rounds // 10),
        output_path=out / "rolling_variance.png"
    )
    plot_threshold_distance_histogram(result.attendance_history, config.threshold, out / "attendance_deviation_hist.png")
    plot_attendance_histogram(result.attendance_history, config.threshold, out / "attendance_hist.png")
    plot_payoff_histogram(result.cumulative_payoffs, out / "payoff_hist.png")
    plot_ranked_final_payoffs(result.cumulative_payoffs, out / "ranked_final_payoffs.png")

    print(f"Inductive ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"  {k}={v}")


def run_heterogeneous(args: argparse.Namespace) -> None:
    """Run repeated game with heterogeneous populations."""
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
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
        )
    elif args.mode == "producer_speculator":
        agents = build_producer_speculator(
            config.n_players,
            n_producers=args.n_producers,
            speculator_type=args.speculator_type,
            beta=args.beta,
            predictors_per_agent=args.predictors_per_agent,
            seed=config.seed,
            producer_base_prediction=args.producer_base_prediction,
            producer_noise_std=args.producer_noise_std,
            threshold=config.threshold,
        )
    else:
        raise ValueError(f"Unknown heterogeneous mode: {args.mode}")

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    predictor_histories = [getattr(a, "predictor_history", []) for a in agents]
    predictor_histories = [h for h in predictor_histories if h]

    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        config.threshold,
        predictor_histories=predictor_histories if predictor_histories else None,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.rounds_dataframe().to_csv(out / "rounds.csv", index=False)

    player_df = result.players_dataframe().copy()
    player_df["agent_type"] = [type(a).__name__ for a in agents]
    player_df.to_csv(out / "players.csv", index=False)

    pd.DataFrame([metrics]).to_csv(out / "summary.csv", index=False)

    plot_attendance_over_time(result.attendance_history, config.threshold, out / "attendance.png")
    plot_attendance_deviation_over_time(result.attendance_history, config.threshold, out / "attendance_deviation.png")
    plot_cumulative_average_attendance(result.attendance_history, config.threshold, out / "cum_avg_attendance.png")
    plot_rolling_variance_from_threshold(
        result.attendance_history, config.threshold,
        window=max(10, config.n_rounds // 10),
        output_path=out / "rolling_variance.png"
    )
    plot_threshold_distance_histogram(result.attendance_history, config.threshold, out / "attendance_deviation_hist.png")
    plot_attendance_histogram(result.attendance_history, config.threshold, out / "attendance_hist.png")
    plot_payoff_histogram(result.cumulative_payoffs, out / "payoff_hist.png")
    plot_ranked_final_payoffs(result.cumulative_payoffs, out / "ranked_final_payoffs.png")

    print(f"Heterogeneous ({args.mode}): {out.resolve()}")
    for k, v in metrics.items():
        print(f"  {k}={v}")


def run_sweep(args: argparse.Namespace) -> None:
    """Run multi-seed parameter sweep."""
    from src.experiments.run_sweep import main as sweep_main
    import sys
    
    sys.argv = [
        "run_sweep",
        f"--n_players={args.n_players}",
        f"--threshold={args.threshold}",
        f"--n_rounds={args.n_rounds}",
        f"--n_seeds={args.n_seeds}",
        f"--output_dir={args.output_dir}",
    ]
    sweep_main()


def run_static_sweep(args: argparse.Namespace) -> None:
    """Run static probability sweep experiment."""
    from src.experiments.run_static_probability_sweep import run_probability_sweep
    
    print(f"Running static probability sweep...")
    print(f"  n_players={args.n_players}, threshold={args.threshold}")
    print(f"  n_samples={args.n_samples}, grid_size={args.grid_size}")
    print(f"  seed={args.seed}")
    
    df = run_probability_sweep(
        n_players=args.n_players,
        threshold=args.threshold,
        n_samples=args.n_samples,
        grid_size=args.grid_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    out_path = Path(args.output_dir).resolve()
    print(f"\nOutputs saved to: {out_path}")
    print(f"  - tables/static_probability_sweep.csv")
    print(f"  - figures/static_payoff_vs_p.png")
    print(f"  - figures/static_attendance_vs_p.png")
    print(f"  - figures/static_overcrowding_vs_p.png")
    
    p_capacity = args.threshold / args.n_players
    idx = (df["p"] - p_capacity).abs().idxmin()
    row = df.iloc[idx]
    print(f"\nAt capacity benchmark p = {p_capacity:.4f}:")
    print(f"  mean_attendance = {row['mean_attendance']:.2f}")
    print(f"  mean_payoff_per_player = {row['mean_payoff_per_player']:.4f}")
    print(f"  overcrowding_rate = {row['overcrowding_rate']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="El Farol threshold minority game runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === static ===
    static_parser = subparsers.add_parser("static", help="Single-shot game")
    static_parser.add_argument("--n_players", type=int, default=101)
    static_parser.add_argument("--threshold", type=int, default=60)
    static_parser.add_argument("--seed", type=int, default=42)
    static_parser.add_argument("--population", choices=["random", "fixed", "mixed"], default="random")
    static_parser.add_argument("--p_attend", type=float, default=0.55)
    static_parser.add_argument("--predicted_attendance", type=int, default=58)

    # === repeated ===
    repeated_parser = subparsers.add_parser("repeated", help="Repeated game with basic populations")
    repeated_parser.add_argument("--n_players", type=int, default=101)
    repeated_parser.add_argument("--threshold", type=int, default=60)
    repeated_parser.add_argument("--n_rounds", type=int, default=200)
    repeated_parser.add_argument("--seed", type=int, default=42)
    repeated_parser.add_argument("--output_dir", type=str, default="outputs/repeated")
    repeated_parser.add_argument("--population", choices=["random", "fixed", "mixed"], default="mixed")
    repeated_parser.add_argument("--p_attend", type=float, default=0.55)
    repeated_parser.add_argument("--predicted_attendance", type=int, default=58)

    # === inductive ===
    inductive_parser = subparsers.add_parser("inductive", help="Repeated game with inductive agents")
    inductive_parser.add_argument("--mode", choices=["best", "softmax", "recency", "turnover"], required=True)
    inductive_parser.add_argument("--n_players", type=int, default=101)
    inductive_parser.add_argument("--threshold", type=int, default=60)
    inductive_parser.add_argument("--n_rounds", type=int, default=200)
    inductive_parser.add_argument("--seed", type=int, default=42)
    inductive_parser.add_argument("--output_dir", type=str, default="outputs/inductive")
    inductive_parser.add_argument("--predictors_per_agent", type=int, default=6)
    inductive_parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature (softmax/recency)")
    inductive_parser.add_argument("--lambda_decay", type=float, default=0.95, help="Score decay (recency/turnover)")
    inductive_parser.add_argument("--selection", choices=["argmax", "softmax"], default="argmax", help="Selection rule (recency)")
    inductive_parser.add_argument("--patience", type=int, default=10, help="Failure patience (turnover)")
    inductive_parser.add_argument("--error_threshold", type=float, default=5.0, help="Error threshold (turnover)")

    # === heterogeneous ===
    hetero_parser = subparsers.add_parser("heterogeneous", help="Repeated game with mixed populations")
    hetero_parser.add_argument("--mode", choices=["mix", "producer_speculator"], required=True)
    hetero_parser.add_argument("--n_players", type=int, default=101)
    hetero_parser.add_argument("--threshold", type=int, default=60)
    hetero_parser.add_argument("--n_rounds", type=int, default=200)
    hetero_parser.add_argument("--seed", type=int, default=42)
    hetero_parser.add_argument("--output_dir", type=str, default="outputs/heterogeneous")
    hetero_parser.add_argument("--predictors_per_agent", type=int, default=6)
    hetero_parser.add_argument("--beta", type=float, default=1.0)
    hetero_parser.add_argument("--p_best", type=float, default=0.5)
    hetero_parser.add_argument("--p_softmax", type=float, default=0.5)
    hetero_parser.add_argument("--p_random", type=float, default=0.0)
    hetero_parser.add_argument("--n_producers", type=int, default=50)
    hetero_parser.add_argument("--speculator_type", choices=["best", "softmax"], default="best")
    hetero_parser.add_argument("--producer_base_prediction", type=float, default=None)
    hetero_parser.add_argument("--producer_noise_std", type=float, default=5.0)

    # === sweep ===
    sweep_parser = subparsers.add_parser("sweep", help="Multi-seed parameter sweep")
    sweep_parser.add_argument("--n_players", type=int, default=101)
    sweep_parser.add_argument("--threshold", type=int, default=60)
    sweep_parser.add_argument("--n_rounds", type=int, default=200)
    sweep_parser.add_argument("--n_seeds", type=int, default=50)
    sweep_parser.add_argument("--output_dir", type=str, default="outputs/sweep")

    # === static-sweep ===
    static_sweep_parser = subparsers.add_parser(
        "static-sweep",
        help="Static probability sweep over p in [0, 1]"
    )
    static_sweep_parser.add_argument("--n_players", type=int, default=101)
    static_sweep_parser.add_argument("--threshold", type=int, default=60)
    static_sweep_parser.add_argument("--n_samples", type=int, default=10_000, help="Monte Carlo samples per p value")
    static_sweep_parser.add_argument("--grid_size", type=int, default=201, help="Number of p values in [0, 1]")
    static_sweep_parser.add_argument("--seed", type=int, default=42)
    static_sweep_parser.add_argument("--output_dir", type=str, default="outputs")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "static":
        run_static(args)
    elif args.command == "repeated":
        run_repeated(args)
    elif args.command == "inductive":
        run_inductive(args)
    elif args.command == "heterogeneous":
        run_heterogeneous(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "static-sweep":
        run_static_sweep(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
