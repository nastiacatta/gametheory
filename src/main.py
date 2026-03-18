"""
CLI for the minority game: static (one shot) or repeated (m rounds).

Run from project root: python -m src.main static | repeated [options]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.agents.base import BaseAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.config import RepeatedGameConfig, StaticGameConfig
from src.game.repeated_game import RepeatedMinorityGame
from src.game.static_game import StaticMinorityGame


def build_agents(args: argparse.Namespace) -> List[BaseAgent]:
    """
    Build agent population based on CLI arguments.

    Populations:
    - random: all RandomAgent with p_attend
    - fixed: all FixedAttendanceAgent with predicted_attendance
    - mixed: half random, half fixed (default)
    """
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
    config = StaticGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        seed=args.seed,
    )
    agents = build_agents(args)

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
    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    agents = build_agents(args)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minority game runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    static_parser = subparsers.add_parser("static")
    static_parser.add_argument("--n_players", type=int, default=101)
    static_parser.add_argument("--threshold", type=int, default=60)
    static_parser.add_argument("--seed", type=int, default=42)
    static_parser.add_argument("--population", choices=["random", "fixed", "mixed"], default="mixed")
    static_parser.add_argument("--p_attend", type=float, default=0.55)
    static_parser.add_argument("--predicted_attendance", type=int, default=58)

    repeated_parser = subparsers.add_parser("repeated")
    repeated_parser.add_argument("--n_players", type=int, default=101)
    repeated_parser.add_argument("--threshold", type=int, default=60)
    repeated_parser.add_argument("--n_rounds", type=int, default=200)
    repeated_parser.add_argument("--seed", type=int, default=42)
    repeated_parser.add_argument("--output_dir", type=str, default="outputs")
    repeated_parser.add_argument("--population", choices=["random", "fixed", "mixed"], default="mixed")
    repeated_parser.add_argument("--p_attend", type=float, default=0.55)
    repeated_parser.add_argument("--predicted_attendance", type=int, default=58)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "static":
        run_static(args)
    elif args.mode == "repeated":
        run_repeated(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
