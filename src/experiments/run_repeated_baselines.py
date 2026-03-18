"""
Run repeated-game baselines (non-adaptive agents, no inductive strategies).

Supports three baseline types:
  - fixed_predictor: Repeated Fixed Strategy — each agent gets one predictor
    at t=0 and uses it forever. Game seeded with bootstrap history. (DEFAULT)
  - all_random: i.i.d. random agents with fixed p_attend.
  - mixed: half random agents, half fixed-attendance agents.

These provide fair comparators for learned inductive strategies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.config import RepeatedGameConfig
from src.experiments.populations import build_fixed_predictor_population
from src.experiments.run_repeated_fixed_strategy import bootstrap_history
from src.game.repeated_game import RepeatedMinorityGame


def _build_baseline_agents(
    n_players: int,
    baseline_type: str = "fixed_predictor",
    p_attend: float = 0.55,
    predicted_attendance: int = 58,
    seed: int = 42,
) -> list:
    """
    Build baseline agents for comparison experiments.

    Args:
        n_players: Number of agents.
        baseline_type: "fixed_predictor" (default) for Repeated Fixed Strategy,
            "all_random" for pure i.i.d. agents, "mixed" for half random /
            half fixed-attendance.
        p_attend: Attendance probability for random agents.
        predicted_attendance: Fixed attendance prediction for FixedAttendanceAgent.
        seed: Random seed for fixed_predictor assignment.

    Returns:
        List of baseline agents.
    """
    if baseline_type == "fixed_predictor":
        return build_fixed_predictor_population(
            n_players=n_players,
            seed=seed,
            cover_all_predictors=True,
        )

    if baseline_type == "all_random":
        return [RandomAgent(p_attend=p_attend) for _ in range(n_players)]

    if baseline_type == "mixed":
        agents = []
        split = n_players // 2
        for _ in range(split):
            agents.append(RandomAgent(p_attend=p_attend))
        for _ in range(n_players - split):
            agents.append(FixedAttendanceAgent(predicted_attendance=predicted_attendance))
        return agents

    raise ValueError(f"Unknown baseline_type: {baseline_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    parser.add_argument(
        "--baseline_type",
        choices=["fixed_predictor", "all_random", "mixed"],
        default="fixed_predictor",
        help="Baseline type (default: fixed_predictor = Repeated Fixed Strategy)"
    )
    parser.add_argument("--p_attend", type=float, default=0.55)
    parser.add_argument("--predicted_attendance", type=int, default=58)
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    agents = _build_baseline_agents(
        config.n_players,
        baseline_type=args.baseline_type,
        p_attend=args.p_attend,
        predicted_attendance=args.predicted_attendance,
        seed=config.seed,
    )

    # Use bootstrap history for fixed_predictor baseline
    init_history = None
    if args.baseline_type == "fixed_predictor":
        init_history = bootstrap_history(
            n_players=config.n_players,
            threshold=config.threshold,
            length=8,
            seed=config.seed,
        )

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
        initial_attendance_history=init_history,
    )
    result = game.play()

    out = Path(args.output_dir)
    result.save_outputs(out)
    print(f"Baseline repeated game ({args.baseline_type}): {out.resolve()}")
    if args.baseline_type == "fixed_predictor":
        print(f"  Bootstrap history: {init_history}")
    for k, v in result.summary().items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
