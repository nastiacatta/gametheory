"""
Run repeated-game baselines (fixed/random agents, no inductive strategies).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.config import RepeatedGameConfig
from src.game.repeated_game import RepeatedMinorityGame


def _build_baseline_agents(n_players: int) -> list:
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
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/baselines")
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )
    agents = _build_baseline_agents(config.n_players)

    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    out = Path(args.output_dir)
    result.save_outputs(out)
    print(f"Baseline repeated game: {out.resolve()}")
    for k, v in result.summary().items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
