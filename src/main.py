from __future__ import annotations

import argparse

from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.config import GameConfig
from src.game.static_game import StaticMinorityGame


def build_agents(n_players: int, threshold: int) -> list:
    """
    Create a simple mixed population for the first static run.
    Half random, half fixed-prediction.
    """
    agents = []
    split = n_players // 2

    for _ in range(split):
        agents.append(RandomAgent(p_attend=0.55))
    for _ in range(n_players - split):
        agents.append(FixedAttendanceAgent(predicted_attendance=58, threshold=threshold))

    return agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one static minority-game simulation.")
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        seed=args.seed,
    )

    agents = build_agents(config.n_players, config.threshold)
    game = StaticMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()

    print("Static minority game result")
    print(f"N = {config.n_players}")
    print(f"L = {config.threshold}")
    print(f"attendance = {result.attendance}")
    print(f"attendance_rate = {result.attendance_rate:.3f}")
    print(f"overcrowded = {result.overcrowded}")
    print(f"number_of_winners = {len(result.winners)}")
    print(f"number_of_losers = {len(result.losers)}")
    print(f"first_10_actions = {result.actions[:10]}")
    print(f"first_10_payoffs = {result.payoffs[:10]}")


if __name__ == "__main__":
    main()
