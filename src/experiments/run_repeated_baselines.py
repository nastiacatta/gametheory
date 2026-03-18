"""
Run repeated-game baselines (i.i.d. random agents, no inductive strategies).

Neutral benchmark: E[A_t] = L, providing a fair comparator for learned strategies.
For n players with p = L/n, the expected threshold-centred MSE is:
    E[(A_t - L)^2] = np(1-p) = L(1 - L/n)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.random_agent import RandomAgent
from src.config import RepeatedGameConfig
from src.game.repeated_game import RepeatedMinorityGame


def _build_baseline_agents(n_players: int, threshold: int) -> list:
    """
    Neutral i.i.d. baseline: choose p so that E[A_t] = L.

    This gives E[A_t] = n * p = L, and variance:
        E[(A_t - L)^2] = n * p * (1 - p)
    For n=101, L=60: E[(A_t - L)^2] ≈ 24.356
    """
    p = threshold / n_players
    return [RandomAgent(p_attend=p) for _ in range(n_players)]


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
    agents = _build_baseline_agents(config.n_players, config.threshold)

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
