"""
Scoring-rule comparison: absolute-error vs virtual-payoff predictor scoring.

Runs two separate repeated games with matched initial conditions:
  - same pre-sampled predictor banks
  - same bootstrap history
  - same (n, L, T, seed)

but different agent objects, different game objects, and different output folders.
Any difference in outcomes is attributable to the scoring rule alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import plot_scoring_rule_comparison
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    sample_predictor_banks,
    build_best_predictor_from_banks,
    build_virtual_payoff_from_banks,
)
from src.game.repeated_game import RepeatedMinorityGame


def _bootstrap_history(
    n_players: int,
    threshold: int,
    length: int,
    seed: int,
) -> list[int]:
    """Generate a short synthetic attendance history for predictor warm-up."""
    if length <= 0:
        return []
    rng = np.random.default_rng(seed + 10_000)
    p = threshold / n_players
    return [int(rng.binomial(n_players, p)) for _ in range(length)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scoring-rule comparison: absolute-error vs virtual-payoff",
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--bootstrap_history_len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/scoring_rule_comparison",
    )
    args = parser.parse_args()

    config = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=args.seed,
    )

    banks = sample_predictor_banks(
        n_players=config.n_players,
        predictors_per_agent=args.predictors_per_agent,
        seed=config.seed,
    )

    initial_history = _bootstrap_history(
        n_players=config.n_players,
        threshold=config.threshold,
        length=args.bootstrap_history_len,
        seed=config.seed,
    )

    abs_agents = build_best_predictor_from_banks(banks)
    virtual_agents = build_virtual_payoff_from_banks(banks)

    abs_game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=abs_agents,
        seed=config.seed,
        initial_attendance_history=initial_history,
    )
    virtual_game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=virtual_agents,
        seed=config.seed,
        initial_attendance_history=initial_history,
    )

    abs_result = abs_game.play()
    virtual_result = virtual_game.play()

    out = Path(args.output_dir)
    out_abs = out / "absolute_error"
    out_virtual = out / "virtual_payoff"
    out_abs.mkdir(parents=True, exist_ok=True)
    out_virtual.mkdir(parents=True, exist_ok=True)

    abs_result.rounds_dataframe().to_csv(out_abs / "rounds.csv", index=False)
    abs_result.players_dataframe().to_csv(out_abs / "players.csv", index=False)
    virtual_result.rounds_dataframe().to_csv(out_virtual / "rounds.csv", index=False)
    virtual_result.players_dataframe().to_csv(out_virtual / "players.csv", index=False)

    abs_histories = [a.predictor_history for a in abs_agents]
    virtual_histories = [a.predictor_history for a in virtual_agents]
    predictor_names = abs_agents[0].predictor_names if abs_agents else []

    abs_metrics = compute_all_metrics(
        abs_result.attendance_history,
        abs_result.cumulative_payoffs,
        config.threshold,
        predictor_histories=abs_histories,
    )
    virtual_metrics = compute_all_metrics(
        virtual_result.attendance_history,
        virtual_result.cumulative_payoffs,
        config.threshold,
        predictor_histories=virtual_histories,
    )

    pd.DataFrame([abs_metrics]).to_csv(out_abs / "summary.csv", index=False)
    pd.DataFrame([virtual_metrics]).to_csv(out_virtual / "summary.csv", index=False)

    pd.DataFrame({
        "player_id": range(config.n_players),
        "predictors": [
            ",".join(name for name, _ in bank) for bank in banks
        ],
    }).to_csv(out / "matched_predictor_banks.csv", index=False)

    pd.DataFrame({
        "bootstrap_history": initial_history,
    }).to_csv(out / "bootstrap_history.csv", index=False)

    comparison_df = pd.DataFrame([
        {"game": "absolute_error", **abs_metrics},
        {"game": "virtual_payoff", **virtual_metrics},
    ])
    comparison_df.to_csv(out / "comparison_summary.csv", index=False)

    plot_scoring_rule_comparison(
        attendance_abs=abs_result.attendance_history,
        attendance_virtual=virtual_result.attendance_history,
        predictor_histories_abs=abs_histories,
        predictor_histories_virtual=virtual_histories,
        predictor_names=predictor_names,
        threshold=config.threshold,
        output_path=out / "scoring_rule_comparison.png",
    )

    print(f"Scoring-rule comparison written to: {out.resolve()}")
    print()
    print("=== ABSOLUTE-ERROR SCORING ===")
    for k, v in abs_metrics.items():
        print(f"  {k}={v}")
    print()
    print("=== VIRTUAL-PAYOFF SCORING ===")
    for k, v in virtual_metrics.items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
