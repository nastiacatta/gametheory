"""
Repeated Fixed Strategy baseline for the El Farol game.

Definition (following Arthur's predictor framework):
    Each player is assigned one predictor at t=0 and applies that same
    predictor to the evolving public attendance history in every round.
    No switching, no score updates, no learning.

    a_i(t) = 1[f_i(H_t) < L]

    where f_i is assigned once at initialisation and H_t is the attendance
    history before round t.

The game is seeded with a bootstrap attendance history of length 8 so that
every predictor has data from round 1.  This avoids the all-attend artefact
when predictors fall back to the threshold on empty history.  The bootstrap
is part of the baseline definition, not an optional feature.

Outputs:
- Standard repeated-game outputs via RepeatedGameResult.save_outputs()
- Additional CSV: predictor_assignment_counts.csv
- Additional plot: predictor_summary.png
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.predictors import default_predictor_library
from src.experiments.populations import build_fixed_predictor_population
from src.game.repeated_game import RepeatedMinorityGame, RepeatedGameResult


def bootstrap_history(
    n_players: int,
    threshold: int,
    length: int,
    seed: int,
) -> list[int]:
    """
    Generate a synthetic initial attendance history by drawing each value
    from Binomial(n_players, threshold/n_players).

    Length 8 covers the longest lookback window in the default predictor
    library (rolling_mean_8, linear_trend_8).
    """
    rng = np.random.default_rng(seed)
    p = threshold / n_players
    return [int(rng.binomial(n_players, p)) for _ in range(length)]


def count_predictor_assignments(agents: List[FixedPredictorAgent]) -> dict[str, int]:
    """Count how many agents are assigned to each predictor."""
    names = [agent.predictor_name for agent in agents]
    return dict(Counter(names))


def compute_predictor_payoffs(
    agents: List[FixedPredictorAgent],
    cumulative_payoffs: List[int],
) -> pd.DataFrame:
    """
    Compute mean cumulative payoff per predictor type.

    Returns DataFrame with columns: predictor_name, n_users, mean_payoff, std_payoff
    """
    from collections import defaultdict

    grouped: dict[str, list[int]] = defaultdict(list)
    for agent, payoff in zip(agents, cumulative_payoffs):
        grouped[agent.predictor_name].append(payoff)

    records = []
    for name in sorted(grouped.keys()):
        payoffs = grouped[name]
        records.append({
            "predictor_name": name,
            "n_users": len(payoffs),
            "mean_payoff": float(np.mean(payoffs)),
            "std_payoff": float(np.std(payoffs)) if len(payoffs) > 1 else 0.0,
        })
    return pd.DataFrame(records)


def plot_predictor_summary(
    predictor_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Bar chart showing predictor assignment counts and mean payoffs.

    Two-axis plot:
    - Left axis (blue bars): number of users assigned to each predictor
    - Right axis (orange bars): mean cumulative payoff per user
    """
    if predictor_df.empty:
        return

    predictor_names = predictor_df["predictor_name"].tolist()
    n_users = predictor_df["n_users"].to_numpy()
    mean_payoff = predictor_df["mean_payoff"].to_numpy()

    x = np.arange(len(predictor_names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, n_users, width=width, color="steelblue", label="# users")
    ax2.bar(x + width / 2, mean_payoff, width=width, color="darkorange", alpha=0.8, label="Mean payoff")

    ax1.set_xlabel("Predictor")
    ax1.set_ylabel("Number of users", color="steelblue")
    ax2.set_ylabel("Mean cumulative payoff", color="darkorange")
    ax1.set_title("Repeated Fixed Strategy: Assignment Counts and Mean Payoff")

    ax1.set_xticks(x)
    ax1.set_xticklabels(predictor_names, rotation=45, ha="right")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.axhline(0, color="black", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.5, linestyle="--")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


_BOOTSTRAP_HISTORY_LEN = 8


def run_repeated_fixed_strategy(
    n_players: int = 101,
    threshold: int = 60,
    n_rounds: int = 200,
    seed: int = 42,
    output_dir: Path | str = "outputs/repeated_fixed_strategy",
    cover_all_predictors: bool = True,
) -> Tuple[RepeatedGameResult, pd.DataFrame]:
    """Run the Repeated Fixed Strategy baseline experiment.

    The game is seeded with a bootstrap attendance history of length 8 so
    that every predictor has data from round 1. This is part of the
    baseline definition, not an optional feature.

    Args:
        n_players: Number of players (default: 101 per coursework).
        threshold: Capacity threshold L (default: 60 per coursework).
        n_rounds: Number of repeated rounds (default: 200 per coursework).
        seed: Random seed for reproducibility.
        output_dir: Directory for outputs.
        cover_all_predictors: If True and n_players >= library size, every
            predictor appears at least once in the population.

    Returns:
        Tuple of (RepeatedGameResult, predictor summary DataFrame).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictor_library = default_predictor_library()

    print(f"Running repeated fixed strategy: n={n_players}, L={threshold}, "
          f"rounds={n_rounds}, seed={seed}")
    print(f"Predictor library size: {len(predictor_library)}")
    print(f"Cover all predictors: {cover_all_predictors}")

    agents = build_fixed_predictor_population(
        n_players=n_players,
        seed=seed,
        cover_all_predictors=cover_all_predictors,
    )

    assignment_counts = count_predictor_assignments(agents)
    print("\nPredictor assignment counts:")
    for name, count in sorted(assignment_counts.items()):
        print(f"  {name}: {count}")

    init_history = bootstrap_history(
        n_players=n_players,
        threshold=threshold,
        length=_BOOTSTRAP_HISTORY_LEN,
        seed=seed,
    )
    print(f"\nBootstrap history (len={_BOOTSTRAP_HISTORY_LEN}): {init_history}")

    game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=agents,
        seed=seed,
        initial_attendance_history=init_history,
    )
    result = game.play()
    print(f"Round 1 attendance: {result.attendance_history[0]}")

    result.save_outputs(output_path)
    print(f"\nSaved standard repeated-game outputs to: {output_path}")

    predictor_df = compute_predictor_payoffs(agents, result.cumulative_payoffs)
    predictor_df.to_csv(output_path / "predictor_summary.csv", index=False)
    print(f"Saved: {output_path / 'predictor_summary.csv'}")

    assignment_df = pd.DataFrame([
        {"predictor_name": name, "n_assigned": count}
        for name, count in sorted(assignment_counts.items())
    ])
    assignment_df.to_csv(output_path / "predictor_assignment_counts.csv", index=False)
    print(f"Saved: {output_path / 'predictor_assignment_counts.csv'}")

    plot_predictor_summary(predictor_df, output_path / "predictor_summary.png")
    print(f"Saved: {output_path / 'predictor_summary.png'}")

    summary = result.summary()
    print("\nSummary metrics:")
    print(f"  Mean attendance: {summary['mean_attendance']:.2f}")
    print(f"  Std attendance: {summary['std_attendance']:.2f}")
    print(f"  Overcrowding rate: {summary['overcrowding_rate']:.4f}")
    print(f"  Mean cumulative payoff: {summary['mean_cumulative_payoff']:.2f}")

    return result, predictor_df


if __name__ == "__main__":
    run_repeated_fixed_strategy()
