"""
Multi-seed comparison across regimes for statistical analysis.

Runs baseline (mixed and all-random), inductive (best-predictor, softmax),
and heterogeneous configurations across multiple seeds. Outputs a summary
CSV and a grouped bar chart with 95% confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.base import BaseAgent
from src.analysis.metrics import (
    mean_attendance,
    variance_from_threshold,
    mad_from_threshold,
    overcrowding_rate,
    mean_cumulative_payoff,
)
from src.config import RepeatedGameConfig
from src.experiments.populations import (
    build_heterogeneous,
    build_homogeneous_best_predictor,
    build_homogeneous_softmax,
)
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.random_agent import RandomAgent
from src.game.repeated_game import RepeatedMinorityGame


def _build_mixed_baseline(
    n_players: int,
    p_attend: float,
    predicted_attendance: int,
    seed: int,
) -> List[BaseAgent]:
    """Half random agents, half fixed-attendance agents."""
    split = n_players // 2
    agents: List[BaseAgent] = []
    for _ in range(split):
        agents.append(RandomAgent(p_attend=p_attend))
    for _ in range(n_players - split):
        agents.append(FixedAttendanceAgent(predicted_attendance=predicted_attendance))
    return agents


def _build_all_random_baseline(
    n_players: int,
    p_attend: float,
    seed: int,
) -> List[BaseAgent]:
    """All i.i.d. random agents."""
    return [RandomAgent(p_attend=p_attend) for _ in range(n_players)]


def run_single_seed(
    config: RepeatedGameConfig,
    agents: List[BaseAgent],
) -> dict:
    """Run a single game and return key metrics."""
    game = RepeatedMinorityGame(
        n_players=config.n_players,
        threshold=config.threshold,
        n_rounds=config.n_rounds,
        agents=agents,
        seed=config.seed,
    )
    result = game.play()
    ah = result.attendance_history
    cp = result.cumulative_payoffs
    L = config.threshold

    return {
        "mean_attendance": mean_attendance(ah),
        "variance_from_threshold": variance_from_threshold(ah, L),
        "mad_from_threshold": mad_from_threshold(ah, L),
        "overcrowding_rate": overcrowding_rate(ah, L),
        "mean_cumulative_payoff": mean_cumulative_payoff(cp),
    }


def run_regime(
    regime_name: str,
    agent_builder: Callable[[int], List[BaseAgent]],
    config_template: RepeatedGameConfig,
    seeds: List[int],
) -> pd.DataFrame:
    """Run a regime across multiple seeds and collect metrics."""
    records = []
    for seed in seeds:
        config = RepeatedGameConfig(
            n_players=config_template.n_players,
            threshold=config_template.threshold,
            n_rounds=config_template.n_rounds,
            seed=seed,
        )
        agents = agent_builder(seed)
        metrics = run_single_seed(config, agents)
        records.append({"regime": regime_name, "seed": seed, **metrics})
    return pd.DataFrame(records)


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot grouped bar chart of metrics across regimes with 95% CI error bars.

    SE = s / sqrt(n), CI = 1.96 * SE.
    """
    metrics = [
        ("mean_attendance", "Mean attendance"),
        ("overcrowding_rate", "Overcrowding rate"),
        ("variance_from_threshold", r"Variance from threshold ($\sigma_L^2$)"),
        ("mean_cumulative_payoff", "Mean cumulative payoff"),
    ]

    regimes = df["regime"].unique()
    n_regimes = len(regimes)
    n_metrics = len(metrics)

    stats = {}
    for regime in regimes:
        regime_df = df[df["regime"] == regime]
        n_seeds = len(regime_df)
        regime_stats = {}
        for metric_key, _ in metrics:
            values = regime_df[metric_key].to_numpy()
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            se = std / np.sqrt(n_seeds)
            ci = 1.96 * se
            regime_stats[metric_key] = (mean, ci)
        stats[regime] = regime_stats

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    x = np.arange(n_regimes)
    width = 0.6

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        means = [stats[r][metric_key][0] for r in regimes]
        cis = [stats[r][metric_key][1] for r in regimes]

        bars = ax.bar(x, means, width, yerr=cis, capsize=4, alpha=0.8, edgecolor="black")
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=30, ha="right")
        ax.set_title(metric_label)

    fig.suptitle("Multi-seed comparison across regimes (95% CI)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed regime comparison")
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=30, help="Number of seeds per regime")
    parser.add_argument("--base_seed", type=int, default=1000, help="Starting seed value")
    parser.add_argument("--predictors_per_agent", type=int, default=6)
    parser.add_argument("--p_attend", type=float, default=0.55, help="Attendance probability for random agents")
    parser.add_argument("--predicted_attendance", type=int, default=58, help="Fixed prediction for baseline")
    parser.add_argument("--beta", type=float, default=1.0, help="Softmax inverse temperature")
    parser.add_argument("--output_dir", type=str, default="outputs/seed_comparison")
    args = parser.parse_args()

    config_template = RepeatedGameConfig(
        n_players=args.n_players,
        threshold=args.threshold,
        n_rounds=args.n_rounds,
        seed=0,
    )

    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    regimes: list[tuple[str, Callable[[int], List[BaseAgent]]]] = [
        (
            "mixed_baseline",
            lambda s: _build_mixed_baseline(
                args.n_players, args.p_attend, args.predicted_attendance, s
            ),
        ),
        (
            "all_random_baseline",
            lambda s: _build_all_random_baseline(args.n_players, args.p_attend, s),
        ),
        (
            "inductive_best",
            lambda s: build_homogeneous_best_predictor(
                args.n_players, predictors_per_agent=args.predictors_per_agent, seed=s
            ),
        ),
        (
            "inductive_softmax",
            lambda s: build_homogeneous_softmax(
                args.n_players,
                beta=args.beta,
                predictors_per_agent=args.predictors_per_agent,
                seed=s,
            ),
        ),
        (
            "heterogeneous_mix",
            lambda s: build_heterogeneous(
                args.n_players,
                p_best=0.5,
                p_softmax=0.5,
                p_random=0.0,
                beta=args.beta,
                predictors_per_agent=args.predictors_per_agent,
                seed=s,
            ),
        ),
    ]

    all_results: List[pd.DataFrame] = []

    for regime_name, agent_builder in regimes:
        print(f"Running regime: {regime_name} ({args.n_seeds} seeds)...", flush=True)
        regime_df = run_regime(regime_name, agent_builder, config_template, seeds)
        all_results.append(regime_df)
        regime_mean = regime_df["mean_attendance"].mean()
        regime_var = regime_df["variance_from_threshold"].mean()
        print(f"  mean_attendance={regime_mean:.2f}, variance_from_threshold={regime_var:.2f}")

    combined_df = pd.concat(all_results, ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(out / "seed_comparison.csv", index=False)

    plot_grouped_bar_chart(combined_df, out / "regime_comparison.png")

    summary = (
        combined_df.groupby("regime")
        .agg(
            mean_attendance_mean=("mean_attendance", "mean"),
            mean_attendance_std=("mean_attendance", "std"),
            variance_from_threshold_mean=("variance_from_threshold", "mean"),
            variance_from_threshold_std=("variance_from_threshold", "std"),
            overcrowding_rate_mean=("overcrowding_rate", "mean"),
            overcrowding_rate_std=("overcrowding_rate", "std"),
            mean_cumulative_payoff_mean=("mean_cumulative_payoff", "mean"),
            mean_cumulative_payoff_std=("mean_cumulative_payoff", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out / "seed_comparison_summary.csv", index=False)

    print(f"\nSaved outputs to {out.resolve()}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
