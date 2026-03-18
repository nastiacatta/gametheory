"""
Static probability sweep experiment for the El Farol threshold game.

Sweeps p from 0 to 1 for a homogeneous mixed-strategy profile where all n players
independently attend with probability p. Each grid point is estimated via
Monte Carlo sampling of independent one-shot games (not repeated play).

Outputs:
- CSV: outputs/tables/static_probability_sweep.csv
- Plots: outputs/static/static_payoff_vs_p.png
         outputs/static/static_attendance_vs_p.png
         outputs/static/static_overcrowding_vs_p.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StaticSweepConfig:
    """Configuration for the static probability sweep experiment."""

    n_players: int = 101
    threshold: int = 60
    n_samples: int = 10_000
    n_grid_points: int = 201
    seed: int = 42


def simulate_static_mixed_profile(
    p: float,
    n_players: int,
    threshold: int,
    n_samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    """
    Vectorised Monte Carlo simulation of n_samples independent one-shot games.

    All players use the same mixed strategy: attend with probability p.

    Payoff convention (weak threshold):
        u_i = +1 if attend and A <= L
        u_i = -1 if attend and A > L
        u_i =  0 if stay home

    Parameters
    ----------
    p : float
        Probability each player attends (common to all players).
    n_players : int
        Number of players n.
    threshold : int
        Capacity threshold L.
    n_samples : int
        Number of independent one-shot game replications.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    dict with keys:
        mean_attendance, std_attendance, mean_payoff_per_player,
        overcrowding_rate, mean_fraction_going,
        mean_n_positive, mean_n_negative, mean_n_zero
    """
    decisions = rng.random((n_samples, n_players)) < p
    attendance = decisions.sum(axis=1)

    overcrowded = attendance > threshold
    payoff_if_go = np.where(overcrowded, -1, 1)

    goers_per_sample = attendance
    total_payoff_per_sample = goers_per_sample * payoff_if_go
    mean_payoff_per_player = total_payoff_per_sample.sum() / (n_samples * n_players)

    mean_attendance = attendance.mean()
    std_attendance = attendance.std()
    overcrowding_rate = overcrowded.mean()
    mean_fraction_going = mean_attendance / n_players

    n_positive_per_sample = np.where(~overcrowded, goers_per_sample, 0)
    n_negative_per_sample = np.where(overcrowded, goers_per_sample, 0)
    n_zero_per_sample = n_players - goers_per_sample

    mean_n_positive = n_positive_per_sample.mean()
    mean_n_negative = n_negative_per_sample.mean()
    mean_n_zero = n_zero_per_sample.mean()

    return {
        "mean_attendance": float(mean_attendance),
        "std_attendance": float(std_attendance),
        "mean_payoff_per_player": float(mean_payoff_per_player),
        "overcrowding_rate": float(overcrowding_rate),
        "mean_fraction_going": float(mean_fraction_going),
        "mean_n_positive": float(mean_n_positive),
        "mean_n_negative": float(mean_n_negative),
        "mean_n_zero": float(mean_n_zero),
    }


def run_probability_sweep(
    config: StaticSweepConfig,
) -> pd.DataFrame:
    """
    Sweep probability p from 0 to 1 and collect summary metrics.

    Returns a DataFrame with one row per grid point.
    """
    rng = np.random.default_rng(config.seed)
    probabilities = np.linspace(0.0, 1.0, config.n_grid_points)

    records: list[dict[str, Any]] = []
    for p in probabilities:
        metrics = simulate_static_mixed_profile(
            p=p,
            n_players=config.n_players,
            threshold=config.threshold,
            n_samples=config.n_samples,
            rng=rng,
        )
        records.append({"p": p, **metrics})

    return pd.DataFrame(records)


def plot_payoff_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Path | None = None,
) -> None:
    """Plot mean payoff per player vs attendance probability p."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["p"], df["mean_payoff_per_player"], linewidth=1.5, color="steelblue")

    p_cap = threshold / n_players
    ax.axvline(
        p_cap,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label=f"capacity-matching p = L/n = {p_cap:.4f}",
    )

    best_idx = df["mean_payoff_per_player"].idxmax()
    best_p = df.loc[best_idx, "p"]
    best_payoff = df.loc[best_idx, "mean_payoff_per_player"]
    ax.axvline(
        best_p,
        linestyle=":",
        color="red",
        alpha=0.7,
        label=f"best p = {best_p:.4f} (payoff = {best_payoff:.4f})",
    )

    ax.set_xlabel("Attendance probability p")
    ax.set_ylabel("Mean payoff per player")
    ax.set_title("Static Game: Mean Payoff vs Attendance Probability")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_attendance_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Path | None = None,
) -> None:
    """Plot mean attendance vs attendance probability p."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["p"], df["mean_attendance"], linewidth=1.5, color="steelblue")

    ax.axhline(
        threshold,
        linestyle="--",
        color="red",
        alpha=0.7,
        label=f"threshold L = {threshold}",
    )

    p_cap = threshold / n_players
    ax.axvline(
        p_cap,
        linestyle="--",
        color="gray",
        alpha=0.5,
        label=f"capacity-matching p = {p_cap:.4f}",
    )

    ax.set_xlabel("Attendance probability p")
    ax.set_ylabel("Mean attendance")
    ax.set_title("Static Game: Mean Attendance vs Probability")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_overcrowding_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Path | None = None,
) -> None:
    """Plot overcrowding rate vs attendance probability p."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["p"], df["overcrowding_rate"], linewidth=1.5, color="steelblue")

    p_cap = threshold / n_players
    ax.axvline(
        p_cap,
        linestyle="--",
        color="gray",
        alpha=0.7,
        label=f"capacity-matching p = {p_cap:.4f}",
    )

    ax.axhline(0.5, linestyle=":", color="black", alpha=0.3)

    ax.set_xlabel("Attendance probability p")
    ax.set_ylabel("Overcrowding rate (fraction of samples with A > L)")
    ax.set_title("Static Game: Overcrowding Rate vs Probability")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def run_static_probability_sweep(
    n_players: int = 101,
    threshold: int = 60,
    n_samples: int = 10_000,
    n_grid_points: int = 201,
    seed: int = 42,
    output_dir: Path | str = "outputs",
) -> pd.DataFrame:
    """
    Main entry point: run the static probability sweep and save outputs.

    Parameters
    ----------
    n_players : int
        Number of players.
    threshold : int
        Capacity threshold L.
    n_samples : int
        Monte Carlo samples per grid point.
    n_grid_points : int
        Number of p values in [0, 1].
    seed : int
        Random seed for reproducibility.
    output_dir : Path or str
        Base output directory.

    Returns
    -------
    pd.DataFrame
        Sweep results with columns: p, mean_attendance, std_attendance,
        mean_payoff_per_player, overcrowding_rate, mean_n_positive,
        mean_n_negative, mean_n_zero.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    static_dir = output_dir / "static"
    tables_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    config = StaticSweepConfig(
        n_players=n_players,
        threshold=threshold,
        n_samples=n_samples,
        n_grid_points=n_grid_points,
        seed=seed,
    )

    print(f"Running static probability sweep: n={n_players}, L={threshold}, "
          f"samples={n_samples}, grid={n_grid_points}, seed={seed}")

    df = run_probability_sweep(config)

    csv_path = tables_dir / "static_probability_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    plot_payoff_vs_p(
        df,
        threshold=threshold,
        n_players=n_players,
        output_path=static_dir / "static_payoff_vs_p.png",
    )
    print(f"Saved: {static_dir / 'static_payoff_vs_p.png'}")

    plot_attendance_vs_p(
        df,
        threshold=threshold,
        n_players=n_players,
        output_path=static_dir / "static_attendance_vs_p.png",
    )
    print(f"Saved: {static_dir / 'static_attendance_vs_p.png'}")

    plot_overcrowding_vs_p(
        df,
        threshold=threshold,
        n_players=n_players,
        output_path=static_dir / "static_overcrowding_vs_p.png",
    )
    print(f"Saved: {static_dir / 'static_overcrowding_vs_p.png'}")

    best_idx = df["mean_payoff_per_player"].idxmax()
    best_row = df.loc[best_idx]
    print(f"\nBest p = {best_row['p']:.4f}")
    print(f"  Mean payoff/player = {best_row['mean_payoff_per_player']:.4f}")
    print(f"  Mean attendance = {best_row['mean_attendance']:.2f}")
    print(f"  Overcrowding rate = {best_row['overcrowding_rate']:.4f}")
    print(f"  Capacity-matching benchmark: p = L/n = {threshold/n_players:.4f}")

    return df


if __name__ == "__main__":
    run_static_probability_sweep()
