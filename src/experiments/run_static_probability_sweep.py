"""
Static probability sweep experiment for the El Farol threshold game.

This module sweeps attendance probability p from 0 to 1 and computes
expected payoff, attendance, and overcrowding rate under symmetric
mixed strategies. Each value of p is evaluated via Monte Carlo simulation
of independent one-shot games.

This is an experiment layer on top of the core static game engine. The
payoff logic mirrors src/game/payoff.py exactly (weak threshold convention).

Output:
    - static_probability_sweep.csv
    - static_payoff_vs_p.png
    - static_attendance_vs_p.png
    - static_overcrowding_vs_p.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.analysis.plots import (
    plot_static_attendance_vs_p,
    plot_static_overcrowding_vs_p,
    plot_static_payoff_vs_p,
)


def simulate_static_mixed_profile(
    p: float,
    n_players: int,
    threshold: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Simulate n_samples independent one-shot games where each player attends
    with probability p.
    
    Args:
        p: Attendance probability for each player.
        n_players: Number of players.
        threshold: Capacity threshold L.
        n_samples: Number of independent game samples.
        rng: NumPy random generator for reproducibility.
    
    Returns:
        Dictionary with summary statistics:
            - p: the input probability
            - mean_attendance: average attendance across samples
            - std_attendance: standard deviation of attendance
            - mean_payoff_per_player: average payoff per player
            - overcrowding_rate: fraction of samples with A > L
            - mean_n_positive: average number of players with payoff +1
            - mean_n_negative: average number of players with payoff -1
            - mean_n_zero: average number of players with payoff 0
    """
    decisions = (rng.random((n_samples, n_players)) < p).astype(int)
    attendance = decisions.sum(axis=1)
    
    overcrowded = attendance > threshold
    overcrowding_rate = overcrowded.mean()
    
    n_attendees = attendance
    n_stayers = n_players - attendance
    
    n_positive = np.where(~overcrowded, n_attendees, 0)
    n_negative = np.where(overcrowded, n_attendees, 0)
    n_zero = n_stayers
    
    total_payoff_per_sample = n_positive * 1 + n_negative * (-1) + n_zero * 0
    mean_payoff_per_player = total_payoff_per_sample.mean() / n_players
    
    return {
        "p": p,
        "mean_attendance": float(attendance.mean()),
        "std_attendance": float(attendance.std()),
        "mean_payoff_per_player": float(mean_payoff_per_player),
        "overcrowding_rate": float(overcrowding_rate),
        "mean_n_positive": float(n_positive.mean()),
        "mean_n_negative": float(n_negative.mean()),
        "mean_n_zero": float(n_zero.mean()),
    }


def run_probability_sweep(
    n_players: int = 101,
    threshold: int = 60,
    n_samples: int = 10_000,
    grid_size: int = 201,
    seed: int = 42,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """
    Run a sweep over attendance probability p from 0 to 1.
    
    For each p in the grid, simulate n_samples independent one-shot games
    and compute summary statistics.
    
    Args:
        n_players: Number of players (default 101).
        threshold: Capacity threshold L (default 60).
        n_samples: Monte Carlo samples per probability value.
        grid_size: Number of points in [0, 1] grid.
        seed: Random seed for reproducibility.
        output_dir: Base directory for outputs (tables/ and figures/ subdirs).
    
    Returns:
        DataFrame with one row per probability value.
    """
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(0.0, 1.0, grid_size)
    
    rows = []
    for p in probabilities:
        row = simulate_static_mixed_profile(
            p=p,
            n_players=n_players,
            threshold=threshold,
            n_samples=n_samples,
            rng=rng,
        )
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    out_path = Path(output_dir)
    tables_dir = out_path / "tables"
    figures_dir = out_path / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = tables_dir / "static_probability_sweep.csv"
    df.to_csv(csv_path, index=False)
    
    plot_static_payoff_vs_p(
        df=df,
        threshold=threshold,
        n_players=n_players,
        output_path=figures_dir / "static_payoff_vs_p.png",
    )
    
    plot_static_attendance_vs_p(
        df=df,
        threshold=threshold,
        n_players=n_players,
        output_path=figures_dir / "static_attendance_vs_p.png",
    )
    
    plot_static_overcrowding_vs_p(
        df=df,
        threshold=threshold,
        output_path=figures_dir / "static_overcrowding_vs_p.png",
    )
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run static probability sweep experiment"
    )
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--grid_size", type=int, default=201)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()
