"""
Generate static-game equilibrium analysis for the El Farol threshold game.

Outputs a CSV with:
  - n_players
  - threshold
  - pure_ne_count (C(n, L))
  - mixed_p_star (symmetric mixed NE probability)
  - expected_attendance_under_mixed (n * p*)

This script bridges the theoretical analysis in docs/game_definition.md
with computational verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.equilibria import static_equilibrium_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute static equilibrium statistics for El Farol threshold game"
    )
    parser.add_argument(
        "--n_players",
        type=int,
        default=101,
        help="Number of players (default: 101)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=60,
        help="Capacity threshold L (default: 60)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/static_theory",
        help="Output directory for CSV",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep over multiple (n, L) combinations",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        results = []
        n_values = [21, 51, 101, 201]
        for n in n_values:
            l_values = [n // 4, n // 3, n // 2, 2 * n // 3]
            for l in l_values:
                if 1 <= l <= n - 1:
                    summary = static_equilibrium_summary(n, l)
                    results.append(summary)
        
        df = pd.DataFrame(results)
        csv_path = out_dir / "static_equilibria_sweep.csv"
        df.to_csv(csv_path, index=False)
        print(f"Sweep results saved to: {csv_path.resolve()}")
        print(df.to_string(index=False))
    else:
        summary = static_equilibrium_summary(args.n_players, args.threshold)
        df = pd.DataFrame([summary])
        csv_path = out_dir / "static_equilibria.csv"
        df.to_csv(csv_path, index=False)

        print("=== STATIC EQUILIBRIUM ANALYSIS ===")
        print(f"n_players: {summary['n_players']}")
        print(f"threshold: {summary['threshold']}")
        print(f"pure_ne_count: {summary['pure_ne_count']}")
        print(f"mixed_p_star: {summary['mixed_p_star']:.6f}")
        print(f"expected_attendance_under_mixed: {summary['expected_attendance_under_mixed']:.2f}")
        print(f"\nResults saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
