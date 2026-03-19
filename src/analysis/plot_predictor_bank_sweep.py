"""Plot results from predictor bank sweep experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictor_bank_sweep(
    csv_path: Path,
    output_path: Path | None = None,
) -> None:
    """Create a clean 2-panel plot of sweep results."""
    df = pd.read_csv(csv_path)
    
    x = df["predictors_per_agent"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel 1: MAD from threshold
    ax1 = axes[0]
    ax1.plot(x, df["mad_from_threshold"], "o-", linewidth=2, markersize=8, color="#2563eb")
    ax1.set_xlabel("Predictors per agent", fontsize=12)
    ax1.set_ylabel("MAD from threshold", fontsize=12)
    ax1.set_title("Coordination efficiency", fontsize=13)
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Mean cumulative payoff
    ax2 = axes[1]
    ax2.plot(x, df["mean_cumulative_payoff"], "o-", linewidth=2, markersize=8, color="#16a34a")
    ax2.set_xlabel("Predictors per agent", fontsize=12)
    ax2.set_ylabel("Mean cumulative payoff", fontsize=12)
    ax2.set_title("Agent welfare", fontsize=13)
    ax2.set_xticks(x)
    ax2.axhline(0, linestyle="--", color="gray", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    mode = df["mode"].iloc[0] if "mode" in df.columns else "unknown"
    fig.suptitle(f"Predictor bank size sweep ({mode})", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved to: {output_path}")
        plt.close()
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot predictor bank sweep results")
    parser.add_argument("csv_path", type=Path, help="Path to sweep CSV file")
    parser.add_argument("--output", "-o", type=Path, help="Output image path")
    args = parser.parse_args()
    
    plot_predictor_bank_sweep(args.csv_path, args.output)


if __name__ == "__main__":
    main()
