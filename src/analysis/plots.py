"""
Plotting utilities for repeated-game analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_attendance_over_time(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Attendance A_t over time with horizontal line at L."""
    plt.figure(figsize=(10, 5))
    plt.plot(attendance_history)
    plt.axhline(threshold, linestyle="--", color="gray", label=f"L={threshold}")
    plt.xlabel("Round")
    plt.ylabel("Attendance")
    plt.title("Attendance over time")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_cumulative_average_attendance(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Cumulative average attendance over time."""
    if not attendance_history:
        return
    cum_avg = np.cumsum(attendance_history) / np.arange(1, len(attendance_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(cum_avg)
    plt.axhline(threshold, linestyle="--", color="gray", label=f"L={threshold}")
    plt.xlabel("Round")
    plt.ylabel("Cumulative average attendance")
    plt.title("Cumulative average attendance")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_payoff_histogram(
    cumulative_payoffs: List[int],
    output_path: Optional[Path] = None,
) -> None:
    """Histogram of final cumulative payoffs."""
    plt.figure(figsize=(8, 5))
    plt.hist(cumulative_payoffs, bins=min(30, max(1, len(set(cumulative_payoffs)))), edgecolor="black", alpha=0.7)
    plt.xlabel("Cumulative payoff")
    plt.ylabel("Count")
    plt.title("Distribution of cumulative payoffs")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_predictor_share_over_time(
    predictor_histories: List[List[int]],
    predictor_names: List[str],
    output_path: Optional[Path] = None,
) -> None:
    """Fraction of agents using each predictor over time."""
    n_agents = len(predictor_histories)
    if n_agents == 0:
        return
    T = len(predictor_histories[0])
    n_predictors = len(predictor_names)
    shares = np.zeros((T, n_predictors))
    for t in range(T):
        for i in range(n_agents):
            j = predictor_histories[i][t]
            shares[t, j] += 1
    shares /= n_agents

    plt.figure(figsize=(10, 5))
    for j in range(n_predictors):
        plt.plot(shares[:, j], label=predictor_names[j])
    plt.xlabel("Round")
    plt.ylabel("Share of agents")
    plt.title("Predictor usage over time")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
