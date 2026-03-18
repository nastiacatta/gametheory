"""
Plotting utilities for repeated-game analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish_plot(output_path: Optional[Path], tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_attendance_over_time(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Attendance A_t over time with threshold line and overcrowding shading."""
    if not attendance_history:
        return

    x = np.arange(1, len(attendance_history) + 1)
    y = np.asarray(attendance_history, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=1.4, label="Attendance")
    plt.axhline(threshold, linestyle="--", color="gray", label=f"L={threshold}")
    plt.fill_between(x, threshold, y, where=(y > threshold), alpha=0.20, interpolate=True)

    plt.xlabel("Round")
    plt.ylabel("Attendance")
    plt.title("Attendance over time")
    plt.legend()
    _finish_plot(output_path)


def plot_attendance_deviation_over_time(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Deviation series A_t - L."""
    if not attendance_history:
        return

    x = np.arange(1, len(attendance_history) + 1)
    deviation = np.asarray(attendance_history, dtype=float) - threshold

    plt.figure(figsize=(10, 5))
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.plot(x, deviation, linewidth=1.4)
    plt.fill_between(x, 0, deviation, where=(deviation > 0), alpha=0.20, interpolate=True)

    plt.xlabel("Round")
    plt.ylabel(r"$A_t - L$")
    plt.title("Attendance deviation from threshold")
    _finish_plot(output_path)


def plot_cumulative_average_attendance(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Cumulative average attendance over time."""
    if not attendance_history:
        return

    x = np.arange(1, len(attendance_history) + 1)
    cum_avg = np.cumsum(attendance_history) / x

    plt.figure(figsize=(10, 5))
    plt.plot(x, cum_avg, linewidth=1.4, label="Cumulative mean attendance")
    plt.axhline(threshold, linestyle="--", color="gray", label=f"L={threshold}")

    plt.xlabel("Round")
    plt.ylabel("Cumulative average attendance")
    plt.title("Cumulative average attendance")
    plt.legend()
    _finish_plot(output_path)


def plot_rolling_variance_from_threshold(
    attendance_history: List[int],
    threshold: int,
    window: int = 20,
    output_path: Optional[Path] = None,
) -> None:
    """Rolling mean of (A_t - L)^2."""
    if not attendance_history:
        return

    arr = np.asarray(attendance_history, dtype=float)
    window = max(2, min(window, len(arr)))
    sq = (arr - threshold) ** 2
    rolling = np.convolve(sq, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(arr) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, rolling, linewidth=1.4)
    plt.xlabel("Round")
    plt.ylabel(r"Rolling mean of $(A_t - L)^2$")
    plt.title(f"Rolling threshold-centred variance (window={window})")
    _finish_plot(output_path)


def plot_threshold_distance_histogram(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Histogram of A_t - L."""
    if not attendance_history:
        return

    deviation = np.asarray(attendance_history, dtype=int) - threshold
    lo = int(deviation.min())
    hi = int(deviation.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(deviation, bins=bins, edgecolor="black", alpha=0.75)
    plt.axvline(0, linestyle="--", color="gray")

    plt.xlabel(r"$A_t - L$")
    plt.ylabel("Count")
    plt.title("Distribution of attendance deviation")
    _finish_plot(output_path)


def plot_attendance_histogram(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Histogram of attendance levels."""
    if not attendance_history:
        return

    arr = np.asarray(attendance_history, dtype=int)
    lo = int(arr.min())
    hi = int(arr.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins, edgecolor="black", alpha=0.75)
    plt.axvline(threshold, linestyle="--", color="gray", label=f"L={threshold}")

    plt.xlabel("Attendance")
    plt.ylabel("Count")
    plt.title("Distribution of attendance")
    plt.legend()
    _finish_plot(output_path)


def plot_payoff_histogram(
    cumulative_payoffs: List[int],
    output_path: Optional[Path] = None,
) -> None:
    """Histogram of final cumulative payoffs with integer-aligned bins."""
    if not cumulative_payoffs:
        return

    arr = np.asarray(cumulative_payoffs, dtype=int)
    lo = int(arr.min())
    hi = int(arr.max())
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins, edgecolor="black", alpha=0.75)

    plt.xlabel("Cumulative payoff")
    plt.ylabel("Count")
    plt.title("Distribution of cumulative payoffs")
    _finish_plot(output_path)


def plot_ranked_final_payoffs(
    cumulative_payoffs: List[int],
    output_path: Optional[Path] = None,
) -> None:
    """Sorted final payoffs to show inequality / dispersion."""
    if not cumulative_payoffs:
        return

    arr = np.sort(np.asarray(cumulative_payoffs, dtype=float))[::-1]
    x = np.arange(1, len(arr) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, arr, linewidth=1.4)

    plt.xlabel("Player rank")
    plt.ylabel("Final cumulative payoff")
    plt.title("Ranked final cumulative payoffs")
    _finish_plot(output_path)


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
    x = np.arange(1, T + 1)
    shares = np.zeros((T, n_predictors), dtype=float)

    for t in range(T):
        for i in range(n_agents):
            shares[t, predictor_histories[i][t]] += 1.0

    shares /= n_agents

    plt.figure(figsize=(10, 5))
    for j in range(n_predictors):
        plt.plot(x, shares[:, j], linewidth=1.2, label=predictor_names[j])

    plt.xlabel("Round")
    plt.ylabel("Share of agents")
    plt.title("Predictor usage over time")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _finish_plot(output_path)


def plot_payoff_by_type(
    player_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """Boxplot of final payoffs by agent type for heterogeneous populations."""
    required = {"agent_type", "cumulative_payoff"}
    if not required.issubset(player_df.columns):
        raise ValueError("player_df must contain 'agent_type' and 'cumulative_payoff' columns.")

    grouped = player_df.groupby("agent_type")["cumulative_payoff"]
    labels = list(grouped.groups.keys())
    data = [grouped.get_group(label).to_numpy() for label in labels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels)

    plt.xlabel("Agent type")
    plt.ylabel("Final cumulative payoff")
    plt.title("Final payoff distribution by agent type")
    _finish_plot(output_path)
