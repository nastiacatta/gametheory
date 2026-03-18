"""
Plotting utilities for repeated-game analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_attendance_over_time(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Attendance A_t over time with horizontal line at L."""
    if not attendance_history:
        return

    rounds = np.arange(1, len(attendance_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, attendance_history, linewidth=1.2)
    ax.axhline(threshold, linestyle="--", color="gray", label=f"L = {threshold}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Attendance")
    ax.set_title("Attendance over time")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=250)
        plt.close(fig)
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


def plot_attendance_deviation_over_time(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot A_t - L over time. This is the most informative single-run plot
    for the threshold game. Values above 0 are overcrowded rounds.
    """
    if not attendance_history:
        return

    rounds = np.arange(1, len(attendance_history) + 1)
    deviation = np.asarray(attendance_history, dtype=float) - threshold

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, deviation, linewidth=1.2, label=r"$A_t - L$")
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax.fill_between(rounds, 0, deviation, where=(deviation > 0), alpha=0.25, label="Overcrowded")
    ax.fill_between(rounds, 0, deviation, where=(deviation <= 0), alpha=0.15, label="At/below capacity")

    ax.set_xlabel("Round")
    ax.set_ylabel(r"Deviation from threshold, $A_t - L$")
    ax.set_title("Attendance deviation from threshold")
    ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_rolling_variance_from_threshold(
    attendance_history: List[int],
    threshold: int,
    window: int = 20,
    output_path: Optional[Path] = None,
) -> None:
    """Plot rolling threshold-centred variance."""
    arr = np.asarray(attendance_history, dtype=float)
    if arr.size < window:
        return

    sq_dev = (arr - threshold) ** 2
    kernel = np.ones(window, dtype=float) / window
    rolling = np.convolve(sq_dev, kernel, mode="valid")
    rounds = np.arange(window, arr.size + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, rolling, linewidth=1.5)

    ax.set_xlabel("Round")
    ax.set_ylabel(r"Rolling $\sigma_L^2$")
    ax.set_title(f"Rolling variance from threshold (window={window})")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_threshold_distance_histogram(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Histogram of A_t - L."""
    if not attendance_history:
        return

    deviation = np.asarray(attendance_history, dtype=float) - threshold

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(deviation, bins="auto", edgecolor="black", alpha=0.7)
    ax.axvline(0.0, linestyle="--", color="black")

    ax.set_xlabel(r"$A_t - L$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of attendance relative to threshold")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_payoff_by_type(
    player_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """Boxplot of cumulative payoff by agent type."""
    if player_df.empty or "agent_type" not in player_df.columns:
        return

    grouped = list(player_df.groupby("agent_type", sort=False))
    labels = [name for name, _ in grouped]
    data = [group["cumulative_payoff"].to_numpy() for _, group in grouped]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=labels)

    ax.set_xlabel("Agent type")
    ax.set_ylabel("Cumulative payoff")
    ax.set_title("Cumulative payoff by agent type")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_ranked_final_payoffs(
    cumulative_payoffs: List[int],
    output_path: Optional[Path] = None,
) -> None:
    """Sorted final payoffs to show inequality across players."""
    if not cumulative_payoffs:
        return

    ranked = np.sort(np.asarray(cumulative_payoffs))[::-1]
    players = np.arange(1, len(ranked) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(players, ranked, linewidth=1.4)

    ax.set_xlabel("Player rank")
    ax.set_ylabel("Final cumulative payoff")
    ax.set_title("Ranked final cumulative payoffs")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


def plot_attendance_histogram(
    attendance_history: List[int],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Distribution of attendance across rounds."""
    if not attendance_history:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(min(attendance_history), max(attendance_history) + 2) - 0.5
    ax.hist(attendance_history, bins=bins, edgecolor="black", alpha=0.75)
    ax.axvline(threshold, linestyle="--", color="gray", label=f"L = {threshold}")

    ax.set_xlabel("Attendance")
    ax.set_ylabel("Count")
    ax.set_title("Attendance distribution across rounds")
    ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
