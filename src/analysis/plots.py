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
    plt.fill_between(x, threshold, y, where=(y >= threshold), alpha=0.20, interpolate=True)

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


def _predictor_share_matrix(
    predictor_histories: List[List[int]],
    n_predictors: int,
) -> np.ndarray:
    """Convert predictor histories into a T x K share matrix."""
    n_agents = len(predictor_histories)
    if n_agents == 0:
        raise ValueError("predictor_histories must be non-empty.")

    lengths = {len(h) for h in predictor_histories}
    if len(lengths) != 1:
        raise ValueError("All predictor histories must have the same length.")

    T = len(predictor_histories[0])
    shares = np.zeros((T, n_predictors), dtype=float)

    for t in range(T):
        for i in range(n_agents):
            j = predictor_histories[i][t]
            shares[t, j] += 1.0

    shares /= n_agents
    return shares


def plot_predictor_share_heatmap(
    predictor_histories: List[List[int]],
    predictor_names: List[str],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> None:
    """
    Heatmap of predictor usage over time.

    X-axis = round, Y-axis = predictor, color = share of agents using that predictor.
    """
    n_agents = len(predictor_histories)
    if n_agents == 0:
        return

    K = len(predictor_names)
    shares = _predictor_share_matrix(predictor_histories, K).T  # K x T

    T = shares.shape[1]

    fig, ax = plt.subplots(figsize=(12, max(4, K * 0.4)))

    im = ax.imshow(
        shares, aspect="auto", origin="lower", interpolation="nearest",
        extent=[1, T, 0, K], vmin=0, vmax=shares.max(),
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Predictor")
    ax.set_yticks(np.arange(K) + 0.5)
    ax.set_yticklabels(predictor_names, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share of agents")

    ax.set_title(title or "Active predictor share over time")

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_scoring_rule_comparison(
    attendance_abs: List[int],
    attendance_virtual: List[int],
    predictor_histories_abs: List[List[int]],
    predictor_histories_virtual: List[List[int]],
    predictor_names: List[str],
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """
    Four-panel comparison of absolute-error vs virtual-payoff scoring.

    Top row: attendance deviation (A_t - L) for each game.
    Bottom row: predictor-share heatmap for each game.
    """
    if not attendance_abs or not attendance_virtual:
        return

    rounds_abs = np.arange(1, len(attendance_abs) + 1)
    rounds_virtual = np.arange(1, len(attendance_virtual) + 1)

    dev_abs = np.asarray(attendance_abs, dtype=float) - threshold
    dev_virtual = np.asarray(attendance_virtual, dtype=float) - threshold

    K = len(predictor_names)
    shares_abs = _predictor_share_matrix(predictor_histories_abs, K).T
    shares_virtual = _predictor_share_matrix(predictor_histories_virtual, K).T

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 10), sharex="col",
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    y_max = max(np.abs(dev_abs).max(), np.abs(dev_virtual).max()) * 1.1

    ax1.plot(rounds_abs, dev_abs, linewidth=1.2)
    ax1.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax1.set_ylim(-y_max, y_max)
    ax1.set_title("Absolute-error scoring")
    ax1.set_ylabel(r"$A_t - L$")

    ax2.plot(rounds_virtual, dev_virtual, linewidth=1.2)
    ax2.axhline(0.0, linestyle="--", color="black", linewidth=1.0)
    ax2.set_ylim(-y_max, y_max)
    ax2.set_title("Virtual-payoff scoring")
    ax2.set_ylabel(r"$A_t - L$")

    vmax = max(shares_abs.max(), shares_virtual.max())

    im1 = ax3.imshow(
        shares_abs, aspect="auto", origin="lower", interpolation="nearest",
        extent=[1, len(attendance_abs), 0, K], vmin=0, vmax=vmax,
    )
    ax3.set_ylabel("Predictor")
    ax3.set_xlabel("Round")
    ax3.set_yticks(np.arange(K) + 0.5)
    ax3.set_yticklabels(predictor_names, fontsize=8)
    ax3.set_title("Active predictor share")

    im2 = ax4.imshow(
        shares_virtual, aspect="auto", origin="lower", interpolation="nearest",
        extent=[1, len(attendance_virtual), 0, K], vmin=0, vmax=vmax,
    )
    ax4.set_ylabel("Predictor")
    ax4.set_xlabel("Round")
    ax4.set_yticks(np.arange(K) + 0.5)
    ax4.set_yticklabels(predictor_names, fontsize=8)
    ax4.set_title("Active predictor share")

    fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04, label="Share of agents")
    fig.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04, label="Share of agents")

    fig.suptitle(
        "Scoring-rule comparison: separate games with matched initial conditions",
        fontsize=13,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_paired_scoring_differences(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """
    Paired difference plot across seeds: virtual-payoff minus absolute-error.

    Expects columns:
      delta_mean_payoff
      delta_overcrowding_rate
      delta_mad_from_threshold
    """
    metrics = [
        ("delta_mean_payoff", r"$\Delta$ mean payoff"),
        ("delta_overcrowding_rate", r"$\Delta$ overcrowding rate"),
        ("delta_mad_from_threshold", r"$\Delta$ MAD from threshold"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, label) in zip(axes, metrics):
        x = df[col].to_numpy()
        y = np.arange(len(x))
        ax.scatter(x, y, s=24, alpha=0.7)
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylabel("Seed index")
        ax.set_xlabel(label)
        
        mean_val = x.mean()
        ax.axvline(mean_val, linestyle="-", color="red", linewidth=1.5, alpha=0.7,
                   label=f"Mean = {mean_val:.3f}")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Virtual-payoff minus cumulative absolute-error (per matched seed)",
        fontsize=13,
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_recency_comparison(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> None:
    """
    Paired difference plot across seeds: recency minus non_recency.

    Expects columns:
      delta_mean_payoff
      delta_overcrowding_rate
      delta_mad_from_threshold
    """
    metrics = [
        ("delta_mean_payoff", r"$\Delta$ mean payoff"),
        ("delta_overcrowding_rate", r"$\Delta$ overcrowding rate"),
        ("delta_mad_from_threshold", r"$\Delta$ MAD from threshold"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, label) in zip(axes, metrics):
        x = df[col].to_numpy()
        y = np.arange(len(x))
        ax.scatter(x, y, s=24, alpha=0.7)
        ax.axvline(0.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylabel("Seed index")
        ax.set_xlabel(label)
        
        mean_val = x.mean()
        ax.axvline(mean_val, linestyle="-", color="red", linewidth=1.5, alpha=0.7,
                   label=f"Mean = {mean_val:.3f}")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Recency minus non-recency (per matched seed)",
        fontsize=13,
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_static_payoff_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Optional[Path] = None,
) -> None:
    """Plot mean payoff per player vs attendance probability p."""
    p = df["p"].to_numpy()
    payoff = df["mean_payoff_per_player"].to_numpy()
    p_capacity = threshold / n_players
    best_idx = int(np.argmax(payoff))
    p_best = float(p[best_idx])
    payoff_best = float(payoff[best_idx])

    plt.figure(figsize=(10, 6))
    plt.plot(p, payoff, linewidth=1.5, label="Mean payoff per player")
    plt.axvline(
        p_best,
        linestyle="--",
        color="tab:green",
        alpha=0.8,
        label=f"best p = {p_best:.3f}",
    )
    plt.axvline(p_capacity, linestyle="--", color="gray", alpha=0.7,
                label=f"p = L/n = {p_capacity:.3f}")
    plt.axhline(0.0, linestyle=":", color="black", alpha=0.5)
    plt.scatter([p_best], [payoff_best], color="tab:green", s=28, zorder=3)

    plt.xlabel("Attendance probability p")
    plt.ylabel("Mean payoff per player")
    plt.title(f"Static game: payoff vs p (n={n_players}, L={threshold})")
    plt.legend()
    _finish_plot(output_path)


def plot_static_attendance_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Optional[Path] = None,
) -> None:
    """Plot mean attendance vs attendance probability p."""
    p = df["p"].to_numpy()
    attendance = df["mean_attendance"].to_numpy()
    p_capacity = threshold / n_players

    plt.figure(figsize=(10, 6))
    plt.plot(p, attendance, linewidth=1.5, label="Mean attendance")
    plt.axhline(threshold, linestyle="--", color="gray", alpha=0.7,
                label=f"L = {threshold}")
    plt.axvline(p_capacity, linestyle=":", color="black", alpha=0.5,
                label=f"p = L/n = {p_capacity:.3f}")

    plt.xlabel("Attendance probability p")
    plt.ylabel("Mean attendance")
    plt.title(f"Static game: attendance vs p (n={n_players}, L={threshold})")
    plt.legend()
    _finish_plot(output_path)


def plot_static_overcrowding_vs_p(
    df: pd.DataFrame,
    threshold: int,
    output_path: Optional[Path] = None,
) -> None:
    """Plot overcrowding rate vs attendance probability p."""
    p = df["p"].to_numpy()
    overcrowding = df["overcrowding_rate"].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(p, overcrowding, linewidth=1.5, label="Overcrowding rate")
    plt.axhline(0.5, linestyle="--", color="gray", alpha=0.7, label="50%")

    plt.xlabel("Attendance probability p")
    plt.ylabel("Overcrowding rate (fraction with A >= L)")
    plt.title(f"Static game: overcrowding rate vs p (L={threshold})")
    plt.legend()
    _finish_plot(output_path)


def plot_static_counts_vs_p(
    df: pd.DataFrame,
    threshold: int,
    n_players: int,
    output_path: Optional[Path] = None,
) -> None:
    """Plot mean number of players in each payoff bucket vs p."""
    p = df["p"].to_numpy()
    n_positive = df["mean_n_positive"].to_numpy()
    n_negative = df["mean_n_negative"].to_numpy()
    n_zero = df["mean_n_zero"].to_numpy()
    p_capacity = threshold / n_players

    plt.figure(figsize=(10, 6))
    plt.plot(p, n_positive, linewidth=1.5, color="green",
             label="+1 (attended, not crowded)")
    plt.plot(p, n_negative, linewidth=1.5, color="red",
             label="-1 (attended, crowded)")
    plt.plot(p, n_zero, linewidth=1.5, color="steelblue",
             label="0 (stayed home)")
    plt.axvline(p_capacity, linestyle="--", color="black", alpha=0.5,
                label=f"p = L/n = {p_capacity:.3f}")

    plt.xlabel("Attendance probability p")
    plt.ylabel("Mean number of players")
    plt.title(f"Static game: payoff decomposition vs p (n={n_players}, L={threshold})")
    plt.legend()
    _finish_plot(output_path)
