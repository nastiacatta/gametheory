"""
Analysis metrics for the repeated El Farol threshold game.

Threshold-centred observables:
  - σ²_L = (1/T) Σ (A_t - L)² (variance from threshold)
  - MAD_L = (1/T) Σ |A_t - L| (mean absolute deviation from threshold)
  - OvercrowdingRate = (1/T) Σ 1[A_t > L] (weak threshold: overcrowded when A > L)

Payoff-side metrics:
  - Mean cumulative payoff
  - Standard deviation of cumulative payoffs
  - Predictor switch rate (for inductive agents)

Note: σ²_L is threshold-deviation volatility, distinct from canonical
Minority Game volatility σ² = (1/T) Σ A_t².
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def mean_attendance(attendance_history: List[int]) -> float:
    return float(np.mean(attendance_history))


def std_attendance(attendance_history: List[int]) -> float:
    return float(np.std(attendance_history))


def variance_from_threshold(attendance_history: List[int], threshold: int) -> float:
    """σ²_L = mean of (A_t - L)²."""
    arr = np.array(attendance_history, dtype=float)
    return float(np.mean((arr - threshold) ** 2))


def mad_from_threshold(attendance_history: List[int], threshold: int) -> float:
    """MAD_L = mean of |A_t - L|."""
    arr = np.array(attendance_history, dtype=float)
    return float(np.mean(np.abs(arr - threshold)))


def overcrowding_rate(attendance_history: List[int], threshold: int) -> float:
    """Fraction of rounds with A_t > L."""
    arr = np.asarray(attendance_history, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr > threshold))


def mean_cumulative_payoff(cumulative_payoffs: List[int]) -> float:
    return float(np.mean(cumulative_payoffs))


def std_cumulative_payoff(cumulative_payoffs: List[int]) -> float:
    return float(np.std(cumulative_payoffs))


def switch_rate(predictor_histories: List[List[int]]) -> float:
    """
    SwitchRate = (1 / (n * (T-1))) * sum over agents and rounds of 1[j_i(t) != j_i(t-1)].
    predictor_histories[i] = list of predictor indices chosen by agent i each round.
    """
    n = len(predictor_histories)
    if n == 0:
        return float("nan")
    T = len(predictor_histories[0])
    if T < 2:
        return 0.0
    total_switches = 0
    for hist in predictor_histories:
        for t in range(1, len(hist)):
            if hist[t] != hist[t - 1]:
                total_switches += 1
    return total_switches / (n * (T - 1))


def compute_all_metrics(
    attendance_history: List[int],
    cumulative_payoffs: List[int],
    threshold: int,
    predictor_histories: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """Compute the full set of analysis metrics."""
    out: Dict[str, float] = {
        "n_rounds": float(len(attendance_history)),
        "mean_attendance": mean_attendance(attendance_history),
        "std_attendance": std_attendance(attendance_history),
        "variance_from_threshold": variance_from_threshold(attendance_history, threshold),
        "mad_from_threshold": mad_from_threshold(attendance_history, threshold),
        "overcrowding_rate": overcrowding_rate(attendance_history, threshold),
        "mean_cumulative_payoff": mean_cumulative_payoff(cumulative_payoffs),
        "std_cumulative_payoff": std_cumulative_payoff(cumulative_payoffs),
        "min_cumulative_payoff": float(np.min(cumulative_payoffs)),
        "max_cumulative_payoff": float(np.max(cumulative_payoffs)),
    }
    if predictor_histories is not None:
        out["switch_rate"] = switch_rate(predictor_histories)
    return out
