"""
Population builders for homogeneous and heterogeneous experiments.

Supports:
  - homogeneous: all BestPredictor, all SoftmaxPredictor, all Random, all Fixed
  - heterogeneous: configurable mix (e.g. 50/50 best/softmax)
  - producer/speculator: producers (non-adaptive) + speculators (adaptive agents with sampled predictor banks)
"""

from __future__ import annotations

from math import isclose
from typing import List

import numpy as np

from src.agents.base import BaseAgent
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.predictors import default_predictor_library, sample_predictor_library
from src.agents.producer_agent import ProducerAgent
from src.agents.random_agent import RandomAgent
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent
from src.agents.turnover_predictor_agent import TurnoverPredictorAgent


def _adaptive_bank(rng: np.random.Generator, predictors_per_agent: int):
    """Sample a predictor bank for a single adaptive agent."""
    return sample_predictor_library(rng=rng, k=predictors_per_agent)


def build_homogeneous_best_predictor(
    n_players: int,
    predictors_per_agent: int = 3,
    seed: int = 42,
) -> List[BaseAgent]:
    """All agents use best-predictor selection over sampled predictor banks."""
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
    rng = np.random.default_rng(seed)
    return [
        BestPredictorAgent(predictors=_adaptive_bank(rng, predictors_per_agent))
        for _ in range(n_players)
    ]


def build_homogeneous_softmax(
    n_players: int,
    beta: float = 1.0,
    predictors_per_agent: int = 3,
    seed: int = 42,
) -> List[BaseAgent]:
    """All agents use softmax predictor selection over sampled predictor banks."""
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
    rng = np.random.default_rng(seed)
    return [
        SoftmaxPredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            beta=beta,
        )
        for _ in range(n_players)
    ]


def build_homogeneous_random(n_players: int, p_attend: float = 0.5) -> List[BaseAgent]:
    """All agents use random mixed strategy."""
    return [RandomAgent(p_attend=p_attend) for _ in range(n_players)]


def build_homogeneous_fixed(n_players: int, predicted_attendance: int) -> List[BaseAgent]:
    """All agents use fixed prediction; threshold is read from context each round."""
    return [FixedAttendanceAgent(predicted_attendance=predicted_attendance) for _ in range(n_players)]


def build_heterogeneous(
    n_players: int,
    p_best: float,
    p_softmax: float,
    p_random: float,
    beta: float = 1.0,
    predictors_per_agent: int = 3,
    seed: int = 42,
) -> List[BaseAgent]:
    """
    Mix of best-predictor, softmax, and random agents with sampled predictor banks.

    Shares must lie in [0, 1] and sum to 1.0.
    Counts are allocated by largest-remainder rounding.
    Adaptive agents receive individually sampled predictor banks.
    """
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")

    shares = [p_best, p_softmax, p_random]

    if any(p < 0.0 or p > 1.0 for p in shares):
        raise ValueError("All shares must be in [0, 1].")

    if not isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("p_best + p_softmax + p_random must equal 1.0.")

    raw_counts = [n_players * p_best, n_players * p_softmax, n_players * p_random]
    counts = [int(x) for x in raw_counts]

    remainder = n_players - sum(counts)
    order = sorted(
        range(len(raw_counts)),
        key=lambda i: raw_counts[i] - counts[i],
        reverse=True,
    )
    for i in order[:remainder]:
        counts[i] += 1

    n_best, n_softmax, n_random = counts

    rng = np.random.default_rng(seed)
    agents: List[BaseAgent] = []
    agents.extend(
        BestPredictorAgent(predictors=_adaptive_bank(rng, predictors_per_agent))
        for _ in range(n_best)
    )
    agents.extend(
        SoftmaxPredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            beta=beta,
        )
        for _ in range(n_softmax)
    )
    agents.extend(RandomAgent(p_attend=0.5) for _ in range(n_random))
    return agents


def build_homogeneous_recency(
    n_players: int,
    lambda_decay: float = 0.95,
    selection: str = "argmax",
    beta: float = 1.0,
    predictors_per_agent: int = 6,
    seed: int = 42,
) -> List[BaseAgent]:
    """All agents use recency-weighted predictor selection with exponential decay."""
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
    rng = np.random.default_rng(seed)
    return [
        RecencyWeightedPredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            lambda_decay=lambda_decay,
            selection=selection,
            beta=beta,
        )
        for _ in range(n_players)
    ]


def build_homogeneous_turnover(
    n_players: int,
    lambda_decay: float = 0.95,
    patience: int = 10,
    error_threshold: float = 5.0,
    predictors_per_agent: int = 6,
    seed: int = 42,
) -> List[BaseAgent]:
    """All agents use turnover predictor selection with hypothesis replacement."""
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
    rng = np.random.default_rng(seed)
    master_lib = default_predictor_library()
    return [
        TurnoverPredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            lambda_decay=lambda_decay,
            patience=patience,
            error_threshold=error_threshold,
            master_library=master_lib,
        )
        for _ in range(n_players)
    ]


def build_producer_speculator(
    n_players: int,
    n_producers: int,
    speculator_type: str = "best",
    beta: float = 1.0,
    predictors_per_agent: int = 3,
    seed: int = 42,
    producer_base_prediction: float | None = None,
    producer_noise_std: float = 5.0,
    threshold: int | None = None,
) -> List[BaseAgent]:
    """
    Producers (non-adaptive noisy-threshold) + speculators (adaptive agents with sampled predictor banks).

    speculator_type: "best" or "softmax".
    producer_base_prediction defaults to threshold if not specified.
    Each speculator receives an individually sampled predictor bank.
    """
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")

    n_speculators = n_players - n_producers
    if n_speculators < 0:
        raise ValueError("n_producers cannot exceed n_players.")

    if producer_base_prediction is None:
        if threshold is None:
            raise ValueError("Pass threshold or producer_base_prediction.")
        producer_base_prediction = float(threshold)

    rng = np.random.default_rng(seed)
    agents: List[BaseAgent] = []
    agents.extend(
        ProducerAgent(
            base_prediction=producer_base_prediction,
            noise_std=producer_noise_std,
        )
        for _ in range(n_producers)
    )

    if speculator_type == "best":
        agents.extend(
            BestPredictorAgent(predictors=_adaptive_bank(rng, predictors_per_agent))
            for _ in range(n_speculators)
        )
    elif speculator_type == "softmax":
        agents.extend(
            SoftmaxPredictorAgent(
                predictors=_adaptive_bank(rng, predictors_per_agent),
                beta=beta,
            )
            for _ in range(n_speculators)
        )
    else:
        raise ValueError("speculator_type must be 'best' or 'softmax'.")

    return agents
