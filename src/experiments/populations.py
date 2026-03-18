"""
Population builders for homogeneous and heterogeneous experiments.

Supports:
  - homogeneous inductive: non_recency and recency modes
  - heterogeneous: configurable mix of non_recency + random
  - producer/speculator: producers (non-adaptive) + speculators (inductive)
  - fixed predictor baselines
"""

from __future__ import annotations

from math import isclose
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.agents.base import BaseAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.predictor_agent import InductivePredictorAgent
from src.agents.predictors import Predictor, default_predictor_library, sample_predictor_library
from src.agents.producer_agent import ProducerAgent
from src.agents.random_agent import RandomAgent
from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater

PredictorBank = List[Tuple[str, Predictor]]


def _adaptive_bank(rng: np.random.Generator, predictors_per_agent: int):
    """Sample a predictor bank for a single adaptive agent."""
    return sample_predictor_library(rng=rng, k=predictors_per_agent)


def sample_predictor_banks(
    n_players: int,
    predictors_per_agent: int,
    seed: int = 42,
) -> List[PredictorBank]:
    """
    Sample predictor banks for all players at once.

    Use this to ensure both non_recency and recency runs use identical
    predictor assignments for a fair paired comparison.
    """
    library = default_predictor_library()
    max_k = len(library)
    if not (1 <= predictors_per_agent <= max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {max_k}.")

    rng = np.random.default_rng(seed)
    banks: List[PredictorBank] = []
    for _ in range(n_players):
        idx = rng.choice(max_k, size=predictors_per_agent, replace=False)
        bank = [library[int(i)] for i in idx]
        banks.append(bank)
    return banks


def build_homogeneous_non_recency(
    n_players: int,
    predictors_per_agent: int = 6,
    seed: int = 42,
    predictor_banks: Optional[Sequence[PredictorBank]] = None,
) -> List[BaseAgent]:
    """
    All agents use non-recency virtual-payoff score updating with hard argmax.

    Score update:
        s_j(t+1) = s_j(t) + \tilde u_j(t)
    where \tilde u_j(t) is the virtual payoff implied by predictor j.

    If predictor_banks is provided, use those instead of sampling new banks.
    """
    updater = CumulativeScoreUpdater()

    if predictor_banks is None:
        _max_k = len(default_predictor_library())
        if not (1 <= predictors_per_agent <= _max_k):
            raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
        predictor_banks = sample_predictor_banks(
            n_players=n_players,
            predictors_per_agent=predictors_per_agent,
            seed=seed,
        )

    if len(predictor_banks) != n_players:
        raise ValueError("predictor_banks length must equal n_players.")

    return [
        InductivePredictorAgent(
            predictors=list(bank),
            score_updater=updater,
        )
        for bank in predictor_banks
    ]


def build_homogeneous_recency(
    n_players: int,
    lambda_decay: float = 0.95,
    predictors_per_agent: int = 6,
    seed: int = 42,
    predictor_banks: Optional[Sequence[PredictorBank]] = None,
) -> List[BaseAgent]:
    """
    All agents use recency-weighted virtual-payoff score updating with hard argmax.

    Score update:
        s_j(t+1) = lambda * s_j(t) + \tilde u_j(t)
    where \tilde u_j(t) is the virtual payoff implied by predictor j.

    If predictor_banks is provided, use those instead of sampling new banks.
    """
    updater = RecencyScoreUpdater(lambda_decay=lambda_decay)

    if predictor_banks is None:
        _max_k = len(default_predictor_library())
        if not (1 <= predictors_per_agent <= _max_k):
            raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")
        predictor_banks = sample_predictor_banks(
            n_players=n_players,
            predictors_per_agent=predictors_per_agent,
            seed=seed,
        )

    if len(predictor_banks) != n_players:
        raise ValueError("predictor_banks length must equal n_players.")

    return [
        InductivePredictorAgent(
            predictors=list(bank),
            score_updater=updater,
        )
        for bank in predictor_banks
    ]


def build_homogeneous_random(n_players: int, p_attend: float = 0.5) -> List[BaseAgent]:
    """All agents use random mixed strategy."""
    return [RandomAgent(p_attend=p_attend) for _ in range(n_players)]


def build_homogeneous_fixed(n_players: int, predicted_attendance: int) -> List[BaseAgent]:
    """All agents use fixed prediction; threshold is read from context each round."""
    return [FixedAttendanceAgent(predicted_attendance=predicted_attendance) for _ in range(n_players)]


def build_heterogeneous(
    n_players: int,
    p_inductive: float,
    p_random: float,
    lambda_decay: float | None = None,
    predictors_per_agent: int = 6,
    seed: int = 42,
) -> List[BaseAgent]:
    """
    Mix of inductive (non_recency or recency virtual-payoff) and random agents.

    Shares must lie in [0, 1] and sum to 1.0.
    If lambda_decay is None, use non_recency; else use recency.
    """
    _max_k = len(default_predictor_library())
    if not (1 <= predictors_per_agent <= _max_k):
        raise ValueError(f"predictors_per_agent must be between 1 and {_max_k}.")

    shares = [p_inductive, p_random]

    if any(p < 0.0 or p > 1.0 for p in shares):
        raise ValueError("All shares must be in [0, 1].")

    if not isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("p_inductive + p_random must equal 1.0.")

    n_inductive = int(round(n_players * p_inductive))
    n_random = n_players - n_inductive

    rng = np.random.default_rng(seed)
    agents: List[BaseAgent] = []

    if lambda_decay is None:
        updater = CumulativeScoreUpdater()
    else:
        updater = RecencyScoreUpdater(lambda_decay=lambda_decay)

    agents.extend(
        InductivePredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            score_updater=updater,
        )
        for _ in range(n_inductive)
    )
    agents.extend(RandomAgent(p_attend=0.5) for _ in range(n_random))
    return agents


def build_producer_speculator(
    n_players: int,
    n_producers: int,
    lambda_decay: float | None = None,
    predictors_per_agent: int = 6,
    seed: int = 42,
    producer_base_prediction: float | None = None,
    producer_noise_std: float = 5.0,
    threshold: int | None = None,
) -> List[BaseAgent]:
    """
    Producers (non-adaptive noisy-threshold) + speculators (inductive agents).

    If lambda_decay is None, speculators use non_recency virtual-payoff; else recency.
    producer_base_prediction defaults to threshold if not specified.
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

    if lambda_decay is None:
        updater = CumulativeScoreUpdater()
    else:
        updater = RecencyScoreUpdater(lambda_decay=lambda_decay)

    agents.extend(
        InductivePredictorAgent(
            predictors=_adaptive_bank(rng, predictors_per_agent),
            score_updater=updater,
        )
        for _ in range(n_speculators)
    )

    return agents


def build_fixed_predictor_population(
    n_players: int,
    seed: int = 42,
    cover_all_predictors: bool = True,
) -> List[BaseAgent]:
    """
    Assign exactly one predictor to each player and keep it fixed for all rounds.

    If cover_all_predictors=True and n_players >= len(library), every predictor
    appears at least once in the population before the remaining slots are
    filled by uniform random draws.
    """
    rng = np.random.default_rng(seed)
    library = default_predictor_library()
    n_pred = len(library)

    if cover_all_predictors and n_players >= n_pred:
        first_pass = rng.permutation(n_pred)
        assignments = [library[int(i)] for i in first_pass]

        extra_idx = rng.choice(n_pred, size=n_players - n_pred, replace=True)
        assignments.extend(library[int(i)] for i in extra_idx)
    else:
        idx = rng.choice(n_pred, size=n_players, replace=True)
        assignments = [library[int(i)] for i in idx]

    return [
        FixedPredictorAgent(predictor_name=name, predictor_fn=fn)
        for name, fn in assignments
    ]


def build_homogeneous_fixed_predictor(
    n_players: int,
    *,
    predictor_name: str = "last_value",
) -> List[BaseAgent]:
    """
    All players get the same fixed predictor and never switch.
    This is a non-adaptive repeated baseline.
    """
    library = dict(default_predictor_library())

    if predictor_name not in library:
        available = ", ".join(library.keys())
        raise ValueError(
            f"Unknown predictor_name={predictor_name!r}. "
            f"Available predictors: {available}"
        )

    predictor_fn = library[predictor_name]
    return [
        FixedPredictorAgent(
            predictor_name=predictor_name,
            predictor_fn=predictor_fn,
        )
        for _ in range(n_players)
    ]


def build_heterogeneous_fixed_predictor(
    n_players: int,
    *,
    seed: int = 42,
    cover_all_predictors: bool = True,
) -> List[BaseAgent]:
    """
    Each player is assigned exactly one predictor at t=0 and never switches.
    Players are heterogeneous across the population, but each player is non-adaptive.
    """
    rng = np.random.default_rng(seed)
    library = default_predictor_library()

    if not library:
        raise ValueError("Predictor library is empty.")

    n_pred = len(library)
    assignments: List[Tuple[str, object]] = []

    if cover_all_predictors and n_players >= n_pred:
        order = rng.permutation(n_pred)
        assignments.extend(library[int(i)] for i in order)

        remaining = n_players - n_pred
        extra_idx = rng.choice(n_pred, size=remaining, replace=True)
        assignments.extend(library[int(i)] for i in extra_idx)
    else:
        idx = rng.choice(n_pred, size=n_players, replace=True)
        assignments = [library[int(i)] for i in idx]

    return [
        FixedPredictorAgent(
            predictor_name=name,
            predictor_fn=fn,
        )
        for name, fn in assignments
    ]
