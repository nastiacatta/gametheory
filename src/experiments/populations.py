"""
Population builders for homogeneous and heterogeneous experiments.

Supports:
  - homogeneous: all BestPredictor, all SoftmaxPredictor, all Random, all Fixed
  - heterogeneous: configurable mix (e.g. 50/50 best/softmax)
  - producer/speculator: producers (non-adaptive) + speculators (adaptive)
"""

from __future__ import annotations

from typing import List

from src.agents.base import BaseAgent
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.producer_agent import ProducerAgent
from src.agents.random_agent import RandomAgent
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent


def build_homogeneous_best_predictor(n_players: int) -> List[BaseAgent]:
    """All agents use Arthur best-predictor."""
    return [BestPredictorAgent(n_players=n_players) for _ in range(n_players)]


def build_homogeneous_softmax(n_players: int, beta: float = 1.0) -> List[BaseAgent]:
    """All agents use softmax predictor selection."""
    return [SoftmaxPredictorAgent(n_players=n_players, beta=beta) for _ in range(n_players)]


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
) -> List[BaseAgent]:
    """
    Mix of best-predictor, softmax, and random agents.
    p_best + p_softmax + p_random should sum to 1.0.
    """
    n_best = int(round(n_players * p_best))
    n_softmax = int(round(n_players * p_softmax))
    n_random = n_players - n_best - n_softmax
    agents: List[BaseAgent] = []
    agents.extend([BestPredictorAgent(n_players=n_players) for _ in range(n_best)])
    agents.extend([SoftmaxPredictorAgent(n_players=n_players, beta=beta) for _ in range(n_softmax)])
    agents.extend([RandomAgent(p_attend=0.5) for _ in range(n_random)])
    return agents


def build_producer_speculator(
    n_players: int,
    n_producers: int,
    speculator_type: str = "best",
    beta: float = 1.0,
) -> List[BaseAgent]:
    """
    Producers (non-adaptive noisy-threshold) + speculators (adaptive predictors).
    speculator_type: "best" or "softmax".
    """
    n_speculators = n_players - n_producers
    if n_speculators < 0:
        raise ValueError("n_producers cannot exceed n_players.")
    agents: List[BaseAgent] = []
    agents.extend([ProducerAgent(base_prediction=50.0, noise_std=5.0) for _ in range(n_producers)])
    if speculator_type == "best":
        agents.extend([BestPredictorAgent(n_players=n_players) for _ in range(n_speculators)])
    elif speculator_type == "softmax":
        agents.extend([SoftmaxPredictorAgent(n_players=n_players, beta=beta) for _ in range(n_speculators)])
    else:
        raise ValueError("speculator_type must be 'best' or 'softmax'.")
    return agents
