"""Agent implementations for the El Farol threshold game."""

from src.agents.base import BaseAgent, RoundContext
from src.agents.best_predictor_agent import BestPredictorAgent
from src.agents.epsilon_greedy_predictor_agent import EpsilonGreedyPredictorAgent
from src.agents.fixed_attendance_agent import FixedAttendanceAgent
from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.nash_initialised_fixed_predictor_agent import NashInitialisedFixedPredictorAgent
from src.agents.predictor_agent import InductivePredictorAgent
from src.agents.predictors import Predictor, default_predictor_library, sample_predictor_library
from src.agents.producer_agent import ProducerAgent
from src.agents.random_agent import RandomAgent
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent
from src.agents.score_updaters import CumulativeScoreUpdater, RecencyScoreUpdater, ScoreUpdater
from src.agents.softmax_predictor_agent import SoftmaxPredictorAgent
from src.agents.turnover_predictor_agent import TurnoverPredictorAgent
from src.agents.virtual_payoff_predictor_agent import VirtualPayoffPredictorAgent

__all__ = [
    # Base classes and protocols
    "BaseAgent",
    "RoundContext",
    "ScoreUpdater",
    # Score updaters
    "CumulativeScoreUpdater",
    "RecencyScoreUpdater",
    # Non-adaptive agents
    "FixedAttendanceAgent",
    "FixedPredictorAgent",
    "NashInitialisedFixedPredictorAgent",
    "ProducerAgent",
    "RandomAgent",
    # Adaptive predictor-selection agents
    "BestPredictorAgent",
    "EpsilonGreedyPredictorAgent",
    "InductivePredictorAgent",
    "RecencyWeightedPredictorAgent",
    "SoftmaxPredictorAgent",
    "TurnoverPredictorAgent",
    "VirtualPayoffPredictorAgent",
    # Predictor utilities
    "Predictor",
    "default_predictor_library",
    "sample_predictor_library",
]
