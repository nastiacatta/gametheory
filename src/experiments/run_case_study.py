"""
Case study runner: heterogeneous populations inspired by real-world contexts.

This script implements a venue/nightlife crowding case study where different
agent types represent different behavioural patterns observed in real crowds:

Agent types:
  - RoutineAgent: Repeats last successful behaviour (habit-forming)
  - TrendAgent: Follows recent attendance trend (bandwagon effect)
  - ContrarianAgent: Uses mirror/contrarian predictor (anti-conformist)
  - AdaptiveAgent: Uses turnover or recency-weighted predictors (sophisticated)

The case study compares:
  1. Homogeneous populations of each type
  2. Heterogeneous populations mixing all types

This maps to the coursework requirement to apply the model to a real-world
minority game context with interpretable agent behaviours.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent, RoundContext
from src.agents.predictors import (
    Predictor,
    last_value,
    make_linear_trend,
    make_rolling_mean,
    mirror,
    mirror_threshold,
)
from src.agents.recency_weighted_predictor_agent import RecencyWeightedPredictorAgent
from src.agents.turnover_predictor_agent import TurnoverPredictorAgent
from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import (
    plot_attendance_over_time,
    plot_attendance_deviation_over_time,
    plot_cumulative_average_attendance,
    plot_payoff_histogram,
    plot_ranked_final_payoffs,
)
from src.config import RepeatedGameConfig
from src.game.repeated_game import RepeatedMinorityGame


class RoutineAgent(BaseAgent):
    """
    Agent that repeats last successful behaviour (habit-forming).
    
    If attending led to positive payoff, attend again.
    If staying home when bar was overcrowded, stay home again.
    Otherwise, flip with small probability.
    """

    def __init__(self, initial_action: int = 1, inertia: float = 0.9) -> None:
        self.initial_action = initial_action
        self.inertia = inertia
        self._last_action = initial_action
        self._last_payoff = 0

    def reset(self) -> None:
        self._last_action = self.initial_action
        self._last_payoff = 0

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        if context.round_index == 0:
            return self.initial_action
        
        if self._last_payoff > 0:
            return self._last_action
        elif self._last_payoff == 0 and self._last_action == 0:
            if context.attendance_history and context.attendance_history[-1] > context.threshold:
                return 0
        
        if rng.random() < self.inertia:
            return self._last_action
        return 1 - self._last_action

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        self._last_action = action
        self._last_payoff = payoff


class TrendAgent(BaseAgent):
    """
    Agent that follows recent attendance trends (bandwagon effect).
    
    Uses rolling mean predictor: if predicted attendance <= threshold, attend.
    """

    def __init__(self, window: int = 4) -> None:
        self.window = window
        self.predictor = make_rolling_mean(window)

    def reset(self) -> None:
        pass

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        prediction = self.predictor(
            context.attendance_history, context.n_players, context.threshold
        )
        return int(prediction <= context.threshold)

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        pass


class ContrarianAgent(BaseAgent):
    """
    Agent that uses contrarian strategy (anti-conformist).
    
    Uses mirror-threshold predictor: expects attendance to flip around threshold.
    """

    def __init__(self, use_mirror_threshold: bool = True) -> None:
        self.use_mirror_threshold = use_mirror_threshold
        self.predictor: Predictor = mirror_threshold if use_mirror_threshold else mirror

    def reset(self) -> None:
        pass

    def choose_action(self, context: RoundContext, rng: np.random.Generator) -> int:
        prediction = self.predictor(
            context.attendance_history, context.n_players, context.threshold
        )
        return int(prediction <= context.threshold)

    def update(
        self,
        context: RoundContext,
        action: int,
        realised_attendance: int,
        payoff: int,
    ) -> None:
        pass


def build_case_study_population(
    n_players: int,
    p_routine: float,
    p_trend: float,
    p_contrarian: float,
    p_adaptive: float,
    adaptive_type: str = "recency",
    seed: int = 42,
) -> List[BaseAgent]:
    """
    Build a heterogeneous population with real-world-inspired agent types.
    
    Args:
        n_players: Total number of agents.
        p_routine: Fraction of RoutineAgents.
        p_trend: Fraction of TrendAgents.
        p_contrarian: Fraction of ContrarianAgents.
        p_adaptive: Fraction of AdaptiveAgents (recency or turnover).
        adaptive_type: "recency" or "turnover" for adaptive agents.
        seed: Random seed for reproducibility.
    
    Returns:
        List of agents.
    """
    rng = np.random.default_rng(seed)
    
    shares = [p_routine, p_trend, p_contrarian, p_adaptive]
    if abs(sum(shares) - 1.0) > 1e-6:
        raise ValueError("Population shares must sum to 1.0")
    
    raw_counts = [n_players * p for p in shares]
    counts = [int(c) for c in raw_counts]
    remainder = n_players - sum(counts)
    order = sorted(range(4), key=lambda i: raw_counts[i] - counts[i], reverse=True)
    for i in order[:remainder]:
        counts[i] += 1
    
    n_routine, n_trend, n_contrarian, n_adaptive = counts
    
    agents: List[BaseAgent] = []
    
    for _ in range(n_routine):
        agents.append(RoutineAgent(initial_action=int(rng.random() < 0.5)))
    
    for _ in range(n_trend):
        agents.append(TrendAgent(window=rng.choice([2, 3, 4, 5])))
    
    for _ in range(n_contrarian):
        agents.append(ContrarianAgent(use_mirror_threshold=rng.random() < 0.5))
    
    from src.agents.predictors import default_predictor_library, sample_predictor_library
    
    for _ in range(n_adaptive):
        preds = sample_predictor_library(rng, k=6)
        if adaptive_type == "recency":
            agents.append(RecencyWeightedPredictorAgent(
                predictors=preds,
                lambda_decay=0.95,
                selection="argmax",
            ))
        else:
            agents.append(TurnoverPredictorAgent(
                predictors=preds,
                lambda_decay=0.95,
                patience=10,
                master_library=default_predictor_library(),
            ))
    
    rng.shuffle(agents)
    return agents


def run_case_study(
    n_players: int,
    threshold: int,
    n_rounds: int,
    population_name: str,
    agents: List[BaseAgent],
    seed: int,
) -> Dict[str, Any]:
    """Run a single case study experiment."""
    game = RepeatedMinorityGame(
        n_players=n_players,
        threshold=threshold,
        n_rounds=n_rounds,
        agents=agents,
        seed=seed,
    )
    result = game.play()
    
    predictor_histories = [
        getattr(a, "predictor_history", []) for a in agents
    ]
    predictor_histories = [h for h in predictor_histories if h]
    
    metrics = compute_all_metrics(
        result.attendance_history,
        result.cumulative_payoffs,
        threshold,
        predictor_histories=predictor_histories if predictor_histories else None,
    )
    
    return {
        "population": population_name,
        "result": result,
        "metrics": metrics,
        "agents": agents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run case study experiments")
    parser.add_argument("--n_players", type=int, default=101)
    parser.add_argument("--threshold", type=int, default=60)
    parser.add_argument("--n_rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/case_study")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_studies = [
        ("homogeneous_routine", {"p_routine": 1.0, "p_trend": 0.0, "p_contrarian": 0.0, "p_adaptive": 0.0}),
        ("homogeneous_trend", {"p_routine": 0.0, "p_trend": 1.0, "p_contrarian": 0.0, "p_adaptive": 0.0}),
        ("homogeneous_contrarian", {"p_routine": 0.0, "p_trend": 0.0, "p_contrarian": 1.0, "p_adaptive": 0.0}),
        ("homogeneous_adaptive", {"p_routine": 0.0, "p_trend": 0.0, "p_contrarian": 0.0, "p_adaptive": 1.0}),
        ("mixed_equal", {"p_routine": 0.25, "p_trend": 0.25, "p_contrarian": 0.25, "p_adaptive": 0.25}),
        ("mixed_routine_heavy", {"p_routine": 0.5, "p_trend": 0.2, "p_contrarian": 0.1, "p_adaptive": 0.2}),
        ("mixed_adaptive_heavy", {"p_routine": 0.2, "p_trend": 0.2, "p_contrarian": 0.1, "p_adaptive": 0.5}),
    ]

    all_results: List[Dict[str, Any]] = []

    for pop_name, pop_params in case_studies:
        print(f"Running: {pop_name}")
        
        agents = build_case_study_population(
            n_players=args.n_players,
            seed=args.seed,
            **pop_params,
        )
        
        case_result = run_case_study(
            n_players=args.n_players,
            threshold=args.threshold,
            n_rounds=args.n_rounds,
            population_name=pop_name,
            agents=agents,
            seed=args.seed,
        )
        
        all_results.append(case_result)
        
        case_dir = out_dir / pop_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        result = case_result["result"]
        result.rounds_dataframe().to_csv(case_dir / "rounds.csv", index=False)
        
        player_df = result.players_dataframe(agents=agents)
        player_df.to_csv(case_dir / "players.csv", index=False)
        
        pd.DataFrame([case_result["metrics"]]).to_csv(case_dir / "summary.csv", index=False)
        
        plot_attendance_over_time(
            result.attendance_history, args.threshold, case_dir / "attendance.png"
        )
        plot_attendance_deviation_over_time(
            result.attendance_history, args.threshold, case_dir / "attendance_deviation.png"
        )
        plot_cumulative_average_attendance(
            result.attendance_history, args.threshold, case_dir / "cum_avg_attendance.png"
        )
        plot_payoff_histogram(result.cumulative_payoffs, case_dir / "payoff_hist.png")
        plot_ranked_final_payoffs(result.cumulative_payoffs, case_dir / "ranked_payoffs.png")

    summary_rows = []
    for r in all_results:
        row = {"population": r["population"], **r["metrics"]}
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "case_study_comparison.csv", index=False)

    print("\n=== CASE STUDY COMPARISON ===")
    print(summary_df[["population", "variance_from_threshold", "overcrowding_rate", "mean_cumulative_payoff"]].to_string(index=False))
    print(f"\nResults saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
