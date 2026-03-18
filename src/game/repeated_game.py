"""
Repeated El Farol threshold game: same agents over m rounds, cumulative payoffs.

Provides summary statistics, DataFrames for rounds/players, and optional
CSV/plot output for report-ready analysis. Uses weak threshold payoff
convention (A <= L for positive payoff).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.agents.base import BaseAgent, RoundContext
from src.analysis.metrics import compute_all_metrics
from src.analysis.plots import (
    plot_attendance_over_time,
    plot_attendance_deviation_over_time,
    plot_cumulative_average_attendance,
    plot_rolling_variance_from_threshold,
    plot_threshold_distance_histogram,
    plot_attendance_histogram,
    plot_payoff_histogram,
    plot_ranked_final_payoffs,
)
from src.game.payoff import build_stage_outcome


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    actions: List[int]
    attendance: int
    payoffs: List[int]
    overcrowded: bool


@dataclass
class RepeatedGameResult:
    n_players: int
    threshold: int
    rounds: List[RoundResult]
    cumulative_payoffs: List[int]
    agent_types: List[str]

    @property
    def attendance_history(self) -> List[int]:
        return [round_result.attendance for round_result in self.rounds]

    @property
    def overcrowded_rounds(self) -> List[bool]:
        return [round_result.overcrowded for round_result in self.rounds]

    def summary(
        self, predictor_histories: List[List[int]] | None = None
    ) -> Dict[str, float]:
        """Delegate to analysis.metrics.compute_all_metrics for consistency."""
        return compute_all_metrics(
            self.attendance_history,
            self.cumulative_payoffs,
            self.threshold,
            predictor_histories=predictor_histories,
        )

    def rounds_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame with round-level statistics.

        The theoretical_total_round_payoff column validates the simulation:
            sum_i u_i(t) = A_t if A_t <= L, else -A_t
        
        Cumulative columns track running averages for report tables.
        """
        records = []
        cumulative_attendance = 0
        cumulative_overcrowded = 0
        
        for round_result in self.rounds:
            cumulative_attendance += round_result.attendance
            cumulative_overcrowded += int(round_result.overcrowded)
            t = round_result.round_index + 1
            
            deviation = round_result.attendance - self.threshold
            records.append(
                {
                    "round": t,
                    "attendance": round_result.attendance,
                    "attendance_rate": round_result.attendance / self.n_players,
                    "deviation_from_threshold": deviation,
                    "abs_deviation_from_threshold": abs(deviation),
                    "squared_deviation_from_threshold": deviation**2,
                    "overcrowded": int(round_result.overcrowded),
                    "mean_round_payoff": float(np.mean(round_result.payoffs)),
                    "total_round_payoff": int(np.sum(round_result.payoffs)),
                    "theoretical_total_round_payoff": int(
                        round_result.attendance
                        if round_result.attendance <= self.threshold
                        else -round_result.attendance
                    ),
                    "cumulative_mean_attendance": cumulative_attendance / t,
                    "cumulative_overcrowding_rate": cumulative_overcrowded / t,
                }
            )
        return pd.DataFrame(records)

    def players_dataframe(self) -> pd.DataFrame:
        """
        Build a DataFrame with player-level statistics.
        
        Returns:
            DataFrame with player_id, agent_type, and cumulative_payoff.
        """
        return pd.DataFrame(
            {
                "player_id": np.arange(self.n_players),
                "agent_type": self.agent_types,
                "cumulative_payoff": self.cumulative_payoffs,
            }
        )

    def save_outputs(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.rounds_dataframe().to_csv(output_path / "repeated_rounds.csv", index=False)
        self.players_dataframe().to_csv(output_path / "repeated_players.csv", index=False)
        pd.DataFrame([self.summary()]).to_csv(output_path / "repeated_summary.csv", index=False)

        plot_attendance_over_time(
            self.attendance_history,
            self.threshold,
            output_path / "attendance_over_time.png",
        )
        plot_attendance_deviation_over_time(
            self.attendance_history,
            self.threshold,
            output_path / "attendance_deviation_over_time.png",
        )
        plot_cumulative_average_attendance(
            self.attendance_history,
            self.threshold,
            output_path / "cumulative_average_attendance.png",
        )
        plot_rolling_variance_from_threshold(
            self.attendance_history,
            self.threshold,
            window=max(10, len(self.attendance_history) // 10),
            output_path=output_path / "rolling_variance_from_threshold.png",
        )
        plot_threshold_distance_histogram(
            self.attendance_history,
            self.threshold,
            output_path / "attendance_deviation_histogram.png",
        )
        plot_attendance_histogram(
            self.attendance_history,
            self.threshold,
            output_path / "attendance_histogram.png",
        )
        plot_payoff_histogram(
            self.cumulative_payoffs,
            output_path / "payoff_histogram.png",
        )
        plot_ranked_final_payoffs(
            self.cumulative_payoffs,
            output_path / "ranked_final_payoffs.png",
        )


class RepeatedMinorityGame:
    """
    Repeated El Farol threshold game using the same agents across rounds.
    Cumulative payoff: U_i(T) = sum_{t=1}^T u_i^{(t)}.
    """

    def __init__(
        self,
        n_players: int,
        threshold: int,
        n_rounds: int,
        agents: List[BaseAgent],
        seed: int = 42,
        initial_attendance_history: list[int] | None = None,
    ) -> None:
        if n_players <= 0:
            raise ValueError("n_players must be positive.")
        if n_rounds <= 0:
            raise ValueError("n_rounds must be positive.")
        if len(agents) != n_players:
            raise ValueError("Number of agents must equal n_players.")
        if not (0 <= threshold <= n_players):
            raise ValueError("threshold must be between 0 and n_players.")

        self.n_players = n_players
        self.threshold = threshold
        self.n_rounds = n_rounds
        self.agents = agents
        self.rng = np.random.default_rng(seed)

        self.initial_attendance_history: list[int] = list(
            initial_attendance_history or []
        )
        for a in self.initial_attendance_history:
            if not (0 <= a <= n_players):
                raise ValueError(
                    f"Initial history value {a} outside [0, {n_players}]."
                )

    def play(self) -> RepeatedGameResult:
        rounds: List[RoundResult] = []
        cumulative_payoffs = [0 for _ in range(self.n_players)]
        attendance_history: List[int] = list(self.initial_attendance_history)

        for agent in self.agents:
            agent.reset()

        for t in range(self.n_rounds):
            history_before = tuple(attendance_history)
            context = RoundContext(
                n_players=self.n_players,
                threshold=self.threshold,
                attendance_history=history_before,
                round_index=t,
            )
            actions = [
                agent.choose_action(context=context, rng=self.rng)
                for agent in self.agents
            ]

            stage = build_stage_outcome(actions, self.threshold)

            for i, payoff in enumerate(stage.payoffs):
                cumulative_payoffs[i] += payoff

            round_result = RoundResult(
                round_index=t,
                actions=stage.actions,
                attendance=stage.attendance,
                payoffs=stage.payoffs,
                overcrowded=stage.overcrowded,
            )
            rounds.append(round_result)
            attendance_history.append(stage.attendance)

            for i, agent in enumerate(self.agents):
                agent.update(
                    context=context,
                    action=stage.actions[i],
                    realised_attendance=stage.attendance,
                    payoff=stage.payoffs[i],
                )

        return RepeatedGameResult(
            n_players=self.n_players,
            threshold=self.threshold,
            rounds=rounds,
            cumulative_payoffs=cumulative_payoffs,
            agent_types=[type(agent).__name__ for agent in self.agents],
        )
