from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents.base import BaseAgent, RoundContext
from src.game.payoff import attendance_from_actions, payoffs_for_actions


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

    @property
    def attendance_history(self) -> List[int]:
        return [round_result.attendance for round_result in self.rounds]

    @property
    def overcrowded_rounds(self) -> List[bool]:
        return [round_result.overcrowded for round_result in self.rounds]

    def summary(self) -> Dict[str, float]:
        attendance = np.array(self.attendance_history, dtype=float)
        cumulative = np.array(self.cumulative_payoffs, dtype=float)

        return {
            "n_rounds": float(len(self.rounds)),
            "mean_attendance": float(attendance.mean()),
            "std_attendance": float(attendance.std()),
            "fraction_overcrowded": float(np.mean(self.overcrowded_rounds)),
            "mean_cumulative_payoff": float(cumulative.mean()),
            "min_cumulative_payoff": float(cumulative.min()),
            "max_cumulative_payoff": float(cumulative.max()),
        }

    def rounds_dataframe(self) -> pd.DataFrame:
        records = []
        for round_result in self.rounds:
            records.append(
                {
                    "round": round_result.round_index,
                    "attendance": round_result.attendance,
                    "attendance_rate": round_result.attendance / self.n_players,
                    "overcrowded": int(round_result.overcrowded),
                    "mean_round_payoff": float(np.mean(round_result.payoffs)),
                }
            )
        return pd.DataFrame(records)

    def players_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player_id": list(range(self.n_players)),
                "cumulative_payoff": self.cumulative_payoffs,
            }
        )

    def save_outputs(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.rounds_dataframe().to_csv(output_path / "repeated_rounds.csv", index=False)
        self.players_dataframe().to_csv(output_path / "repeated_players.csv", index=False)

        summary_df = pd.DataFrame([self.summary()])
        summary_df.to_csv(output_path / "repeated_summary.csv", index=False)

        plt.figure(figsize=(10, 5))
        plt.plot(self.attendance_history)
        plt.axhline(self.threshold, linestyle="--")
        plt.xlabel("Round")
        plt.ylabel("Attendance")
        plt.title("Attendance over time")
        plt.tight_layout()
        plt.savefig(output_path / "attendance_over_time.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.attendance_history) / np.arange(1, len(self.attendance_history) + 1))
        plt.axhline(self.threshold, linestyle="--")
        plt.xlabel("Round")
        plt.ylabel("Cumulative average attendance")
        plt.title("Cumulative average attendance")
        plt.tight_layout()
        plt.savefig(output_path / "cumulative_average_attendance.png", dpi=200)
        plt.close()


class RepeatedMinorityGame:
    """
    Repeated threshold minority game using the same agents across rounds.
    """

    def __init__(
        self,
        n_players: int,
        threshold: int,
        n_rounds: int,
        agents: List[BaseAgent],
        seed: int = 42,
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

    def play(self) -> RepeatedGameResult:
        rounds: List[RoundResult] = []
        cumulative_payoffs = [0 for _ in range(self.n_players)]
        attendance_history: List[int] = []

        for t in range(self.n_rounds):
            context = RoundContext(
                round_index=t,
                n_players=self.n_players,
                threshold=self.threshold,
                attendance_history=tuple(attendance_history),
            )

            actions = [agent.choose_action(context, self.rng) for agent in self.agents]
            attendance = attendance_from_actions(actions)
            payoffs = payoffs_for_actions(actions, self.threshold)

            for i, payoff in enumerate(payoffs):
                cumulative_payoffs[i] += payoff

            round_result = RoundResult(
                round_index=t,
                actions=actions,
                attendance=attendance,
                payoffs=payoffs,
                overcrowded=attendance > self.threshold,
            )
            rounds.append(round_result)
            attendance_history.append(attendance)

            for agent, action, payoff in zip(self.agents, actions, payoffs):
                agent.update(
                    context=context,
                    action=action,
                    realised_attendance=attendance,
                    payoff=payoff,
                )

        return RepeatedGameResult(
            n_players=self.n_players,
            threshold=self.threshold,
            rounds=rounds,
            cumulative_payoffs=cumulative_payoffs,
        )
