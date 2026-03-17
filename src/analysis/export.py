"""
Export utilities for experiment outputs (CSV, summary tables).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def export_rounds_csv(
    rounds: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Export round-level data to CSV."""
    df = pd.DataFrame(rounds)
    df.to_csv(output_path, index=False)


def export_players_csv(
    player_ids: List[int],
    cumulative_payoffs: List[int],
    agent_types: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Export player-level data to DataFrame; optionally save to CSV."""
    data: Dict[str, List[Any]] = {
        "player_id": player_ids,
        "cumulative_payoff": cumulative_payoffs,
    }
    if agent_types is not None:
        data["agent_type"] = agent_types
    df = pd.DataFrame(data)
    if output_path:
        df.to_csv(output_path, index=False)
    return df


def export_experiment_summary(
    metrics: Dict[str, float],
    run_name: str,
    params: Dict[str, Any],
    output_path: Path,
) -> None:
    """Export a single experiment's summary (metrics + params) to CSV."""
    row = {"run": run_name, **params, **metrics}
    df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)
