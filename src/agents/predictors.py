"""
Arthur-style attendance predictor library for the El Farol game.

Each predictor is a callable (history, n_players, threshold) -> float
returning a predicted attendance for the next round.  When history is
too short the predictor falls back to the threshold as a default.

The six predictors below follow Arthur (1994):
    1. last_value           — same as last week
    2. mirror               — n - last week  (contrarian)
    3. rolling_mean(k=4)    — average of last k weeks
    4. linear_trend(k=8)    — extrapolate slope over last k weeks
    5. lag_cycle(lag=2)      — same as 2 weeks ago
    6. lag_cycle(lag=5)      — same as 5 weeks ago
"""

from __future__ import annotations

from typing import Callable, List, Tuple

Predictor = Callable[[Tuple[int, ...], int, int], float]


def last_value(history: Tuple[int, ...], n_players: int, threshold: int) -> float:
    """Predict same attendance as last round."""
    if not history:
        return float(threshold)
    return float(history[-1])


def mirror(history: Tuple[int, ...], n_players: int, threshold: int) -> float:
    """Contrarian: predict n - last_value (mirror around n/2)."""
    if not history:
        return float(threshold)
    return float(n_players - history[-1])


def make_rolling_mean(window: int = 4) -> Predictor:
    """Factory: average attendance over the last *window* rounds."""

    def predictor(history: Tuple[int, ...], n_players: int, threshold: int) -> float:
        if not history:
            return float(threshold)
        recent = history[-window:]
        return sum(recent) / len(recent)

    predictor.__name__ = f"rolling_mean_{window}"
    predictor.__qualname__ = f"rolling_mean_{window}"
    return predictor


def make_linear_trend(window: int = 8) -> Predictor:
    """Factory: linear extrapolation of the trend over last *window* rounds."""

    def predictor(history: Tuple[int, ...], n_players: int, threshold: int) -> float:
        if len(history) < 2:
            return float(threshold)
        recent = history[-window:]
        if len(recent) < 2:
            return float(recent[-1])
        slope = (recent[-1] - recent[0]) / (len(recent) - 1)
        predicted = recent[-1] + slope
        return max(0.0, min(float(n_players), predicted))

    predictor.__name__ = f"linear_trend_{window}"
    predictor.__qualname__ = f"linear_trend_{window}"
    return predictor


def make_lag_cycle(lag: int) -> Predictor:
    """Factory: repeat the attendance from *lag* rounds ago."""

    def predictor(history: Tuple[int, ...], n_players: int, threshold: int) -> float:
        if len(history) < lag:
            return float(threshold)
        return float(history[-lag])

    predictor.__name__ = f"lag_cycle_{lag}"
    predictor.__qualname__ = f"lag_cycle_{lag}"
    return predictor


def default_predictor_library() -> List[Tuple[str, Predictor]]:
    """Return the standard six-predictor library based on Arthur (1994)."""
    return [
        ("last_value", last_value),
        ("mirror", mirror),
        ("rolling_mean_4", make_rolling_mean(4)),
        ("linear_trend_8", make_linear_trend(8)),
        ("lag_2_cycle", make_lag_cycle(2)),
        ("lag_5_cycle", make_lag_cycle(5)),
    ]
