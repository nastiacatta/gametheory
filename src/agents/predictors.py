"""
Arthur-inspired fixed predictor library for coursework experiments.

Each predictor is a callable (history, n_players, threshold) -> float
returning a predicted attendance for the next round.  When history is
too short the predictor falls back to the threshold as a default.

A fixed master library used by this repo, inspired by predictor-based
inductive strategies.  Not intended as an exact reconstruction of
Arthur's original predictor assignment process.  The library includes:
    1. last_value           — same as last week
    2. mirror               — n - last week  (contrarian)
    3. rolling_mean(k)      — average of last k weeks
    4. linear_trend(k)      — extrapolate slope over last k weeks
    5. lag_cycle(lag)       — same as *lag* weeks ago
    6. rolling_median(k)    — median of last k weeks
    7. mean_all_history     — mean of entire history
    8. mirror_threshold     — mirror around threshold
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import median

Predictor = Callable[[tuple[int, ...], int, int], float]


def _clip_prediction(value: float, n_players: int) -> float:
    """Clip prediction to valid range [0, n_players]."""
    return max(0.0, min(float(n_players), float(value)))


def _fallback_threshold(history: tuple[int, ...], threshold: int) -> float:
    """Default fallback when history is too short.

    Returns threshold - 1, the pure-NE attendance level for the
    strict-threshold game.  This ensures that in the first round
    (empty history) agents attend rather than all staying home,
    providing an informative learning signal from the start.
    """
    return float(max(threshold - 1, 0))


def last_value(history: tuple[int, ...], n_players: int, threshold: int) -> float:
    """Predict same attendance as last round."""
    if not history:
        return _fallback_threshold(history, threshold)
    return float(history[-1])


def mirror(history: tuple[int, ...], n_players: int, threshold: int) -> float:
    """Contrarian: predict n - last_value (mirror around n/2)."""
    if not history:
        return _fallback_threshold(history, threshold)
    return float(n_players - history[-1])


def make_rolling_mean(window: int = 4) -> Predictor:
    """Factory: average attendance over the last *window* rounds."""

    def predictor(history: tuple[int, ...], n_players: int, threshold: int) -> float:
        if not history:
            return _fallback_threshold(history, threshold)
        recent = history[-window:]
        return sum(recent) / len(recent)

    predictor.__name__ = f"rolling_mean_{window}"
    predictor.__qualname__ = f"rolling_mean_{window}"
    return predictor


def make_linear_trend(window: int = 8) -> Predictor:
    """Factory: linear extrapolation of the trend over last *window* rounds."""

    def predictor(history: tuple[int, ...], n_players: int, threshold: int) -> float:
        if len(history) < 2:
            return _fallback_threshold(history, threshold)
        recent = history[-window:]
        slope = (recent[-1] - recent[0]) / (len(recent) - 1)
        predicted = recent[-1] + slope
        return max(0.0, min(float(n_players), predicted))

    predictor.__name__ = f"linear_trend_{window}"
    predictor.__qualname__ = f"linear_trend_{window}"
    return predictor


def make_lag_cycle(lag: int) -> Predictor:
    """Factory: repeat the attendance from *lag* rounds ago."""

    def predictor(history: tuple[int, ...], n_players: int, threshold: int) -> float:
        if len(history) < lag:
            return _fallback_threshold(history, threshold)
        return float(history[-lag])

    predictor.__name__ = f"lag_cycle_{lag}"
    predictor.__qualname__ = f"lag_cycle_{lag}"
    return predictor


def make_rolling_median(window: int) -> Predictor:
    """Factory: median attendance over the last *window* rounds."""

    def predictor(history: tuple[int, ...], n_players: int, threshold: int) -> float:
        if len(history) < window:
            return _fallback_threshold(history, threshold)
        return _clip_prediction(median(history[-window:]), n_players)

    predictor.__name__ = f"rolling_median_{window}"
    predictor.__qualname__ = f"rolling_median_{window}"
    return predictor


def mean_all_history(history: tuple[int, ...], n_players: int, threshold: int) -> float:
    """Predict using the mean of all recorded attendance."""
    if not history:
        return _fallback_threshold(history, threshold)
    return _clip_prediction(sum(history) / len(history), n_players)


def mirror_threshold(history: tuple[int, ...], n_players: int, threshold: int) -> float:
    """Mirror last attendance around the threshold: pred = 2*threshold - last."""
    if not history:
        return _fallback_threshold(history, threshold)
    return _clip_prediction(2.0 * threshold - history[-1], n_players)


def default_predictor_library() -> list[tuple[str, Predictor]]:
    """Return the fixed master predictor library used by this repo.
    
    Inspired by predictor-based inductive strategies, not intended as an
    exact reconstruction of Arthur's original predictor assignment process.
    """
    return [
        ("last_value", last_value),
        ("mirror", mirror),
        ("rolling_mean_2", make_rolling_mean(2)),
        ("rolling_mean_3", make_rolling_mean(3)),
        ("rolling_mean_4", make_rolling_mean(4)),
        ("rolling_mean_5", make_rolling_mean(5)),
        ("rolling_mean_8", make_rolling_mean(8)),
        ("mean_all_history", mean_all_history),
        ("mirror_threshold", mirror_threshold),
        ("linear_trend_3", make_linear_trend(3)),
        ("linear_trend_5", make_linear_trend(5)),
        ("linear_trend_8", make_linear_trend(8)),
        ("lag_2_cycle", make_lag_cycle(2)),
        ("lag_5_cycle", make_lag_cycle(5)),
        ("rolling_median_3", make_rolling_median(3)),
        ("rolling_median_5", make_rolling_median(5)),
    ]


def sample_predictor_library(
    rng,
    k: int,
) -> list[tuple[str, Predictor]]:
    """
    Sample k distinct predictors without replacement from the fixed master library.
    Used to give each adaptive agent its own predictor bank while keeping a common
    master library for reproducibility.
    """
    library = default_predictor_library()
    if not (1 <= k <= len(library)):
        raise ValueError(f"k must be between 1 and {len(library)}.")
    idx = rng.choice(len(library), size=k, replace=False)
    return [library[int(i)] for i in idx]
