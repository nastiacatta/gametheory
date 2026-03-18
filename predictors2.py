from typing import Union, Tuple
import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

def _to_array(history: Union[list, np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """Normalize history input to a 1-D numpy float array."""
    if isinstance(history, pd.DataFrame):
        history = history.iloc[:, 0]
    return np.asarray(history, dtype=float).ravel()


def _clip_and_round(value: float, lo: int, hi: int) -> int:
    """Clip a prediction to [lo, hi] and round to nearest int."""
    return int(np.clip(np.round(value), lo, hi))


def _decide(predicted_others: int, capacity: int) -> int:
    """
    Player-level go decision using prediction of OTHER players only.

    If predicted attendance of the other players is <= capacity, go.
    Otherwise stay.

    Example:
    - 101 players total
    - total bar capacity = 60
    - predictor estimates attendance among the other 100 players
    - then capacity should be passed as 59
    """
    return 1 if predicted_others <= capacity else 0


def _output(
    value: float,
    capacity: int,
    lo: int,
    hi: int,
) -> Tuple[float, int]:
    """Clip, round, decide — standard deterministic return."""
    pred = _clip_and_round(value, lo, hi)
    return float(pred), _decide(pred, capacity)


def _fallback_random(
    fallback_go_probability: float,
) -> Tuple[float, int]:
    """
    Random fallback used only when history is too short.
    predicted_attendance is NaN by design.
    """
    go = int(np.random.random() < fallback_go_probability)
    return np.nan, go


def _enough_history_or_random(
    h: np.ndarray,
    required_len: int,
    fallback_go_probability: float,
) -> Tuple[bool, Tuple[float, int] | None]:
    """Return (True, None) if enough history, else (False, random fallback output)."""
    if len(h) >= required_len:
        return True, None
    return False, _fallback_random(fallback_go_probability)


# ============================================================
# Predictors
# ============================================================

def same_as_last_week(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict attendance equals last week's attendance."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-1], capacity, min_attendance, max_attendance)


def same_as_2_weeks_ago(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict attendance equals two weeks ago."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-2], capacity, min_attendance, max_attendance)


def same_as_3_weeks_ago(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict attendance equals three weeks ago."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-3], capacity, min_attendance, max_attendance)


def same_as_5_weeks_ago(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict attendance equals five weeks ago."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-5], capacity, min_attendance, max_attendance)


def moving_average_2(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of the last 2 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-2:].mean(), capacity, min_attendance, max_attendance)


def moving_average_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of the last 3 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-3:].mean(), capacity, min_attendance, max_attendance)


def moving_average_4(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of the last 4 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 4, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-4:].mean(), capacity, min_attendance, max_attendance)


def moving_average_5(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of the last 5 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-5:].mean(), capacity, min_attendance, max_attendance)


def moving_average_8(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of the last 8 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 8, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-8:].mean(), capacity, min_attendance, max_attendance)


def average_of_entire_history(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict using the average of the entire available history."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h.mean(), capacity, min_attendance, max_attendance)


def mirror_around_50_last(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Mirror last attendance around 50: pred = 100 - last."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(100 - h[-1], capacity, min_attendance, max_attendance)


def mirror_around_60_last(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Mirror last attendance around 60: pred = 120 - last."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(120 - h[-1], capacity, min_attendance, max_attendance)


def mirror_around_capacity_last(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Mirror last attendance around the player-level threshold."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(2 * capacity - h[-1], capacity, min_attendance, max_attendance)


def linear_trend_2(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Extrapolate using the last 2 observations."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    delta = h[-1] - h[-2]
    return _output(h[-1] + delta, capacity, min_attendance, max_attendance)


def linear_trend_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Fit a linear trend to the last 3 points and extrapolate one step."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    window = h[-3:]
    x = np.arange(3)
    slope = np.polyfit(x, window, 1)[0]
    return _output(window[-1] + slope, capacity, min_attendance, max_attendance)


def linear_trend_5(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Fit a linear trend to the last 5 points and extrapolate one step."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    window = h[-5:]
    x = np.arange(5)
    slope = np.polyfit(x, window, 1)[0]
    return _output(window[-1] + slope, capacity, min_attendance, max_attendance)


def repeat_last_change(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Apply the same change that happened last week."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    delta = h[-1] - h[-2]
    return _output(h[-1] + delta, capacity, min_attendance, max_attendance)


def mean_of_all_history(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the mean of all recorded attendance."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h.mean(), capacity, min_attendance, max_attendance)


def median_of_last_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the median of the last 3 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(np.median(h[-3:]), capacity, min_attendance, max_attendance)


def median_of_last_5(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the median of the last 5 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    return _output(np.median(h[-5:]), capacity, min_attendance, max_attendance)


def min_of_last_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the minimum of the last 3 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-3:].min(), capacity, min_attendance, max_attendance)


def max_of_last_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict the maximum of the last 3 weeks."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-3:].max(), capacity, min_attendance, max_attendance)


def weighted_average_recent(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Weighted average of last 3 weeks, heavier on most recent."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    window = h[-3:]
    weights = np.arange(1, 4, dtype=float)
    return _output(np.average(window, weights=weights), capacity, min_attendance, max_attendance)


def weighted_average_recent_5(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Weighted average of last 5 weeks, heavier on most recent."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    window = h[-5:]
    weights = np.arange(1, 6, dtype=float)
    return _output(np.average(window, weights=weights), capacity, min_attendance, max_attendance)


def alternating_cycle_2(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Assume a period-2 cycle."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-2], capacity, min_attendance, max_attendance)


def alternating_cycle_3(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Assume a period-3 cycle."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    return _output(h[-3], capacity, min_attendance, max_attendance)


def bounded_trend_last_8(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Fit a linear trend over last 8 weeks and bound within observed window range."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 3, fallback_go_probability)
    if not ok:
        return fallback
    window = h[-8:] if len(h) >= 8 else h
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    raw = window[-1] + slope
    raw = np.clip(raw, window.min(), window.max())
    return _output(raw, capacity, min_attendance, max_attendance)


def revert_to_mean_all_history(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict halfway between last value and overall mean."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    pred = 0.5 * h[-1] + 0.5 * h.mean()
    return _output(pred, capacity, min_attendance, max_attendance)


def revert_to_capacity(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Predict halfway between last value and capacity."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    pred = 0.5 * h[-1] + 0.5 * capacity
    return _output(pred, capacity, min_attendance, max_attendance)


def average_of_last_and_5_ago(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Average of last week's and 5 weeks ago attendance."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 5, fallback_go_probability)
    if not ok:
        return fallback
    return _output((h[-1] + h[-5]) / 2, capacity, min_attendance, max_attendance)


def average_of_last_2_and_last_5(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Average of the 2-week and 5-week moving averages."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 2, fallback_go_probability)
    if not ok:
        return fallback
    ma2 = h[-2:].mean()
    ma5 = h[-5:].mean() if len(h) >= 5 else h.mean()
    return _output((ma2 + ma5) / 2, capacity, min_attendance, max_attendance)


def contrarian_last(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Contrarian mirror of last attendance."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    return _output(max_attendance - h[-1], capacity, min_attendance, max_attendance)


def exponential_smoothing(
    history,
    fallback_go_probability: float,
    capacity: int = 59,
    min_attendance: int = 0,
    max_attendance: int = 100,
) -> Tuple[float, int]:
    """Simple exponential smoothing with alpha = 0.3."""
    h = _to_array(history)
    ok, fallback = _enough_history_or_random(h, 1, fallback_go_probability)
    if not ok:
        return fallback
    alpha = 0.3
    smoothed = h[0]
    for val in h[1:]:
        smoothed = alpha * val + (1 - alpha) * smoothed
    return _output(smoothed, capacity, min_attendance, max_attendance)