"""Tests for the predictor library."""

import numpy as np

from src.agents.predictors import (
    default_predictor_library,
    mean_all_history,
    mirror_threshold,
    make_rolling_median,
    sample_predictor_library,
)


def test_mean_all_history_empty_returns_fallback():
    """Empty history returns threshold - 1 (strict-threshold NE level)."""
    assert mean_all_history((), 101, 60) == 59.0


def test_mean_all_history_basic():
    assert mean_all_history((40, 50, 70), 101, 60) == 160 / 3


def test_mirror_threshold_basic():
    assert mirror_threshold((70,), 101, 60) == 50.0


def test_mirror_threshold_clips_low():
    assert mirror_threshold((200,), 101, 60) == 0.0


def test_median_3_basic():
    pred = make_rolling_median(3)
    assert pred((40, 90, 60), 101, 60) == 60.0


def test_default_predictor_library_names_unique():
    names = [name for name, _ in default_predictor_library()]
    assert len(names) == len(set(names))


def test_default_predictor_library_returns_floats():
    history = (55, 61, 59, 58, 62)
    for _, pred in default_predictor_library():
        value = pred(history, 101, 60)
        assert isinstance(value, float)
        assert 0.0 <= value <= 101.0


def test_sample_predictor_library_size():
    """k=3 returns exactly 3 distinct predictor-name/function pairs."""
    rng = np.random.default_rng(42)
    sampled = sample_predictor_library(rng, k=3)
    assert len(sampled) == 3
    names = [name for name, _ in sampled]
    assert len(names) == len(set(names))


def test_sample_predictor_library_reproducible():
    """Two RNGs with the same seed give the same sampled bank."""
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    sampled1 = sample_predictor_library(rng1, k=4)
    sampled2 = sample_predictor_library(rng2, k=4)
    names1 = [name for name, _ in sampled1]
    names2 = [name for name, _ in sampled2]
    assert names1 == names2
