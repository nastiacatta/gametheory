"""Tests for the Arthur-style predictor library."""

from src.agents.predictors import (
    default_predictor_library,
    last_value,
    make_lag_cycle,
    make_linear_trend,
    make_rolling_mean,
    mirror,
)


def test_last_value() -> None:
    assert last_value((), 100, 50) == 50.0
    assert last_value((60, 70), 100, 50) == 70.0


def test_mirror() -> None:
    assert mirror((), 100, 50) == 50.0
    assert mirror((70,), 100, 50) == 30.0


def test_rolling_mean() -> None:
    pred = make_rolling_mean(4)
    assert pred((), 100, 50) == 50.0
    assert pred((10, 20, 30, 40), 100, 50) == 25.0
    assert pred((10, 20), 100, 50) == 15.0


def test_linear_trend() -> None:
    pred = make_linear_trend(8)
    assert pred((), 100, 50) == 50.0
    assert pred((10,), 100, 50) == 50.0
    # [10, 20, 30, 40]: slope 10, last 40, extrapolate 50
    assert 48 <= pred((10, 20, 30, 40), 100, 50) <= 52


def test_lag_cycle() -> None:
    pred = make_lag_cycle(2)
    assert pred((), 100, 50) == 50.0
    assert pred((10,), 100, 50) == 50.0
    assert pred((10, 20, 30), 100, 50) == 20.0


def test_default_library_size() -> None:
    lib = default_predictor_library()
    assert len(lib) == 6
    names = [n for n, _ in lib]
    assert "last_value" in names
    assert "mirror" in names
