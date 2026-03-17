"""Tests for analysis metrics."""

from src.analysis.metrics import (
    mad_from_threshold,
    mean_attendance,
    overcrowding_rate,
    switch_rate,
    variance_from_threshold,
)


def test_mean_attendance() -> None:
    assert mean_attendance([50, 60, 70]) == 60.0


def test_variance_from_threshold() -> None:
    assert variance_from_threshold([60, 60, 60], 60) == 0.0
    assert variance_from_threshold([58, 62], 60) == 4.0


def test_mad_from_threshold() -> None:
    assert mad_from_threshold([60, 60], 60) == 0.0
    assert mad_from_threshold([58, 62], 60) == 2.0


def test_overcrowding_rate() -> None:
    assert overcrowding_rate([59, 60, 61], 60) == 1 / 3


def test_switch_rate() -> None:
    # Agent 0: [0, 1, 1] -> 1 switch. Agent 1: [1, 1, 0] -> 1 switch.
    # n=2, T=3, so (T-1)=2. Total switches=2. Rate = 2/(2*2)=0.5
    assert switch_rate([[0, 1, 1], [1, 1, 0]]) == 0.5
    assert switch_rate([[0, 0, 0]]) == 0.0
