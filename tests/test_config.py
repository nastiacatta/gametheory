"""
Config validation: ensure invalid parameters raise clear errors.
"""

import pytest

from src.config import RepeatedGameConfig, StaticGameConfig


def test_static_config_valid() -> None:
    StaticGameConfig(n_players=101, threshold=60, seed=42)


def test_static_config_n_players_positive() -> None:
    with pytest.raises(ValueError, match="n_players must be positive"):
        StaticGameConfig(n_players=0, threshold=60)
    with pytest.raises(ValueError, match="n_players must be positive"):
        StaticGameConfig(n_players=-1, threshold=60)


def test_static_config_even_n_allowed() -> None:
    # Config does not enforce odd n; game engine does
    StaticGameConfig(n_players=10, threshold=5)


def test_static_config_threshold_in_range() -> None:
    with pytest.raises(ValueError, match="threshold must be between"):
        StaticGameConfig(n_players=10, threshold=-1)
    with pytest.raises(ValueError, match="threshold must be between"):
        StaticGameConfig(n_players=10, threshold=11)


def test_repeated_config_valid() -> None:
    RepeatedGameConfig(n_players=101, threshold=60, n_rounds=200, seed=42)


def test_repeated_config_n_rounds_positive() -> None:
    with pytest.raises(ValueError, match="n_rounds must be positive"):
        RepeatedGameConfig(n_players=10, threshold=5, n_rounds=0)
