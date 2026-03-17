"""
Configuration for the static and repeated minority game.

Parameters follow the coursework notation: n (players), L (threshold), m (rounds).
An odd n is often used in the literature; defaults here use n=101, L=60, m=200.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticGameConfig:
    """Immutable configuration for a single-shot game."""

    n_players: int = 101
    threshold: int = 60
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_players <= 0:
            raise ValueError("n_players must be positive.")
        if not (0 <= self.threshold <= self.n_players):
            raise ValueError("threshold must be between 0 and n_players (inclusive).")


@dataclass(frozen=True)
class RepeatedGameConfig:
    """Immutable configuration for the repeated game (m rounds)."""

    n_players: int = 101
    threshold: int = 60
    n_rounds: int = 200
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_players <= 0:
            raise ValueError("n_players must be positive.")
        if not (0 <= self.threshold <= self.n_players):
            raise ValueError("threshold must be between 0 and n_players (inclusive).")
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive.")
