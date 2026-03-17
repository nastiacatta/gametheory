from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    """Configuration for the static minority game."""

    n_players: int = 101
    threshold: int = 60
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_players <= 0:
            raise ValueError("n_players must be positive.")
        if self.n_players % 2 == 0:
            raise ValueError("n_players should be odd for this coursework setup.")
        if not (0 <= self.threshold <= self.n_players):
            raise ValueError("threshold must lie between 0 and n_players.")
