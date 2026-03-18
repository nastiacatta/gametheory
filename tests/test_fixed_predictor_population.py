from src.agents.fixed_predictor_agent import FixedPredictorAgent
from src.agents.predictors import default_predictor_library
from src.experiments.populations import (
    build_homogeneous_fixed_predictor,
    build_heterogeneous_fixed_predictor,
)


def test_build_homogeneous_fixed_predictor():
    agents = build_homogeneous_fixed_predictor(11, predictor_name="last_value")

    assert len(agents) == 11
    assert all(isinstance(a, FixedPredictorAgent) for a in agents)
    assert all(a.predictor_name == "last_value" for a in agents)


def test_build_heterogeneous_fixed_predictor_size():
    agents = build_heterogeneous_fixed_predictor(17, seed=42)

    assert len(agents) == 17
    assert all(isinstance(a, FixedPredictorAgent) for a in agents)


def test_build_heterogeneous_fixed_predictor_covers_library():
    library_size = len(default_predictor_library())
    n_players = max(20, library_size)

    agents = build_heterogeneous_fixed_predictor(
        n_players,
        seed=42,
        cover_all_predictors=True,
    )

    names = {a.predictor_name for a in agents}
    expected = {name for name, _ in default_predictor_library()}

    assert expected.issubset(names)
