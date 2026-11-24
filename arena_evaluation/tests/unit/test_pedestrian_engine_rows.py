"""A recording without arena_peds scores its pedestrians on the engine's states."""

import polars as pl

from arena_evaluation.processing.pipeline import _engine_as_arena_peds

ARENA = "pedestrian_arena_peds"
ENGINE = "pedestrian_agent_states"


def _peds(n: int) -> pl.DataFrame:
    return pl.DataFrame({"t_ns": list(range(n)), "env_id": "env_0", "ped_id": [f"Pedestrian_{i}" for i in range(n)], "x": [0.0] * n})


def test_the_engine_states_stand_in_for_missing_or_empty_arena_peds():
    engine = _peds(2)
    assert _engine_as_arena_peds({ENGINE: engine})[ARENA] is engine
    assert _engine_as_arena_peds({ENGINE: engine, ARENA: _peds(0)})[ARENA] is engine
    # recorded arena_peds are kept, and without engine states nothing is added
    arena = _peds(3)
    assert _engine_as_arena_peds({ENGINE: engine, ARENA: arena})[ARENA] is arena
    assert ARENA not in _engine_as_arena_peds({"robot_tracks": engine})
    assert ARENA not in _engine_as_arena_peds({ENGINE: _peds(0)})
