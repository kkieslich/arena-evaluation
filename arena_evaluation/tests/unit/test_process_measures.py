"""`evaluation process --measures`: the calculator selection, the episode stamp of a selection, and the exposure grid limit."""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest
from robot_params_factory import make_robot_params

from arena_evaluation.processing.metrics.ecological.acoustic_exposure import (
    _PEDESTRIAN_KEYS,
    AcousticExposureCalculator,
    max_exposure_grid_cells,
)
from arena_evaluation.processing.metrics.registry import MEASURE_SETS, MetricRegistry, select_calculators, shipped_calculators
from arena_evaluation.storage.layout import (
    episode_is_complete,
    episode_stamp_is_current,
    write_episode_stamp,
)
from arena_evaluation.storage.registry import MetricScope
from arena_evaluation.storage.registry.null_reasons import NullReason
from arena_evaluation.storage.schemas import RobotParams


def _params() -> RobotParams:
    return make_robot_params()


def test_flow_selects_no_calculator() -> None:
    assert MEASURE_SETS["flow"] == frozenset()
    assert select_calculators(["flow"]) == frozenset()


def test_a_name_selects_it_with_its_dependencies() -> None:
    classes = {calc.NAME: calc for calc in shipped_calculators()}
    name = next(name for name, calc in classes.items() if calc.DEPENDS_ON)
    chosen = select_calculators([name])
    assert name in chosen
    assert set(classes[name].DEPENDS_ON) & set(classes) <= chosen


def test_a_category_selects_its_calculators() -> None:
    classes = {calc.NAME: calc for calc in shipped_calculators()}
    chosen = select_calculators(["ecological"])
    assert {name for name, calc in classes.items() if calc.CATEGORY == "ecological"} <= chosen
    assert "acoustic_exposure" in chosen


def test_an_unknown_measure_is_refused() -> None:
    with pytest.raises(ValueError, match="names no calculator"):
        select_calculators(["no_such_measure"])


def test_a_selection_keeps_every_calculator_registered_and_runs_only_its_series() -> None:
    every = MetricRegistry(_params())
    selected = MetricRegistry(_params(), selected=frozenset())
    assert set(selected.calculators) == set(every.calculators)
    assert not selected.is_selected("acoustic_exposure")
    assert every.is_selected("acoustic_exposure")
    assert selected.series_calculators() == []
    assert every.series_calculators()


def test_an_unselected_calculator_publishes_null_with_not_selected() -> None:
    registry = MetricRegistry(_params(), selected=frozenset())
    episode = types.SimpleNamespace(topics={}, series={}, episode_id=0)
    outcome = registry.run_scope(MetricScope.EPISODE, episode, available_topics=set())  # type: ignore[arg-type]
    assert outcome.results
    assert all(value is None for value in outcome.results.values())
    assert all(reason.endswith(f":{NullReason.NOT_SELECTED.value}") for reason in outcome.provenance.values())


def test_a_selection_stamp_covers_only_its_calculators(tmp_path: pathlib.Path) -> None:
    write_episode_stamp(tmp_path, frozenset({"a", "b"}))
    assert episode_stamp_is_current(tmp_path)
    assert not episode_is_complete(tmp_path)
    assert episode_is_complete(tmp_path, frozenset({"a"}))
    assert episode_is_complete(tmp_path, frozenset())
    assert not episode_is_complete(tmp_path, frozenset({"a", "c"}))


def test_a_full_stamp_covers_every_selection(tmp_path: pathlib.Path) -> None:
    write_episode_stamp(tmp_path)
    assert episode_is_complete(tmp_path)
    assert episode_is_complete(tmp_path, frozenset({"a"}))


def test_the_exposure_grid_limit_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARENA_ACOUSTIC_MAX_GRID_CELLS", raising=False)
    assert max_exposure_grid_cells() == 4_000_000
    monkeypatch.setenv("ARENA_ACOUSTIC_MAX_GRID_CELLS", "0")
    assert max_exposure_grid_cells() is None
    monkeypatch.setenv("ARENA_ACOUSTIC_MAX_GRID_CELLS", "100")
    assert max_exposure_grid_cells() == 100


def test_an_unsolved_grid_publishes_null_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    calc = AcousticExposureCalculator(_params())
    monkeypatch.setattr(calc, "_episode_result", lambda episode, prior: calc._null_result(NullReason.GRID_TOO_LARGE.value))
    pedestrian = types.SimpleNamespace(scope=MetricScope.PEDESTRIAN, full_episode=None, identity={"ped_id": "p0"}, slice_id="p0")
    out = calc.calculate_for_scope(None, {}, pedestrian)  # type: ignore[arg-type]
    assert all(out[key] is None and out[f"{key}_null_reason"] == "grid_too_large" for key in _PEDESTRIAN_KEYS)
    encounter = types.SimpleNamespace(scope=MetricScope.ENCOUNTER, full_episode=None, identity={"actor_2": "p0", "actor_2_kind": "pedestrian"}, slice_id="e0", t_start_ns=0, t_end_ns=1)
    out = calc.calculate_for_scope(None, {}, encounter)  # type: ignore[arg-type]
    assert out == {"sound_exposure_dose_pa2s": None, "sound_exposure_dose_pa2s_null_reason": "grid_too_large"}


def test_a_grid_above_the_limit_is_not_solved(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_evaluation.processing.metrics.ecological import acoustic_exposure

    barrier = types.SimpleNamespace(occupancy=np.zeros((20, 20), dtype=np.uint8), resolution=0.05, origin=(0.0, 0.0), door_masks={}, door_cells_across={})
    monkeypatch.setattr(acoustic_exposure, "load_barrier_map", lambda name, run_dir=None: barrier)
    monkeypatch.setenv("ARENA_ACOUSTIC_MAX_GRID_CELLS", "100")
    calc = AcousticExposureCalculator(_params())
    monkeypatch.setattr(calc, "_sample_grid", lambda episode: (np.array([0]), np.array([0.0]), np.array([0.0]), np.array([True])))
    monkeypatch.setattr(calc, "native_ped_frame", lambda episode: [1])
    episode = types.SimpleNamespace(map="big", run_dir=None, folder_manager=None, run=None)
    result = calc._compute(episode, {})  # type: ignore[arg-type]
    assert result.null_reason == "grid_too_large"
    assert result.pedestrian_scalars == {}
