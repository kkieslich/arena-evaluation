import math

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams

S = 1_000_000_000
POOLED = ("ped_path_deflection_m", "ped_velocity_delay_ratio", "ped_round_trips_completed")
ALL_NONE = dict.fromkeys(POOLED)
deflection = PedestrianDisturbanceCalculator._compute_trajectory_deflection


def _frame(rows: list, seconds: list[float] | None = None, names: list[list[str]] | None = None) -> pl.DataFrame:
    """Flat x,y,z pedestrian rows, the agent ids by name or by entry index."""
    columns: dict = {"peds_positions": pl.Series(rows, dtype=pl.List(pl.Float64))}
    if seconds is not None:
        columns["time_ns"] = pl.Series([int(s * S) for s in seconds], dtype=pl.Int64)
    if names is not None:
        columns["peds_names"] = pl.Series(names, dtype=pl.List(pl.Utf8))
        ids = {name: i for i, name in enumerate(dict.fromkeys(n for row in names for n in row))}
        columns["peds_ids"] = pl.Series([[ids[n] for n in row] for row in names], dtype=pl.List(pl.Int64))
    else:
        columns["peds_ids"] = pl.Series([list(range(len(row) // 3)) if row else [] for row in rows], dtype=pl.List(pl.Int64))
    return pl.DataFrame(columns)


def _physics(data: pl.DataFrame, desired: float | dict[str, float]) -> pl.DataFrame:
    """The crowd engine's frames of `data`'s pedestrians, all at `desired` m/s or each at its name's speed (0, unknown, when unlisted)."""
    names = data["peds_names"].to_list() if "peds_names" in data.columns else [None] * len(data)
    ids = data["peds_ids"].to_list()
    speeds = [[desired.get(n, 0.0) for n in row_names] if isinstance(desired, dict) else [desired] * len(row_ids) for row_names, row_ids in zip(names, ids, strict=True)]
    return pl.DataFrame({"peds_ids": ids, "peds_desired_velocities": speeds}, schema={"peds_ids": pl.List(pl.Int64), "peds_desired_velocities": pl.List(pl.Float64)})


def _result(data: pl.DataFrame | None, desired: float | dict[str, float] | None = 1.2) -> dict:
    topics = {"peds_physics": _physics(data, desired)} if data is not None and desired is not None and "peds_ids" in data.columns else None
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], topics=topics)
    return PedestrianDisturbanceCalculator(RobotParams()).calculate(episode, {})


def _calc(data: pl.DataFrame | None, desired: float | dict[str, float] | None = 1.2) -> dict:
    """The pooled episode values."""
    result = _result(data, desired)
    return {key: result[key] for key in POOLED}


def _per_ped(data: pl.DataFrame, key: str, desired: float | dict[str, float] | None = 1.2) -> dict[str, float]:
    return {record["ped"]: record["value"] for record in _result(data, desired)[f"{key}_per_ped"]}


def _walk(*points: tuple[float, float]) -> list[list[float]]:
    """One single-pedestrian x,y,z row per point."""
    return [[x, y, 0.0] for x, y in points]


def test_output_keys():
    assert PedestrianDisturbanceCalculator.output_keys() == [*POOLED, *(f"{key}_per_ped" for key in POOLED)]


def test_deflection_of_empty_input_is_zero():
    assert deflection(np.empty((0, 2)), np.array([[0.0, 0.0]])) == 0.0
    assert deflection(np.array([[0.0, 0.0]]), np.empty((0, 2))) == 0.0


def test_deflection_of_all_nan_rows_is_zero():
    assert deflection(np.array([[np.nan, 0.0], [1.0, np.nan]]), np.array([[0.0, 0.0]])) == 0.0
    assert deflection(np.array([[0.0, 0.0]]), np.array([[np.nan, np.nan]])) == 0.0


def test_deflection_is_mean_nearest_reference_distance_ignoring_nan_rows():
    actual = np.array([[0.0, 1.0], [2.0, 1.0], [np.nan, 0.0], [1.0, 3.0]])
    reference = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert deflection(actual, reference) == pytest.approx((1.0 + 1.0 + 3.0) / 3.0, abs=1e-12)


def test_deflection_ignores_z():
    assert deflection(np.array([[0.0, 0.0, 5.0]]), np.array([[3.0, 4.0, -2.0]])) == pytest.approx(5.0, abs=1e-12)


def test_deflection_with_infinite_reference_falls_back_to_dense_distances():
    actual = np.array([[0.0, 2.0], [4.0, 0.0]])
    reference = np.array([[0.0, 0.0], [np.inf, 0.0], [4.0, 1.0]])
    assert deflection(actual, reference) == pytest.approx((2.0 + 1.0) / 2.0, abs=1e-12)


def test_missing_frame_yields_none():
    assert _calc(None) == ALL_NONE


def test_frame_without_peds_column_yields_none():
    assert _calc(pl.DataFrame({"time_ns": [0, S]})) == ALL_NONE


def test_empty_frame_yields_none():
    assert _calc(_frame([], [])) == ALL_NONE


def test_rows_without_valid_points_yield_zeros():
    rows = [None, [], [math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0], [0.0, -math.inf, 0.0], [2e6, 0.0, 0.0], [0.0, 1e6, 0.0]]
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5, 6])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0}


def test_agents_under_five_samples_count_only_toward_speed():
    rows = _walk((0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0))
    assert _calc(_frame(rows, [0, 2, 4, 6])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": round(1.0 - (3.0 * math.sqrt(2.0) / 6.0) / 1.2, 3),
        "ped_round_trips_completed": 0.0,
    }


def test_back_and_forth_walk_counts_round_trips():
    rows = _walk((0.0, 0.0), (2.0, 0.0), (0.0, 0.0), (2.0, 0.0), (0.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": round(1.0 - 1.0 / 1.2, 3),
        "ped_round_trips_completed": 2.0,
    }


def test_straight_walk_at_half_speed_has_no_deflection_and_no_round_trip():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 2, 4, 6, 8])) == {
        "ped_path_deflection_m": 0.0,
        "ped_velocity_delay_ratio": 0.583,
        "ped_round_trips_completed": 0.0,
    }


def test_walk_faster_than_desired_speed_clamps_delay_to_zero():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 0.5, 1, 1.5, 2]))["ped_velocity_delay_ratio"] == 0.0


def test_deflection_and_delay_average_over_all_agents():
    detour = [(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0)]
    rows = [[x, y, 0.0, 10.0, 10.0, 0.0] for x, y in detour]
    result = _calc(_frame(rows, [0, 1, 2, 3, 4]))
    detour_length = 2.0 + 2.0 * math.sqrt(2.0)
    assert result["ped_path_deflection_m"] == round(0.6 / 2.0, 3)
    assert result["ped_velocity_delay_ratio"] == round(1.0 - (detour_length / 2.0 / 4.0) / 1.2, 3)


def test_stationary_pedestrians_report_zero_delay():
    rows = [[1.0, 1.0, 0.0]] * 5
    assert _calc(_frame(rows, [0, 1, 2, 3, 4])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_delay_is_zero_without_timestamps():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows))["ped_velocity_delay_ratio"] == 0.0


def test_delay_is_zero_for_single_sample():
    assert _calc(_frame([[1.0, 1.0, 0.0]], [0])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_delay_is_zero_for_zero_duration():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 0, 0, 0, 0]))["ped_velocity_delay_ratio"] == 0.0


def test_delay_is_zero_when_every_agent_has_one_sample():
    assert _calc(_frame([[1.0, 1.0, 0.0], []], [0, 1]))["ped_velocity_delay_ratio"] == 0.0


def test_invalid_sample_does_not_shift_pedestrian_identity():
    rows = [[0.0, 0.0, 0.0, 10.0, 0.0, 0.0] for _ in range(6)]
    rows[2] = [math.nan, math.nan, 0.0, 10.0, 0.0, 0.0]
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5])) == {"ped_path_deflection_m": 0.0, "ped_velocity_delay_ratio": 0.0, "ped_round_trips_completed": 0.0}


def test_one_way_detour_completes_no_round_trip():
    rows = _walk((0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4]))["ped_round_trips_completed"] == 0.0


def test_return_short_of_start_completes_no_round_trip():
    rows = _walk((0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (3.0, 0.0), (2.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4]))["ped_round_trips_completed"] == 0.0


def test_round_trip_counts_return_near_start_after_reaching_far_end():
    rows = _walk((0.0, 0.0), (1.0, 0.0), (4.0, 0.0), (3.9, 0.0), (0.9, 0.0), (0.5, 0.0), (4.0, 0.0))
    assert _calc(_frame(rows, [0, 1, 2, 3, 4, 5, 6]))["ped_round_trips_completed"] == 1.0


def test_missing_frame_yields_no_per_ped_values():
    assert _result(None) == dict.fromkeys(PedestrianDisturbanceCalculator.output_keys())


def test_rows_without_valid_points_yield_no_pedestrians():
    result = _result(_frame([None, [], [math.nan, 0.0, 0.0]], [0, 1, 2]))
    assert [result[f"{key}_per_ped"] for key in POOLED] == [[], [], []]


def test_unnamed_pedestrians_are_keyed_by_entry_index():
    rows = [[x, 0.0, 0.0, x, 2.0, 0.0] for x in (0.0, 1.0, 2.0, 3.0, 4.0)]
    assert _per_ped(_frame(rows, [0, 1, 2, 3, 4]), "ped_round_trips_completed") == {"#0": 0.0, "#1": 0.0}


def test_named_pedestrian_keeps_its_track_when_another_leaves():
    """Ped B moves to entry 0 once A has left; keyed by index the two tracks would mix into one detour."""
    a = [(0.0, 0.0), (1.0, 0.0)]
    b = [(0.0, 5.0), (1.0, 6.0), (2.0, 6.0), (3.0, 6.0), (4.0, 5.0)]
    rows = [[*a[0], 0.0, *b[0], 0.0], [*a[1], 0.0, *b[1], 0.0], [*b[2], 0.0], [*b[3], 0.0], [*b[4], 0.0]]
    names = [["/env_0/A", "/env_0/B"], ["/env_0/A", "/env_0/B"], ["/env_0/B"], ["/env_0/B"], ["/env_0/B"]]
    data = _frame(rows, [0, 1, 2, 3, 4], names)

    assert _per_ped(data, "ped_path_deflection_m") == {"B": 0.6}
    assert _per_ped(data, "ped_round_trips_completed") == {"B": 0.0}
    assert _calc(data)["ped_path_deflection_m"] == 0.3


def test_per_ped_velocity_delay_counts_only_the_time_a_pedestrian_was_recorded():
    rows = [[0.0, 9.0, 0.0], [0.0, 9.0, 0.0], [0.0, 9.0, 0.0, 0.0, 0.0, 0.0], [0.0, 9.0, 0.0, 0.6, 0.0, 0.0]]
    names = [["visitor"], ["visitor"], ["visitor", "walker"], ["visitor", "walker"]]
    data = _frame(rows, [0, 1, 2, 3], names)
    assert _per_ped(data, "ped_velocity_delay_ratio") == {"visitor": 0.0, "walker": 0.5}
    assert _calc(data)["ped_velocity_delay_ratio"] == round(1.0 - (0.6 / 2.0 / 3.0) / 1.2, 3)


def test_single_sample_pedestrian_has_no_velocity_delay():
    rows = [[0.0, 0.0, 0.0, 5.0, 5.0, 0.0], [1.2, 0.0, 0.0]]
    assert _per_ped(_frame(rows, [0, 1], [["walker", "passer"], ["walker"]]), "ped_velocity_delay_ratio") == {"walker": 0.0}


def _two_walkers() -> pl.DataFrame:
    rows = [[x, 0.0, 0.0, x, 5.0, 0.0] for x in (0.0, 0.6, 1.2, 1.8)]
    return _frame(rows, [0, 1, 2, 3], [["chair", "walker"]] * 4)


def test_velocity_delay_holds_each_pedestrian_to_its_own_desired_speed():
    data = _two_walkers()
    assert _per_ped(data, "ped_velocity_delay_ratio", {"chair": 0.6, "walker": 1.2}) == {"chair": 0.0, "walker": 0.5}
    assert _calc(data, {"chair": 0.6, "walker": 1.2})["ped_velocity_delay_ratio"] == round(1.0 - 0.6 / 0.9, 3)


def test_pedestrian_without_a_desired_speed_has_no_velocity_delay():
    data = _two_walkers()
    assert _per_ped(data, "ped_velocity_delay_ratio", {"walker": 1.2}) == {"walker": 0.5}
    assert _calc(data, {"walker": 1.2})["ped_velocity_delay_ratio"] == 0.5


def test_velocity_delay_is_none_without_recorded_desired_speeds():
    result = _result(_two_walkers(), desired=None)
    assert (result["ped_velocity_delay_ratio"], result["ped_velocity_delay_ratio_per_ped"]) == (None, None)
    assert result["ped_round_trips_completed"] == 0.0


def test_desired_speed_is_the_median_of_the_positive_recorded_ones():
    data = _frame([[x, 0.0, 0.0] for x in (0.0, 0.6, 1.2, 1.8)], [0, 1, 2, 3], [["walker"]] * 4)
    physics = pl.DataFrame({"peds_ids": [[0]] * 4, "peds_desired_velocities": [[1.0], [0.0], [1.4], [1.2]]})
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[], topics={"peds_physics": physics})
    result = PedestrianDisturbanceCalculator(RobotParams()).calculate(episode, {})
    assert result["ped_velocity_delay_ratio_per_ped"] == [{"ped": "walker", "value": 0.5}]
