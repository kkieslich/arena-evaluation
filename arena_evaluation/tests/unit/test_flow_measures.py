"""Flow measures on hand-made tracks: the grid, the divergence, and an edit against its base world."""

from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")
pytest.importorskip("shapely")

from arena_evaluation.processing.flow import FlowGrid, jsd, occupancy, pooled
from arena_evaluation.processing.flow.benchmark import add_flow_reference_metrics, parse_stage
from arena_evaluation.processing.metrics.flow.flow_episode import object_uses
from arena_evaluation.processing.ped_track_store import TRACK_SCHEMA

ROOM = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]


def _tracks(rows: list[tuple[str, float, float, float, float, float]]) -> pl.DataFrame:
    """(key, t, x, y, vx, vy) rows as a track table."""
    data: dict[str, list] = {c: [] for c in TRACK_SCHEMA}
    for key, t, x, y, vx, vy in rows:
        row = {"episode_id": 0, "time_ns": int(t * 1e9), "t": t, "agent_id": 1, "key": key, "name": key, "source_name": "", "source_seq": 0, "agent_type": "adult", "x": x, "y": y, "vx": vx, "vy": vy, "radius": 0.25}
        for c in TRACK_SCHEMA:
            data[c].append(row[c])
    return pl.DataFrame(data, schema=TRACK_SCHEMA)


def _stand(key: str, x: float, y: float, seconds: float = 2.0) -> list[tuple]:
    return [(key, k * 0.5, x, y, 0.0, 0.0) for k in range(int(seconds / 0.5) + 1)]


def _walk_east(key: str, y: float = 0.75) -> list[tuple]:
    return [(key, k * 0.5, 0.25 + 0.5 * k, y, 1.0, 0.0) for k in range(5)]


GRID = FlowGrid.from_polygons([ROOM])


def test_the_grid_covers_the_rooms_in_half_metre_cells() -> None:
    assert (GRID.x0, GRID.y0, GRID.nx, GRID.ny, GRID.bins) == (0.0, 0.0, 8, 4, 9)
    assert FlowGrid.from_polygons([]) is None and FlowGrid.from_polygons([[(0.0, 0.0), (1.0, 1.0)]]) is None


def test_walking_and_standing_time_land_in_their_cells_and_bins() -> None:
    walked = GRID.counts(_tracks(_walk_east("a")))
    # five frames 0.5 s apart: four steps of 0.5 s, and the last frame lasts one median step
    assert walked.sum() == pytest.approx(2.5)
    # heading 0 rad (east) is bin 4 of 8: [-180 + 4 * 45, -180 + 5 * 45) degrees
    assert walked[..., 4].sum() == pytest.approx(2.5) and walked[1, :, 4].tolist() == [0.5] * 5 + [0.0] * 3
    stood = GRID.counts(_tracks(_stand("b", 3.25, 1.75)))
    assert stood[3, 6, 8] == pytest.approx(2.5) and stood.sum() == pytest.approx(2.5)


def test_only_time_inside_the_rooms_counts() -> None:
    l_shape = FlowGrid.from_polygons([[(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]])
    # (3.25, 1.75) lies in the bounding box but outside the L
    assert l_shape.counts(_tracks(_stand("b", 3.25, 1.75))).sum() == 0
    assert GRID.counts(_tracks(_stand("c", 9.0, 9.0))).sum() == 0


def test_velocities_missing_from_the_recording_come_from_the_positions() -> None:
    rows = [(k, t, x, y, float("nan"), float("nan")) for k, t, x, y, _, _ in _walk_east("a")]
    assert GRID.counts(_tracks(rows))[..., 4].sum() == pytest.approx(2.5)


def test_a_gap_in_the_recording_counts_at_most_four_median_steps() -> None:
    rows = _stand("a", 1.25, 0.75, seconds=1.0) + [("a", 11.0, 1.25, 0.75, 0.0, 0.0)]
    assert GRID.counts(_tracks(rows)).sum() == pytest.approx(1.0 + 4 * 0.5 + 0.5)


def test_divergence_is_zero_for_the_same_flow_and_one_for_disjoint_flow() -> None:
    a = GRID.counts(_tracks(_stand("a", 0.25, 0.25)))
    b = GRID.counts(_tracks(_stand("a", 3.75, 1.75)))
    assert jsd(a, a) == pytest.approx(0.0) and jsd(a, 3 * a) == pytest.approx(0.0)
    assert jsd(a, b) == pytest.approx(1.0)
    # same cell, other heading: disjoint in the full map, identical occupancy
    walked = GRID.counts(_tracks([("w", 0.0, 0.25, 0.25, 1.0, 0.0), ("w", 0.5, 0.25, 0.25, 1.0, 0.0)]))
    stood = GRID.counts(_tracks([("s", 0.0, 0.25, 0.25, 0.0, 0.0), ("s", 0.5, 0.25, 0.25, 0.0, 0.0)]))
    assert jsd(walked, stood) == pytest.approx(1.0) and jsd(occupancy(walked), occupancy(stood)) == pytest.approx(0.0)
    assert jsd(a, GRID.empty()) is None


def test_pooling_averages_each_seed_as_a_share() -> None:
    a = GRID.counts(_tracks(_stand("a", 0.25, 0.25)))
    b = GRID.counts(_tracks(_stand("a", 3.75, 1.75, seconds=6.0)))
    mixed = pooled([a, b])
    assert mixed.sum() == pytest.approx(1.0) and mixed[0, 0, 8] == pytest.approx(0.5) and mixed[3, 7, 8] == pytest.approx(0.5)
    assert pooled([GRID.empty()]) is None


def test_stage_names_parse_to_world_and_seed() -> None:
    assert parse_stage("hospital_1__move_fountain__6e1d6873__s12") == ("hospital_1__move_fountain__6e1d6873", 12)
    assert parse_stage("hospital_1__s1") == ("hospital_1", 1)
    assert parse_stage("hospital_1") is None and parse_stage("a planner stage") is None and parse_stage(None) is None


# The fountain moves from (0.25, 0.25) to (3.75, 1.75). Seed 1: the drinkers all move with it. Seed 2: one of the two
# keeps going to the old spot. A held-out edit (seed 6) has no base run at its seed.
EPISODES = {
    "hospital_1__s1": _stand("p#1", 0.25, 0.25) + _stand("p#2", 0.25, 0.25),
    "hospital_1__s2": _stand("p#1", 0.25, 0.25) + _stand("p#2", 0.25, 0.25),
    "hospital_1__move_fountain__ab__s1": _stand("p#1", 3.75, 1.75) + _stand("p#2", 3.75, 1.75),
    "hospital_1__move_fountain__ab__s2": _stand("p#1", 3.75, 1.75) + _stand("p#3", 0.25, 0.25),
    "hospital_1__move_fountain__ab__s6": _stand("p#1", 3.75, 1.75),
}


def _rows(planner: str = "peds_only", first: int = 0) -> list[dict]:
    rows = [{"episode": first + i, "planner": planner, "robot": "env_0_jackal", "stage": stage, "status": "evaluated", "is_reference": False} for i, stage in enumerate(EPISODES)]
    rows.append({"episode": first + 99, "planner": planner, "robot": "env_0_jackal", "stage": "hospital_1__s1", "status": "evaluated", "is_reference": True})
    rows.append({"episode": first + 98, "planner": planner, "robot": "env_0_jackal", "stage": "a planner stage", "status": "evaluated", "is_reference": False})
    return rows


def _measure(rows: list[dict], recordings: dict[tuple[int, str], list[tuple]] | None = None, loaded: list[tuple[int, str]] | None = None) -> int:
    """Flow outputs of `rows`, episode `i` (and `i + 100`) recording EPISODES' i-th stage unless `recordings` names its (episode, env)."""
    stages = list(EPISODES)

    def tracks(directory: pathlib.Path, env: str) -> pl.DataFrame:
        episode = int(directory.name)
        if loaded is not None:
            loaded.append((episode, env))
        if recordings is not None and (episode, env) in recordings:
            return _tracks(recordings[(episode, env)])
        return _tracks(EPISODES[stages[episode % 100]])

    return add_flow_reference_metrics(rows, lambda episode: pathlib.Path("/runs/r/episodes") / str(episode), tracks=tracks, grid=lambda world, run_dir: GRID)


def test_each_edit_is_scored_against_its_base_world_at_the_same_seed() -> None:
    rows = _rows()
    assert _measure(rows) == 5
    by = {r["stage"]: r for r in rows[:5]}
    base, s1, s2, s6 = by["hospital_1__s1"], by["hospital_1__move_fountain__ab__s1"], by["hospital_1__move_fountain__ab__s2"], by["hospital_1__move_fountain__ab__s6"]
    assert base["flow_reference_stage"] is None and base["flow_jsd"] is None and base["flow_base"] == "hospital_1"
    assert base["flow_room_ped_seconds"] == pytest.approx(5.0)
    assert (s1["flow_reference_stage"], s1["flow_world"], s1["flow_seed"]) == ("hospital_1__s1", "hospital_1__move_fountain__ab", 1)
    assert s1["flow_jsd"] == pytest.approx(1.0) and s1["flow_jsd_occupancy"] == pytest.approx(1.0)
    assert (s1["flow_paired_peds"], s1["flow_unpaired_peds"]) == (2, 0)
    # half the time stays at the old spot: JSD of (1/2, 1/2) against (1, 0) is 1.5 - 0.75 log2(3), about 0.311
    assert s2["flow_jsd"] == pytest.approx(1.5 - 0.75 * np.log2(3)) and (s2["flow_paired_peds"], s2["flow_unpaired_peds"]) == (1, 2)
    # pooled over seeds 1 and 2: three quarters moved
    pooled_jsd = s1["flow_jsd_pooled"]
    assert pooled_jsd == s2["flow_jsd_pooled"] == s6["flow_jsd_pooled"] and s1["flow_pooled_seeds"] == 2
    assert 0.311 < pooled_jsd < 1.0
    assert s6["flow_reference_stage"] is None and s6["flow_jsd"] is None


def test_reference_planner_and_failed_rows_are_left_alone() -> None:
    rows = _rows()
    rows[1]["status"] = "error"
    _measure(rows)
    assert "flow_jsd" not in rows[1] and "flow_jsd" not in rows[5] and "flow_jsd" not in rows[6]
    # the edit at seed 2 has no evaluated base run any more
    assert rows[3]["flow_reference_stage"] is None and rows[2]["flow_pooled_seeds"] == 1


def test_each_contestant_is_scored_against_its_own_base_run() -> None:
    # contestant b's base run at seed 1 already had the drinkers at the new spot, so its edit changes nothing
    rows = _rows("a") + _rows("b", first=100)
    _measure(rows, {(100, "env_0"): _stand("p#1", 3.75, 1.75) + _stand("p#2", 3.75, 1.75)})
    a = {r["stage"]: r for r in rows[:5]}
    b = {r["stage"]: r for r in rows[7:12]}
    assert a["hospital_1__move_fountain__ab__s1"]["flow_jsd"] == pytest.approx(1.0)
    assert b["hospital_1__move_fountain__ab__s1"]["flow_jsd"] == pytest.approx(0.0)
    assert b["hospital_1__move_fountain__ab__s1"]["flow_reference_stage"] == "hospital_1__s1"
    assert a["hospital_1__move_fountain__ab__s1"]["flow_jsd_pooled"] != pytest.approx(b["hospital_1__move_fountain__ab__s1"]["flow_jsd_pooled"])


def test_each_env_of_an_episode_reads_its_own_tracks_once() -> None:
    rows = [{"episode": 20, "planner": "peds_only", "robot": robot, "stage": "hospital_1__s1", "status": "evaluated", "is_reference": False} for robot in ("env_0_jackal", "env_0_burger", "env_1_jackal")]
    loaded: list[tuple[int, str]] = []
    _measure(rows, {(20, "env_0"): _stand("p#1", 0.25, 0.25), (20, "env_1"): _stand("p#1", 0.25, 0.25) + _stand("p#2", 3.75, 1.75)}, loaded)
    assert [r["flow_room_ped_seconds"] for r in rows] == pytest.approx([2.5, 2.5, 5.0])
    assert loaded == [(20, "env_0"), (20, "env_1")]


def test_object_uses_count_held_interactions_and_the_longest_queue() -> None:
    interactions = pl.DataFrame(
        {
            "time_ns": [1, 2, 3, 4, 5],
            "interaction_id": [7, 7, 8, 9, 10],
            "object_id": ["env_0/fountain", "env_0/fountain", "env_0/fountain", "env_0/sofa", ""],
            "queue": [[], [3, 4], [5], [], []],
            "holding": [False, True, True, False, True],
        }
    )
    assert object_uses(interactions) == [{"object": "fountain", "uses": 2, "max_queue": 2}, {"object": "sofa", "uses": 0, "max_queue": 0}]
    assert object_uses(None) == []
