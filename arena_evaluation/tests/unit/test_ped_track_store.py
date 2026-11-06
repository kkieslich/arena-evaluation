from __future__ import annotations

import math

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.ped_track_store import pair, strip_env, track_table

S = 1_000_000_000


def _peds(rows: list[tuple[int, list[int], list[tuple[float, float]], list[tuple[float, float]] | None, list[str] | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "peds_ids": [r[1] for r in rows],
            "peds_positions": [[v for x, y in r[2] for v in (x, y, 0.0)] for r in rows],
            "peds_twists": [None if r[3] is None else [v for vx, vy in r[3] for v in (vx, vy, 0.0)] for r in rows],
            "peds_names": [r[4] for r in rows],
        },
        schema_overrides={"peds_twists": pl.List(pl.Float64), "peds_names": pl.List(pl.String)},
    )


def _meta(rows: list[tuple[int, int, str, str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(("time_ns", "agent_id", "name", "agent_type", "source_name", "source_seq"))})


ENGINE = _peds(
    [
        (10 * S, [4, 11, 12], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], [(0.5, 0.0), (0.0, 1.0), (0.0, 0.0)], ["4", "11", "12"]),
        (11 * S, [4, 11, 12], [(1.5, 2.0), (3.0, 5.0), (5.0, 6.0)], [(0.5, 0.0), (0.0, 1.0), (0.0, 0.0)], ["4", "11", "12"]),
    ]
)
# the scenario person is named from the first frame on, the released visitor's meta arrives a second late
META = _meta([(9 * S, 4, "env_2/flow_1", "", "", 0), (11 * S, 11, "", "elder", "door_a", 7)])


def test_keys_pair_released_people_by_source_and_placed_people_by_name() -> None:
    tracks = track_table({"peds_physics": ENGINE, "agent_meta": META})

    first = {r["agent_id"]: r for r in tracks.filter(pl.col("time_ns") == 10 * S).iter_rows(named=True)}
    assert first[4]["key"] == "flow_1" and first[4]["name"] == "flow_1" and first[4]["agent_type"] == "adult"
    # the first later meta row names an agent whose meta came after its first frame
    assert (first[11]["key"], first[11]["source_name"], first[11]["source_seq"], first[11]["agent_type"]) == ("door_a#7", "door_a", 7, "elder")
    assert first[12]["key"] == "id12" and first[12]["name"] == ""
    assert tracks.filter(pl.col("agent_id") == 11)["y"].to_list() == [4.0, 5.0]
    assert tracks["t"].to_list() == [0.0] * 3 + [1.0] * 3


def test_tracks_are_cut_to_the_episode_and_timed_from_its_start() -> None:
    tracks = track_table({"peds_physics": ENGINE, "agent_meta": META}, start_ns=10 * S + S // 2, end_ns=20 * S)
    assert set(tracks["time_ns"].to_list()) == {11 * S}
    assert tracks["t"].to_list() == [0.5] * 3


def test_arena_peds_are_the_fallback_and_lose_their_env_prefix() -> None:
    peds = _peds([(10 * S, [4, 11], [(1.0, 2.0), (3.0, 4.0)], None, ["env_3/flow_1", "env_3/door_a_7"])])
    tracks = track_table({"peds": peds})
    assert sorted(tracks["key"].to_list()) == ["door_a_7", "flow_1"]
    # no twists recorded: velocities are NaN, left to derive from the positions
    assert all(math.isnan(v) for v in tracks["vx"].to_list())


def test_rows_whose_lists_disagree_with_their_ids_are_dropped() -> None:
    bad = pl.DataFrame({"time_ns": [10 * S], "peds_ids": [[4, 11]], "peds_positions": [[1.0, 2.0, 0.0]]})
    assert len(track_table({"peds_physics": bad})) == 0
    assert len(track_table({})) == 0


def test_pairing_lists_shared_and_unmatched_pedestrians() -> None:
    base = track_table({"peds_physics": ENGINE, "agent_meta": META})
    edit = base.filter(pl.col("key") != "id12").vstack(base.filter(pl.col("key") == "flow_1").with_columns(pl.lit("door_b#0").alias("key")))
    pairing = pair(base, edit)
    assert pairing.keys == ["door_a#7", "flow_1"] and pairing.only_ref == ["id12"] and pairing.only_run == ["door_b#0"]


def test_strip_env_drops_only_a_leading_env_segment() -> None:
    assert strip_env("env_12/flow_3") == "flow_3" and strip_env("/env_1/sofa/L1") == "sofa/L1" and strip_env("flow_env_1/x") == "flow_env_1/x"
