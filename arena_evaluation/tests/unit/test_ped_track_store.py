from __future__ import annotations

import math
import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.ped_track_store import TRACK_SCHEMA, pair, strip_env, track_table, tracks_for_episode, tracks_path, write_tracks
from arena_evaluation.storage.schemas import TopicBundle

S = 1_000_000_000


def _peds(rows: list[tuple[int, list[int], list[tuple[float, float]], list[tuple[float, float]] | None, list[str] | None]], radii: list[list[float]] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [r[0] for r in rows],
            "peds_ids": [r[1] for r in rows],
            "peds_positions": [[v for x, y in r[2] for v in (x, y, 0.0)] for r in rows],
            "peds_twists": [None if r[3] is None else [v for vx, vy in r[3] for v in (vx, vy, 0.0)] for r in rows],
            "peds_names": [r[4] for r in rows],
            "peds_radii": radii if radii is not None else [[] for _ in rows],
        },
        schema_overrides={"peds_twists": pl.List(pl.Float64), "peds_names": pl.List(pl.String), "peds_radii": pl.List(pl.Float64)},
    )


def _meta(rows: list[tuple[int, int, str, str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame({k: [r[i] for r in rows] for i, k in enumerate(("time_ns", "agent_id", "name", "agent_type", "source_name", "source_seq"))})


ENGINE = _peds(
    [
        (10 * S, [4, 11, 12], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], [(0.5, 0.0), (0.0, 1.0), (0.0, 0.0)], ["4", "11", "12"]),
        (11 * S, [4, 11, 12], [(1.5, 2.0), (3.0, 5.0), (5.0, 6.0)], [(0.5, 0.0), (0.0, 1.0), (0.0, 0.0)], ["4", "11", "12"]),
    ],
    radii=[[0.25, 0.325, 0.18], [0.25, 0.325, 0.18]],
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


def test_tracks_carry_the_episode_and_each_pedestrians_radius() -> None:
    tracks = track_table({"peds_physics": ENGINE, "agent_meta": META}, episode_id=7)
    assert list(tracks.columns) == list(TRACK_SCHEMA)
    assert set(tracks["episode_id"].to_list()) == {7}
    assert dict(zip(tracks["agent_id"].to_list(), tracks["radius"].to_list(), strict=True)) == {4: 0.25, 11: 0.325, 12: 0.18}


def test_an_unrecorded_or_zero_radius_is_nan() -> None:
    frame = [(10 * S, [4, 11], [(1.0, 2.0), (3.0, 4.0)], None, ["env_3/flow_1", "env_3/flow_2"])]
    for radii in (None, [[0.3, 0.0]]):
        tracks = track_table({"peds": _peds(frame, radii)}).sort("agent_id")
        assert math.isnan(tracks["radius"][1]) and (radii is None) == math.isnan(tracks["radius"][0])
    assert track_table({"peds": _peds(frame)})["episode_id"].null_count() == 2


def _engine(x: float) -> pl.DataFrame:
    return _peds([(10 * S, [4], [(x, 2.0)], [(0.5, 0.0)], ["4"]), (11 * S, [4], [(x + 0.5, 2.0)], [(0.5, 0.0)], ["4"])], radii=[[0.25], [0.25]])


def _episode(tmp_path: pathlib.Path, x_by_env: dict[str, float]) -> pathlib.Path:
    """An extracted episode_007 whose envs each ran one pedestrian, starting at x."""
    episode = tmp_path / "episode_007"
    bundles = {f"{env}_jackal": TopicBundle(odom=pl.DataFrame({"time_ns": [10 * S]}), peds_physics=_engine(x), agent_meta=_meta([(9 * S, 4, f"{env}/flow_1", "", "", 0)])) for env, x in x_by_env.items()}
    TopicParquetStore.write(bundles, episode / "topics")
    return episode


def test_each_env_of_an_episode_has_its_own_kept_tracks(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path, {"env_1": 1.0, "env_2": 8.0})
    first, second = tracks_for_episode(episode, "env_1"), tracks_for_episode(episode, "env_2")
    assert first["x"].to_list() == [1.0, 1.5] and second["x"].to_list() == [8.0, 8.5]
    assert set(first["episode_id"].to_list()) == {7} and first["key"].to_list() == ["flow_1", "flow_1"]
    assert tracks_path(episode, "env_1").is_file() and tracks_path(episode, "env_2").is_file()
    with pytest.raises(ValueError, match="name the env"):
        tracks_for_episode(episode)
    with pytest.raises(ValueError, match="no env_3"):
        tracks_for_episode(episode, "env_3")


def test_an_episode_with_one_env_needs_no_env_named(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path, {"env_1": 1.0})
    assert tracks_for_episode(episode)["x"].to_list() == [1.0, 1.5]
    assert len(tracks_for_episode(tmp_path / "episode_008")) == 0


def test_kept_tracks_are_read_back_until_the_topic_cache_is_written_again(tmp_path: pathlib.Path) -> None:
    episode = _episode(tmp_path, {"env_1": 1.0})
    built = write_tracks(episode, TopicParquetStore.read(episode / "topics") or {})
    assert list(built) == ["env_1"] and tracks_for_episode(episode, "env_1").equals(built["env_1"])
    # the kept file, not the topics, answers a second read
    pl.DataFrame(schema=TRACK_SCHEMA).write_parquet(tracks_path(episode, "env_1"))
    assert len(tracks_for_episode(episode, "env_1")) == 0
    _episode(tmp_path, {"env_1": 4.0})
    assert tracks_for_episode(episode, "env_1")["x"].to_list() == [4.0, 4.5]


def test_tracks_are_timed_by_the_engine_stamps_from_the_episodes_start_time() -> None:
    # the recorder logged the first frame 0.4 s late and the second 30 ms late; the engine stamped them at its ticks
    stamped = ENGINE.with_columns(pl.Series("stamp_ns", [100 * S + S // 20, 100 * S + S // 10]))
    logged = stamped.with_columns(pl.Series("time_ns", [10 * S + 2 * S // 5, 10 * S + 13 * S // 100]))
    tracks = track_table({"peds_physics": logged, "agent_meta": META}, start_ns=10 * S, end_ns=20 * S, start_stamp_ns=100 * S)
    assert tracks["t"].to_list() == pytest.approx([0.05] * 3 + [0.1] * 3)
    assert set(tracks["time_ns"].to_list()) == {100 * S + S // 20, 100 * S + S // 10}
    # the same frames logged on time give the same tracks: the log delay is gone
    on_time = stamped.with_columns(pl.Series("time_ns", [10 * S + S // 20, 10 * S + S // 10]))
    again = track_table({"peds_physics": on_time, "agent_meta": META}, start_ns=10 * S, end_ns=20 * S, start_stamp_ns=100 * S)
    assert again.select("key", "t", "x", "y").equals(tracks.select("key", "t", "x", "y"))
    # a late copy of the previous episode's frame (stamped before the start) is dropped
    stale = pl.concat([logged.head(1).with_columns(pl.lit(99 * S).alias("stamp_ns")), logged])
    assert track_table({"peds_physics": stale, "agent_meta": META}, start_ns=10 * S, end_ns=20 * S, start_stamp_ns=100 * S)["t"].min() == pytest.approx(0.05)
    # without the start time, or frames without stamps, the log time stands in
    assert track_table({"peds_physics": logged, "agent_meta": META}, start_ns=10 * S, end_ns=20 * S)["t"].to_list() == pytest.approx([0.13] * 3 + [0.4] * 3)
    assert track_table({"peds_physics": ENGINE, "agent_meta": META}, start_ns=10 * S, end_ns=20 * S, start_stamp_ns=100 * S)["t"].to_list() == [0.0] * 3 + [1.0] * 3


def test_the_episode_start_stamp_is_the_running_rows_start_time() -> None:
    from arena_evaluation.processing.episode_topics import episode_start_stamp

    record = pl.DataFrame({"time_ns": [5 * S, 10 * S, 20 * S], "start_ns": [0, 100 * S, 100 * S], "outcome_state": [0, 1, 3]})
    assert episode_start_stamp(record) == 100 * S
    assert episode_start_stamp(record.drop("start_ns")) is None and episode_start_stamp(None) is None
    assert episode_start_stamp(record.with_columns(pl.lit(0).alias("start_ns"))) is None
