"""`evaluation process --measures`: only the selected calculators run, and extraction decodes only the tables they and the pipeline read."""

from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import AGENT_STATES, EPISODE_RECORD, ODOMETRY, PEDESTRIANS, SEC, TF_MESSAGE, _every_topic_bag, _header, _transform, _write_bag

from arena_evaluation.processing.mcap_reader import MCAPReader, channel_tables, recorded_tables, tf_transforms
from arena_evaluation.processing.metrics.registry import MetricRegistry, calculator_tables, select_calculators
from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.pipeline import PIPELINE_TABLES, ProcessingPipeline, resolve_tables
from arena_evaluation.storage.folder_manager import FolderManager
from arena_evaluation.storage.schemas import RobotParams

FLOW_TABLES = frozenset({"odom", "odom_controller", "tf_gt", "tf_static", "episode_record", "semantic_snapshot", "plan", "task_pose", "peds_physics", "agent_meta", "interactions"})


def _files(root: pathlib.Path) -> dict[str, pl.DataFrame]:
    return {str(p.relative_to(root)): pl.read_parquet(p) for p in sorted(root.rglob("*.parquet"))}


def test_channel_tables_name_every_table_a_full_extraction_fills(tmp_path: pathlib.Path):
    bag = _every_topic_bag(tmp_path / "every.mcap")
    MCAPReader(bag).read(tmp_path / "full")
    filled = {p.stem for p in (tmp_path / "full").rglob("*.parquet")}
    recorded = recorded_tables(bag)
    assert recorded is not None
    # every table read() fills comes from a channel channel_tables names it for (peds.parquet may be the engine copy)
    assert filled <= recorded | {"peds"}
    assert channel_tables("/env_0/jackal/unrecognized") == frozenset()
    assert channel_tables("tf") == {"tf", "tf_gt"}
    assert channel_tables("/env_0/jackal/diff_drive_controller/odom") == {"odom_controller"}


def test_a_selective_extraction_writes_the_selected_tables_as_a_full_one_does(tmp_path: pathlib.Path):
    bag = _every_topic_bag(tmp_path / "every.mcap")
    MCAPReader(bag).read(tmp_path / "full")
    tables = {"odom", "odom_controller", "episode_record", "peds_physics", "tf_gt", "tf_static"}
    MCAPReader(bag).read(tmp_path / "some", tables=tables)
    full, some = _files(tmp_path / "full"), _files(tmp_path / "some")
    for name, frame in some.items():
        stem = pathlib.Path(name).stem
        if stem in tables:
            assert frame.equals(full[name]), name
    assert {pathlib.Path(n).stem for n in some} - tables == {"tf", "peds"}
    # no arena_peds decoded: peds is the engine's agent_states copy, and /tf keeps only its env anchors
    assert some["env_0/peds.parquet"].equals(full["env_0/peds_physics.parquet"])
    assert some["tf.parquet"]["child_frame_id"].to_list() == ["env_0/map"]
    assert full["tf.parquet"].height == 3


def test_the_tf_walker_reads_what_the_generic_decoder_reads(tmp_path: pathlib.Path):
    from mcap.reader import NonSeekingReader
    from mcap_ros2.decoder import DecoderFactory

    # frame names of every length mod 8, so each string ends at every alignment before the next stamp and the doubles
    names = ["m", "ma", "map", "map_", "env_0/map", "env_10/jackal/base_link", "body_env_0_agent_12", "x" * 31]
    transforms = [{**_transform(a, b, 0.1 * i, -0.2 * i), "header": _header(i, 7 * i, a)} for i, (a, b) in enumerate(zip(names, reversed(names)))]
    bag = _write_bag(tmp_path / "tf.mcap", [("/tf", TF_MESSAGE, {"transforms": transforms}, SEC), ("/tf", TF_MESSAGE, {"transforms": []}, 2 * SEC)])
    with bag.open("rb") as f:
        for schema, channel, message in NonSeekingReader(f).iter_messages():
            generic = DecoderFactory().decoder_for(channel.message_encoding, schema)(message.data).transforms
            fast = tf_transforms(message.data)
            assert len(fast) == len(generic)
            for a, b in zip(fast, generic):
                assert (a.header.frame_id, a.child_frame_id, a.header.stamp.sec, a.header.stamp.nanosec) == (b.header.frame_id, b.child_frame_id, b.header.stamp.sec, b.header.stamp.nanosec)
                assert tuple(a.transform.translation) == (b.transform.translation.x, b.transform.translation.y, b.transform.translation.z)
                assert tuple(a.transform.rotation) == (b.transform.rotation.x, b.transform.rotation.y, b.transform.rotation.z, b.transform.rotation.w)
            assert [t.child_frame_id for t in tf_transforms(message.data, lambda frame, child: child.endswith(b"base_link"))] == [t.child_frame_id for t in generic if t.child_frame_id.endswith("base_link")]
            # a big-endian payload is left to the generic decoder
            assert tf_transforms(b"\x00\x00" + bytes(message.data[2:])) is None


def test_without_the_tf_table_only_robot_poses_and_env_anchors_are_kept_from_tf(tmp_path: pathlib.Path):
    peds = {"transforms": [_transform("map", f"body_env_0_agent_{i}", float(i), 0.0) for i in range(5)]}
    mixed = {"transforms": [*peds["transforms"], _transform("env_0/jackal/odom", "env_0/jackal/base_link", 1.0, 2.0), _transform("map", "env_0/map", 0.0, 0.0)]}
    odom = {"header": _header(1, 0, "odom"), "pose": {"pose": {"position": {"x": 1.0, "y": 2.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
    bag = _write_bag(tmp_path / "tf.mcap", [("/env_0/jackal/odom", ODOMETRY, odom, SEC), ("/tf", TF_MESSAGE, peds, SEC), ("/tf", TF_MESSAGE, mixed, 2 * SEC)])
    full = MCAPReader(bag).read(tmp_path / "full")["env_0_jackal"]
    some = MCAPReader(bag).read(tmp_path / "some", tables={"odom", "tf_gt"})["env_0_jackal"]
    assert some.tf_gt.collect().equals(full.tf_gt.collect())
    assert some.tf_gt.collect()["pos_x_gt"].to_list() == [1.0]
    assert full.tf.collect().height == 12
    assert some.tf.collect()["child_frame_id"].to_list() == ["env_0/map"]


def test_flow_selects_the_flow_calculator_and_names_select_with_their_dependencies():
    assert select_calculators(["flow"]) == {"flow_episode"}
    registry = MetricRegistry(RobotParams.load("jackal"), measures=["flow"])
    assert list(registry.calculators) == ["flow_episode"]
    classes = MetricRegistry(RobotParams.load("jackal")).calculators
    dependent = next(name for name, calc in classes.items() if calc.DEPENDS_ON)
    assert set(classes[dependent].DEPENDS_ON) <= select_calculators([dependent])
    with pytest.raises(ValueError, match="selects no calculator"):
        select_calculators(["nonsense"])


def test_the_flow_tables_are_the_tracks_the_pipeline_and_the_object_uses():
    assert calculator_tables(["flow_episode"]) == [("peds_physics", "peds"), "interactions"]
    recorded = frozenset({"odom", "tf", "tf_gt", "tf_static", "episode_record", "peds", "peds_physics", "agent_meta", "interactions", "scan"})
    assert resolve_tables([*PIPELINE_TABLES, *calculator_tables(["flow_episode"])], recorded) == FLOW_TABLES
    # without agent_states in the recording the tracks come from arena_peds; unknown recordings take both
    assert "peds" in resolve_tables([*PIPELINE_TABLES], recorded - {"peds_physics"})
    assert {"peds", "peds_physics"} <= resolve_tables([*PIPELINE_TABLES], None)


def test_a_partial_cache_serves_only_the_tables_it_holds(tmp_path: pathlib.Path):
    bag = _every_topic_bag(tmp_path / "every.mcap")
    tables = frozenset({"odom", "episode_record"})
    TopicParquetStore.write(MCAPReader(bag).read(tmp_path / "scratch", tables=tables), tmp_path / "topics", overwrite=True, tables=tables)
    assert TopicParquetStore.cached_tables(tmp_path / "topics") == tables
    assert TopicParquetStore.is_current(tmp_path / "topics", {"odom"})
    assert not TopicParquetStore.is_current(tmp_path / "topics", {"odom", "scan"})
    assert not TopicParquetStore.is_current(tmp_path / "topics")
    TopicParquetStore.write(MCAPReader(bag).read(tmp_path / "scratch2"), tmp_path / "topics", overwrite=True)
    assert TopicParquetStore.cached_tables(tmp_path / "topics") is None
    assert TopicParquetStore.is_current(tmp_path / "topics", {"odom", "scan"})


def _peds_episode(root: pathlib.Path) -> pathlib.Path:
    """A benchmark of one episode: a parked robot (tf ground truth), two walkers in agent_states and arena_peds, a 10 s window."""
    episode = root / "bench" / "episodes" / "episode_000"
    episode.mkdir(parents=True)
    (episode / "episode_000.yaml").write_text("benchmark_id: bench\nplanner: peds_only\nmap: no_such_world\nstage: no_such_world__s1\nepisode_id: 0\nrobot_model: [jackal]\npedsim_available: true\nrecording_started_at: '2026-11-09T15:05:30+00:00'\npython_version: 3.12.3\nros_distro: jazzy\n")
    records = [("/env_0/state/episode", EPISODE_RECORD, {"episode_id": 0, "outcome_state": 1, "modes": [], "params": [], "phases": ""}, 0)]
    for k in range(101):
        t = k * SEC // 10
        odom = {"header": _header(t // SEC, t % SEC, "env_0/jackal/odom"), "pose": {"pose": {"position": {"x": 0.0, "y": 0.0}, "orientation": {"w": 1.0}}}, "twist": {"twist": {}}}
        records.append(("/arena/env_0/arena_env_node/jackal/odom", ODOMETRY, odom, t))
        records.append(("tf", TF_MESSAGE, {"transforms": [_transform("env_0/jackal/odom", "env_0/jackal/base_link", 0.0, 0.0)]}, t))
        records.append(("tf", TF_MESSAGE, {"transforms": [_transform("map", f"body_env_0_agent_{i}", 0.1 * k, float(i)) for i in (1, 2)]}, t))
        agents = [{"agent_id": i, "kind": 0, "pose": {"x": 0.1 * k, "y": float(i), "theta": 0.0}, "velocity": {"x": 1.0}} for i in (1, 2)]
        records.append(("/arena/env_0/arena_env_node/agent_states", AGENT_STATES, {"header": _header(frame_id="map"), "agents": agents}, t))
        peds = [{"name": f"flow_{i}", "id": i, "pose": {"position": {"x": 0.1 * k, "y": float(i)}, "orientation": {"w": 1.0}}, "twist": {"linear": {"x": 1.0}}} for i in (1, 2)]
        records.append(("/arena/env_0/arena_peds", PEDESTRIANS, {"header": _header(frame_id="map"), "pedestrians": peds}, t))
    records.append(("/env_0/state/episode", EPISODE_RECORD, {"episode_id": 0, "outcome_state": 3, "outcome_info": "timeout", "modes": [], "params": [], "phases": ""}, 10 * SEC))
    _write_bag(episode / "episode_000.mcap", sorted(records, key=lambda r: r[3]))
    return episode


def test_process_episode_with_the_flow_measures_gives_the_flow_columns_and_tracks_of_a_full_run(tmp_path: pathlib.Path):
    full_dir, flow_dir = _peds_episode(tmp_path / "full"), _peds_episode(tmp_path / "flow")
    full = ProcessingPipeline(FolderManager(data_root=full_dir.parents[1].parent), workers=1)
    flow = ProcessingPipeline(FolderManager(data_root=flow_dir.parents[1].parent), workers=1, measures=["flow"])
    (full_row,) = full.process_episode(full.folder_manager.describe_episode(full_dir))
    (flow_row,) = flow.process_episode(flow.folder_manager.describe_episode(flow_dir))
    assert flow_row["status"] == full_row["status"] == "evaluated"
    assert flow_row["flow_peds"] == full_row["flow_peds"] == 2
    assert flow_row["flow_object_uses"] == full_row["flow_object_uses"]
    assert set(flow_row) < set(full_row)
    tracks = "topics/.derived/ped_tracks.v1/env_0.parquet"
    assert pl.read_parquet(flow_dir / tracks).equals(pl.read_parquet(full_dir / tracks))
    assert TopicParquetStore.cached_tables(flow_dir / "topics") == FLOW_TABLES
    assert not (flow_dir / "topics" / "env_0_jackal" / "scan.parquet").exists()
