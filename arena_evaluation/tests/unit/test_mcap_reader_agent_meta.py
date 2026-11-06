from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")
rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")

import pyarrow.parquet as pq
from arena_humansim_msgs.msg import AgentMeta
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import ENGINE_PEDS_MARKER, MCAPReader

_TOPIC = "/arena/env_2/arena_env_node/agent_meta"


def _bag(tmp_path: pathlib.Path, msg: AgentMeta) -> pathlib.Path:
    bag = tmp_path / "meta"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    writer.create_topic(rosbag2_py.TopicMetadata(0, _TOPIC, "arena_humansim_msgs/msg/AgentMeta", "cdr"))
    writer.write(_TOPIC, serialize_message(msg), 3_000_000_000)
    writer.close()
    return bag


def test_agent_meta_rows_name_each_agent_and_its_releasing_source(tmp_path: pathlib.Path) -> None:
    msg = AgentMeta(agent_id=[4, 11], name=["env_2/flow_1", ""], handedness=["", "l"], agent_type=["adult", "elder"], source_name=["", "door_a"], source_seq=[0, 7])
    msg.header.stamp.sec = 3
    out = tmp_path / "out"
    MCAPReader(_bag(tmp_path, msg)).read(out)

    rows = pq.read_table(out / "env_2" / "agent_meta.parquet").to_pylist()
    assert rows == [
        {"time_ns": 3_000_000_000, "stamp_ns": 3_000_000_000, "agent_id": 4, "name": "env_2/flow_1", "agent_type": "adult", "source_name": "", "source_seq": 0},
        {"time_ns": 3_000_000_000, "stamp_ns": 3_000_000_000, "agent_id": 11, "name": "", "agent_type": "elder", "source_name": "door_a", "source_seq": 7},
    ]


def test_agent_meta_without_source_fields_reads_as_spawned_directly(tmp_path: pathlib.Path) -> None:
    # an engine that sends no release index for an agent: the columns stay aligned, the agent counts as unreleased
    msg = AgentMeta(agent_id=[4], name=["env_2/flow_1"], handedness=[""], agent_type=[""], source_name=[], source_seq=[])
    out = tmp_path / "out"
    MCAPReader(_bag(tmp_path, msg)).read(out)

    (row,) = pq.read_table(out / "env_2" / "agent_meta.parquet").to_pylist()
    assert (row["source_name"], row["source_seq"]) == ("", 0)


def _tf_static(topics: pathlib.Path, ox: float, oy: float) -> None:
    pl.DataFrame(
        {
            "time_ns": [0],
            "frame_id": ["map"],
            "child_frame_id": ["env_1/map"],
            "trans_x": [ox],
            "trans_y": [oy],
            "trans_z": [0.0],
            "rot_x": [0.0],
            "rot_y": [0.0],
            "rot_z": [0.0],
            "rot_w": [1.0],
        }
    ).write_parquet(topics / "tf_static.parquet")


def _robot(topics: pathlib.Path) -> None:
    robot_dir = topics / "env_1_jackal"
    robot_dir.mkdir(parents=True)
    pl.DataFrame({"time_ns": [0], "pos_x": [0.0], "pos_y": [0.0], "yaw": [0.0]}).write_parquet(robot_dir / "odom.parquet")


def _peds(path: pathlib.Path, x: float, y: float) -> None:
    pl.DataFrame({"time_ns": [10], "peds_frame_id": ["map"], "peds_ids": [[4]], "peds_positions": [[x, y, 0.0]]}).write_parquet(path)


def test_engine_positions_are_world_frame_and_take_no_env_offset(tmp_path: pathlib.Path) -> None:
    # the adapter publishes arena_peds at engine pose + env offset (5.25, 5.5): both tables must land on one world point
    topics = tmp_path / "topics"
    env_dir = topics / "env_1"
    env_dir.mkdir(parents=True)
    _robot(topics)
    _tf_static(topics, 5.25, 5.5)
    _peds(env_dir / "peds.parquet", 7.25, 8.5)
    _peds(env_dir / "peds_physics.parquet", 2.0, 3.0)

    bundle = MCAPReader(topics / "run.mcap").load_bundles(topics, topics)["env_1_jackal"]
    assert bundle.peds.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]
    assert bundle.peds_physics.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]


def test_peds_copied_from_the_engine_take_no_env_offset(tmp_path: pathlib.Path) -> None:
    # no arena_peds recorded: peds.parquet is the engine's table and is in the world frame like it
    topics = tmp_path / "topics"
    env_dir = topics / "env_1"
    env_dir.mkdir(parents=True)
    _robot(topics)
    _tf_static(topics, 5.25, 5.5)
    _peds(env_dir / "peds_physics.parquet", 2.0, 3.0)
    _peds(env_dir / "peds.parquet", 2.0, 3.0)
    (env_dir / ENGINE_PEDS_MARKER).touch()

    bundle = MCAPReader(topics / "run.mcap").load_bundles(topics, topics)["env_1_jackal"]
    assert bundle.peds.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]
