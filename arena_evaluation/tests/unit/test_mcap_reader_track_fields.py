"""What the track store and the INFRA measures read of a recording: the pedestrian frames' stamps, the agent meta, the fleet task log."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

pl = pytest.importorskip("polars")
rosbag2_py = pytest.importorskip("rosbag2_py")
pytest.importorskip("arena_humansim_msgs.msg")

from arena_humansim_msgs.msg import AgentFrame, AgentMeta
from rclpy.serialization import serialize_message

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry.context import FleetLogRecorder
from arena_evaluation.processing.telemetry.messages import Envelope
from arena_evaluation.processing.telemetry.writer import TableSink

_META = "/arena/env_2/arena_env_node/agent_meta"
_STATES = "/arena/env_2/arena_env_node/agent_states"
_META_FIELDS = AgentMeta.get_fields_and_field_types()


def _bag(tmp_path: pathlib.Path, messages: list[tuple[str, str, object, int]]) -> pathlib.Path:
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    for i, topic in enumerate(dict.fromkeys(topic for topic, _, _, _ in messages)):
        msg_type = next(t for name, t, _, _ in messages if name == topic)
        writer.create_topic(rosbag2_py.TopicMetadata(i, topic, msg_type, "cdr"))
    for topic, _, msg, t_ns in messages:
        writer.write(topic, serialize_message(msg), t_ns)
    writer.close()
    return bag


def _meta(**fields: list) -> AgentMeta:
    known = {name: value for name, value in fields.items() if name in _META_FIELDS}
    # the crowd engine sends its robot settings for every agent
    n = len(fields["agent_id"])
    for name, kind in _META_FIELDS.items():
        if name.startswith("robot_"):
            known[name] = [False] * n if "boolean" in kind else [1.0] * n
    msg = AgentMeta(**known)
    msg.header.stamp.sec = 3
    return msg


@pytest.mark.skipif("source_name" not in _META_FIELDS, reason="arena_humansim_msgs built without AgentMeta sources")
def test_agent_meta_rows_name_each_agent_and_its_releasing_source(tmp_path: pathlib.Path) -> None:
    msg = _meta(agent_id=[4, 11], name=["env_2/flow_1", ""], handedness=["", "l"], agent_type=["adult", "elder"], source_name=["", "door_a"], source_seq=[0, 7])
    out = tmp_path / "out"
    MCAPReader(_bag(tmp_path, [(_META, "arena_humansim_msgs/msg/AgentMeta", msg, 3_000_000_000)])).read(out)

    rows = pl.read_parquet(out / "pedestrian_agent_meta.parquet").select("t_ns", "env_id", "stamp_ns", "agent_id", "name", "agent_type", "source_name", "source_seq").to_dicts()
    assert rows == [
        {"t_ns": 3_000_000_000, "env_id": "env_2", "stamp_ns": 3_000_000_000, "agent_id": 4, "name": "env_2/flow_1", "agent_type": "adult", "source_name": "", "source_seq": 0},
        {"t_ns": 3_000_000_000, "env_id": "env_2", "stamp_ns": 3_000_000_000, "agent_id": 11, "name": "", "agent_type": "elder", "source_name": "door_a", "source_seq": 7},
    ]


@pytest.mark.skipif("source_name" not in _META_FIELDS, reason="arena_humansim_msgs built without AgentMeta sources")
def test_agent_meta_without_one_source_per_agent_reads_as_spawned_directly(tmp_path: pathlib.Path) -> None:
    # an engine that sends no release index for an agent: the columns stay aligned, the agent counts as unreleased
    msg = _meta(agent_id=[4], name=["env_2/flow_1"], handedness=[""], agent_type=[""], source_name=[], source_seq=[])
    out = tmp_path / "out"
    MCAPReader(_bag(tmp_path, [(_META, "arena_humansim_msgs/msg/AgentMeta", msg, 3_000_000_000)])).read(out)

    (row,) = pl.read_parquet(out / "pedestrian_agent_meta.parquet").to_dicts()
    assert (row["agent_id"], row["name"], row["source_name"], row["source_seq"]) == (4, "env_2/flow_1", "", 0)


def test_engine_frames_carry_their_header_stamp_beside_the_log_time(tmp_path: pathlib.Path) -> None:
    fields = AgentFrame.get_fields_and_field_types()
    two = {"agent_id": [4, 11], "x": [1.0, 2.0], "y": [0.0, 0.0], "theta": [0.0, 0.0], "vx": [0.0, 0.0], "vy": [0.0, 0.0], "desired_velocity": [1.2, 0.8], "kind": [0, 0], "animation_state": [0, 0], "radius": [0.25, 0.325]}
    frames = []
    # the engine sends every per-agent list for every agent
    for name, kind in fields.items():
        if name not in two and kind.startswith("sequence"):
            two[name] = [""] * 2 if "string" in kind else [0] * 2
    for log_ns, stamp_s in ((3_400_000_000, 3), (3_430_000_000, 3)):
        msg = AgentFrame(**{k: v for k, v in two.items() if k in fields})
        msg.header.stamp.sec = stamp_s
        msg.header.stamp.nanosec = 100_000_000 if log_ns == 3_430_000_000 else 0
        frames.append((_STATES, "arena_humansim_msgs/msg/AgentFrame", msg, log_ns))
    out = tmp_path / "out"
    MCAPReader(_bag(tmp_path, frames)).read(out)

    rows = pl.read_parquet(out / "pedestrian_agent_states.parquet").sort("t_ns", "ped_id")
    assert rows.select("t_ns", "stamp_ns", "ped_id").rows() == [
        (3_400_000_000, 3_000_000_000, "Pedestrian_11"),
        (3_400_000_000, 3_000_000_000, "Pedestrian_4"),
        (3_430_000_000, 3_100_000_000, "Pedestrian_11"),
        (3_430_000_000, 3_100_000_000, "Pedestrian_4"),
    ]
    if "radius" in fields:
        assert rows.filter(pl.col("ped_id") == "Pedestrian_11")["radius_m"].to_list() == [0.325, 0.325]


def _entity(entity: str, kind: str, fields: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(entity=entity, kind=kind, discrete_names=list(fields), discrete_values=list(fields.values()), continuous_names=["x"], continuous_values=[1.0], predicate_names=[], predicate_values=[], members=[])


def test_the_fleet_log_keeps_two_messages_of_one_stamp_apart(tmp_path: pathlib.Path) -> None:
    # the latched copy of the previous episode's log and this episode's reset can carry one stamp
    out = tmp_path / "out"
    sink = TableSink(out)
    recorder = FleetLogRecorder(sink)
    latched = SimpleNamespace(env_id=0, world="w", entities=[_entity("env_0/fleet", "fleet", {"t0": "1.0", "job.0.outcome": "failed"}), _entity("env_0/door_0", "door", {"state": "open"})])
    reset = SimpleNamespace(env_id=0, world="w", entities=[_entity("env_0/fleet", "fleet", {"t0": "2.0"})])
    for body in (latched, reset):
        recorder.record(Envelope(t_ns=2_000_000_000, ref=None, schema=None, body=body))
    sink.close()

    rows = pl.read_parquet(out / "episode_fleet_log.parquet")
    assert rows.select("t_ns", "env_id", "msg_index", "entity_name", "field_name", "value_str").rows() == [
        (2_000_000_000, "env_0", 0, "env_0/fleet", "t0", "1.0"),
        (2_000_000_000, "env_0", 0, "env_0/fleet", "job.0.outcome", "failed"),
        (2_000_000_000, "env_0", 1, "env_0/fleet", "t0", "2.0"),
    ]
