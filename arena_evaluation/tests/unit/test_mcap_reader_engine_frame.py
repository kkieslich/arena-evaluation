from __future__ import annotations

import pathlib

import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.mcap_reader import ENGINE_PEDS_MARKER, MCAPReader


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


def _bag(tmp_path: pathlib.Path) -> pathlib.Path:
    """env_1 records only the engine's agent_states, env_2 also arena_peds; both envs have a robot and a map anchor."""
    rosbag2_py = pytest.importorskip("rosbag2_py")
    agent_msgs = pytest.importorskip("arena_humansim_msgs.msg")
    people_msgs = pytest.importorskip("arena_people_msgs.msg")
    nav_msgs = pytest.importorskip("nav_msgs.msg")
    tf2_msgs = pytest.importorskip("tf2_msgs.msg")
    geometry_msgs = pytest.importorskip("geometry_msgs.msg")
    serialization = pytest.importorskip("rclpy.serialization")

    tf = tf2_msgs.TFMessage()
    for env, (ox, oy) in {"env_1": (5.25, 5.5), "env_2": (3.0, 1.0)}.items():
        anchor = geometry_msgs.TransformStamped()
        anchor.header.frame_id = "map"
        anchor.child_frame_id = f"{env}/map"
        anchor.transform.translation.x, anchor.transform.translation.y = ox, oy
        anchor.transform.rotation.w = 1.0
        tf.transforms.append(anchor)
    engine = agent_msgs.AgentFrame()
    engine.header.frame_id = "map"
    engine.agent_id, engine.kind, engine.x, engine.y, engine.theta, engine.vx, engine.vy = [4], [agent_msgs.AgentState.KIND_HUMAN], [2.0], [3.0], [0.0], [0.0], [0.0]
    arena_peds = people_msgs.Pedestrians()
    arena_peds.header.frame_id = "map"
    ped = people_msgs.Pedestrian(name="env_2/flow_1")
    ped.pose.position.x, ped.pose.position.y = 7.0, 4.0
    ped.pose.orientation.w = 1.0
    arena_peds.pedestrians.append(ped)
    odom = nav_msgs.Odometry()
    odom.pose.pose.orientation.w = 1.0

    messages = [
        ("/tf_static", "tf2_msgs/msg/TFMessage", tf),
        ("/arena/env_1/arena_env_node/jackal/odom", "nav_msgs/msg/Odometry", odom),
        ("/arena/env_2/arena_env_node/jackal/odom", "nav_msgs/msg/Odometry", odom),
        ("/arena/env_1/arena_env_node/agent_states", "arena_humansim_msgs/msg/AgentFrame", engine),
        ("/arena/env_2/arena_env_node/agent_states", "arena_humansim_msgs/msg/AgentFrame", engine),
        ("/arena/env_2/arena_env_node/arena_peds", "arena_people_msgs/msg/Pedestrians", arena_peds),
    ]
    bag = tmp_path / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))
    for i, (topic, type_name, _) in enumerate(messages):
        writer.create_topic(rosbag2_py.TopicMetadata(i, topic, type_name, "cdr"))
    for topic, _, msg in messages:
        writer.write(topic, serialization.serialize_message(msg), 1_000_000_000)
    writer.close()
    return bag


def test_read_marks_peds_it_copies_from_the_engine_and_keeps_them_in_the_world_frame(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "out"
    bundles = MCAPReader(_bag(tmp_path)).read(out)

    assert (out / "env_1" / "peds.parquet").exists() and (out / "env_1" / ENGINE_PEDS_MARKER).exists()
    assert bundles["env_1_jackal"].peds.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]
    assert bundles["env_1_jackal"].peds_physics.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]
    # env_2 recorded arena_peds: no marker, and its positions (engine pose + env offset) do take the offset
    assert not (out / "env_2" / ENGINE_PEDS_MARKER).exists()
    assert bundles["env_2_jackal"].peds.collect()["peds_positions"].to_list() == [[4.0, 3.0, 0.0]]
    assert bundles["env_2_jackal"].peds_physics.collect()["peds_positions"].to_list() == [[2.0, 3.0, 0.0]]


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
