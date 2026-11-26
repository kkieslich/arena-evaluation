from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("rosbag2_py")
pytest.importorskip("rclpy")

import yaml
from arena_runtime_msgs.msg import EpisodeRecord
from rclpy.parameter import Parameter

from arena_evaluation.ingestion.recorder import DataRecorderNode


@pytest.fixture
def recorder_node(tmp_path: pathlib.Path):
    node = DataRecorderNode(parameter_overrides=[Parameter("record_data_dir", value=str(tmp_path / "episodes"))])
    yield node
    node.destroy_node()


def _record(episode_id: int, outcome_state: int, *integrity: str) -> EpisodeRecord:
    msg = EpisodeRecord()
    msg.episode_id = episode_id
    msg.outcome_state = outcome_state
    msg.integrity = list(integrity)
    return msg


def _yaml_integrity(node: DataRecorderNode) -> list[str]:
    return yaml.safe_load(node.current_metadata_path.read_text())["integrity"]


def test_recorder_keeps_integrity_reasons_arriving_after_stop(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.episode_record_callback(_record(4, EpisodeRecord.RUNNING))
    path = recorder_node.current_metadata_path
    recorder_node._stop_episode(4, outcome_state=EpisodeRecord.FAILED, outcome_info="timeout")

    recorder_node.episode_record_callback(_record(4, EpisodeRecord.FAILED, "env_0/jackal moved by hand"))

    assert yaml.safe_load(path.read_text())["integrity"] == ["env_0/jackal moved by hand"]


def test_queued_record_of_the_next_episode_leaves_integrity_of_the_current_one(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.episode_record_callback(
        _record(
            5,
            EpisodeRecord.RUNNING,
            "jackal: elevator teleport 'a' -> 'b' failed: TimeoutError()",
        )
    )
    recorder_node.episode_record_callback(_record(6, EpisodeRecord.QUEUED))

    assert recorder_node.current_sim_episode_id == 5
    assert _yaml_integrity(recorder_node) == ["jackal: elevator teleport 'a' -> 'b' failed: TimeoutError()"]


def _health_record(episode_id: int, outcome_state: int, **health: str) -> EpisodeRecord:
    from diagnostic_msgs.msg import KeyValue

    msg = _record(episode_id, outcome_state)
    msg.health = [KeyValue(key=key, value=value) for key, value in health.items()]
    return msg


def test_the_health_of_the_running_and_the_terminal_record_lands_in_the_metadata_as_seconds(
    recorder_node: DataRecorderNode,
) -> None:
    recorder_node.episode_record_callback(_health_record(7, EpisodeRecord.RUNNING, prev_teardown_loop_lag_peak_s="9.250", setup_loop_lag_peak_s="0.120"))
    recorder_node.episode_record_callback(_health_record(7, EpisodeRecord.SUCCESS, prev_teardown_loop_lag_peak_s="9.250", setup_loop_lag_peak_s="0.120", episode_loop_lag_peak_s="14.000", broken="n/a"))

    assert yaml.safe_load(recorder_node.current_metadata_path.read_text())["health"] == {
        "prev_teardown_loop_lag_peak_s": 9.25,
        "setup_loop_lag_peak_s": 0.12,
        "episode_loop_lag_peak_s": 14.0,
    }
