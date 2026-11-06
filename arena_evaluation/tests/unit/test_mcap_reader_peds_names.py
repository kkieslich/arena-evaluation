import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from arena_evaluation.processing.mcap_reader import MCAPReader
from test_mcap_reader_topics import HEADER, PEDESTRIAN, PEDESTRIANS, POINT, POSE, QUAT, TIME, TWIST, VEC3, _header, _msgdef, _write_bag

TYPED_PEDESTRIANS = ("arena_people_msgs/msg/Pedestrians", _msgdef("std_msgs/Header header\nPedestrian[] pedestrians", HEADER, TIME, PEDESTRIAN + "\nstring agent_type", POSE, POINT, QUAT, TWIST, VEC3))


def _ped(name: str, ped_id: int, x: float, y: float) -> dict:
    return {"name": name, "id": ped_id, "pose": {"position": {"x": x, "y": y, "z": 0.0}, "orientation": {"w": 1.0}}, "twist": {"linear": {"x": 0.1}}}


def _peds(*peds: dict) -> dict:
    return {"header": _header(frame_id="map"), "pedestrians": list(peds)}


def test_reader_types_ped_columns_when_the_first_sample_is_empty(tmp_path: pathlib.Path):
    bag = _write_bag(
        tmp_path / "episode.mcap",
        [
            ("/env_0/arena_peds", PEDESTRIANS, _peds(), 1),
            ("/env_0/arena_peds", PEDESTRIANS, _peds(_ped("env_0/Pedestrian_1", 1, 1.0, 2.0), _ped("env_0/Pedestrian_2", 2, 3.0, 4.0)), 2),
        ],
    )

    MCAPReader(bag).read(tmp_path / "raw")

    peds = pl.read_parquet(tmp_path / "raw" / "env_0" / "peds.parquet")
    assert peds.schema["peds_names"] == pl.List(pl.Utf8)
    assert {peds.schema[name] for name in ("peds_positions", "peds_headings", "peds_twists")} == {pl.List(pl.Float64)}
    assert peds["peds_names"].to_list() == [[], ["env_0/Pedestrian_1", "env_0/Pedestrian_2"]]
    assert peds["peds_positions"].to_list()[1][:2] == [1.0, 2.0]


def test_reader_keeps_each_pedestrians_agent_type(tmp_path: pathlib.Path):
    wheelchair = {**_ped("env_0/Pedestrian_1", 1, 1.0, 2.0), "agent_type": "wheelchair_manual"}
    unset = {**_ped("env_0/Pedestrian_2", 2, 3.0, 4.0), "agent_type": ""}
    bag = _write_bag(tmp_path / "episode.mcap", [("/env_0/arena_peds", TYPED_PEDESTRIANS, _peds(wheelchair, unset), 1)])

    MCAPReader(bag).read(tmp_path / "raw")

    peds = pl.read_parquet(tmp_path / "raw" / "env_0" / "peds.parquet")
    assert peds.schema["peds_agent_types"] == pl.List(pl.Utf8)
    assert peds["peds_agent_types"].to_list() == [["wheelchair_manual", ""]]


def test_reader_records_no_agent_types_when_the_message_has_none(tmp_path: pathlib.Path):
    bag = _write_bag(tmp_path / "episode.mcap", [("/env_0/arena_peds", PEDESTRIANS, _peds(_ped("env_0/Pedestrian_1", 1, 1.0, 2.0)), 1)])

    MCAPReader(bag).read(tmp_path / "raw")

    assert pl.read_parquet(tmp_path / "raw" / "env_0" / "peds.parquet")["peds_agent_types"].to_list() == [[]]
