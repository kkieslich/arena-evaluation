"""The direct nav_msgs/Path reader: the values of the generic decoder, and the same plan table."""

from __future__ import annotations

import math
import pathlib

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("mcap_ros2.writer")

from test_mcap_reader_topics import PATH, SEC, _header, _write_bag

from arena_evaluation.processing.mcap_reader import MCAPReader
from arena_evaluation.processing.telemetry import cdr


def _plan(frames: list[str], header_frame: str = "map", n: int | None = None) -> dict:
    """A nav_msgs/Path whose i-th pose has frame id frames[i % len(frames)] and distinct values."""
    count = len(frames) if n is None else n
    poses = []
    for i in range(count):
        yaw = 0.01 * i - 1.0
        poses.append({"header": _header(i, 3 * i, frames[i % len(frames)]), "pose": {"position": {"x": 0.5 * i + 0.1, "y": -0.25 * i, "z": 0.001 * i}, "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2), "w": math.cos(yaw / 2)}}})
    return {"header": _header(1, 2, header_frame), "poses": poses}


def test_the_plan_walker_reads_what_the_generic_decoder_reads(tmp_path: pathlib.Path) -> None:
    from mcap.reader import NonSeekingReader
    from mcap_ros2.decoder import DecoderFactory

    plans = [
        _plan(["map"], n=4000),  # one frame id: one strided gather
        _plan(["map"], header_frame="", n=50),  # the poses start at the other alignment
        _plan(["m", "ma", "map", "map_", "env_0/map", "x" * 31]),  # frame ids of every length mod 8
        _plan(["map"] * 5 + ["odom"] + ["map"] * 7),  # one odd frame id in a run of equal ones
        _plan(["map"], n=1),
        _plan(["map"], n=0),
    ]
    bag = _write_bag(tmp_path / "plan.mcap", [("/env_0/jackal/plan", PATH, plan, (i + 1) * SEC) for i, plan in enumerate(plans)])
    seen = 0
    with bag.open("rb") as f:
        for schema, channel, message in NonSeekingReader(f).iter_messages():
            generic = DecoderFactory().decoder_for(channel.message_encoding, schema)(message.data).poses
            fast = cdr.path_poses(message.data)
            assert fast.shape == (len(generic), 7)
            assert fast.tolist() == [[p.pose.position.x, p.pose.position.y, p.pose.position.z, p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w] for p in generic]
            # a big-endian payload is left to the generic decoder
            assert cdr.path_poses(b"\x00\x00" + bytes(message.data[2:])) is None
            seen += 1
    assert seen == len(plans)


def test_the_plan_table_is_the_same_with_the_plan_walker_and_the_generic_decoder(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plans = [_plan(["map"], n=300), _plan(["m", "map", "env_0/map"]), _plan(["map"], n=0)]
    bag = _write_bag(tmp_path / "plan.mcap", [("/env_0/jackal/plan", PATH, plan, (i + 1) * SEC) for i, plan in enumerate(plans)])
    MCAPReader(bag).read(tmp_path / "fast")
    fast = pl.read_parquet(tmp_path / "fast" / "robot_plans.parquet")
    monkeypatch.setattr(cdr, "path_poses", lambda data: None)
    MCAPReader(bag).read(tmp_path / "generic")
    generic = pl.read_parquet(tmp_path / "generic" / "robot_plans.parquet")
    assert fast.height == 303 and fast.equals(generic)
    assert fast.filter(pl.col("plan_id") == 0)["waypoint_idx"].to_list() == list(range(300))


def test_a_batch_of_rows_is_written_as_the_same_rows_one_by_one(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.processing.telemetry.writer import TableSink

    rows = [{"t_ns": t, "env_id": "env_0", "robot_id": "env_0_jackal", "plan_id": p, "waypoint_idx": w, "x": float(w), "y": -float(w), "yaw": 0.5} for t, p in ((5, 0), (3, 1)) for w in range(4)]
    # a repeated key replaces the earlier row, in a batch as one by one
    rows.append({**rows[1], "x": 99.0})
    one, batch = TableSink(tmp_path / "one", flush_rows=3), TableSink(tmp_path / "batch", flush_rows=3)
    for row in rows:
        one.append("robot_plans", row)
    for chunk in (rows[:4], rows[4:8], rows[8:]):
        batch.extend("robot_plans", {name: [row[name] for row in chunk] for name in chunk[0] if name != "yaw"} | {"yaw": [row["yaw"] for row in chunk]})
    one.close()
    batch.close()
    assert pl.read_parquet(tmp_path / "batch" / "robot_plans.parquet").equals(pl.read_parquet(tmp_path / "one" / "robot_plans.parquet"))
    with pytest.raises(KeyError, match="no columns"):
        TableSink(tmp_path / "bad").extend("robot_plans", {"t_ns": [1], "nope": [1]})
