import concurrent.futures
import functools
import pathlib
import shutil

import polars as pl
import pytest
import yaml

from arena_evaluation.processing.metrics.social import paired_peds
from arena_evaluation.processing import pipeline
from arena_evaluation.processing.pipeline import ProcessingPipeline
from arena_evaluation.processing.ped_track_store import TRACK_SCHEMA
from arena_evaluation.storage.folder_manager import FolderManager

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "sample_benchmark"


def test_process_benchmark_writes_the_metrics_of_a_recorded_episode(tmp_path: pathlib.Path):
    shutil.copytree(FIXTURE, tmp_path / "sample_benchmark")
    fm = FolderManager(data_root=tmp_path)

    written = ProcessingPipeline(fm, workers=1).process_benchmark("sample_benchmark")

    assert written == fm.combined_metrics_path("sample_benchmark")
    metrics = pl.read_parquet(written)
    assert metrics.height == 1
    row = metrics.row(0, named=True)
    assert row["planner"] == "dwb"
    assert row["map"] == "barn_0"
    assert row["path_length"] > 0.0


def _walker(seconds: float) -> pl.DataFrame:
    """Tracks of a clock pedestrian spanning the episode and one walking for `seconds` inside it."""
    rows = [(float(t), "clock") for t in range(11)] + [(t, "door#1") for t in (1.0, 1.0 + seconds)]
    return pl.DataFrame([{**dict.fromkeys(TRACK_SCHEMA), "time_ns": int(t * 1e9), "t": t, "key": key, "agent_type": "elder", "x": 0.0, "y": 0.0} for t, key in rows], schema=TRACK_SCHEMA)


def test_process_benchmark_pairs_each_run_with_its_robot_free_reference(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    bench = tmp_path / "sample_benchmark"
    shutil.copytree(FIXTURE, bench)
    reference = bench / "episodes" / "episode_001"
    shutil.copytree(bench / "episodes" / "episode_000", reference)
    (reference / "episode_000.mcap").rename(reference / "episode_001.mcap")
    meta = yaml.safe_load((reference / "episode_000.yaml").read_text())
    (reference / "episode_000.yaml").unlink()
    (reference / "episode_001.yaml").write_text(yaml.safe_dump({**meta, "episode_id": 1, "is_reference": True, "reference_type": "unhindered_peds"}))
    tracks = {("episode_000", "env_0"): _walker(7.0), ("episode_001", "env_0"): _walker(4.0)}
    monkeypatch.setattr(pipeline, "add_paired_ped_measures", functools.partial(paired_peds.add_paired_ped_measures, tracks=lambda episode_dir, env: tracks[episode_dir.name, env]))

    written = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=1).process_benchmark("sample_benchmark")

    run = pl.read_parquet(written).filter(~pl.col("is_reference")).row(0, named=True)
    assert run["ped_paired_delay_s_per_ped"] == [{"ped": "door#1", "value": 3.0}]
    assert run["ped_paired_relative_delay_by_group"] == [{"group": "elder", "value": 0.75, "peds": 1}]


def test_process_benchmark_spawns_its_workers_after_the_parent_ran_polars(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A forked worker copies the parent's polars thread pool with its locks held by threads it does not have, and hangs."""
    shutil.copytree(FIXTURE, tmp_path / "sample_benchmark")
    pl.DataFrame({"a": list(range(10_000))}).group_by(pl.col("a") % 7).len()
    contexts = []
    pool = concurrent.futures.ProcessPoolExecutor

    def recording(*args, **kwargs):
        contexts.append(kwargs.get("mp_context"))
        return pool(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", recording)

    written = ProcessingPipeline(FolderManager(data_root=tmp_path), workers=2).process_benchmark("sample_benchmark")

    assert pl.read_parquet(written).height == 1
    assert [c.get_start_method() for c in contexts] == ["spawn", "spawn"]


def test_spawned_workers_run_polars_on_one_thread():
    with concurrent.futures.ProcessPoolExecutor(1, initializer=pipeline._worker_init, mp_context=pipeline._spawn()) as pool:
        assert pool.submit(pl.thread_pool_size).result(timeout=120) == 1
