"""An episode's lifecycle as the cross-run measures read it: its window and the sim time it started at."""

from __future__ import annotations

import polars as pl

RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING


def _collected(record: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if record is None:
        return None
    return record.collect() if isinstance(record, pl.LazyFrame) else record


def episode_start_stamp(record: pl.DataFrame | pl.LazyFrame | None) -> int | None:
    """The sim time the episode started at (the first RUNNING row's `start_t_ns`, the record's start_time), None when unrecorded.

    The crowd engine stamps its frames on the same clock, so a frame's time in the episode is its stamp minus this,
    free of the recorder's delivery delay."""
    df = _collected(record)
    if df is None or len(df) == 0 or "start_t_ns" not in df.columns:
        return None
    rows = df.sort("t_ns", maintain_order=True)
    if "outcome_state" in rows.columns:
        rows = rows.filter(pl.col("outcome_state") == RECORD_RUNNING)
    starts = [int(v) for v in rows["start_t_ns"].to_list() if v]
    return starts[0] if starts else None


def episode_window(record: pl.DataFrame | pl.LazyFrame | None) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of an `episode_lifecycle` table, as the pipeline windows the episode; a bound is
    None when its row is missing."""
    from arena_evaluation.processing.pipeline import _episode_window

    df = _collected(record)
    if df is None or len(df) == 0 or "t_ns" not in df.columns:
        return None, None
    return _episode_window(df)


__all__ = ["RECORD_RUNNING", "episode_start_stamp", "episode_window"]
