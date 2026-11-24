"""An episode's lifecycle records: its own rows, its window and the sim time it started at."""

from __future__ import annotations

import polars as pl

RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING
RECORD_SUCCESS = 2  # arena_runtime_msgs/msg/EpisodeRecord.SUCCESS


def _collected(record: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if record is None:
        return None
    return record.collect() if isinstance(record, pl.LazyFrame) else record


def episode_rows(record: pl.DataFrame) -> pl.DataFrame:
    """The episode's own records in publish order, up to its first terminal record after it ran, without the next episode's queued records that share its last timestamp."""
    df = record.sort("t_ns", maintain_order=True)
    states = df["outcome_state"].to_numpy()
    started = next((i for i, state in enumerate(states) if state == RECORD_RUNNING), None)
    if started is None:
        return df
    if df["episode_id"][started] is not None:
        df = df.filter(pl.col("episode_id") == df["episode_id"][started])
        states = df["outcome_state"].to_numpy()
        started = next(i for i, state in enumerate(states) if state == RECORD_RUNNING)
    end = next(
        (i for i in range(started, len(states)) if states[i] is not None and states[i] >= RECORD_SUCCESS),
        None,
    )
    return df if end is None else df.head(end + 1)


def episode_window(
    record: pl.DataFrame | pl.LazyFrame | None,
) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of the episode record, a bound is None when its row is missing."""
    df = _collected(record)
    if df is None or len(df) == 0:
        return None, None
    df = episode_rows(df)
    states = df["outcome_state"].to_numpy()
    stamps = df["t_ns"].to_numpy()
    start = next(
        (int(t) for t, s in zip(stamps, states, strict=True) if s == RECORD_RUNNING),
        None,
    )
    end = None if states[-1] == RECORD_RUNNING else int(stamps[-1])
    return start, end


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


__all__ = ["RECORD_RUNNING", "RECORD_SUCCESS", "episode_rows", "episode_start_stamp", "episode_window"]
