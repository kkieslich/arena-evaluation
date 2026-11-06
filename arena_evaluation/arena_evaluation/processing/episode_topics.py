"""An episode's topic frames as the metrics read them: its window and its native-rate tables."""

from __future__ import annotations

import typing
from collections.abc import Iterable

import polars as pl

from arena_evaluation.processing.mcap_reader import KEPT_EMPTY_TABLES

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import TopicBundle

# Frames shared across robots or scoped to the whole episode, never per-robot sample data.
SHARED_FRAMES = frozenset({"tf", "tf_static", "semantic_snapshot", "episode_record"})
RECORD_RUNNING = 1  # arena_runtime_msgs/msg/EpisodeRecord.RUNNING


def episode_window(record: pl.DataFrame | pl.LazyFrame | None) -> tuple[int | None, int | None]:
    """[RUNNING stamp, terminal stamp] of the episode record, a bound is None when its row is missing."""
    if record is None:
        return None, None
    df = record.collect() if isinstance(record, pl.LazyFrame) else record
    if len(df) == 0 or "time_ns" not in df.columns:
        return None, None
    df = df.sort("time_ns")
    if "outcome_state" not in df.columns:
        return int(df["time_ns"][0]), int(df["time_ns"][-1])
    states = df["outcome_state"].to_list()
    stamps = df["time_ns"].to_list()
    start = next((int(t) for t, s in zip(stamps, states, strict=True) if s == RECORD_RUNNING), None)
    end = None if states[-1] == RECORD_RUNNING else int(stamps[-1])
    return start, end


def native_topics(bundle: TopicBundle, names: Iterable[str] | None = None) -> dict[str, pl.DataFrame]:
    """Native-rate topic frames by name, collected: `names`, else every topic but the shared frames; empty ones only when kept."""
    available = bundle.available()
    chosen = sorted(available - SHARED_FRAMES) if names is None else [name for name in names if name in available]
    topics: dict[str, pl.DataFrame] = {}
    for name in chosen:
        df = getattr(bundle, name)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        if len(df) > 0 or name in KEPT_EMPTY_TABLES:
            topics[name] = df
    return topics


__all__ = ["RECORD_RUNNING", "SHARED_FRAMES", "episode_window", "native_topics"]
