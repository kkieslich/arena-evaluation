"""Per-episode inputs of the FLOWEDIT flow measures: who walked, and how often each object was used.

The flow change itself compares two runs (an edit against its base world) and is computed across the benchmark, in
`processing.flow.benchmark`; this calculator reports what one episode alone says:

    flow_peds          pedestrians in the episode, counted by their pairing key (`ped_track_store`)
    flow_object_uses   per object an interaction was anchored at: `uses`, the interactions that reached holding (a
                       drink at a fountain, a seat taken), and `max_queue`, the longest queue it had
"""

from __future__ import annotations

import typing

import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.ped_track_store import strip_env, track_table
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

FLOW_EPISODE_DTYPES = {
    "flow_peds": pl.Int64,
    "flow_object_uses": pl.List(pl.Struct({"object": pl.String, "uses": pl.Int64, "max_queue": pl.Int64})),
}


def object_uses(interactions: pl.DataFrame | None) -> list[dict[str, typing.Any]]:
    """Uses and longest queue per anchor object of HumanSim's interactions table, sorted by object."""
    if interactions is None or len(interactions) == 0 or "object_id" not in interactions.columns:
        return []
    df = interactions.filter(pl.col("object_id").fill_null("") != "")
    if len(df) == 0:
        return []
    queue = pl.col("queue").list.len() if "queue" in df.columns else pl.lit(0)
    held = pl.col("holding").fill_null(False) if "holding" in df.columns else pl.lit(False)
    per = df.group_by("object_id").agg(pl.col("interaction_id").filter(held).n_unique().alias("uses"), queue.max().fill_null(0).alias("max_queue"))
    out: dict[str, dict[str, typing.Any]] = {}
    for object_id, uses, max_queue in per.iter_rows():
        name = strip_env(str(object_id))
        entry = out.setdefault(name, {"object": name, "uses": 0, "max_queue": 0})
        entry["uses"] += int(uses)
        entry["max_queue"] = max(entry["max_queue"], int(max_queue))
    return [out[k] for k in sorted(out)]


class FlowEpisodeCalculator(BaseMetricCalculator):
    """Pedestrians and object uses of one episode, the per-episode side of the FLOWEDIT flow measures."""

    NAME = "flow_episode"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    REQUIRED_TOPICS = [("peds_physics", "peds")]
    UNITS = {"flow_peds": "count", "flow_object_uses": "list"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return list(FLOW_EPISODE_DTYPES)

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        del prior_results
        topics = self.native_topics(episode)
        start = end = None
        if episode.data is not None and "time_ns" in episode.data.columns and len(episode.data) > 0:
            start, end = int(episode.data["time_ns"].min()), int(episode.data["time_ns"].max())
        tracks = track_table(topics, start, end)
        uses = topics.get("interactions")
        if uses is not None and start is not None and "time_ns" in uses.columns:
            uses = uses.filter(pl.col("time_ns").is_between(start, end))
        return {"flow_peds": int(tracks["key"].n_unique()) if len(tracks) else 0, "flow_object_uses": object_uses(uses)}


__all__ = ["FLOW_EPISODE_DTYPES", "FlowEpisodeCalculator", "object_uses"]
