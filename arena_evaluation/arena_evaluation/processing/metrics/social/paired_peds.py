"""Per-pedestrian measures of a run against its stage's robot-free reference run, the pedestrians paired by track key.

The tracks are `ped_track_store.track_table` frames. A pedestrian completes a run when it appears after the table's
first frame and leaves before its last, so a table cut at the end of a warm-up counts only pedestrians spawned after it.
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.mcap_reader import bundle_env
from arena_evaluation.processing.metrics.social.pedestrian_disturbance import PedestrianDisturbanceCalculator
from arena_evaluation.processing.metrics.social.pedestrian_groups import agent_group, by_group, worst_group
from arena_evaluation.processing.ped_track_store import Pairing, empty_tracks, pair, tracks_for_episode

_PER_PED = pl.List(pl.Struct({"ped": pl.String, "value": pl.Float64}))
_BY_GROUP = pl.List(pl.Struct({"group": pl.String, "value": pl.Float64, "peds": pl.Int64}))

# The paired measures with a worst group, with the direction that is better. `ped_paired_dropped` counts pedestrians
# per group and names none.
PAIRED_MEASURES = {
    "ped_paired_deflection_m": "lower",
    "ped_paired_delay_s": "lower",
    "ped_paired_relative_delay": "lower",
}

PAIRED_DTYPES = {
    "ped_paired_deflection_m": pl.Float64,
    "ped_paired_deflection_m_per_ped": _PER_PED,
    "ped_paired_deflection_m_by_group": _BY_GROUP,
    "ped_paired_deflection_m_worst_group": pl.String,
    "ped_paired_delay_s": pl.Float64,
    "ped_paired_delay_s_per_ped": _PER_PED,
    "ped_paired_delay_s_by_group": _BY_GROUP,
    "ped_paired_delay_s_worst_group": pl.String,
    "ped_paired_relative_delay": pl.Float64,
    "ped_paired_relative_delay_by_group": _BY_GROUP,
    "ped_paired_relative_delay_worst_group": pl.String,
    "ped_paired_dropped": pl.Int64,
    "ped_paired_dropped_by_group": _BY_GROUP,
}


def _points(tracks: pl.DataFrame, keys: list[str]) -> dict[str, np.ndarray]:
    """Each key's finite (x, y) points, as views into one array sorted by key."""
    rows = tracks.lazy().select("key", "x", "y").filter(pl.col("key").is_in(keys) & pl.col("x").is_finite() & pl.col("y").is_finite()).sort("key", maintain_order=True).collect()
    counts = rows.group_by("key", maintain_order=True).len()
    xy = rows.select("x", "y").to_numpy()
    bounds = np.cumsum([0, *counts["len"].to_list()])
    return {key: xy[start:end] for key, start, end in zip(counts["key"].to_list(), bounds[:-1], bounds[1:], strict=True)}


def paired_deflection(ref_tracks: pl.DataFrame, run_tracks: pl.DataFrame, pairing: Pairing) -> list[dict[str, typing.Any]]:
    """Each paired pedestrian's mean distance from its run points to the nearest point of its own reference track.

    The pooled reference `ped_path_deflection_m` of `process_benchmark`, per pedestrian.
    """
    ref = _points(ref_tracks, pairing.keys)
    run = _points(run_tracks, pairing.keys)
    return [{"ped": key, "value": round(PedestrianDisturbanceCalculator._compute_trajectory_deflection(run[key], ref[key]), 3)} for key in pairing.keys if key in ref and key in run]


def _travel_times(tracks: pl.DataFrame) -> pl.DataFrame:
    """key, agent_type and travel time in s of each pedestrian that completes the run."""
    if len(tracks) == 0:
        return pl.DataFrame(schema={"key": pl.String, "agent_type": pl.String, "travel_s": pl.Float64})
    first, last = tracks["time_ns"].min(), tracks["time_ns"].max()
    spans = tracks.group_by("key").agg(pl.col("time_ns").min().alias("start"), pl.col("time_ns").max().alias("end"), pl.col("agent_type").first())
    return spans.filter((pl.col("start") > first) & (pl.col("end") < last)).select("key", "agent_type", ((pl.col("end") - pl.col("start")) / 1e9).alias("travel_s"))


def _completed(ref_tracks: pl.DataFrame, run_tracks: pl.DataFrame, pairing: Pairing) -> pl.DataFrame:
    """key, agent_type, reference and run travel time of the paired pedestrians that complete both runs, by key."""
    ref = _travel_times(ref_tracks).select("key", pl.col("travel_s").alias("ref_s"))
    run = _travel_times(run_tracks).rename({"travel_s": "run_s"})
    return run.join(ref, on="key").filter(pl.col("key").is_in(pairing.keys)).sort("key")


def paired_travel_time_delay(ref_tracks: pl.DataFrame, run_tracks: pl.DataFrame, pairing: Pairing) -> list[dict[str, typing.Any]]:
    """Each paired pedestrian's travel time in the run minus in the reference, for those that complete both runs."""
    done = _completed(ref_tracks, run_tracks, pairing)
    return [{"ped": key, "value": run_s - ref_s} for key, run_s, ref_s in done.select("key", "run_s", "ref_s").iter_rows()]


def _groups(tracks: pl.DataFrame) -> dict[str, str]:
    """Group by key, from each pedestrian's agent type."""
    return {key: agent_group(agent_type) for key, agent_type in tracks.group_by("key").agg(pl.col("agent_type").first()).iter_rows()}


def paired_ped_measures(ref_tracks: pl.DataFrame, run_tracks: pl.DataFrame) -> dict[str, typing.Any]:
    """The paired deflection and travel-time delay of a run against its reference, per pedestrian and per group, None without both tracks.

    The relative delay of a group is its mean extra travel time over its mean reference travel time (a ratio of
    means). Dropped are paired pedestrians that do not complete both runs, per group of the paired ones.
    """
    if len(ref_tracks) == 0 or len(run_tracks) == 0:
        return dict.fromkeys(PAIRED_DTYPES)
    pairing = pair(ref_tracks, run_tracks)
    groups = _groups(run_tracks)
    deflection = paired_deflection(ref_tracks, run_tracks, pairing)
    done = _completed(ref_tracks, run_tracks, pairing).with_columns((pl.col("run_s") - pl.col("ref_s")).alias("delay_s"))
    delay = [{"ped": key, "value": value} for key, value in done.select("key", "delay_s").iter_rows()]
    done_groups = done.with_columns(pl.Series("group", [groups[key] for key in done["key"].to_list()], dtype=pl.String))
    relative = done_groups.group_by("group").agg((pl.col("delay_s").sum() / pl.col("ref_s").sum()).alias("value"), pl.len().cast(pl.Int64).alias("peds")).sort("group").to_dicts() or None
    finished = set(done["key"].to_list())
    dropped: dict[str, list[float]] = {}
    for key in pairing.keys:
        dropped.setdefault(groups[key], []).append(float(key not in finished))
    deflection_groups = by_group(deflection, groups)
    delay_groups = by_group(delay, groups)
    ref_total = float(done["ref_s"].sum())
    return {
        "ped_paired_deflection_m": round(float(np.mean([r["value"] for r in deflection])), 3) if deflection else None,
        "ped_paired_deflection_m_per_ped": deflection,
        "ped_paired_deflection_m_by_group": deflection_groups,
        "ped_paired_deflection_m_worst_group": worst_group(deflection_groups, PAIRED_MEASURES["ped_paired_deflection_m"]),
        "ped_paired_delay_s": float(done["delay_s"].mean()) if len(done) else None,
        "ped_paired_delay_s_per_ped": delay,
        "ped_paired_delay_s_by_group": delay_groups,
        "ped_paired_delay_s_worst_group": worst_group(delay_groups, PAIRED_MEASURES["ped_paired_delay_s"]),
        "ped_paired_relative_delay": float(done["delay_s"].sum()) / ref_total if ref_total > 0 else None,
        "ped_paired_relative_delay_by_group": relative,
        "ped_paired_relative_delay_worst_group": worst_group(relative, PAIRED_MEASURES["ped_paired_relative_delay"]),
        "ped_paired_dropped": len(pairing.keys) - len(finished),
        "ped_paired_dropped_by_group": [{"group": group, "value": float(sum(flags)), "peds": len(flags)} for group, flags in sorted(dropped.items())] or None,
    }


def _run(row: dict[str, typing.Any]) -> tuple[int, str] | None:
    """The (episode id, env) whose tracks a row scores, None for a row without an episode."""
    episode = row.get("episode")
    return None if episode is None else (int(episode), bundle_env(str(row.get("robot") or "")))


def add_paired_ped_measures(
    rows: list[dict[str, typing.Any]],
    ref_rows_by_stage: typing.Mapping[str | None, list[dict[str, typing.Any]]],
    episode_dir: typing.Callable[[int], pathlib.Path],
    *,
    evaluated: str,
    tracks: typing.Callable[[pathlib.Path, str], pl.DataFrame] = tracks_for_episode,
) -> None:
    """Give every run row its paired measures against the first `evaluated` reference row of its stage.

    Each row is scored on the tracks of its own env, the reference on those of the reference row's env; each stage's
    reference and each (episode, env) are read once.
    """
    by_stage: dict[str | None, dict[tuple[int, str] | None, list[dict[str, typing.Any]]]] = {}
    for row in rows:
        if not row.get("is_reference"):
            by_stage.setdefault(row.get("stage"), {}).setdefault(_run(row), []).append(row)
    for stage, runs in by_stage.items():
        ref_row = next((r for r in ref_rows_by_stage.get(stage, []) if r.get("status") == evaluated), None)
        ref = None if ref_row is None else _run(ref_row)
        if ref is None:
            continue
        ref_tracks = tracks(episode_dir(ref[0]), ref[1])
        for run, run_rows in runs.items():
            measures = paired_ped_measures(ref_tracks, empty_tracks() if run is None else tracks(episode_dir(run[0]), run[1]))
            for row in run_rows:
                row.update(measures)
