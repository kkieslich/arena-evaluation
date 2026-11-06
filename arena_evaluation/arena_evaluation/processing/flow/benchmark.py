"""FLOWEDIT flow change across the episodes of a benchmark: each edit against its base world, seed by seed and pooled.

A FLOWEDIT suite (`flowedit suite`) names every stage `<world>__s<seed>`, one episode each. An edited world is
`<base>__<edit id>`; its base stage at the same seed ran the same crowd, so the two pair. Per episode row:

    flow_world, flow_seed, flow_base    what the stage name says
    flow_room_ped_seconds               pedestrian-seconds inside the base world's rooms
    flow_reference_stage                the base stage the row is compared with, None for a base row
    flow_jsd, flow_jsd_occupancy        divergence from that base run (paired): occupancy and heading, occupancy alone
    flow_paired_peds, flow_unpaired_peds
                                        pedestrians the two runs share by key, and those only one of them has
    flow_jsd_pooled, flow_jsd_pooled_occupancy, flow_pooled_seeds
                                        divergence of the seed-averaged maps over the seeds both worlds ran (pooled),
                                        the same on every row of a world: the systematic flow change

Both runs of a pair are counted on the base world's grid. The pooled divergence is the primary score: the paired one
also counts the crowd running differently after one pedestrian was turned another way.
"""

from __future__ import annotations

import pathlib
import re
import typing
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import polars as pl

from ..ped_track_store import pair, tracks_for_episode
from .grid import FlowGrid
from .maps import jsd, occupancy, pooled

_STAGE = re.compile(r"^(?P<world>.+)__s(?P<seed>\d+)$")

FLOW_REFERENCE_DTYPES = {
    "flow_world": pl.String,
    "flow_seed": pl.Int64,
    "flow_base": pl.String,
    "flow_room_ped_seconds": pl.Float64,
    "flow_reference_stage": pl.String,
    "flow_jsd": pl.Float64,
    "flow_jsd_occupancy": pl.Float64,
    "flow_paired_peds": pl.Int64,
    "flow_unpaired_peds": pl.Int64,
    "flow_jsd_pooled": pl.Float64,
    "flow_jsd_pooled_occupancy": pl.Float64,
    "flow_pooled_seeds": pl.Int64,
}


def parse_stage(stage: typing.Any) -> tuple[str, int] | None:
    """(world, seed) of a FLOWEDIT stage name `<world>__s<seed>`, None for any other."""
    match = _STAGE.match(stage) if isinstance(stage, str) else None
    return None if match is None else (match["world"], int(match["seed"]))


def base_world(world: str) -> str:
    """The base of an edited world `<base>__<edit id>`, or the world itself."""
    return world.partition("__")[0]


def world_grid(world: str, run_dir: pathlib.Path | None) -> FlowGrid | None:
    """The grid over `world`'s rooms, from the run's world snapshot first, None when the world is not available."""
    from ..metrics.edge_case.provenance import world_view

    view = world_view(world, run_dir)
    return None if view is None else FlowGrid.from_world(view)


def add_flow_reference_metrics(
    rows: list[dict],
    episode_dir: Callable[[dict], pathlib.Path | None],
    *,
    evaluated: str = "evaluated",
    tracks: Callable[[pathlib.Path], pl.DataFrame] = lambda d: tracks_for_episode(d, window=True),
    grid: Callable[[str, pathlib.Path | None], FlowGrid | None] = world_grid,
) -> int:
    """Write the flow outputs into every evaluated FLOWEDIT row of `rows` in place; returns how many rows took part."""
    plan: list[tuple[int, str, int]] = []
    for i, row in enumerate(rows):
        parsed = parse_stage(row.get("stage"))
        if parsed is None or row.get("is_reference") or row.get("status") != evaluated:
            continue
        plan.append((i, *parsed))
    if not plan:
        return 0

    grids: dict[str, FlowGrid | None] = {}
    maps: dict[int, np.ndarray] = {}
    keys: dict[int, pl.DataFrame] = {}
    by_stage: dict[tuple[str, int], int] = {}
    for i, world, seed in plan:
        row = rows[i]
        base = base_world(world)
        row.update(dict.fromkeys(FLOW_REFERENCE_DTYPES))
        row.update({"flow_world": world, "flow_seed": seed, "flow_base": base})
        by_stage.setdefault((world, seed), i)
        directory = episode_dir(row)
        if directory is None:
            continue
        if base not in grids:
            grids[base] = grid(base, pathlib.Path(directory).parent.parent)
        g = grids[base]
        track = tracks(pathlib.Path(directory))
        keys[i] = track.select("key").unique()
        if g is not None:
            maps[i] = g.counts(track)
            row["flow_room_ped_seconds"] = float(maps[i].sum())

    for i, world, seed in plan:
        base = base_world(world)
        if world == base:
            continue
        row = rows[i]
        ref = by_stage.get((base, seed))
        if ref is None:
            continue
        row["flow_reference_stage"] = f"{base}__s{seed}"
        if i in maps and ref in maps:
            row["flow_jsd"] = jsd(maps[i], maps[ref])
            row["flow_jsd_occupancy"] = jsd(occupancy(maps[i]), occupancy(maps[ref]))
        if i in keys and ref in keys:
            pairing = pair(keys[ref], keys[i])
            row["flow_paired_peds"] = len(pairing.keys)
            row["flow_unpaired_peds"] = len(pairing.only_ref) + len(pairing.only_run)

    seeds: dict[str, dict[int, int]] = defaultdict(dict)
    for (world, seed), i in by_stage.items():
        if i in maps:
            seeds[world][seed] = i
    for world, runs in seeds.items():
        base = base_world(world)
        if world == base or base not in seeds:
            continue
        shared = sorted(set(runs) & set(seeds[base]))
        edit = pooled([maps[runs[s]] for s in shared])
        ref = pooled([maps[seeds[base][s]] for s in shared])
        result = {"flow_pooled_seeds": len(shared)}
        if edit is not None and ref is not None:
            result.update({"flow_jsd_pooled": jsd(edit, ref), "flow_jsd_pooled_occupancy": jsd(occupancy(edit), occupancy(ref))})
        for i, w, _ in plan:
            if w == world:
                rows[i].update(result)
    return len(plan)


__all__ = ["FLOW_REFERENCE_DTYPES", "add_flow_reference_metrics", "base_world", "parse_stage", "world_grid"]
