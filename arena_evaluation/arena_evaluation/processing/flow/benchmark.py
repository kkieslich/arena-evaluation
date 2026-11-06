"""Flow change across the episodes of a `flowedit suite` benchmark: each edit against its base world, seed by seed and pooled."""

from __future__ import annotations

import dataclasses
import pathlib
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import polars as pl
from arena_simulation_setup.generative.flowedit.split import base_of
from arena_simulation_setup.generative.flowedit.suite import parse_stage_name, stage_name

from ..mcap_reader import bundle_env
from ..ped_track_store import pair, tracks_for_episode
from .grid import CELL_SCHEMA, FlowGrid
from .maps import jsd, occupancy, pooled

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
# the cell rows of every evaluated run, as process_benchmark writes them to grid.FLOW_CELLS_FILE
FLOW_CELL_SCHEMA = {"planner": pl.String, "stage": pl.String, "flow_world": pl.String, "flow_base": pl.String, "flow_seed": pl.Int64, **CELL_SCHEMA}

Run = tuple[int, str]


def parse_stage(stage: str | None) -> tuple[str, int] | None:
    """(world, seed) of a `flowedit suite` stage name, None for any other."""
    if stage is None:
        return None
    try:
        return parse_stage_name(stage)
    except ValueError:
        return None


def world_grid(world: str, run_dir: pathlib.Path | None) -> FlowGrid | None:
    """The grid over `world`'s rooms, from the run's world snapshot first, None when the world is not available."""
    from ..metrics.edge_case.provenance import world_view

    view = world_view(world, run_dir)
    return None if view is None else FlowGrid.from_world(view)


@dataclasses.dataclass(frozen=True)
class FlowOutput:
    """How many rows took part in the flow measures, and the flow map of each run they scored (FLOW_CELL_SCHEMA rows)."""

    rows: int
    cells: pl.DataFrame


@dataclasses.dataclass(frozen=True)
class _Entry:
    """An evaluated row of a `flowedit suite` stage: its contestant, world and seed, and the recording (episode, env) it read."""

    row: int
    planner: str
    world: str
    seed: int
    run: Run | None

    @property
    def base(self) -> str:
        return base_of(self.world)


def add_flow_reference_metrics(
    rows: list[dict],
    episode_dir: Callable[[int], pathlib.Path],
    *,
    run_dir: pathlib.Path | None = None,
    evaluated: str = "evaluated",
    tracks: Callable[[pathlib.Path, str], pl.DataFrame] = tracks_for_episode,
    grid: Callable[[str, pathlib.Path | None], FlowGrid | None] = world_grid,
) -> FlowOutput:
    """Write the flow outputs into every evaluated `flowedit suite` row of `rows` in place, each edit against its contestant's base run."""
    entries: list[_Entry] = []
    for i, row in enumerate(rows):
        parsed = parse_stage(row.get("stage"))
        if parsed is None or row.get("is_reference") or row.get("status") != evaluated:
            continue
        episode = row.get("episode")
        run = None if episode is None else (int(episode), bundle_env(str(row.get("robot") or "")))
        entries.append(_Entry(i, str(row.get("planner") or ""), *parsed, run))
    if not entries:
        return FlowOutput(0, pl.DataFrame(schema=FLOW_CELL_SCHEMA))

    grids: dict[str, FlowGrid | None] = {}
    maps: dict[Run, np.ndarray] = {}
    keys: dict[Run, pl.DataFrame] = {}
    for entry in entries:
        row = rows[entry.row]
        row.update(dict.fromkeys(FLOW_REFERENCE_DTYPES))
        row.update({"flow_world": entry.world, "flow_seed": entry.seed, "flow_base": entry.base})
        if entry.run is None:
            continue
        if entry.run not in keys:
            if entry.base not in grids:
                grids[entry.base] = grid(entry.base, run_dir)
            g = grids[entry.base]
            track = tracks(episode_dir(entry.run[0]), entry.run[1])
            keys[entry.run] = track.select("key").unique()
            if g is not None:
                maps[entry.run] = g.counts(track)
        if entry.run in maps:
            row["flow_room_ped_seconds"] = float(maps[entry.run].sum())

    stages: dict[tuple[str, str, int], Run | None] = {}
    for entry in entries:
        stages.setdefault((entry.planner, entry.world, entry.seed), entry.run)
    cells = [pl.DataFrame(schema=FLOW_CELL_SCHEMA)]
    for (planner, world, seed), run in stages.items():
        if run in maps:
            labels = {"planner": planner, "stage": stage_name(world, seed), "flow_world": world, "flow_base": base_of(world), "flow_seed": seed}
            cells.append(grids[base_of(world)].cells(maps[run]).with_columns(pl.lit(v, FLOW_CELL_SCHEMA[k]).alias(k) for k, v in labels.items()).select(list(FLOW_CELL_SCHEMA)))

    for entry in entries:
        if entry.world == entry.base or (entry.planner, entry.base, entry.seed) not in stages:
            continue
        row = rows[entry.row]
        row["flow_reference_stage"] = stage_name(entry.base, entry.seed)
        ref = stages[(entry.planner, entry.base, entry.seed)]
        if entry.run is None or ref is None:
            continue
        if entry.run in maps and ref in maps:
            row["flow_jsd"] = jsd(maps[entry.run], maps[ref])
            row["flow_jsd_occupancy"] = jsd(occupancy(maps[entry.run]), occupancy(maps[ref]))
        if entry.run in keys and ref in keys:
            pairing = pair(keys[ref], keys[entry.run])
            row["flow_paired_peds"] = len(pairing.keys)
            row["flow_unpaired_peds"] = len(pairing.only_ref) + len(pairing.only_run)

    seeds: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
    for (planner, world, seed), run in stages.items():
        if run is not None and run in maps:
            seeds[(planner, world)][seed] = run
    for (planner, world), runs in seeds.items():
        base = base_of(world)
        if world == base or (planner, base) not in seeds:
            continue
        reference = seeds[(planner, base)]
        shared = sorted(set(runs) & set(reference))
        edit = pooled([maps[runs[s]] for s in shared])
        ref = pooled([maps[reference[s]] for s in shared])
        result = {"flow_pooled_seeds": len(shared)}
        if edit is not None and ref is not None:
            result.update({"flow_jsd_pooled": jsd(edit, ref), "flow_jsd_pooled_occupancy": jsd(occupancy(edit), occupancy(ref))})
        for entry in entries:
            if (entry.planner, entry.world) == (planner, world):
                rows[entry.row].update(result)
    return FlowOutput(len(entries), pl.concat(cells))


__all__ = ["FLOW_CELL_SCHEMA", "FLOW_REFERENCE_DTYPES", "FlowOutput", "add_flow_reference_metrics", "parse_stage", "world_grid"]
