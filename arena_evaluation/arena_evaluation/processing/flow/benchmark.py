"""Flow change across the runs of a `flowedit suite` benchmark: each edit against its base world, seed by seed and pooled."""

from __future__ import annotations

import dataclasses
import pathlib
import typing
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from arena_evaluation.storage.run_tables import FLOW_CELL_SCHEMA, FLOW_CELLS, FLOW_MAP_PAIRS, FLOW_OBJECT_USES, FLOW_RUN_SCHEMA, FLOW_RUNS, FLOW_USES_SCHEMA, run_table_path

from ..benchmark_runs import EpisodeRun, episode_runs
from ..episode_topics import episode_window
from ..ped_track_store import TOPICS_DIR, pair, strip_env, tracks_for_episode
from .grid import FlowGrid
from .maps import jsd, occupancy, pooled
from .plots import map_pairs

#: The flow scores a run takes against its base world (FLOW_RUN_SCHEMA without the run's identity and its own counts)
FLOW_REFERENCE_COLUMNS = ("flow_reference_stage", "flow_jsd", "flow_jsd_occupancy", "flow_paired_peds", "flow_unpaired_peds", "flow_jsd_pooled", "flow_jsd_pooled_occupancy", "flow_pooled_seeds")
_QUEUE_ROLE = "queue"

Run = tuple[str, str]


def parse_stage(stage: str | None) -> tuple[str, int] | None:
    """(world, seed) of a `flowedit suite` stage name, None for any other."""
    from arena_simulation_setup.generative.flowedit.suite import parse_stage_name

    if stage is None:
        return None
    try:
        return parse_stage_name(stage)
    except ValueError:
        return None


def _base_of(world: str) -> str:
    from arena_simulation_setup.generative.flowedit.split import base_of

    return base_of(world)


def _stage_name(world: str, seed: int) -> str:
    from arena_simulation_setup.generative.flowedit.suite import stage_name

    return stage_name(world, seed)


def world_grid(world: str, run_dir: pathlib.Path | None) -> FlowGrid | None:
    """The grid over `world`'s rooms, from the run's world snapshot first, None when the world is not available."""
    from ..metrics.edge_case.provenance import world_view

    view = world_view(world, run_dir)
    return None if view is None else FlowGrid.from_world(view)


def object_uses(interactions: pl.DataFrame | None, members: pl.DataFrame | None = None) -> list[dict[str, typing.Any]]:
    """Uses and longest queue per anchor object of HumanSim's interactions (`pedestrian_interactions`, its queued
    members in `pedestrian_interaction_members`), sorted by object: a use is an interaction that reached holding, the
    queue the members waiting in one interaction's frame."""
    if interactions is None or len(interactions) == 0 or "object_id" not in interactions.columns:
        return []
    df = interactions.filter(pl.col("object_id").fill_null("") != "")
    if len(df) == 0:
        return []
    if members is not None and len(members) > 0 and {"t_ns", "interaction_id", "member_role"} <= set(members.columns):
        queued = members.filter(pl.col("member_role") == _QUEUE_ROLE).group_by("t_ns", "interaction_id").agg(pl.len().cast(pl.Int64).alias("queue"))
        df = df.join(queued, on=["t_ns", "interaction_id"], how="left").with_columns(pl.col("queue").fill_null(0))
    else:
        df = df.with_columns(pl.lit(0, dtype=pl.Int64).alias("queue"))
    held = pl.col("holding").fill_null(False) if "holding" in df.columns else pl.lit(False)
    per = df.group_by("object_id").agg(pl.col("interaction_id").filter(held).n_unique().alias("uses"), pl.col("queue").max().fill_null(0).alias("max_queue"))
    out: dict[str, dict[str, typing.Any]] = {}
    for object_id, uses, max_queue in per.iter_rows():
        name = strip_env(str(object_id))
        entry = out.setdefault(name, {"object": name, "uses": 0, "max_queue": 0})
        entry["uses"] += int(uses)
        entry["max_queue"] = max(entry["max_queue"], int(max_queue))
    return [out[k] for k in sorted(out)]


def _env_table(topics: pathlib.Path, name: str, env: str) -> pl.DataFrame | None:
    path = topics / f"{name}.parquet"
    return pl.scan_parquet(path).filter(pl.col("env_id") == env).collect() if path.is_file() else None


def run_object_uses(episode_dir: pathlib.Path, env: str) -> list[dict[str, typing.Any]]:
    """`object_uses` of one env of an extracted episode, inside its episode window."""
    topics = pathlib.Path(episode_dir) / TOPICS_DIR
    interactions = _env_table(topics, "pedestrian_interactions", env)
    if interactions is None:
        return []
    start, end = episode_window(_env_table(topics, "episode_lifecycle", env))
    if start is not None:
        interactions = interactions.filter(pl.col("t_ns") >= start)
    if end is not None:
        interactions = interactions.filter(pl.col("t_ns") <= end)
    return object_uses(interactions, _env_table(topics, "pedestrian_interaction_members", env))


@dataclasses.dataclass(frozen=True)
class FlowOutput:
    """The flow scores of every scored run (FLOW_RUN_SCHEMA), its flow map (FLOW_CELL_SCHEMA) and its object uses (FLOW_USES_SCHEMA)."""

    runs: pl.DataFrame
    cells: pl.DataFrame
    uses: pl.DataFrame


@dataclasses.dataclass(frozen=True)
class _Entry:
    """A scored run of a `flowedit suite` stage: its contestant, world and seed, and the recording (episode dir, env) it read."""

    run: EpisodeRun
    world: str
    seed: int

    @property
    def key(self) -> Run:
        return (str(self.run.episode_dir), self.run.env)

    @property
    def base(self) -> str:
        return _base_of(self.world)


def flow_measures(
    benchmark_dir: pathlib.Path,
    *,
    runs: Sequence[EpisodeRun] | None = None,
    tracks: Callable[[pathlib.Path, str], pl.DataFrame] = tracks_for_episode,
    grid: Callable[[str, pathlib.Path | None], FlowGrid | None] = world_grid,
    uses: Callable[[pathlib.Path, str], list[dict[str, typing.Any]]] = run_object_uses,
) -> FlowOutput:
    """The flow measures of every run of a `flowedit suite` stage of a processed benchmark (`benchmark_runs.episode_runs`,
    reference runs left out), each edit against its contestant's base run at the same seed and pooled over seeds."""
    benchmark_dir = pathlib.Path(benchmark_dir)
    entries: list[_Entry] = []
    for run in episode_runs(benchmark_dir) if runs is None else runs:
        parsed = parse_stage(run.stage)
        if parsed is None or run.is_reference:
            continue
        entries.append(_Entry(run, *parsed))
    if not entries:
        return FlowOutput(pl.DataFrame(schema=FLOW_RUN_SCHEMA), pl.DataFrame(schema=FLOW_CELL_SCHEMA), pl.DataFrame(schema=FLOW_USES_SCHEMA))

    grids: dict[str, FlowGrid | None] = {}
    maps: dict[Run, np.ndarray] = {}
    keys: dict[Run, pl.DataFrame] = {}
    rows: list[dict[str, typing.Any]] = []
    use_rows: list[dict[str, typing.Any]] = []
    for entry in entries:
        run = entry.run
        row: dict[str, typing.Any] = {**dict.fromkeys(FLOW_RUN_SCHEMA), "planner": run.planner, "stage": run.stage, "episode": run.episode, "env": run.env, "flow_world": entry.world, "flow_seed": entry.seed, "flow_base": entry.base}
        if entry.key not in keys:
            if entry.base not in grids:
                grids[entry.base] = grid(entry.base, benchmark_dir)
            g = grids[entry.base]
            track = tracks(run.episode_dir, run.env)
            keys[entry.key] = track.select("key").unique()
            if g is not None:
                maps[entry.key] = g.counts(track)
        row["flow_peds"] = len(keys[entry.key])
        if entry.key in maps:
            row["flow_room_ped_seconds"] = float(maps[entry.key].sum())
        rows.append(row)
        use_rows.extend({"planner": run.planner, "stage": run.stage, "episode": run.episode, "env": run.env, **use} for use in uses(run.episode_dir, run.env))

    stages: dict[tuple[str, str, int], Run] = {}
    for entry in entries:
        stages.setdefault((entry.run.planner, entry.world, entry.seed), entry.key)
    cells = [pl.DataFrame(schema=FLOW_CELL_SCHEMA)]
    for (planner, world, seed), key in stages.items():
        if key in maps:
            base = _base_of(world)
            labels = {"planner": planner, "stage": _stage_name(world, seed), "flow_world": world, "flow_base": base, "flow_seed": seed}
            g = grids[base]
            assert g is not None, f"{base}: a flow map without its grid"  # maps holds only runs counted on a grid
            cells.append(g.cells(maps[key]).with_columns(pl.lit(v, FLOW_CELL_SCHEMA[k]).alias(k) for k, v in labels.items()).select(list(FLOW_CELL_SCHEMA)))

    for entry, row in zip(entries, rows, strict=True):
        if entry.world == entry.base or (entry.run.planner, entry.base, entry.seed) not in stages:
            continue
        row["flow_reference_stage"] = _stage_name(entry.base, entry.seed)
        ref = stages[(entry.run.planner, entry.base, entry.seed)]
        if entry.key in maps and ref in maps:
            row["flow_jsd"] = jsd(maps[entry.key], maps[ref])
            row["flow_jsd_occupancy"] = jsd(occupancy(maps[entry.key]), occupancy(maps[ref]))
        if entry.key in keys and ref in keys:
            pairing = pair(keys[ref], keys[entry.key])
            row["flow_paired_peds"] = len(pairing.keys)
            row["flow_unpaired_peds"] = len(pairing.only_ref) + len(pairing.only_run)

    seeds: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
    for (planner, world, seed), key in stages.items():
        if key in maps:
            seeds[(planner, world)][seed] = key
    for (planner, world), by_seed in seeds.items():
        base = _base_of(world)
        if world == base or (planner, base) not in seeds:
            continue
        reference = seeds[(planner, base)]
        shared = sorted(set(by_seed) & set(reference))
        edit = pooled([maps[by_seed[s]] for s in shared])
        ref = pooled([maps[reference[s]] for s in shared])
        result: dict[str, typing.Any] = {"flow_pooled_seeds": len(shared)}
        if edit is not None and ref is not None:
            result.update({"flow_jsd_pooled": jsd(edit, ref), "flow_jsd_pooled_occupancy": jsd(occupancy(edit), occupancy(ref))})
        for entry, row in zip(entries, rows, strict=True):
            if (entry.run.planner, entry.world) == (planner, world):
                row.update(result)
    return FlowOutput(pl.DataFrame(rows, schema=FLOW_RUN_SCHEMA), pl.concat(cells), pl.DataFrame(use_rows, schema=FLOW_USES_SCHEMA))


def write_flow_tables(benchmark_dir: pathlib.Path, output: FlowOutput | None = None) -> FlowOutput:
    """Compute the flow measures of a processed benchmark (`flow_measures`) and write them at its root (`flow_runs`,
    `flow_cells`, `flow_object_uses`, and the maps the report draws, `flow_map_pairs`; `storage.run_tables`); a
    benchmark without `flowedit suite` runs has none, and a stale table is removed."""
    benchmark_dir = pathlib.Path(benchmark_dir)
    output = flow_measures(benchmark_dir) if output is None else output
    tables = ((FLOW_RUNS, output.runs), (FLOW_CELLS, output.cells), (FLOW_OBJECT_USES, output.uses), (FLOW_MAP_PAIRS, map_pairs(output.cells)))
    for name, frame in tables:
        path = run_table_path(benchmark_dir, name)
        if len(output.runs) == 0:
            path.unlink(missing_ok=True)
            continue
        temp = path.with_name(f".{path.name}.partial")
        frame.write_parquet(temp)
        temp.replace(path)
    return output


__all__ = ["FLOW_CELL_SCHEMA", "FLOW_REFERENCE_COLUMNS", "FLOW_RUN_SCHEMA", "FLOW_USES_SCHEMA", "FlowOutput", "flow_measures", "object_uses", "parse_stage", "run_object_uses", "world_grid", "write_flow_tables"]
