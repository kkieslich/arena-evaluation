"""Benchmark-level tables the cross-run measures write at the run root, beside the dimensions: report data sources."""

from __future__ import annotations

import pathlib
from typing import Final

import polars as pl

#: The flow measures of a `flowedit suite` benchmark (`processing/flow`): the scores of each run against its base
#: world, the flow map of each run (kkieslich/Arena's flowedit search and forecaster read `flow_cells.parquet` too),
#: the uses of each anchor object per run, and the maps the report draws per edit against its base world
FLOW_RUNS: Final[str] = "flow_runs"
FLOW_CELLS: Final[str] = "flow_cells"
FLOW_OBJECT_USES: Final[str] = "flow_object_uses"
FLOW_MAP_PAIRS: Final[str] = "flow_map_pairs"
RUN_TABLES: Final[tuple[str, ...]] = (FLOW_RUNS, FLOW_CELLS, FLOW_OBJECT_USES, FLOW_MAP_PAIRS)

#: One row per scored run (an env of an episode of a `flowedit suite` stage that is no reference run)
FLOW_RUN_SCHEMA = {
    "planner": pl.String,
    "stage": pl.String,
    "episode": pl.Int64,
    "env": pl.String,
    "flow_world": pl.String,
    "flow_seed": pl.Int64,
    "flow_base": pl.String,
    "flow_peds": pl.Int64,
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
#: The cell rows of every scored run's map (grid.FLOW_CELLS_FILE at the benchmark root)
FLOW_CELL_SCHEMA = {"planner": pl.String, "stage": pl.String, "flow_world": pl.String, "flow_base": pl.String, "flow_seed": pl.Int64, "x": pl.Float64, "y": pl.Float64, "cell_m": pl.Float64, "heading_rad": pl.Float64, "ped_seconds": pl.Float64}
#: Uses and the longest queue per anchor object of HumanSim's interactions, per scored run
FLOW_USES_SCHEMA = {"planner": pl.String, "stage": pl.String, "episode": pl.Int64, "env": pl.String, "object": pl.String, "uses": pl.Int64, "max_queue": pl.Int64}
#: One row per drawn cell of each pair: the pair (planner, edit world, base world, seeds, rank: the largest change
#: first per contestant), its raster (origin, size, cell), the cell (row i, column j), the base and edit occupancy in
#: pedestrian-seconds per m² per run, the walking time summed as a vector along the headings (hx, hy), whether the cell
#: carries an arrow, and the colour scales shared by every pair (occupancy top, half-range of the difference)
MAP_PAIR_SCHEMA = {
    "planner": pl.String,
    "flow_world": pl.String,
    "flow_base": pl.String,
    "seeds": pl.String,
    "rank": pl.Int64,
    "x0": pl.Float64,
    "y0": pl.Float64,
    "nx": pl.Int64,
    "ny": pl.Int64,
    "cell_m": pl.Float64,
    "i": pl.Int64,
    "j": pl.Int64,
    "ref": pl.Float64,
    "edit": pl.Float64,
    "ref_hx": pl.Float64,
    "ref_hy": pl.Float64,
    "edit_hx": pl.Float64,
    "edit_hy": pl.Float64,
    "ref_arrow": pl.Boolean,
    "edit_arrow": pl.Boolean,
    "occ_top": pl.Float64,
    "diff_lim": pl.Float64,
}

#: The columns of each run table
RUN_TABLE_SCHEMAS: Final[dict[str, dict[str, pl.DataType]]] = {FLOW_RUNS: FLOW_RUN_SCHEMA, FLOW_CELLS: FLOW_CELL_SCHEMA, FLOW_OBJECT_USES: FLOW_USES_SCHEMA, FLOW_MAP_PAIRS: MAP_PAIR_SCHEMA}


def run_table_path(benchmark_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Where the run table `name` lies in a benchmark dir."""
    if name not in RUN_TABLES:
        raise ValueError(f"unknown run table {name!r}, known: {', '.join(RUN_TABLES)}")
    return pathlib.Path(benchmark_dir) / f"{name}.parquet"


def scan_run_tables(benchmark_dir: pathlib.Path, *, empty: bool = False) -> dict[str, pl.LazyFrame]:
    """The run tables a benchmark dir holds, by name, lazily; with `empty`, an empty frame of its columns for each one
    it does not hold (a benchmark without flowedit suite stages)."""
    out: dict[str, pl.LazyFrame] = {}
    for name in RUN_TABLES:
        path = run_table_path(benchmark_dir, name)
        if path.is_file():
            out[name] = pl.scan_parquet(path)
        elif empty:
            out[name] = pl.LazyFrame(schema=RUN_TABLE_SCHEMAS[name])
    return out


__all__ = ["FLOW_CELL_SCHEMA", "FLOW_RUN_SCHEMA", "FLOW_USES_SCHEMA", "MAP_PAIR_SCHEMA", "RUN_TABLE_SCHEMAS", "FLOW_CELLS", "FLOW_MAP_PAIRS", "FLOW_OBJECT_USES", "FLOW_RUNS", "RUN_TABLES", "run_table_path", "scan_run_tables"]
