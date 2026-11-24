"""The flow maps the report draws (`flow_map`): per edit and contestant, its base world's map, the edit's and their
difference on one raster, each the mean of the seeds both ran, with the shared colour scales and the arrow cells."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import polars as pl

from arena_evaluation.storage.run_tables import MAP_PAIR_SCHEMA

# low occupancy stays visible next to a queue: its colour follows value ** GAMMA (the difference stays linear)
GAMMA = 0.45
# arrows mark the cells whose walking time is at or above this percentile of the map's
ARROW_PERCENTILE = 60



@dataclasses.dataclass(frozen=True)
class Raster:
    """Cell centres of one base world's grid, covering every map drawn on it."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float

    @classmethod
    def over(cls, cells: pl.DataFrame) -> Raster:
        """The raster spanning every cell of `cells` (FLOW_CELL_SCHEMA rows of one base world)."""
        cell = float(cells["cell_m"][0])
        x0, y0 = float(cells["x"].min()), float(cells["y"].min())
        return cls(x0, y0, round((float(cells["x"].max()) - x0) / cell) + 1, round((float(cells["y"].max()) - y0) / cell) + 1, cell)

    def index(self, cells: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(row, column) of each cell of `cells`."""
        return np.rint((cells["y"].to_numpy() - self.y0) / self.cell).astype(np.int64), np.rint((cells["x"].to_numpy() - self.x0) / self.cell).astype(np.int64)

    def occupancy(self, cells: pl.DataFrame) -> np.ndarray:
        """Pedestrian-seconds per m² of each cell."""
        out = np.zeros((self.ny, self.nx))
        i, j = self.index(cells)
        np.add.at(out, (i, j), cells["ped_seconds"].to_numpy() / self.cell**2)
        return out

    def heading(self, cells: pl.DataFrame) -> np.ndarray:
        """The walking time of each cell summed as a vector along its headings, (ny, nx, 2)."""
        out = np.zeros((self.ny, self.nx, 2))
        moving = cells.filter(pl.col("heading_rad").is_not_null())
        i, j = self.index(moving)
        theta, w = moving["heading_rad"].to_numpy(), moving["ped_seconds"].to_numpy()
        np.add.at(out, (i, j, 0), w * np.cos(theta))
        np.add.at(out, (i, j, 1), w * np.sin(theta))
        return out


@dataclasses.dataclass(frozen=True)
class _Pair:
    planner: str
    world: str
    base: str
    seeds: list[int]
    raster: Raster
    ref: np.ndarray
    edit: np.ndarray
    ref_heading: np.ndarray
    edit_heading: np.ndarray


def _mean(runs: list[pl.DataFrame], fn: Callable[[pl.DataFrame], np.ndarray]) -> np.ndarray:
    return np.mean([fn(run) for run in runs], axis=0)


def _pairs(cells: pl.DataFrame) -> list[_Pair]:
    """Every edit with its base world per contestant, over the seeds both ran, the largest change first."""
    out = []
    rasters = {base: Raster.over(group) for (base,), group in cells.group_by("flow_base")}
    runs = {key: group for key, group in cells.group_by("planner", "flow_world", "flow_seed")}
    for planner, world, base in cells.select("planner", "flow_world", "flow_base").unique().sort("planner", "flow_world").iter_rows():
        if world == base:
            continue
        seeds = sorted(s for p, w, s in runs if (p, w) == (planner, world) and (planner, base, s) in runs)
        if not seeds:
            continue
        raster = rasters[base]
        ref, edit = [runs[(planner, base, s)] for s in seeds], [runs[(planner, world, s)] for s in seeds]
        out.append(_Pair(planner, world, base, seeds, raster, _mean(ref, raster.occupancy), _mean(edit, raster.occupancy), _mean(ref, raster.heading), _mean(edit, raster.heading)))
    return sorted(out, key=lambda p: (p.planner, -float(np.abs(p.edit - p.ref).sum()), p.world))


def _scales(pairs: list[_Pair]) -> tuple[float, float]:
    """Top of the occupancy scale and half-range of the difference scale, shared by every pair so maps compare."""
    occ = np.concatenate([np.concatenate([p.ref.ravel(), p.edit.ravel()]) for p in pairs])
    diff = np.concatenate([(p.edit - p.ref).ravel() for p in pairs])
    top = float(np.percentile(occ[occ > 0], 99)) if (occ > 0).any() else 1.0
    lim = float(np.percentile(np.abs(diff[diff != 0]), 99)) if (diff != 0).any() else 1.0
    return top, lim


def _arrows(heading: np.ndarray) -> np.ndarray:
    """The cells walked through most, one arrow per cell, as a mask."""
    mag = np.hypot(heading[..., 0], heading[..., 1])
    if not (mag > 0).any():
        return np.zeros(mag.shape, dtype=bool)
    return (mag > 0) & (mag >= np.percentile(mag[mag > 0], ARROW_PERCENTILE))


def map_pairs(cells: pl.DataFrame) -> pl.DataFrame:
    """The drawn flow maps of the flow cells of a benchmark (MAP_PAIR_SCHEMA), empty without an edit and its base run."""
    pairs = _pairs(cells) if len(cells) else []
    if not pairs:
        return pl.DataFrame(schema=MAP_PAIR_SCHEMA)
    top, lim = _scales(pairs)
    frames = []
    for rank, pair in enumerate(pairs):
        r = pair.raster
        ref_arrow, edit_arrow = _arrows(pair.ref_heading), _arrows(pair.edit_heading)
        drawn = (pair.ref != 0) | (pair.edit != 0) | ref_arrow | edit_arrow
        i, j = np.nonzero(drawn)
        frames.append(
            pl.DataFrame(
                {
                    "i": i,
                    "j": j,
                    "ref": pair.ref[i, j],
                    "edit": pair.edit[i, j],
                    "ref_hx": pair.ref_heading[i, j, 0],
                    "ref_hy": pair.ref_heading[i, j, 1],
                    "edit_hx": pair.edit_heading[i, j, 0],
                    "edit_hy": pair.edit_heading[i, j, 1],
                    "ref_arrow": ref_arrow[i, j],
                    "edit_arrow": edit_arrow[i, j],
                }
            ).with_columns(
                pl.lit(pair.planner).alias("planner"),
                pl.lit(pair.world).alias("flow_world"),
                pl.lit(pair.base).alias("flow_base"),
                pl.lit(", ".join(map(str, pair.seeds))).alias("seeds"),
                pl.lit(rank).alias("rank"),
                pl.lit(r.x0).alias("x0"),
                pl.lit(r.y0).alias("y0"),
                pl.lit(r.nx).alias("nx"),
                pl.lit(r.ny).alias("ny"),
                pl.lit(r.cell).alias("cell_m"),
                pl.lit(top).alias("occ_top"),
                pl.lit(lim).alias("diff_lim"),
            )
        )
    return pl.concat(frames).select([pl.col(c).cast(t) for c, t in MAP_PAIR_SCHEMA.items()])


__all__ = ["ARROW_PERCENTILE", "GAMMA", "MAP_PAIR_SCHEMA", "Raster", "map_pairs"]
