"""The flow grid: square cells over a world's rooms, each holding the pedestrian-seconds per walking direction."""

from __future__ import annotations

import dataclasses
import math
import typing
from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl

if typing.TYPE_CHECKING:
    import shapely
    from arena_simulation_setup.tree.World import WorldView

CELL_M = 0.5
HEADINGS = 8
STANDING_MPS = 0.2
# a frame lasts until the pedestrian's next one, at most this many of its median steps (a gap in the recording)
MAX_STEP_FACTOR = 4.0


@dataclasses.dataclass(frozen=True)
class FlowGrid:
    """Square cells from (x0, y0), each with `headings` heading bins and a last bin for standing (speed under `standing`)."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float = CELL_M
    headings: int = HEADINGS
    standing: float = STANDING_MPS
    rooms: shapely.Geometry | None = dataclasses.field(default=None, compare=False, repr=False)

    @property
    def bins(self) -> int:
        return self.headings + 1

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ny, self.nx, self.bins)

    def empty(self) -> np.ndarray:
        return np.zeros(self.shape)

    @classmethod
    def from_polygons(cls, polygons: Iterable[Sequence[tuple[float, float]]], cell: float = CELL_M) -> FlowGrid | None:
        """The grid over the union of `polygons` (world frame), None without one of at least three corners."""
        import shapely

        shapes = [shapely.Polygon(p) for p in polygons if len(p) >= 3]
        shapes = [s if s.is_valid else s.buffer(0) for s in shapes]
        if not shapes:
            return None
        rooms = shapely.union_all(shapes)
        if rooms.is_empty:
            return None
        x0, y0, x1, y1 = rooms.bounds
        shapely.prepare(rooms)
        return cls(x0=float(x0), y0=float(y0), nx=max(1, math.ceil((x1 - x0) / cell)), ny=max(1, math.ceil((y1 - y0) / cell)), cell=cell, rooms=rooms)

    @classmethod
    def from_world(cls, view: WorldView, cell: float = CELL_M) -> FlowGrid | None:
        """The grid over every zone of a world (`arena_simulation_setup` WorldView), each moved by its level's origin."""
        try:
            world = view.load(validate=False)
        except (FileNotFoundError, ValueError):
            return None
        origins = view.level_origins() or {}
        polygons = []
        for level_id, level in world.levels.items():
            ox, oy = origins.get(level_id, (0.0, 0.0))
            polygons.extend([(c.x + ox, c.y + oy) for c in zone.polygon] for zone in level.zones)
        return cls.from_polygons(polygons, cell)

    def counts(self, tracks: pl.DataFrame) -> np.ndarray:
        """Pedestrian-seconds per cell and bin of `tracks` (`ped_track_store.track_table`), inside the rooms only."""
        h = self.empty()
        if tracks is None or len(tracks) == 0:
            return h
        df = tracks.select("key", "t", "x", "y", "vx", "vy").sort("key", "t")
        step = (pl.col("t").shift(-1) - pl.col("t")).over("key")
        df = df.with_columns(step.alias("dt"))
        df = df.with_columns(pl.col("dt").median().over("key").alias("median_dt"))
        df = df.with_columns(pl.when(pl.col("dt").is_null()).then(pl.col("median_dt")).otherwise(pl.min_horizontal("dt", MAX_STEP_FACTOR * pl.col("median_dt"))).fill_null(0.0).alias("w"))
        # velocities the recording lacks come from the positions: the step to the next frame, else the one before
        derived_x = ((pl.col("x").shift(-1) - pl.col("x")) / pl.col("dt")).over("key")
        derived_y = ((pl.col("y").shift(-1) - pl.col("y")) / pl.col("dt")).over("key")
        df = df.with_columns(
            pl.when(pl.col("vx").is_nan() | pl.col("vx").is_null()).then(derived_x).otherwise(pl.col("vx")).alias("vx"),
            pl.when(pl.col("vy").is_nan() | pl.col("vy").is_null()).then(derived_y).otherwise(pl.col("vy")).alias("vy"),
        ).with_columns(pl.col("vx").forward_fill().over("key").fill_null(0.0), pl.col("vy").forward_fill().over("key").fill_null(0.0))
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        vx, vy, w = df["vx"].to_numpy(), df["vy"].to_numpy(), df["w"].to_numpy()
        i = np.floor((y - self.y0) / self.cell).astype(np.int64)
        j = np.floor((x - self.x0) / self.cell).astype(np.int64)
        keep = (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx) & (w > 0)
        if self.rooms is not None:
            import shapely

            keep &= shapely.contains_xy(self.rooms, x, y)
        speed = np.hypot(vx, vy)
        heading = np.floor((np.arctan2(vy, vx) + math.pi) / (2 * math.pi) * self.headings).astype(np.int64) % self.headings
        b = np.where(speed < self.standing, self.headings, heading)
        np.add.at(h, (i[keep], j[keep], b[keep]), w[keep])
        return h


__all__ = ["CELL_M", "HEADINGS", "STANDING_MPS", "FlowGrid"]
