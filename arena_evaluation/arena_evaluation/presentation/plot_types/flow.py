"""Flow plots of a `flowedit suite` benchmark: where pedestrians spend time in each edit against its base world, and how far each edit moved the flow."""

from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Callable

import numpy as np
import polars as pl

from .base import BasePlotRenderer

# a log axis cannot show 0 bits: smaller divergences are drawn here
JSD_FLOOR = 1e-4
# low occupancy stays visible next to a queue: its colour follows value ** GAMMA (the difference stays linear)
GAMMA = 0.45
# arrows mark the cells whose walking time is at or above this percentile of the map's
ARROW_PERCENTILE = 60


@dataclasses.dataclass(frozen=True)
class _Raster:
    """Cell centres of one base world's grid, covering every map drawn on it."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float

    @classmethod
    def over(cls, cells: pl.DataFrame) -> _Raster:
        """The raster spanning every cell of `cells` (FLOW_CELL_SCHEMA rows of one base world)."""
        cell = float(cells["cell_m"][0])
        x0, y0 = float(cells["x"].min()), float(cells["y"].min())
        return cls(x0, y0, round((float(cells["x"].max()) - x0) / cell) + 1, round((float(cells["y"].max()) - y0) / cell) + 1, cell)

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.cell

    @property
    def ys(self) -> np.ndarray:
        return self.y0 + np.arange(self.ny) * self.cell

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
    """An edit and its base world run by one contestant, each the mean of the seeds both ran."""

    planner: str
    world: str
    base: str
    seeds: list[int]
    raster: _Raster
    ref: np.ndarray
    edit: np.ndarray
    ref_heading: np.ndarray
    edit_heading: np.ndarray

    @property
    def edit_name(self) -> str:
        return self.world.removeprefix(f"{self.base}__")

    @property
    def diff(self) -> np.ndarray:
        return self.edit - self.ref

    def title(self, many_planners: bool) -> str:
        who = f" ({self.planner})" if many_planners else ""
        return f"{self.edit_name}{who} against {self.base}, mean of seeds {', '.join(map(str, self.seeds))}"


def _mean(runs: list[pl.DataFrame], fn: Callable[[pl.DataFrame], np.ndarray]) -> np.ndarray:
    return np.mean([fn(run) for run in runs], axis=0)


def _pairs(cells: pl.DataFrame) -> list[_Pair]:
    """Every edit with its base world per contestant, over the seeds both ran, the largest change first."""
    out = []
    rasters = {base: _Raster.over(group) for (base,), group in cells.group_by("flow_base")}
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
    return sorted(out, key=lambda p: (p.planner, -float(np.abs(p.diff).sum()), p.world))


def _scales(pairs: list[_Pair]) -> tuple[float, float]:
    """Top of the occupancy scale and half-range of the difference scale, shared by every pair so maps compare."""
    occ = np.concatenate([np.concatenate([p.ref.ravel(), p.edit.ravel()]) for p in pairs])
    diff = np.concatenate([p.diff.ravel() for p in pairs])
    top = float(np.percentile(occ[occ > 0], 99)) if (occ > 0).any() else 1.0
    lim = float(np.percentile(np.abs(diff[diff != 0]), 99)) if (diff != 0).any() else 1.0
    return top, lim


def _arrows(raster: _Raster, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, direction in rad) of the cells walked through most, one arrow per cell."""
    mag = np.hypot(heading[..., 0], heading[..., 1])
    if not (mag > 0).any():
        return np.empty(0), np.empty(0), np.empty(0)
    i, j = np.nonzero((mag > 0) & (mag >= np.percentile(mag[mag > 0], ARROW_PERCENTILE)))
    return raster.xs[j], raster.ys[i], np.arctan2(heading[i, j, 1], heading[i, j, 0])


def _walls(world: str, run_dir: pathlib.Path | None) -> list[list[tuple[float, float]]]:
    """Zone outlines of `world`, from the run's snapshot first; none when the world is not available."""
    from arena_evaluation.processing.flow import zone_polygons
    from arena_evaluation.processing.metrics.edge_case.provenance import world_view

    view = world_view(world, run_dir)
    polygons = None if view is None else zone_polygons(view)
    return polygons or []


def _plotly_scale(name: str, gamma: float = 1.0) -> list[list]:
    """A plotly colorscale of matplotlib's `name` whose colour follows value ** gamma."""
    import matplotlib
    from matplotlib.colors import to_hex

    cmap = matplotlib.colormaps[name]
    stops = np.linspace(0.0, 1.0, 21)
    return [[float(p), to_hex(cmap(float(p**gamma)))] for p in stops]


class FlowMapRenderer(BasePlotRenderer):
    """Per edit: its base world's flow, the edit's, and the difference, from the `flow_cells` data source."""

    PLOT_TYPE = "flow_map"

    def render_plotly(self, df: pl.DataFrame) -> list[str] | None:
        pairs = _pairs(self._apply_filters(df))
        if not pairs:
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        top, lim = _scales(pairs)
        many = len({p.planner for p in pairs}) > 1
        walls = {base: _walls(base, self.run_dir) for base in {p.base for p in pairs}}
        out = []
        for pair in pairs:
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Base world", "Edit", "Edit minus base"), horizontal_spacing=0.04)
            xs, ys = pair.raster.xs, pair.raster.ys
            panels = ((pair.ref, pair.ref_heading, "coloraxis"), (pair.edit, pair.edit_heading, "coloraxis"), (pair.diff, None, "coloraxis2"))
            for col, (values, heading, axis) in enumerate(panels, start=1):
                z = np.where(values != 0, values, np.nan)
                fig.add_trace(go.Heatmap(z=z, x=xs, y=ys, coloraxis=axis, hovertemplate="x %{x:.2f} m, y %{y:.2f} m<br>%{z:.1f} ped·s/m²<extra></extra>"), row=1, col=col)
                if walls[pair.base]:
                    wx = [x for poly in walls[pair.base] for x in [*(p[0] for p in poly), poly[0][0], None]]
                    wy = [y for poly in walls[pair.base] for y in [*(p[1] for p in poly), poly[0][1], None]]
                    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line={"color": "#555", "width": 1}, hoverinfo="skip", showlegend=False), row=1, col=col)
                if heading is not None:
                    arrow_x, arrow_y, theta = _arrows(pair.raster, heading)
                    fig.add_trace(go.Scatter(x=arrow_x, y=arrow_y, mode="markers", marker={"symbol": "arrow", "size": 7, "angle": (90 - np.degrees(theta)).tolist(), "color": "white", "line": {"color": "black", "width": 0.5}}, hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.update_xaxes(title_text="x [m]", constrain="domain")
            fig.update_yaxes(constrain="domain")
            fig.update_yaxes(title_text="y [m]", scaleanchor="x", scaleratio=1, row=1, col=1)
            for col in (2, 3):
                fig.update_xaxes(matches="x", row=1, col=col)
                fig.update_yaxes(matches="y", row=1, col=col)
            bar = {"orientation": "h", "y": -0.2, "yanchor": "top", "thickness": 12, "title": {"side": "top"}}
            fig.update_layout(
                title=pair.title(many),
                template="plotly_white",
                height=640,
                margin={"b": 130},
                plot_bgcolor="#eeeeee",
                coloraxis={"colorscale": _plotly_scale("magma", GAMMA), "cmin": 0, "cmax": top, "colorbar": {**bar, "x": 0.33, "len": 0.55, "title": {"text": "pedestrian-seconds per m² per run", "side": "top"}}},
                coloraxis2={"colorscale": _plotly_scale("RdBu_r"), "cmin": -lim, "cmax": lim, "colorbar": {**bar, "x": 0.85, "len": 0.28, "title": {"text": "edit minus base", "side": "top"}}},
            )
            out.append(fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
        return out

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pairs = _pairs(self._apply_filters(df))
        if not pairs:
            return
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm, TwoSlopeNorm

        top, lim = _scales(pairs)
        many = len({p.planner for p in pairs}) > 1
        walls = {base: _walls(base, self.run_dir) for base in {p.base for p in pairs}}
        for pair in pairs:
            r = pair.raster
            extent = (r.x0 - r.cell / 2, r.x0 + (r.nx - 0.5) * r.cell, r.y0 - r.cell / 2, r.y0 + (r.ny - 0.5) * r.cell)
            fig, axes = plt.subplots(1, 3, figsize=(19, 8), constrained_layout=True)
            for ax, values, heading, title in ((axes[0], pair.ref, pair.ref_heading, "Base world"), (axes[1], pair.edit, pair.edit_heading, "Edit")):
                im = ax.imshow(np.ma.masked_equal(values, 0), origin="lower", extent=extent, cmap="magma", norm=PowerNorm(GAMMA, vmin=0, vmax=top), interpolation="nearest")
                arrow_x, arrow_y, theta = _arrows(r, heading)
                if len(theta):
                    ax.quiver(arrow_x, arrow_y, np.cos(theta), np.sin(theta), color="white", scale=45, width=0.0035, headwidth=4, zorder=4)
                ax.set_title(title)
            fig.colorbar(im, ax=axes[:2], shrink=0.75, label="pedestrian-seconds per m² per run")
            im = axes[2].imshow(np.ma.masked_equal(pair.diff, 0), origin="lower", extent=extent, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
            axes[2].set_title("Edit minus base (red: more time, blue: less)")
            fig.colorbar(im, ax=axes[2], shrink=0.75, label="Δ pedestrian-seconds per m²")
            for ax in axes:
                for poly in walls[pair.base]:
                    ax.plot(*np.vstack([poly, poly[:1]]).T, color="0.35", lw=0.6)
                ax.set_aspect("equal")
                ax.set_facecolor("0.93")
                ax.set_xlabel("x [m]")
            axes[0].set_ylabel("y [m]")
            fig.suptitle(pair.title(many), fontsize=14)
            suffix = f"{pair.world}__{pair.planner}" if many else pair.world
            fig.savefig(out_path.with_name(f"{out_path.stem}_{suffix}.png"), dpi=110)
            plt.close(fig)


@dataclasses.dataclass(frozen=True)
class _Score:
    """One edit's flow change for one contestant: pooled over seeds, and per seed against the base run at that seed."""

    planner: str
    edit: str
    pooled: float
    seeds: list[tuple[int, float]]


def _scores(df: pl.DataFrame) -> list[_Score]:
    """Every edit with a pooled score, largest change first."""
    need = {"planner", "flow_world", "flow_base", "flow_seed", "flow_jsd", "flow_jsd_pooled"}
    if not need <= set(df.columns):
        return []
    rows = df.filter(pl.col("flow_jsd_pooled").is_not_null())
    out = []
    for (planner, world, base), group in rows.group_by("planner", "flow_world", "flow_base"):
        seeds = sorted((int(s), float(j)) for s, j in group.select("flow_seed", "flow_jsd").iter_rows() if j is not None)
        out.append(_Score(str(planner), str(world).removeprefix(f"{base}__"), float(group["flow_jsd_pooled"][0]), seeds))
    return sorted(out, key=lambda s: (-s.pooled, s.edit, s.planner))


class FlowJsdRenderer(BasePlotRenderer):
    """Flow change per edit on a log axis: a bar pooled over seeds, a dot per seed."""

    PLOT_TYPE = "flow_jsd"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        scores = _scores(self._apply_filters(df))
        if not scores:
            return None
        import plotly.graph_objects as go

        fig = go.Figure()
        planners = sorted({s.planner for s in scores})
        order = list(dict.fromkeys(s.edit for s in scores))[::-1]
        for planner in planners:
            mine = [s for s in scores if s.planner == planner]
            fig.add_trace(go.Bar(x=[max(s.pooled, JSD_FLOOR) for s in mine], y=[s.edit for s in mine], orientation="h", name=f"{planner}: pooled", text=[f"{s.pooled:.3g}" for s in mine], textposition="inside", insidetextanchor="start", hovertemplate="%{y}<br>pooled %{text} bits<extra></extra>"))
            fig.add_trace(
                go.Scatter(
                    x=[max(j, JSD_FLOOR) for s in mine for _, j in s.seeds],
                    y=[s.edit for s in mine for _ in s.seeds],
                    mode="markers",
                    marker={"color": "black", "size": 7},
                    name=f"{planner}: per seed",
                    text=[f"seed {seed}: {j:.3g} bits" for s in mine for seed, j in s.seeds],
                    hovertemplate="%{y}<br>%{text}<extra></extra>",
                )
            )
        fig.update_layout(
            template="plotly_white",
            barmode="group",
            height=160 + 50 * len(order) * len(planners),
            xaxis={"type": "log", "range": [math.log10(JSD_FLOOR), 0.2], "title": "Flow JSD against the base world [bits, log scale]"},
            yaxis={"categoryorder": "array", "categoryarray": order},
            legend={"orientation": "h", "yanchor": "top", "y": -0.25, "xanchor": "center", "x": 0.5},
        )
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})

    def render_seaborn(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        scores = _scores(self._apply_filters(df))
        if not scores:
            return
        import matplotlib.pyplot as plt

        planners = sorted({s.planner for s in scores})
        order = list(dict.fromkeys(s.edit for s in scores))
        height = 0.8 / len(planners)
        fig, ax = plt.subplots(figsize=(10, 1.2 + 0.6 * len(order) * len(planners)), constrained_layout=True)
        for k, planner in enumerate(planners):
            for n, s in enumerate(s for s in scores if s.planner == planner):
                y = len(order) - 1 - order.index(s.edit) + (k - (len(planners) - 1) / 2) * height
                value = max(s.pooled, JSD_FLOOR)
                ax.barh(y, value - JSD_FLOOR, left=JSD_FLOOR, height=height * 0.9, color=f"C{k}", label=planner if n == 0 else None)
                ax.scatter([max(j, JSD_FLOOR) for _, j in s.seeds], [y] * len(s.seeds), color="black", s=20, zorder=3)
                ax.text(max([value, *(j for _, j in s.seeds)]) * 1.3, y, f"{s.pooled:.3g}", va="center")
        ax.set_xscale("log")
        ax.set_xlim(JSD_FLOOR, 3)
        ax.set_yticks(range(len(order)), order[::-1])
        ax.set_xlabel("Flow JSD against the base world [bits, log scale]")
        ax.set_title(self.spec.title)
        if len(planners) > 1:
            ax.legend(loc="lower right")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


__all__ = ["FlowJsdRenderer", "FlowMapRenderer"]
