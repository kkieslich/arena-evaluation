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
# low occupancy stays visible next to a queue: its colour follows value ** GAMMA (the difference stays linear);
# `processing.flow.plots.GAMMA`, which the drawn maps are prepared with
GAMMA = 0.45


@dataclasses.dataclass(frozen=True)
class _Pair:
    """An edit and its base world run by one contestant, each the mean of the seeds both ran, as `flow_map_pairs` holds it."""

    planner: str
    world: str
    base: str
    seeds: str
    x0: float
    y0: float
    cell: float
    ref: np.ndarray
    edit: np.ndarray
    ref_heading: np.ndarray
    edit_heading: np.ndarray
    ref_arrows: np.ndarray
    edit_arrows: np.ndarray
    top: float
    lim: float

    @property
    def nx(self) -> int:
        return self.ref.shape[1]

    @property
    def ny(self) -> int:
        return self.ref.shape[0]

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + np.arange(self.nx) * self.cell

    @property
    def ys(self) -> np.ndarray:
        return self.y0 + np.arange(self.ny) * self.cell

    @property
    def edit_name(self) -> str:
        return self.world.removeprefix(f"{self.base}__")

    @property
    def diff(self) -> np.ndarray:
        return self.edit - self.ref

    def title(self, many_planners: bool) -> str:
        who = f" ({self.planner})" if many_planners else ""
        return f"{self.edit_name}{who} against {self.base}, mean of seeds {self.seeds}"

    def arrows(self, mask: np.ndarray, heading: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, direction in rad) of the arrow cells of one panel."""
        i, j = np.nonzero(mask)
        return self.xs[j], self.ys[i], np.arctan2(heading[i, j, 1], heading[i, j, 0])


_PAIR_COLUMNS = {"planner", "flow_world", "flow_base", "seeds", "rank", "x0", "y0", "nx", "ny", "cell_m", "i", "j", "ref", "edit", "ref_hx", "ref_hy", "edit_hx", "edit_hy", "ref_arrow", "edit_arrow", "occ_top", "diff_lim"}


def _pairs(df: pl.DataFrame) -> list[_Pair]:
    """The drawn maps of the `flow_map_pairs` data source, in its rank order (the largest change first per contestant)."""
    if df.is_empty() or not _PAIR_COLUMNS <= set(df.columns):
        return []
    out = []
    for frame in sorted(df.partition_by("planner", "rank", maintain_order=True), key=lambda f: (str(f["planner"][0]), int(f["rank"][0]))):
        first = frame.row(0, named=True)
        ny, nx = int(first["ny"]), int(first["nx"])
        i, j = frame["i"].to_numpy(), frame["j"].to_numpy()

        def grid(*columns: str) -> np.ndarray:
            shape = (ny, nx, len(columns)) if len(columns) > 1 else (ny, nx)
            out = np.zeros(shape, dtype=bool if frame.schema[columns[0]] == pl.Boolean else float)
            if len(columns) > 1:
                for k, column in enumerate(columns):
                    out[i, j, k] = frame[column].to_numpy()
            else:
                out[i, j] = frame[columns[0]].to_numpy()
            return out

        out.append(
            _Pair(
                str(first["planner"]), str(first["flow_world"]), str(first["flow_base"]), str(first["seeds"]), float(first["x0"]), float(first["y0"]), float(first["cell_m"]),
                grid("ref"), grid("edit"), grid("ref_hx", "ref_hy"), grid("edit_hx", "edit_hy"), grid("ref_arrow"), grid("edit_arrow"), float(first["occ_top"]), float(first["diff_lim"]),
            )
        )
    return out


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
    """Per edit: its base world's flow, the edit's, and the difference, from the `flow_map_pairs` data source (one figure per edit)."""

    PLOT_TYPE = "flow_map"

    def render_plotly(self, df: pl.DataFrame) -> list[str] | None:
        pairs = _pairs(df)
        if not pairs:
            return None
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        many = len({p.planner for p in pairs}) > 1
        walls = {base: _walls(base, self.run_dir) for base in {p.base for p in pairs}}
        out = []
        for pair in pairs:
            top, lim = pair.top, pair.lim
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Base world", "Edit", "Edit minus base"), horizontal_spacing=0.04)
            xs, ys = pair.xs, pair.ys
            panels = ((pair.ref, (pair.ref_arrows, pair.ref_heading), "coloraxis"), (pair.edit, (pair.edit_arrows, pair.edit_heading), "coloraxis"), (pair.diff, None, "coloraxis2"))
            for col, (values, heading, axis) in enumerate(panels, start=1):
                z = np.where(values != 0, values, np.nan)
                fig.add_trace(go.Heatmap(z=z, x=xs, y=ys, coloraxis=axis, hovertemplate="x %{x:.2f} m, y %{y:.2f} m<br>%{z:.1f} ped·s/m²<extra></extra>"), row=1, col=col)
                if walls[pair.base]:
                    wx = [x for poly in walls[pair.base] for x in [*(p[0] for p in poly), poly[0][0], None]]
                    wy = [y for poly in walls[pair.base] for y in [*(p[1] for p in poly), poly[0][1], None]]
                    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", line={"color": "#555", "width": 1}, hoverinfo="skip", showlegend=False), row=1, col=col)
                if heading is not None:
                    arrow_x, arrow_y, theta = pair.arrows(*heading)
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

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        pairs = _pairs(df)
        if not pairs:
            return
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm, TwoSlopeNorm

        many = len({p.planner for p in pairs}) > 1
        walls = {base: _walls(base, self.run_dir) for base in {p.base for p in pairs}}
        for n, pair in enumerate(pairs):
            top, lim = pair.top, pair.lim
            extent = (pair.x0 - pair.cell / 2, pair.x0 + (pair.nx - 0.5) * pair.cell, pair.y0 - pair.cell / 2, pair.y0 + (pair.ny - 0.5) * pair.cell)
            fig, axes = plt.subplots(1, 3, figsize=(19, 8), constrained_layout=True)
            for ax, values, heading, title in ((axes[0], pair.ref, (pair.ref_arrows, pair.ref_heading), "Base world"), (axes[1], pair.edit, (pair.edit_arrows, pair.edit_heading), "Edit")):
                im = ax.imshow(np.ma.masked_equal(values, 0), origin="lower", extent=extent, cmap="magma", norm=PowerNorm(GAMMA, vmin=0, vmax=top), interpolation="nearest")
                arrow_x, arrow_y, theta = pair.arrows(*heading)
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
            # every edit's figure beside the plot's, the largest change also as the plot's own file
            out_path = pathlib.Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path.with_name(f"{out_path.stem}_{suffix}{out_path.suffix}"), dpi=110)
            if n == 0:
                fig.savefig(out_path, dpi=110)
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
    by_edit: dict[tuple[str, str, str], list[dict]] = {}
    for row in df.filter(pl.col("flow_jsd_pooled").is_not_null()).iter_rows(named=True):
        by_edit.setdefault((str(row["planner"]), str(row["flow_world"]), str(row["flow_base"])), []).append(row)
    out = []
    for (planner, world, base), rows in by_edit.items():
        seeds = sorted((int(r["flow_seed"]), float(r["flow_jsd"])) for r in rows if r["flow_jsd"] is not None)
        out.append(_Score(planner, world.removeprefix(f"{base}__"), float(rows[0]["flow_jsd_pooled"]), seeds))
    return sorted(out, key=lambda s: (-s.pooled, s.edit, s.planner))


class FlowJsdRenderer(BasePlotRenderer):
    """Flow change per edit on a log axis: a bar pooled over seeds, a dot per seed."""

    PLOT_TYPE = "flow_jsd"

    def render_plotly(self, df: pl.DataFrame) -> str | None:
        scores = _scores(df)
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

    def render_static(self, df: pl.DataFrame, out_path: pathlib.Path) -> None:
        scores = _scores(df)
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
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


__all__ = ["FlowJsdRenderer", "FlowMapRenderer"]
