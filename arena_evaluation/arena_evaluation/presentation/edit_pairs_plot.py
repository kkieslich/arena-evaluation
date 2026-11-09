"""One figure of `arena evaluation edits`: per planner, every edited world's difference to its base world at each seed."""

from __future__ import annotations

import pathlib

import polars as pl

from arena_evaluation.processing.edit_pairs import CONTROL, EDIT, short_world

# Categorical slots 1 and 2 of the reference chart palette (blue, orange), a validated adjacent pair on the light surface;
# role is also told by marker shape and by the x labels, never by color alone.
_ROLE_STYLE = {EDIT: ("#2a78d6", "o", "function-carrying edit"), CONTROL: ("#eb6834", "s", "geometry-only control")}
_OTHER_STYLE = ("#52514e", "D", "edited world")
_SURFACE, _INK, _INK_2, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

_UNITS = {"personal_space_time": "s", "personal_space_intrusion_integral": "m·s", "time_in_intimate_zone": "s", "path_length": "m"}
_TITLES = {
    "personal_space_time": "Time with a person inside the robot's personal space (1.2 m)",
    "personal_space_intrusion_integral": "Personal-space intrusion integral",
    "time_in_intimate_zone": "Time with a person inside the intimate zone (0.45 m)",
    "psi_personal_events": "Personal-space encounters",
    "ped_robot_contacts": "Robot-pedestrian contacts",
    "path_length": "Path length",
}


def _p(value: float | None) -> str:
    if value is None:
        return "p = -"
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def plot_edit_pairs(pairs: pl.DataFrame, tests: pl.DataFrame, measure: str, out: pathlib.Path, planners: list[str] | None = None) -> pathlib.Path:
    """Save the figure of `measure` to `out`: one panel per planner on a shared y axis (in the order of `planners`, then
    the rest), the edited worlds along x (edit first), each seed's difference as a point, the mean with its 95 % t
    interval, and the t-test's p and d_z above."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = pairs.filter(pl.col("measure") == measure)
    stats = tests.filter(pl.col("measure") == measure)
    present = data["planner"].unique(maintain_order=True).to_list()
    planners = [p for p in planners or [] if p in present] + [p for p in present if p not in (planners or [])]
    worlds = data.select("world", "role").unique(maintain_order=True)
    order = {EDIT: 0, CONTROL: 1}
    worlds = sorted(worlds.iter_rows(), key=lambda wr: (order.get(wr[1], 2), wr[0]))
    unit = _UNITS.get(measure, "")

    fig, axes = plt.subplots(1, len(planners), figsize=(3.2 * len(planners) + 0.6, 4.4), sharey=True, squeeze=False, facecolor=_SURFACE)
    for ax, planner in zip(axes[0], planners, strict=True):
        ax.set_facecolor(_SURFACE)
        ax.axhline(0.0, color=_INK_2, linewidth=1.0, zorder=1)
        ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
        for x, (world, role) in enumerate(worlds):
            color, marker, _ = _ROLE_STYLE.get(role, _OTHER_STYLE)
            block = data.filter((pl.col("planner") == planner) & (pl.col("world") == world)).sort("seed")
            diffs = block["diff"].to_list()
            offsets = [x + (i - (len(diffs) - 1) / 2) * 0.07 for i in range(len(diffs))]
            ax.scatter(offsets, diffs, s=36, color=color, marker=marker, edgecolors=_SURFACE, linewidths=1.5, zorder=3)
            row = stats.filter((pl.col("planner") == planner) & (pl.col("world") == world))
            if len(row) == 0:
                continue
            r = row.row(0, named=True)
            if r["ci_low"] is not None:
                ax.plot([x + 0.28, x + 0.28], [r["ci_low"], r["ci_high"]], color=_INK, linewidth=2.0, solid_capstyle="round", zorder=2)
            if r["mean_diff"] is not None:
                ax.plot([x + 0.2, x + 0.36], [r["mean_diff"], r["mean_diff"]], color=_INK, linewidth=2.0, solid_capstyle="round", zorder=2)
            d_z = "-" if r["d_z"] is None else f"{r['d_z']:.2f}"
            ax.annotate(f"{_p(r['p_t'])}\nd_z = {d_z}", xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -4), textcoords="offset points", ha="center", va="top", fontsize=7.5, color=_INK_2)
        ax.set_xticks(range(len(worlds)), [short_world(w).replace("_", " ") for w, _ in worlds], rotation=20, ha="right", fontsize=8, color=_INK_2)
        ax.set_xlim(-0.6, len(worlds) - 0.4)
        ax.set_title(planner, fontsize=10, color=_INK)
        ax.tick_params(axis="y", labelsize=8, colors=_INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
    low, high = axes[0][0].get_ylim()
    axes[0][0].set_ylim(low, high + 0.22 * (high - low))
    axes[0][0].set_ylabel(f"edited minus base{f' ({unit})' if unit else ''}", fontsize=9, color=_INK)
    roles = [role for role in (EDIT, CONTROL) if any(r == role for _, r in worlds)]
    handles = [Line2D([], [], linestyle="", marker=_ROLE_STYLE[r][1], color=_ROLE_STYLE[r][0], markersize=7, label=_ROLE_STYLE[r][2]) for r in roles]
    handles.append(Line2D([], [], color=_INK, linewidth=2.0, label="mean, 95 % t interval"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8, labelcolor=_INK_2)
    seeds = data["seed"].n_unique()
    fig.suptitle(f"{_TITLES.get(measure, measure)}: edited world minus base world, paired by seed (n = {seeds})", fontsize=10.5, color=_INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=_SURFACE)
    plt.close(fig)
    return out


__all__ = ["plot_edit_pairs"]
