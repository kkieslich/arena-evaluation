"""Each pedestrian measure per group of agent types across the runs of a benchmark, per seed and as the mean over seeds.

The per-pedestrian measures are the columns of each episode's `metrics/fact_pedestrians.parquet`, a pedestrian's agent
type comes from the benchmark's `dim_pedestrians.parquet` (`adult` without one), its group from `agent_group`. The
robot contacts come from `metrics/fact_contacts.parquet` (the contacts of a robot whose other body is a pedestrian):
`ped_robot_contacts` per robot, `ped_fleet_contacts` summed over the env's robots and `ped_contact_free_rate`, the share
of pedestrians no robot touched. The paired measures by group come from `ped_paired.parquet` (`paired_peds`).
`arena evaluation groups` prints the mean over seeds and writes `ped_groups.parquet` in the benchmark dir.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import typing

import polars as pl

from arena_evaluation.processing.benchmark_runs import EpisodeRun, episode_runs
from arena_evaluation.processing.edit_pairs import parse_stage
from arena_evaluation.processing.metrics.social.paired_peds import PAIRED_FILE, PAIRED_MEASURES
from arena_evaluation.processing.metrics.social.pedestrian_groups import FLEET_MEASURES, PED_MEASURES, agent_group, worst_group
from arena_evaluation.processing.ped_track_store import DEFAULT_AGENT_TYPE, strip_env

PED_GROUPS_FILE = "ped_groups.parquet"
DIM_PEDESTRIANS_FILE = "dim_pedestrians.parquet"
PEDESTRIAN_KIND = "pedestrian"

# The direction that is better of every measure scored by group, the other end names the worst group. A measure without
# one (`ped_paired_dropped`, a count) names no worst group.
GROUP_DIRECTIONS = {**PED_MEASURES, **FLEET_MEASURES, **PAIRED_MEASURES}

# `level` is `seed` for a row of one seed and `mean` for the equal-weight mean over the seeds, whose `seed` is null.
PED_GROUPS_SCHEMA = {
    "planner": pl.String,
    "stage": pl.String,
    "level": pl.String,
    "seed": pl.Int64,
    "measure": pl.String,
    "group": pl.String,
    "value": pl.Float64,
    "peds": pl.Int64,
    "runs": pl.Int64,
    "seeds": pl.Int64,
    "worst_group": pl.String,
    "worst": pl.Boolean,
}

_BY_GROUP = "_by_group"
_RUN = ["planner", "stage", "seed", "episode", "env"]
_PARTITION = ["planner", "stage", "level", "seed", "measure"]
#: One group's value of one measure in one run (or one robot of it): the mean over the group's pedestrians and their number
_LONG_SCHEMA = {"planner": pl.String, "stage": pl.String, "episode": pl.Int64, "env": pl.String, "measure": pl.String, "group": pl.String, "value": pl.Float64, "peds": pl.Int64}
_DIGITS = re.compile(r"\d+")


def _read(path: pathlib.Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.is_file() else None


def _agent_types(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """episode_id, env_id, ped_id and agent_type of every pedestrian of the benchmark's `dim_pedestrians.parquet`."""
    schema = {"episode_id": pl.Int64, "env_id": pl.String, "ped_id": pl.String, "agent_type": pl.String}
    dims = _read(benchmark_dir / DIM_PEDESTRIANS_FILE)
    if dims is None or "agent_type" not in dims.columns:
        return pl.DataFrame(schema=schema)
    return dims.select([pl.col(c).cast(t) for c, t in schema.items()]).unique(subset=["episode_id", "env_id", "ped_id"], keep="first")


def _pedestrians(run: EpisodeRun, types: pl.DataFrame) -> pl.DataFrame | None:
    """The run's pedestrian fact rows with each pedestrian's group, None without any."""
    facts = _read(run.episode_dir / "metrics" / "fact_pedestrians.parquet")
    if facts is None or "ped_id" not in facts.columns:
        return None
    rows = facts.filter(pl.col("env_id") == run.env) if "env_id" in facts.columns else facts
    rows = rows.unique(subset=["ped_id"], keep="first", maintain_order=True)
    rows = rows.join(types.filter((pl.col("episode_id") == run.episode) & (pl.col("env_id") == run.env)).select("ped_id", "agent_type"), on="ped_id", how="left")
    agent_type = pl.col("agent_type").fill_null("")
    rows = rows.with_columns(pl.when(agent_type == "").then(pl.lit(DEFAULT_AGENT_TYPE)).otherwise(agent_type).alias("agent_type"))
    return rows.with_columns(pl.col("agent_type").map_elements(agent_group, return_dtype=pl.String).alias("group"))


def _group_means(run: EpisodeRun, peds: pl.DataFrame, measure: str, values: pl.Expr) -> pl.DataFrame:
    """One `_LONG_SCHEMA` row per group: the mean of `values` over the group's pedestrians with a finite one."""
    scored = peds.select("group", values.cast(pl.Float64).alias("value")).filter(pl.col("value").is_not_null() & pl.col("value").is_finite())
    out = scored.group_by("group").agg(pl.col("value").mean(), pl.len().cast(pl.Int64).alias("peds"))
    return out.with_columns(pl.lit(run.planner).alias("planner"), pl.lit(run.stage).alias("stage"), pl.lit(run.episode, pl.Int64).alias("episode"), pl.lit(run.env).alias("env"), pl.lit(measure).alias("measure")).select(list(_LONG_SCHEMA)).cast(_LONG_SCHEMA)


def _contact_ped(obstacle: str, peds: pl.DataFrame) -> str | None:
    """The ped_id of the pedestrian a contact's obstacle id names: its id or name (without the env prefix), else the
    pedestrian whose id ends in the obstacle's last number; None when none matches."""
    name = strip_env("/".join(part for part in str(obstacle).split("/") if part))
    for ped_id, ped_name in peds.select("ped_id", pl.col("ped_name").fill_null("")).iter_rows():
        if name in (ped_id, strip_env(str(ped_name))):
            return ped_id
    digits = _DIGITS.findall(name)
    if digits:
        for ped_id in peds["ped_id"].to_list():
            found = _DIGITS.findall(str(ped_id))
            if found and found[-1] == digits[-1]:
                return ped_id
    return None


def _contacts(run: EpisodeRun, peds: pl.DataFrame) -> dict[str, dict[str, float]] | None:
    """Contacts of each robot of the run with each of its pedestrians (0 for every pedestrian the robot never touched),
    by robot; None without the run's contact fact rows."""
    facts = _read(run.episode_dir / "metrics" / "fact_contacts.parquet")
    if facts is None:
        return None
    contacts = {robot: dict.fromkeys(peds["ped_id"].to_list(), 0.0) for robot in run.robots}
    for robot, kind, obstacle in facts.select("robot_id", "collider_kind", "obstacle_id").iter_rows():
        if robot not in contacts or kind != PEDESTRIAN_KIND:
            continue
        ped = _contact_ped(str(obstacle or ""), peds)
        if ped is not None:
            contacts[robot][ped] += 1.0
    return contacts


def _paired_rows(benchmark_dir: pathlib.Path) -> list[pl.DataFrame]:
    """The paired measures by group of `ped_paired.parquet` as `_LONG_SCHEMA` rows."""
    paired = _read(benchmark_dir / PAIRED_FILE)
    if paired is None:
        return []
    frames = []
    for column in [c for c in paired.columns if c.endswith(_BY_GROUP)]:
        measure = column.removesuffix(_BY_GROUP)
        frames.append(
            paired.select("planner", "stage", "episode", "env", pl.lit(measure).alias("measure"), pl.col(column).alias("record"))
            .explode("record")
            .drop_nulls("record")
            .unnest("record")
            .select(list(_LONG_SCHEMA))
            .cast(_LONG_SCHEMA)
        )
    return frames


def group_rows(benchmark_dir: pathlib.Path) -> pl.DataFrame:
    """Every group's value of every measure in every contestant run of a processed benchmark dir (`_LONG_SCHEMA`):
    the per-pedestrian measures and `ped_fleet_contacts`, `ped_contact_free_rate` once per run, `ped_robot_contacts`
    once per robot of the run, and the paired measures. Reference runs are left out."""
    benchmark_dir = pathlib.Path(benchmark_dir)
    types = _agent_types(benchmark_dir)
    frames: list[pl.DataFrame] = []
    for run in episode_runs(benchmark_dir):
        if run.is_reference:
            continue
        peds = _pedestrians(run, types)
        if peds is None or len(peds) == 0:
            continue
        frames.extend(_group_means(run, peds, measure, pl.col(measure)) for measure in PED_MEASURES if measure in peds.columns)
        contacts = _contacts(run, peds)
        if contacts:
            for by_ped in contacts.values():
                frames.append(_group_means(run, peds.with_columns(pl.col("ped_id").replace_strict(by_ped, default=0.0, return_dtype=pl.Float64).alias("_c")), "ped_robot_contacts", pl.col("_c")))
            fleet = {ped: sum(robot[ped] for robot in contacts.values()) for ped in peds["ped_id"].to_list()}
            with_fleet = peds.with_columns(pl.col("ped_id").replace_strict(fleet, return_dtype=pl.Float64).alias("_f"))
            frames.append(_group_means(run, with_fleet, "ped_fleet_contacts", pl.col("_f")))
            frames.append(_group_means(run, with_fleet, "ped_contact_free_rate", (pl.col("_f") == 0).cast(pl.Float64)))
    frames.extend(_paired_rows(benchmark_dir))
    return pl.concat(frames) if frames else pl.DataFrame(schema=_LONG_SCHEMA)


def ped_group_table(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (planner, stage, seed, measure, group) and the equal-weight mean over seeds, each with the worst group.

    `rows` are `group_rows`. Each (episode, env) counts once, with the mean of its rows (a per-robot value like
    `ped_robot_contacts` enters as the mean over the env's robots) and the most pedestrians any of them scored. A seed's
    value is the mean over its (episode, env) runs, its `peds` their sum; the mean over seeds weights every seed a group
    has a value in equally and sums the seeds' `peds` and `runs`. A `flowedit suite` stage `<world>__s<seed>` is the
    world at that seed, any other stage one seed of unknown value. `worst_group` is the group with the worst value of
    its (planner, stage, level, seed, measure) by `GROUP_DIRECTIONS`, null when groups tie or the measure has no
    direction.
    """
    long = rows.filter(pl.col("group").is_not_null() & pl.col("value").is_not_null() & pl.col("value").is_not_nan())
    if len(long) == 0:
        return pl.DataFrame(schema=PED_GROUPS_SCHEMA)
    measures = list(dict.fromkeys(long["measure"].to_list()))
    stages = [stage for stage in long["stage"].unique().to_list() if stage is not None]
    parsed = {stage: parse_stage(stage) for stage in stages}
    lookup = pl.DataFrame(
        {"stage": stages, "world": [p[0] if p else stage for stage, p in parsed.items()], "seed": [p[1] if p else None for p in parsed.values()]},
        schema={"stage": pl.String, "world": pl.String, "seed": pl.Int64},
    )
    long = long.join(lookup, on="stage", how="left").with_columns(pl.coalesce("world", "stage").alias("stage")).drop("world")
    per_run = long.group_by(*_RUN, "measure", "group").agg(pl.col("value").mean(), pl.col("peds").max())
    per_seed = per_run.group_by("planner", "stage", "seed", "measure", "group").agg(pl.col("value").mean(), pl.col("peds").sum(), pl.len().cast(pl.Int64).alias("runs"))
    mean = per_seed.group_by("planner", "stage", "measure", "group").agg(pl.col("value").mean(), pl.col("peds").sum(), pl.col("runs").sum(), pl.len().cast(pl.Int64).alias("seeds"))
    table = pl.concat(
        [
            per_seed.with_columns(pl.lit("seed").alias("level"), pl.lit(1, pl.Int64).alias("seeds")),
            mean.with_columns(pl.lit("mean").alias("level"), pl.lit(None, pl.Int64).alias("seed")),
        ],
        how="diagonal",
    )
    order = {measure: i for i, measure in enumerate(measures)}
    table = table.sort(
        [pl.col("planner"), pl.col("stage"), pl.col("level") == "mean", pl.col("seed"), pl.col("measure").replace_strict(order, return_dtype=pl.Int64), pl.col("group")],
        nulls_last=True,
    )
    records: dict[tuple, list[dict[str, typing.Any]]] = {}
    for *key, group, value in table.select(*_PARTITION, "group", "value").iter_rows():
        records.setdefault(tuple(key), []).append({"group": group, "value": value})
    worst = {key: worst_group(group_records, GROUP_DIRECTIONS[key[-1]]) if key[-1] in GROUP_DIRECTIONS else None for key, group_records in records.items()}
    table = table.with_columns(pl.Series("worst_group", [worst[key] for key in table.select(_PARTITION).iter_rows()], dtype=pl.String))
    table = table.with_columns((pl.col("group") == pl.col("worst_group")).fill_null(False).alias("worst"))
    return table.select(list(PED_GROUPS_SCHEMA)).cast(PED_GROUPS_SCHEMA)


def _label(measure: str) -> str:
    return measure.removeprefix("ped_")


def _value(value: float, worst: bool) -> str:
    return f"{value:.4g}{'*' if worst else ''}"


def format_group_table(table: pl.DataFrame, width: int | None = None) -> str:
    """The mean over seeds of every (planner, stage) as a measure by group table, `*` on the worst group's value.

    Group columns that do not fit in `width` characters (the terminal's) continue in another table below.
    """
    mean = table.filter(pl.col("level") == "mean")
    if len(mean) == 0:
        return "No per-group pedestrian measures: no pedestrian fact rows with a value (not processed, or no pedestrians)."
    width = width or shutil.get_terminal_size((120, 24)).columns
    seeds = table.filter(pl.col("level") == "seed")
    out: list[str] = []
    for (planner, stage), block in mean.group_by("planner", "stage", maintain_order=True):
        stage_seeds = seeds.filter(pl.col("planner").eq_missing(planner) & pl.col("stage").eq_missing(stage))
        seed_values = sorted({seed for seed in stage_seeds["seed"].to_list() if seed is not None})
        runs = stage_seeds.group_by("seed").agg(pl.col("runs").max())["runs"].sum()
        seed_text = f"mean of {len(seed_values)} seed{'s' if len(seed_values) != 1 else ''} ({', '.join(map(str, seed_values))})" if seed_values else "seed not recorded, mean of its runs"
        out.append(f"{planner} / {stage}: {seed_text}, {runs} run{'s' if runs != 1 else ''}")
        groups = sorted(block["group"].unique().to_list())
        measures = block["measure"].unique(maintain_order=True).to_list()
        cells = {(measure, group): _value(value, worst) for measure, group, value, worst in block.select("measure", "group", "value", "worst").iter_rows()}
        worst_of = {measure: name for measure, name in block.select("measure", "worst_group").unique(maintain_order=True).iter_rows()}
        peds = {group: n for group, n in block.group_by("group").agg(pl.col("peds").max()).iter_rows()}
        labels = ["measure", *map(_label, measures), "peds"]
        label_w = max(map(len, labels))
        worst_w = max(len("worst"), *(len(name or "-") for name in worst_of.values()))
        col_w = {group: max(len(group), *(len(cells.get((m, group), "")) for m in measures), len(str(peds.get(group, "")))) for group in groups}
        chunks: list[list[str]] = [[]]
        used = label_w + 2 + worst_w
        for group in groups:
            if chunks[-1] and used + 2 + col_w[group] > width:
                chunks.append([])
                used = label_w + 2 + worst_w
            chunks[-1].append(group)
            used += 2 + col_w[group]
        for chunk in chunks:
            out.append("  ".join([f"{'measure':<{label_w}}", f"{'worst':<{worst_w}}", *(f"{group:>{col_w[group]}}" for group in chunk)]).rstrip())
            for measure in measures:
                row = [f"{_label(measure):<{label_w}}", f"{worst_of.get(measure) or '-':<{worst_w}}", *(f"{cells.get((measure, group), ''):>{col_w[group]}}" for group in chunk)]
                out.append("  ".join(row).rstrip())
            out.append("  ".join([f"{'peds':<{label_w}}", " " * worst_w, *(f"{peds.get(group, ''):>{col_w[group]}}" for group in chunk)]).rstrip())
        out.append("")
    higher = ", ".join(_label(measure) for measure, direction in GROUP_DIRECTIONS.items() if direction == "higher")
    out.append(f"* worst group: the highest mean, the lowest for {higher}; - when groups tie or the measure has no direction.")
    out.append("peds: pedestrians the group's values average, summed over seeds (the most of any measure).")
    return "\n".join(out)


def groups_command(directories: list[pathlib.Path]) -> int:
    """`arena evaluation groups`: print each benchmark dir's group table and write it there, 1 when a dir has no
    processed runs."""
    failed = False
    for directory in directories:
        try:
            rows = group_rows(pathlib.Path(directory))
        except (FileNotFoundError, ValueError) as error:
            print(f"Error: {error}")
            failed = True
            continue
        table = ped_group_table(rows)
        out = pathlib.Path(directory) / PED_GROUPS_FILE
        table.write_parquet(out)
        print(f"{directory}:")
        print(format_group_table(table))
        print(f"Per-seed rows and means written to: {out}")
    return 1 if failed else 0
