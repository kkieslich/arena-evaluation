"""Each pedestrian measure per group of agent types across the runs of a benchmark, per seed and as the mean over seeds.

`ped_group_table` reads the `<measure>_by_group` columns that `pedestrian_groups` and `paired_peds` write on every
robot row. `arena evaluation groups` prints the mean over seeds and writes the table next to the metrics.
"""

from __future__ import annotations

import pathlib
import shutil
import typing

import polars as pl

from arena_evaluation.processing.flow.benchmark import parse_stage
from arena_evaluation.processing.metrics.social.paired_peds import PAIRED_MEASURES
from arena_evaluation.processing.metrics.social.pedestrian_groups import FLEET_MEASURES, PED_MEASURES, worst_group
from arena_evaluation.processing.parquet_store import ParquetStore

PED_GROUPS_FILE = "ped_groups.parquet"

# The direction that is better of every measure with a `<measure>_by_group` column, the other end names the worst
# group. A measure without one (`ped_paired_dropped`, a count) names no worst group.
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


def grouped_measures(metrics: pl.DataFrame) -> list[str]:
    """The measures with a `<measure>_by_group` column of `{group, value, peds}` records, in column order."""
    measures = []
    for name, dtype in metrics.schema.items():
        if name.endswith(_BY_GROUP) and isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
            if {"group", "value", "peds"} <= {field.name for field in dtype.inner.fields}:
                measures.append(name.removesuffix(_BY_GROUP))
    return measures


def _runs(metrics: pl.DataFrame, measures: list[str]) -> pl.DataFrame:
    """The contestant rows with their planner, stage, seed, episode and env (as `mcap_reader.bundle_env`) and the group columns.

    A `flowedit suite` stage `<world>__s<seed>` is the world at that seed; the seed of any other stage is null.
    """

    def column(name: str, dtype: type[pl.DataType]) -> pl.Expr:
        return pl.col(name).cast(dtype) if name in metrics.columns else pl.lit(None, dtype)

    rows = metrics.filter(~pl.col("is_reference").fill_null(False)) if "is_reference" in metrics.columns else metrics
    rows = rows.select(
        column("planner", pl.String).alias("planner"),
        column("stage", pl.String).alias("stage"),
        column("episode", pl.Int64).alias("episode"),
        column("robot", pl.String).str.extract(r"(env_\d+)", 1).fill_null("env_0").alias("env"),
        *(pl.col(f"{measure}{_BY_GROUP}") for measure in measures),
    )
    stages = [stage for stage in rows["stage"].unique().to_list() if stage is not None]
    parsed = {stage: parse_stage(stage) for stage in stages}
    lookup = pl.DataFrame(
        {"stage": stages, "world": [p[0] if p else stage for stage, p in parsed.items()], "seed": [p[1] if p else None for p in parsed.values()]},
        schema={"stage": pl.String, "world": pl.String, "seed": pl.Int64},
    )
    return rows.join(lookup, on="stage", how="left").with_columns(pl.coalesce("world", "stage").alias("stage")).drop("world")


def ped_group_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """One row per (planner, stage, seed, measure, group) and the equal-weight mean over seeds, each with the worst group.

    Reference rows are left out. The pedestrian measures repeat on every robot row of an env, so each (episode, env)
    counts once, with the mean of its robot rows (a per-robot value like `ped_robot_contacts` enters as the mean over
    the env's robots) and the most pedestrians any of them scored. A seed's value is the mean over its (episode, env)
    runs, its `peds` their sum; the mean over seeds weights every seed a group has a value in equally and sums the
    seeds' `peds` and `runs`. `worst_group` is the group with the worst value of its (planner, stage, level, seed,
    measure) by `GROUP_DIRECTIONS`, null when groups tie or the measure has no direction.
    """
    measures = grouped_measures(metrics)
    if not measures or len(metrics) == 0:
        return pl.DataFrame(schema=PED_GROUPS_SCHEMA)
    runs = _runs(metrics, measures)
    long = pl.concat(
        [
            runs.select(*_RUN, pl.lit(measure).alias("measure"), pl.col(f"{measure}{_BY_GROUP}").alias("record"))
            .explode("record")
            .drop_nulls("record")
            .unnest("record")
            .select(*_RUN, "measure", pl.col("group").cast(pl.String), pl.col("value").cast(pl.Float64), pl.col("peds").cast(pl.Int64))
            for measure in measures
        ]
    ).filter(pl.col("group").is_not_null() & pl.col("value").is_not_null() & pl.col("value").is_not_nan())
    if len(long) == 0:
        return pl.DataFrame(schema=PED_GROUPS_SCHEMA)
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
        return "No per-group pedestrian measures: the metrics have no `<measure>_by_group` values (recordings without agent types, or not processed)."
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


def _metrics_file(directory: pathlib.Path) -> pathlib.Path | None:
    """The directory's combined_metrics.parquet, else its metrics.parquet, None without either."""
    return next((path for path in (directory / "combined_metrics.parquet", directory / "metrics.parquet") if path.is_file()), None)


def groups_command(directories: list[pathlib.Path]) -> int:
    """`arena evaluation groups`: print each directory's group table and write it next to its metrics, 1 when a directory has no metrics."""
    failed = False
    for directory in directories:
        source = _metrics_file(directory)
        if source is None:
            print(f"Error: no combined_metrics.parquet or metrics.parquet in {directory}, run `arena evaluation process` first.")
            failed = True
            continue
        table = ped_group_table(ParquetStore.read(source)[0])
        out = source.with_name(PED_GROUPS_FILE)
        ParquetStore.write(table, out)
        print(f"{source}:")
        print(format_group_table(table))
        print(f"Per-seed rows and means written to: {out}")
    return 1 if failed else 0
