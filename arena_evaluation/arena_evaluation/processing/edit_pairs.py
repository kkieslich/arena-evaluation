"""Each robot measure of an edited world minus its base world at the same seed, per planner, with a paired test across seeds.

A `flowedit suite` stage `<world>__s<seed>` is a world at a seed, and an edited world `<base>__<edit id>` pairs with
`<base>__s<seed>` run by the same planner: the same people on the same seed, so the difference is the edit's. A split's
`control_worlds` names which edited worlds are function-carrying edits and which are their geometry-only controls.

`arena evaluation edits` prints the tests, writes `edit_pairs.parquet` (one row per planner, world, seed and measure) and
`edit_tests.csv` next to the metrics, and with `--plot` one figure of the first measure.

The test is a paired t-test on the seeds' differences (df = n - 1), with Cohen's d_z (mean difference over the standard
deviation of the differences) and the 95 % t interval of the mean difference. The exact Wilcoxon signed-rank p is listed
as a check that assumes no normality, but with n seeds its two-sided p cannot fall below 2 / 2^n (0.0625 at n = 5), so at
five seeds it can never reach 0.05; the t-test can, at the price of assuming roughly normal differences.
"""

from __future__ import annotations

import math
import pathlib
import typing
from collections.abc import Mapping, Sequence

import polars as pl

from arena_evaluation.processing.flow.benchmark import parse_stage

EDIT_PAIRS_FILE = "edit_pairs.parquet"
EDIT_TESTS_FILE = "edit_tests.csv"

#: Seconds with a pedestrian inside Hall's personal distance of the robot (1.2 m edge to edge): the intimate and the
#: personal band of `proxemics_extended` together
PERSONAL_SPACE_TIME = "personal_space_time"
DERIVED = {PERSONAL_SPACE_TIME: ("time_in_intimate_zone", "time_in_personal_zone")}

#: Personal-space intrusion first, then its depth, the closest band, the encounters and contacts, and the path driven
#: (how much the robot was exposed to the people)
DEFAULT_MEASURES = (
    PERSONAL_SPACE_TIME,
    "personal_space_intrusion_integral",
    "time_in_intimate_zone",
    "psi_personal_events",
    "ped_robot_contacts",
    "path_length",
)

EDIT, CONTROL, UNLABELLED = "edit", "control", "edited"

#: Seeds up to which tied differences get the signed-rank p by enumerating every sign flip (2^n of them)
_MAX_ENUMERATED = 16

PAIRS_SCHEMA = {
    "planner": pl.String,
    "world": pl.String,
    "base": pl.String,
    "role": pl.String,
    "seed": pl.Int64,
    "measure": pl.String,
    "base_value": pl.Float64,
    "value": pl.Float64,
    "diff": pl.Float64,
}

TESTS_SCHEMA = {
    "planner": pl.String,
    "world": pl.String,
    "role": pl.String,
    "measure": pl.String,
    "n": pl.Int64,
    "base_mean": pl.Float64,
    "mean": pl.Float64,
    "mean_diff": pl.Float64,
    "ci_low": pl.Float64,
    "ci_high": pl.Float64,
    "sd_diff": pl.Float64,
    "d_z": pl.Float64,
    "t": pl.Float64,
    "p_t": pl.Float64,
    "p_t_holm": pl.Float64,
    "p_wilcoxon": pl.Float64,
}


def roles_from_split(split: Mapping[str, typing.Any]) -> dict[str, str]:
    """`{world: "edit" | "control"}` from a split's `control_worlds`: each edit world and each of its controls."""
    roles: dict[str, str] = {}
    for edit, controls in (split.get("control_worlds") or {}).items():
        roles[str(edit)] = EDIT
        for control in controls or []:
            roles[str(control)] = CONTROL
    return roles


def planners_from_split(split: Mapping[str, typing.Any]) -> list[str]:
    """The contestant names of a split's `contest`, in its order."""
    return [str(c["name"]) for c in split.get("contest") or [] if isinstance(c, Mapping) and "name" in c]


def episode_table(metrics: pl.DataFrame, measures: Sequence[str] = DEFAULT_MEASURES) -> pl.DataFrame:
    """One row per (planner, world, seed) of the `flowedit suite` stages in `metrics`, each measure the mean of its episodes.

    Reference rows and stages that are no `<world>__s<seed>` are left out. A derived measure (`personal_space_time`) is
    the sum of its parts; a measure without a column is left out.
    """
    rows = metrics.filter(~pl.col("is_reference").fill_null(False)) if "is_reference" in metrics.columns else metrics
    present = [m for m in measures if m in rows.columns or (m in DERIVED and all(part in rows.columns for part in DERIVED[m]))]
    if len(rows) == 0 or not present or "stage" not in rows.columns:
        return pl.DataFrame(schema={"planner": pl.String, "world": pl.String, "seed": pl.Int64})
    parsed = {stage: parse_stage(stage) for stage in rows["stage"].drop_nulls().unique().to_list()}
    parsed = {stage: p for stage, p in parsed.items() if p is not None}
    lookup = pl.DataFrame({"stage": list(parsed), "world": [p[0] for p in parsed.values()], "seed": [p[1] for p in parsed.values()]}, schema={"stage": pl.String, "world": pl.String, "seed": pl.Int64})

    def value(measure: str) -> pl.Expr:
        if measure in DERIVED and measure not in rows.columns:
            return pl.sum_horizontal(*(pl.col(part).cast(pl.Float64) for part in DERIVED[measure])).alias(measure)
        return pl.col(measure).cast(pl.Float64).alias(measure)

    planner = pl.col("planner").cast(pl.String).fill_null("unknown") if "planner" in rows.columns else pl.lit("unknown", pl.String)
    table = rows.select(planner.alias("planner"), pl.col("stage").cast(pl.String), *(value(m) for m in present)).join(lookup, on="stage", how="inner")
    return table.group_by("planner", "world", "seed").agg(*(pl.col(m).mean() for m in present)).sort("planner", "world", "seed")


def pair_table(episodes: pl.DataFrame, roles: Mapping[str, str] | None = None) -> pl.DataFrame:
    """Every edited world's measures minus its base world's at the same seed and planner, long: one row per measure.

    `roles` labels each edited world `edit` or `control` (`roles_from_split`); others are `edited`.
    """
    measures = [c for c in episodes.columns if c not in ("planner", "world", "seed")]
    if len(episodes) == 0 or not measures:
        return pl.DataFrame(schema=PAIRS_SCHEMA)
    roles = dict(roles or {})
    long = episodes.unpivot(index=["planner", "world", "seed"], on=measures, variable_name="measure", value_name="value")
    long = long.with_columns(pl.col("world").str.split("__").list.first().alias("base"))
    base = long.filter(pl.col("world") == pl.col("base")).select("planner", "base", "seed", "measure", pl.col("value").alias("base_value"))
    edited = long.filter(pl.col("world") != pl.col("base"))
    pairs = edited.join(base, on=["planner", "base", "seed", "measure"], how="inner")
    pairs = pairs.with_columns(
        pl.col("world").replace_strict(roles, default=UNLABELLED, return_dtype=pl.String).alias("role"),
        (pl.col("value") - pl.col("base_value")).alias("diff"),
    )
    order = {EDIT: 0, CONTROL: 1, UNLABELLED: 2}
    pairs = pairs.sort([pl.col("planner"), pl.col("base"), pl.col("role").replace_strict(order, return_dtype=pl.Int64), pl.col("world"), pl.col("measure"), pl.col("seed")], nulls_last=True)
    return pairs.select(list(PAIRS_SCHEMA)).cast(PAIRS_SCHEMA)


def paired_test(diffs: Sequence[float]) -> dict[str, float | int | None]:
    """The paired t-test, d_z, the 95 % t interval and the exact Wilcoxon signed-rank p of one world's seed differences."""
    from scipy import stats

    d = [float(x) for x in diffs if x is not None and not math.isnan(float(x))]
    n = len(d)
    out: dict[str, float | int | None] = {"n": n, "mean_diff": None, "ci_low": None, "ci_high": None, "sd_diff": None, "d_z": None, "t": None, "p_t": None, "p_wilcoxon": None}
    if n == 0:
        return out
    mean = sum(d) / n
    out["mean_diff"] = mean
    if n < 2:
        return out
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    out["sd_diff"] = sd
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    out["ci_low"], out["ci_high"] = mean - half, mean + half
    if sd > 0:
        out["d_z"] = mean / sd
        out["t"] = mean / (sd / math.sqrt(n))
        out["p_t"] = float(2 * stats.t.sf(abs(out["t"]), n - 1))
    elif mean == 0:
        out["p_t"] = 1.0
    if any(x != 0 for x in d):
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d, method=_signed_rank_method(d)).pvalue)
        except ValueError:
            out["p_wilcoxon"] = None
    return out


def _signed_rank_method(d: Sequence[float]) -> typing.Any:
    """scipy's `method` that makes the signed-rank p exact for `d`: its exact null distribution without ties, else every
    sign flip of the differences enumerated (exact given the tied ranks; scipy's `exact` is not, gh-19872).

    Count measures (contacts, encounters) tie often. Zeros are dropped either way (Wilcoxon's zero method)."""
    from scipy import stats

    nonzero = [abs(x) for x in d if x != 0]
    if len(set(nonzero)) == len(nonzero):
        return "exact"
    if len(d) > _MAX_ENUMERATED:
        return "asymptotic"
    return stats.PermutationMethod(n_resamples=2 ** len(d))


def holm(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm's step-down adjustment of `p_values`, None kept in place."""
    ranked = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    m = len(ranked)
    out: list[float | None] = [None] * len(p_values)
    running = 0.0
    for k, (p, i) in enumerate(ranked):
        running = max(running, min(1.0, (m - k) * p))
        out[i] = running
    return out


def edit_tests(pairs: pl.DataFrame) -> pl.DataFrame:
    """One paired test per (planner, world, measure) over its seeds; `p_t_holm` adjusts the t-tests of one measure across
    every planner and world of the table."""
    if len(pairs) == 0:
        return pl.DataFrame(schema=TESTS_SCHEMA)
    rows = []
    for (planner, world, role, measure), block in pairs.group_by("planner", "world", "role", "measure", maintain_order=True):
        test = paired_test(block["diff"].to_list())
        rows.append({"planner": planner, "world": world, "role": role, "measure": measure, "base_mean": block["base_value"].mean(), "mean": block["value"].mean(), **test})
    table = pl.DataFrame(rows, schema={k: v for k, v in TESTS_SCHEMA.items() if k != "p_t_holm"})
    adjusted = []
    for measure in table["measure"].unique(maintain_order=True).to_list():
        block = table.with_row_index("row").filter(pl.col("measure") == measure)
        adjusted.extend(zip(block["row"].to_list(), holm(block["p_t"].to_list()), strict=True))
    by_row = dict(adjusted)
    table = table.with_columns(pl.Series("p_t_holm", [by_row.get(i) for i in range(len(table))], dtype=pl.Float64))
    return table.select(list(TESTS_SCHEMA)).cast(TESTS_SCHEMA)


def short_world(world: str) -> str:
    """`hospital_1__move_fountain__6e1d6873` as `move_fountain`: the edit's name without its base and id."""
    parts = world.split("__")
    return parts[1] if len(parts) >= 2 else world


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}g}"


def format_tests(tests: pl.DataFrame, measures: Sequence[str] | None = None) -> str:
    """The tests as one text table per measure: edited world (role), seeds, base and edited means, mean difference with its
    95 % interval, d_z, the t-test's p (raw and Holm) and the exact Wilcoxon p."""
    if len(tests) == 0:
        return "No edit/base pairs: the metrics have no `<base>__<edit>__s<seed>` stage with a `<base>__s<seed>` stage of the same planner."
    out: list[str] = []
    for measure in measures or tests["measure"].unique(maintain_order=True).to_list():
        block = tests.filter(pl.col("measure") == measure)
        if len(block) == 0:
            continue
        out.append(f"{measure}: edited world minus base world, paired by seed")
        header = ["planner", "world", "role", "n", "base", "edited", "diff", "95% CI", "d_z", "p_t", "p_holm", "p_wilcoxon"]
        lines = [header]
        for r in block.iter_rows(named=True):
            ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]"
            lines.append([r["planner"] or "-", short_world(r["world"]), r["role"], str(r["n"]), _fmt(r["base_mean"]), _fmt(r["mean"]), _fmt(r["mean_diff"]), ci, _fmt(r["d_z"], 2), _fmt(r["p_t"], 2), _fmt(r["p_t_holm"], 2), _fmt(r["p_wilcoxon"], 2)])
        widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
        out.extend("  ".join(cell.ljust(w) if i < 3 else cell.rjust(w) for i, (cell, w) in enumerate(zip(line, widths, strict=True))).rstrip() for line in lines)
        out.append("")
    out.append("diff: mean of edited minus base over the seeds; 95% CI: t interval of that mean; d_z: mean over sd of the differences.")
    out.append("p_t: paired t-test (df = n - 1), p_holm: Holm-adjusted across the measure's rows; p_wilcoxon: exact signed-rank (two-sided, at least 2/2^n).")
    return "\n".join(out)


def _metrics_file(directory: pathlib.Path) -> pathlib.Path | None:
    return next((path for path in (directory / "combined_metrics.parquet", directory / "metrics.parquet") if path.is_file()), None)


def edits_command(directories: list[pathlib.Path], split: pathlib.Path | None = None, measures: Sequence[str] | None = None, plot: bool = False) -> int:
    """`arena evaluation edits`: print each directory's tests, write `edit_pairs.parquet` and `edit_tests.csv` (and with
    `plot` the figure of the first measure) next to its metrics; 1 when a directory has no metrics or no pair."""
    import yaml

    split_doc = (yaml.safe_load(split.read_text()) or {}) if split is not None else {}
    roles = roles_from_split(split_doc)
    chosen = list(measures or DEFAULT_MEASURES)
    failed = False
    for directory in directories:
        source = _metrics_file(directory)
        if source is None:
            print(f"Error: no combined_metrics.parquet or metrics.parquet in {directory}, run `arena evaluation process` first.")
            failed = True
            continue
        pairs = pair_table(episode_table(pl.read_parquet(source), chosen), roles)
        tests = edit_tests(pairs)
        print(f"{directory}:")
        print(format_tests(tests, [m for m in chosen if m in set(tests["measure"].to_list())]))
        if len(pairs) == 0:
            failed = True
            continue
        pairs.write_parquet(source.parent / EDIT_PAIRS_FILE)
        tests.write_csv(source.parent / EDIT_TESTS_FILE)
        print(f"wrote {source.parent / EDIT_PAIRS_FILE} and {source.parent / EDIT_TESTS_FILE}")
        if plot:
            measure = next(m for m in chosen if m in set(pairs["measure"].to_list()))
            from arena_evaluation.presentation.edit_pairs_plot import plot_edit_pairs

            out = plot_edit_pairs(pairs, tests, measure, source.parent / "plots" / f"edit_pairs_{measure}.png", planners_from_split(split_doc))
            print(f"wrote {out}")
    return 1 if failed else 0


__all__ = [
    "CONTROL",
    "DEFAULT_MEASURES",
    "EDIT",
    "EDIT_PAIRS_FILE",
    "EDIT_TESTS_FILE",
    "PAIRS_SCHEMA",
    "PERSONAL_SPACE_TIME",
    "TESTS_SCHEMA",
    "edit_tests",
    "edits_command",
    "episode_table",
    "format_tests",
    "holm",
    "pair_table",
    "paired_test",
    "planners_from_split",
    "roles_from_split",
    "short_world",
]
