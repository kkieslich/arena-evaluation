"""The per-group table over the runs of a benchmark: per seed, the mean over seeds, the worst group, the print and the CLI."""

import pathlib
import sys

import polars as pl
import pytest

from arena_evaluation import cli
from arena_evaluation.presentation.report_builder import ReportBuilder
from arena_evaluation.processing.metrics.social.paired_peds import PAIRED_DTYPES
from arena_evaluation.processing.metrics.social.pedestrian_groups import PED_MEASURE_DTYPES
from arena_evaluation.processing.ped_groups import GROUP_DIRECTIONS, PED_GROUPS_FILE, PED_GROUPS_SCHEMA, format_group_table, grouped_measures, ped_group_table

BY_GROUP = pl.List(pl.Struct({"group": pl.String, "value": pl.Float64, "peds": pl.Int64}))
MEASURES = ["ped_path_deflection_m", "ped_round_trips_completed", "ped_robot_contacts", "ped_paired_dropped"]
SCHEMA = {"episode": pl.Int64, "robot": pl.String, "planner": pl.String, "stage": pl.String, "is_reference": pl.Boolean, **{f"{m}_by_group": BY_GROUP for m in MEASURES}}


def _g(**values: tuple[float, int]) -> list[dict]:
    return [{"group": group, "value": value, "peds": peds} for group, (value, peds) in values.items()]


def _row(episode: int, robot: str, stage: str, *, reference: bool = False, **measures: list[dict] | None) -> dict:
    return {"episode": episode, "robot": robot, "planner": "dwa", "stage": stage, "is_reference": reference, **{f"{m}_by_group": measures.get(m) for m in MEASURES}}


def _env(episode: int, env: str, stage: str, contacts: tuple[list[dict], list[dict]] | None = None, **measures: list[dict]) -> list[dict]:
    """The two robot rows of one env, the same pedestrian measures on both, each robot's own contacts."""
    contacts = contacts or (None, None)
    return [_row(episode, f"{env}_{robot}", stage, ped_robot_contacts=own, **measures) for robot, own in zip(("jackal", "burger"), contacts, strict=True)]


ZERO = _g(adult=(0.0, 2), long_cane=(0.0, 1), wheelchair=(0.0, 1))
DROPPED = _g(adult=(1.0, 2), long_cane=(0.0, 1), wheelchair=(1.0, 1))
# Seed 1 runs in two envs, seed 2 in one, every env with two robots; the reference row must not count.
ROWS = [
    *_env(
        0,
        "env_0",
        "w__s1",
        contacts=(_g(adult=(2.0, 2), long_cane=(0.0, 1), wheelchair=(0.0, 1)), _g(adult=(0.0, 2), long_cane=(1.0, 1), wheelchair=(0.0, 1))),
        ped_path_deflection_m=_g(adult=(1.0, 2), long_cane=(3.0, 1), wheelchair=(2.0, 1)),
        ped_round_trips_completed=_g(adult=(2.0, 2), long_cane=(1.0, 1), wheelchair=(2.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    *_env(
        1,
        "env_1",
        "w__s1",
        contacts=(ZERO, ZERO),
        ped_path_deflection_m=_g(adult=(2.0, 2), long_cane=(5.0, 1), wheelchair=(2.0, 1)),
        ped_round_trips_completed=_g(adult=(2.0, 2), long_cane=(1.0, 1), wheelchair=(2.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    *_env(
        2,
        "env_0",
        "w__s2",
        contacts=(ZERO, ZERO),
        ped_path_deflection_m=_g(adult=(2.5, 2), long_cane=(1.0, 1), wheelchair=(3.0, 1)),
        ped_round_trips_completed=_g(adult=(3.0, 2), long_cane=(0.0, 1), wheelchair=(1.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    _row(3, "env_2_jackal", "w__s1", reference=True, ped_path_deflection_m=_g(adult=(100.0, 2), long_cane=(100.0, 1), wheelchair=(0.0, 1))),
]


@pytest.fixture
def table() -> pl.DataFrame:
    return ped_group_table(pl.DataFrame(ROWS, schema=SCHEMA))


def _pick(table: pl.DataFrame, measure: str, level: str, seed: int | None = None) -> dict[str, dict]:
    rows = table.filter((pl.col("measure") == measure) & (pl.col("level") == level) & pl.col("seed").eq_missing(seed))
    return {row["group"]: row for row in rows.iter_rows(named=True)}


def test_the_table_has_its_schema_and_a_row_per_seed_measure_and_group_then_the_means(table):
    assert table.schema == pl.Schema(PED_GROUPS_SCHEMA)
    assert len(table) == 3 * len(MEASURES) * 3
    assert table["planner"].unique().to_list() == ["dwa"] and table["stage"].unique().to_list() == ["w"]
    assert table["level"].to_list() == ["seed"] * 24 + ["mean"] * 12
    assert table.filter(pl.col("level") == "seed")["seed"].unique(maintain_order=True).to_list() == [1, 2]
    assert table.filter(pl.col("level") == "mean")["seed"].is_null().all()
    assert table["measure"].unique(maintain_order=True).to_list() == MEASURES


def test_each_env_counts_once_and_a_seed_averages_its_runs(table):
    seed_1 = _pick(table, "ped_path_deflection_m", "seed", 1)
    assert {group: row["value"] for group, row in seed_1.items()} == {"adult": 1.5, "long_cane": 4.0, "wheelchair": 2.0}
    assert {group: (row["peds"], row["runs"], row["seeds"]) for group, row in seed_1.items()} == {"adult": (4, 2, 1), "long_cane": (2, 2, 1), "wheelchair": (2, 2, 1)}


def test_a_value_of_each_robot_enters_as_the_mean_over_the_envs_robots(table):
    seed_1 = _pick(table, "ped_robot_contacts", "seed", 1)
    assert {group: row["value"] for group, row in seed_1.items()} == {"adult": 0.5, "long_cane": 0.25, "wheelchair": 0.0}
    assert seed_1["adult"]["peds"] == 4


def test_an_env_counts_once_whatever_its_number_of_robots():
    rows = [
        *_env(0, "env_0", "w__s1", ped_path_deflection_m=_g(adult=(1.0, 1))),
        _row(1, "env_1_jackal", "w__s1", ped_path_deflection_m=_g(adult=(4.0, 1))),
    ]
    assert _pick(ped_group_table(pl.DataFrame(rows, schema=SCHEMA)), "ped_path_deflection_m", "seed", 1)["adult"]["value"] == 2.5


def test_the_mean_weights_every_seed_equally(table):
    mean = _pick(table, "ped_path_deflection_m", "mean")
    assert {group: row["value"] for group, row in mean.items()} == {"adult": 2.0, "long_cane": 2.5, "wheelchair": 2.5}
    assert {group: (row["peds"], row["runs"], row["seeds"]) for group, row in mean.items()} == {"adult": (6, 3, 2), "long_cane": (3, 3, 2), "wheelchair": (3, 3, 2)}


def test_a_seed_without_the_group_leaves_it_out_of_that_groups_mean():
    rows = [
        *_env(0, "env_0", "w__s1", ped_path_deflection_m=_g(adult=(1.0, 1), elder=(2.0, 1))),
        *_env(1, "env_0", "w__s2", ped_path_deflection_m=_g(adult=(3.0, 1))),
    ]
    mean = _pick(ped_group_table(pl.DataFrame(rows, schema=SCHEMA)), "ped_path_deflection_m", "mean")
    assert {group: (row["value"], row["seeds"]) for group, row in mean.items()} == {"adult": (2.0, 2), "elder": (2.0, 1)}


def test_the_worst_group_is_the_highest_and_the_lowest_for_round_trips(table):
    assert {group: row["worst_group"] for group, row in _pick(table, "ped_path_deflection_m", "seed", 1).items()} == dict.fromkeys(("adult", "long_cane", "wheelchair"), "long_cane")
    assert _pick(table, "ped_path_deflection_m", "seed", 2)["adult"]["worst_group"] == "wheelchair"
    trips = _pick(table, "ped_round_trips_completed", "mean")
    assert {group: (row["value"], row["worst"]) for group, row in trips.items()} == {"adult": (2.5, False), "long_cane": (0.5, True), "wheelchair": (1.5, False)}
    assert trips["adult"]["worst_group"] == "long_cane"
    assert _pick(table, "ped_robot_contacts", "mean")["wheelchair"]["worst_group"] == "adult"


def test_groups_that_tie_for_the_worst_value_name_none(table):
    for picked in (_pick(table, "ped_path_deflection_m", "mean"), _pick(table, "ped_robot_contacts", "seed", 2)):
        assert {row["worst_group"] for row in picked.values()} == {None}
        assert not any(row["worst"] for row in picked.values())


def test_a_measure_without_a_direction_names_no_worst_group(table):
    assert "ped_paired_dropped" not in GROUP_DIRECTIONS
    assert table.filter(pl.col("measure") == "ped_paired_dropped")["worst_group"].is_null().all()


def test_every_grouped_measure_but_the_dropped_count_has_a_direction():
    grouped = {name.removesuffix("_by_group") for name in {**PED_MEASURE_DTYPES, **PAIRED_DTYPES} if name.endswith("_by_group")}
    assert set(GROUP_DIRECTIONS) == grouped - {"ped_paired_dropped"}
    assert {measure for measure, direction in GROUP_DIRECTIONS.items() if direction == "higher"} == {"ped_round_trips_completed", "ped_contact_free_rate"}


def test_reference_rows_are_left_out(table):
    assert _pick(table, "ped_path_deflection_m", "seed", 1)["adult"]["runs"] == 2


def test_a_stage_without_a_seed_in_its_name_is_one_seed_of_unknown_value():
    rows = [*_env(0, "env_0", "corridor", ped_path_deflection_m=_g(adult=(1.0, 1))), *_env(1, "env_0", "corridor", ped_path_deflection_m=_g(adult=(2.0, 1)))]
    table = ped_group_table(pl.DataFrame(rows, schema=SCHEMA)).filter(pl.col("measure") == "ped_path_deflection_m")
    assert table.select("stage", "level", "seed", "value", "runs", "seeds").rows() == [("corridor", "seed", None, 1.5, 2, 1), ("corridor", "mean", None, 1.5, 2, 1)]


def test_empty_and_missing_columns_give_an_empty_table():
    empty = pl.DataFrame(schema=PED_GROUPS_SCHEMA)
    assert ped_group_table(pl.DataFrame(schema=SCHEMA)).equals(empty)
    assert ped_group_table(pl.DataFrame({"episode": [0], "robot": ["env_0_jackal"], "success": [True]})).equals(empty)
    assert ped_group_table(pl.DataFrame([_row(0, "env_0_jackal", "w__s1")], schema=SCHEMA)).equals(empty)
    assert ped_group_table(pl.DataFrame({"ped_path_deflection_m_by_group": [None]})).equals(empty)
    assert ped_group_table(pl.DataFrame({"ped_path_deflection_m_by_group": [[{"group": "adult", "value": None, "peds": 1}]]}, schema={"ped_path_deflection_m_by_group": BY_GROUP})).equals(empty)
    assert grouped_measures(pl.DataFrame({"x_by_group": [[1]], "y_by_group": ["z"]})) == []


def test_a_frame_without_identity_columns_is_one_run():
    frame = pl.DataFrame({"ped_path_deflection_m_by_group": [_g(adult=(1.0, 2)), _g(adult=(1.0, 2))]}, schema={"ped_path_deflection_m_by_group": BY_GROUP})
    table = ped_group_table(frame)
    assert table.select("planner", "stage", "level", "seed", "group", "value", "peds", "runs").rows() == [(None, None, "seed", None, "adult", 1.0, 2, 1), (None, None, "mean", None, "adult", 1.0, 2, 1)]
    assert "None / None: seed not recorded, mean of its runs, 1 run" in format_group_table(table)


def test_the_print_shows_the_mean_per_measure_and_group_with_the_worst_marked(table):
    text = format_group_table(table, width=200)
    lines = text.splitlines()
    assert lines[0] == "dwa / w: mean of 2 seeds (1, 2), 3 runs"
    assert lines[1].split() == ["measure", "worst", "adult", "long_cane", "wheelchair"]
    assert lines[2].split() == ["path_deflection_m", "-", "2", "2.5", "2.5"]
    assert lines[3].split() == ["round_trips_completed", "long_cane", "2.5", "0.5*", "1.5"]
    assert lines[4].split() == ["robot_contacts", "adult", "0.25*", "0.125", "0"]
    assert lines[5].split() == ["paired_dropped", "-", "1", "0", "1"]
    assert lines[6].split() == ["peds", "6", "3", "3"]
    assert "the lowest for round_trips_completed, contact_free_rate" in text


def test_groups_that_do_not_fit_continue_below(table):
    lines = format_group_table(table, width=45).splitlines()
    headers = [line.split() for line in lines if line.startswith("measure")]
    assert headers == [["measure", "worst", "adult"], ["measure", "worst", "long_cane"], ["measure", "worst", "wheelchair"]]
    assert all(len(line) <= 45 for line in lines[:-2])


def test_the_print_of_an_empty_table_says_why():
    assert format_group_table(pl.DataFrame(schema=PED_GROUPS_SCHEMA)).startswith("No per-group pedestrian measures")


def _cli(monkeypatch, *argv: str) -> int | None:
    monkeypatch.setattr(sys, "argv", ["evaluation", *argv])
    return cli._main_impl()


def test_the_groups_command_prints_the_table_and_writes_it_next_to_the_metrics(tmp_path: pathlib.Path, monkeypatch, capsys):
    run = tmp_path / "bench"
    run.mkdir()
    pl.DataFrame(ROWS, schema=SCHEMA).write_parquet(run / "combined_metrics.parquet")
    assert _cli(monkeypatch, "groups", "--benchmark-dir", str(run)) == 0
    out = capsys.readouterr().out
    assert "dwa / w: mean of 2 seeds (1, 2), 3 runs" in out and "0.5*" in out
    assert f"Per-seed rows and means written to: {run.resolve() / PED_GROUPS_FILE}" in out
    assert pl.read_parquet(run / PED_GROUPS_FILE).equals(ped_group_table(pl.DataFrame(ROWS, schema=SCHEMA)))


def test_the_groups_command_reads_an_episodes_metrics(tmp_path: pathlib.Path, monkeypatch, capsys):
    episode = tmp_path / "episode_000"
    episode.mkdir()
    pl.DataFrame(ROWS[:2], schema=SCHEMA).write_parquet(episode / "metrics.parquet")
    assert _cli(monkeypatch, "groups", "--run-dir", str(episode)) == 0
    assert "dwa / w: mean of 1 seed (1), 1 run" in capsys.readouterr().out
    assert (episode / PED_GROUPS_FILE).is_file()


def test_the_groups_command_fails_without_metrics(tmp_path: pathlib.Path, monkeypatch, capsys):
    assert _cli(monkeypatch, "groups", "--benchmark-dir", str(tmp_path)) == 1
    assert "no combined_metrics.parquet or metrics.parquet" in capsys.readouterr().out
    assert not (tmp_path / PED_GROUPS_FILE).exists()


def test_the_groups_report_tables_the_means_and_the_seeds(tmp_path: pathlib.Path):
    run = tmp_path / "bench"
    run.mkdir()
    ped_group_table(pl.DataFrame(ROWS, schema=SCHEMA)).write_parquet(run / PED_GROUPS_FILE)
    ReportBuilder.from_dirs([run], run, manifest="groups").build()
    report = (run / "report.html").read_text()
    assert "Pedestrian Groups Report" in report and "Each measure per group, mean over seeds" in report and "Each measure per group and seed" in report
    assert "ped_round_trips_completed" in report and "long_cane" in report
