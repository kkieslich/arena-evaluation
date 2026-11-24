"""The per-group table over the runs of a benchmark: per seed, the mean over seeds, the worst group, the print and the CLI."""

import json
import pathlib
import re
import sys

import polars as pl
import pytest

from arena_evaluation import cli
from arena_evaluation.processing.paired_peds import PAIRED_DTYPES, PAIRED_FILE
from arena_evaluation.processing.ped_groups import GROUP_DIRECTIONS, PED_GROUPS_FILE, PED_GROUPS_SCHEMA, format_group_table, group_rows, ped_group_table

BY_GROUP = pl.List(pl.Struct({"group": pl.String, "value": pl.Float64, "peds": pl.Int64}))
MEASURES = ["ped_path_deflection_m", "round_trips_completed", "ped_robot_contacts", "ped_paired_dropped"]
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
        round_trips_completed=_g(adult=(2.0, 2), long_cane=(1.0, 1), wheelchair=(2.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    *_env(
        1,
        "env_1",
        "w__s1",
        contacts=(ZERO, ZERO),
        ped_path_deflection_m=_g(adult=(2.0, 2), long_cane=(5.0, 1), wheelchair=(2.0, 1)),
        round_trips_completed=_g(adult=(2.0, 2), long_cane=(1.0, 1), wheelchair=(2.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    *_env(
        2,
        "env_0",
        "w__s2",
        contacts=(ZERO, ZERO),
        ped_path_deflection_m=_g(adult=(2.5, 2), long_cane=(1.0, 1), wheelchair=(3.0, 1)),
        round_trips_completed=_g(adult=(3.0, 2), long_cane=(0.0, 1), wheelchair=(1.0, 1)),
        ped_paired_dropped=DROPPED,
    ),
    _row(3, "env_2_jackal", "w__s1", reference=True, ped_path_deflection_m=_g(adult=(100.0, 2), long_cane=(100.0, 1), wheelchair=(0.0, 1))),
]


_LONG = {"planner": pl.String, "stage": pl.String, "episode": pl.Int64, "env": pl.String, "measure": pl.String, "group": pl.String, "value": pl.Float64, "peds": pl.Int64}


def _long(rows: list[dict]) -> pl.DataFrame:
    """`group_rows` of robot rows that carry each measure's group records, the reference rows left out."""
    out = []
    for row in rows:
        if row.get("is_reference"):
            continue
        env = re.search(r"(env_\d+)", row.get("robot") or "")
        for measure in MEASURES:
            for record in row.get(f"{measure}_by_group") or ():
                out.append({"planner": row.get("planner"), "stage": row.get("stage"), "episode": row.get("episode"), "env": env.group(1) if env else "env_0", "measure": measure, **record})
    return pl.DataFrame(out, schema=_LONG)


@pytest.fixture
def table() -> pl.DataFrame:
    return ped_group_table(_long(ROWS))


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
    assert _pick(ped_group_table(_long(rows)), "ped_path_deflection_m", "seed", 1)["adult"]["value"] == 2.5


def test_the_mean_weights_every_seed_equally(table):
    mean = _pick(table, "ped_path_deflection_m", "mean")
    assert {group: row["value"] for group, row in mean.items()} == {"adult": 2.0, "long_cane": 2.5, "wheelchair": 2.5}
    assert {group: (row["peds"], row["runs"], row["seeds"]) for group, row in mean.items()} == {"adult": (6, 3, 2), "long_cane": (3, 3, 2), "wheelchair": (3, 3, 2)}


def test_a_seed_without_the_group_leaves_it_out_of_that_groups_mean():
    rows = [
        *_env(0, "env_0", "w__s1", ped_path_deflection_m=_g(adult=(1.0, 1), elder=(2.0, 1))),
        *_env(1, "env_0", "w__s2", ped_path_deflection_m=_g(adult=(3.0, 1))),
    ]
    mean = _pick(ped_group_table(_long(rows)), "ped_path_deflection_m", "mean")
    assert {group: (row["value"], row["seeds"]) for group, row in mean.items()} == {"adult": (2.0, 2), "elder": (2.0, 1)}


def test_the_worst_group_is_the_highest_and_the_lowest_for_round_trips(table):
    assert {group: row["worst_group"] for group, row in _pick(table, "ped_path_deflection_m", "seed", 1).items()} == dict.fromkeys(("adult", "long_cane", "wheelchair"), "long_cane")
    assert _pick(table, "ped_path_deflection_m", "seed", 2)["adult"]["worst_group"] == "wheelchair"
    trips = _pick(table, "round_trips_completed", "mean")
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
    paired = {name.removesuffix("_by_group") for name in PAIRED_DTYPES if name.endswith("_by_group")}
    assert paired - {"ped_paired_dropped"} <= set(GROUP_DIRECTIONS) and "ped_paired_dropped" not in GROUP_DIRECTIONS
    assert {measure for measure, direction in GROUP_DIRECTIONS.items() if direction == "higher"} == {"round_trips_completed", "ped_contact_free_rate"}


def test_reference_rows_are_left_out(table):
    assert _pick(table, "ped_path_deflection_m", "seed", 1)["adult"]["runs"] == 2


def test_a_stage_without_a_seed_in_its_name_is_one_seed_of_unknown_value():
    rows = [*_env(0, "env_0", "corridor", ped_path_deflection_m=_g(adult=(1.0, 1))), *_env(1, "env_0", "corridor", ped_path_deflection_m=_g(adult=(2.0, 1)))]
    table = ped_group_table(_long(rows)).filter(pl.col("measure") == "ped_path_deflection_m")
    assert table.select("stage", "level", "seed", "value", "runs", "seeds").rows() == [("corridor", "seed", None, 1.5, 2, 1), ("corridor", "mean", None, 1.5, 2, 1)]


def test_empty_rows_and_rows_without_a_value_give_an_empty_table():
    empty = pl.DataFrame(schema=PED_GROUPS_SCHEMA)
    assert ped_group_table(pl.DataFrame(schema=_LONG)).equals(empty)
    assert ped_group_table(_long([_row(0, "env_0_jackal", "w__s1")])).equals(empty)
    assert ped_group_table(_long([_row(0, "env_0_jackal", "w__s1", ped_path_deflection_m=[{"group": "adult", "value": None, "peds": 1}])])).equals(empty)


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


def _episode(root: pathlib.Path, episode: int, stage: str, robots: tuple[str, ...], *, reference: str | None = None, peds: dict[str, float] | None = None, contacts: list[tuple[str, str, str]] | None = None) -> pathlib.Path:
    """A processed episode of env_0: its yaml, its fact rows (one per robot), its pedestrians' deflection and, unless
    None, its contacts (robot, kind, obstacle id)."""
    from arena_evaluation.storage.schemas import RunMetadata

    episode_dir = root / "episodes" / f"episode_{episode:03d}"
    (episode_dir / "metrics").mkdir(parents=True)
    planner = "unhindered_peds" if reference else "dwa"
    metadata = RunMetadata(benchmark_id=root.name, planner=planner, map="w", stage=stage, episode_id=episode, recording_started_at="2026-01-01T00:00:00", python_version="3.12", ros_distro="jazzy", modes={}, params={}, is_reference=reference is not None, reference_type=reference)
    (episode_dir / f"episode_{episode:03d}.yaml").write_text(json.dumps(metadata.model_dump(mode="json")))
    pl.DataFrame({"run_id": root.name, "episode_id": [episode] * len(robots), "robot_id": list(robots)}).write_parquet(episode_dir / "metrics" / "fact_episodes.parquet")
    peds = peds or {}
    pl.DataFrame({"run_id": root.name, "episode_id": episode, "env_id": "env_0", "ped_id": list(peds), "ped_name": [p.removeprefix("Pedestrian_") for p in peds], "ped_path_deflection_m": list(peds.values())}, schema_overrides={"ped_path_deflection_m": pl.Float64}).write_parquet(
        episode_dir / "metrics" / "fact_pedestrians.parquet"
    )
    if contacts is not None:
        pl.DataFrame({"robot_id": [c[0] for c in contacts], "collider_kind": [c[1] for c in contacts], "obstacle_id": [c[2] for c in contacts]}, schema={"robot_id": pl.String, "collider_kind": pl.String, "obstacle_id": pl.String}).write_parquet(episode_dir / "metrics" / "fact_contacts.parquet")
    return episode_dir


def _processed(tmp_path: pathlib.Path) -> pathlib.Path:
    """Two seeds of world w with two robots, three pedestrians (an adult, a wheelchair and a white cane user), and the
    reference of seed 1, whose values must not count."""
    root = tmp_path / "bench"
    robots = ("env_0_jackal", "env_0_burger")
    peds = {"Pedestrian_1": 1.0, "Pedestrian_2": 3.0, "Pedestrian_3": 2.0}
    # the jackal touches the adult twice (named by id, and by its number with its env), the burger the wheelchair user
    # once; a wall contact and the reference's contacts are not the pedestrians'
    contacts = [("env_0_jackal", "pedestrian", "Pedestrian_1"), ("env_0_jackal", "pedestrian", "env_0/1"), ("env_0_burger", "pedestrian", "agent_2"), ("env_0_burger", "wall", "wall_3")]
    _episode(root, 0, "w__s1", robots, peds=peds, contacts=contacts)
    _episode(root, 1, "w__s2", robots, peds={k: v + 1.0 for k, v in peds.items()})
    _episode(root, 2, "w__s1", ("env_0_jackal",), reference="unhindered_peds", peds=dict.fromkeys(peds, 100.0), contacts=[("env_0_jackal", "pedestrian", "Pedestrian_3")])
    types = ["adult", "wheelchair_manual", "/humansim/agents/blind_white_cane.yaml"]
    pl.DataFrame({"run_id": "bench", "episode_id": [e for e in (0, 1, 2) for _ in peds], "env_id": "env_0", "ped_id": list(peds) * 3, "agent_type": types * 3}).write_parquet(root / "dim_pedestrians.parquet")
    paired = pl.DataFrame(
        {"planner": ["dwa"], "stage": ["w__s1"], "episode": [0], "env": ["env_0"], "ped_paired_delay_s_by_group": [[{"group": "adult", "value": 4.0, "peds": 1}]]},
        schema={"planner": pl.String, "stage": pl.String, "episode": pl.Int64, "env": pl.String, "ped_paired_delay_s_by_group": PAIRED_DTYPES["ped_paired_delay_s_by_group"]},
    )
    paired.write_parquet(root / PAIRED_FILE)
    return root


def test_the_groups_come_from_the_fact_rows_the_agent_types_and_the_contacts(tmp_path: pathlib.Path):
    table = ped_group_table(group_rows(_processed(tmp_path)))
    deflection = _pick(table, "ped_path_deflection_m", "mean")
    assert {group: row["value"] for group, row in deflection.items()} == {"adult": 1.5, "wheelchair": 3.5, "long_cane": 2.5}
    assert {group: row["peds"] for group, row in deflection.items()} == {"adult": 2, "wheelchair": 2, "long_cane": 2}
    # per robot, then the mean over the env's robots: the jackal's 2 contacts with the adult, the burger's 1 with the wheelchair user
    contacts = _pick(table, "ped_robot_contacts", "seed", 1)
    assert {group: row["value"] for group, row in contacts.items()} == {"adult": 1.0, "wheelchair": 0.5, "long_cane": 0.0}
    fleet = _pick(table, "ped_fleet_contacts", "seed", 1)
    assert {group: row["value"] for group, row in fleet.items()} == {"adult": 2.0, "wheelchair": 1.0, "long_cane": 0.0}
    free = _pick(table, "ped_contact_free_rate", "seed", 1)
    assert {group: row["value"] for group, row in free.items()} == {"adult": 0.0, "wheelchair": 0.0, "long_cane": 1.0} and free["adult"]["worst_group"] is None
    assert _pick(table, "ped_paired_delay_s", "seed", 1)["adult"]["value"] == 4.0
    # seed 2 recorded no contacts table: no contact rows for it
    assert _pick(table, "ped_robot_contacts", "seed", 2) == {}


def test_the_groups_command_prints_the_table_and_writes_it_in_the_benchmark_dir(tmp_path: pathlib.Path, monkeypatch, capsys):
    run = _processed(tmp_path)
    assert _cli(monkeypatch, "groups", "--benchmark-dir", str(run)) == 0
    out = capsys.readouterr().out
    assert "dwa / w: mean of 2 seeds (1, 2), 2 runs" in out and "3.5*" in out
    assert f"Per-seed rows and means written to: {run.resolve() / PED_GROUPS_FILE}" in out
    assert pl.read_parquet(run / PED_GROUPS_FILE).equals(ped_group_table(group_rows(run)))


def test_the_groups_command_fails_without_a_benchmark(tmp_path: pathlib.Path, monkeypatch, capsys):
    assert _cli(monkeypatch, "groups", "--benchmark-dir", str(tmp_path)) == 1
    assert "no episodes/" in capsys.readouterr().out
    assert not (tmp_path / PED_GROUPS_FILE).exists()
