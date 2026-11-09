"""Edited world minus base world per planner and seed: the episode table, the pairs, the paired tests, the print, the CLI and the figure."""

import math
import pathlib
import sys

import polars as pl
import pytest

from arena_evaluation import cli
from arena_evaluation.processing.edit_pairs import (
    CONTRAST,
    CONTROL,
    EDIT,
    EDIT_PAIRS_FILE,
    EDIT_TESTS_FILE,
    PAIRS_SCHEMA,
    TESTS_SCHEMA,
    contrast_table,
    controls_from_split,
    edit_tests,
    episode_table,
    format_tests,
    holm,
    pair_table,
    paired_test,
    planners_from_split,
    roles_from_split,
    short_world,
)

BASE = "hospital_1"
MOVE = "hospital_1__move_fountain__6e1d6873"
INERT = "hospital_1__insert_fountain_inert__4a32a4fc"
SPLIT = {"control_worlds": {MOVE: [INERT]}}
SCHEMA = {
    "episode": pl.Int64,
    "robot": pl.String,
    "planner": pl.String,
    "stage": pl.String,
    "is_reference": pl.Boolean,
    "time_in_intimate_zone": pl.Float64,
    "time_in_personal_zone": pl.Float64,
    "path_length": pl.Float64,
}


def _row(planner: str, world: str, seed: int, intimate: float, personal: float, *, reference: bool = False, path: float = 30.0) -> dict:
    return {"episode": 0, "robot": "env_0_jackal", "planner": planner, "stage": f"{world}__s{seed}", "is_reference": reference, "time_in_intimate_zone": intimate, "time_in_personal_zone": personal, "path_length": path}


# dwb: the move adds 10 s of personal-space time at every seed (plus 0..4 s), the inert copy nothing (+-1 s)
ROWS = [
    *(_row("dwb", BASE, s, 1.0, 20.0 + s) for s in range(1, 6)),
    *(_row("dwb", MOVE, s, 2.0, 29.0 + 2 * s) for s in range(1, 6)),
    *(_row("dwb", INERT, s, 1.0, 20.0 + s + (-1) ** s) for s in range(1, 6)),
    # a reference row and a stage of no flowedit suite are left out
    _row("dwb", BASE, 1, 50.0, 50.0, reference=True),
    {**_row("dwb", BASE, 1, 50.0, 50.0), "stage": "warmup"},
]


def _frame(rows: list[dict] = ROWS) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


def test_the_episode_table_has_one_row_per_planner_world_and_seed_with_personal_space_time_derived():
    table = episode_table(_frame(), ["personal_space_time", "path_length", "no_such_measure"])
    assert table.columns == ["planner", "world", "seed", "personal_space_time", "path_length"]
    assert len(table) == 15
    base_1 = table.filter((pl.col("world") == BASE) & (pl.col("seed") == 1)).row(0, named=True)
    assert base_1["personal_space_time"] == pytest.approx(22.0)


def test_each_edited_world_pairs_with_its_base_at_the_same_seed_and_planner():
    pairs = pair_table(episode_table(_frame(), ["personal_space_time"]), roles_from_split(SPLIT))
    assert dict(pairs.schema) == PAIRS_SCHEMA
    assert pairs["role"].unique(maintain_order=True).to_list() == [EDIT, CONTROL]
    move = pairs.filter(pl.col("world") == MOVE).sort("seed")
    assert move["base"].unique().to_list() == [BASE] and move["seed"].to_list() == [1, 2, 3, 4, 5]
    assert move["diff"].to_list() == pytest.approx([10.0 + s for s in range(1, 6)])
    # another planner's base never pairs: a world without its base at a seed gives no row
    lone = pair_table(episode_table(_frame([*ROWS, _row("mppi", MOVE, 1, 0.0, 1.0)]), ["personal_space_time"]))
    assert lone.filter(pl.col("planner") == "mppi").is_empty() and set(lone["role"].to_list()) == {"edited"}


def test_the_paired_test_matches_scipy_and_gives_d_z_and_the_t_interval():
    from scipy import stats

    diffs = [11.0, 12.0, 13.0, 14.0, 15.0]
    test = paired_test(diffs)
    assert test["n"] == 5 and test["mean_diff"] == pytest.approx(13.0)
    assert test["sd_diff"] == pytest.approx(math.sqrt(2.5)) and test["d_z"] == pytest.approx(13.0 / math.sqrt(2.5))
    assert test["p_t"] == pytest.approx(stats.ttest_1samp(diffs, 0.0).pvalue)
    half = stats.t.ppf(0.975, 4) * math.sqrt(2.5) / math.sqrt(5)
    assert (test["ci_low"], test["ci_high"]) == pytest.approx((13.0 - half, 13.0 + half))
    # five seeds all one way: the exact signed-rank test cannot go below 2 / 2^5
    assert test["p_wilcoxon"] == pytest.approx(0.0625)
    assert paired_test([0.0, 0.0])["p_t"] == 1.0 and paired_test([0.0, 0.0])["p_wilcoxon"] is None
    assert paired_test([3.0])["mean_diff"] == 3.0 and paired_test([3.0])["p_t"] is None
    assert paired_test([])["n"] == 0


def test_holm_adjusts_step_down_and_keeps_missing_values():
    assert holm([0.01, None, 0.04, 0.03]) == pytest.approx([0.03, None, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_the_tests_find_the_move_and_not_the_inert_copy():
    tests = edit_tests(pair_table(episode_table(_frame(), ["personal_space_time", "path_length"]), roles_from_split(SPLIT)))
    assert dict(tests.schema) == TESTS_SCHEMA
    by = {(r["world"], r["measure"]): r for r in tests.iter_rows(named=True)}
    move, inert = by[(MOVE, "personal_space_time")], by[(INERT, "personal_space_time")]
    assert move["mean_diff"] == pytest.approx(13.0) and move["p_t"] < 0.001 and move["d_z"] > 5
    assert abs(inert["mean_diff"]) < 1 and inert["p_t"] > 0.5
    assert move["p_t_holm"] == pytest.approx(min(1.0, 2 * move["p_t"]))
    # path length did not change: zero differences, nothing to test
    assert by[(MOVE, "path_length")]["mean_diff"] == 0.0 and by[(MOVE, "path_length")]["p_wilcoxon"] is None


def test_the_print_has_a_table_per_measure_and_says_why_when_empty():
    tests = edit_tests(pair_table(episode_table(_frame(), ["personal_space_time"]), roles_from_split(SPLIT)))
    text = format_tests(tests)
    assert text.startswith("personal_space_time: edited world minus base world, paired by seed")
    assert "move_fountain" in text and "insert_fountain_inert" in text and "control" in text
    assert format_tests(pl.DataFrame(schema=TESTS_SCHEMA)).startswith("No edit/base pairs")
    assert short_world(MOVE) == "move_fountain" and short_world(BASE) == BASE


def _cli(monkeypatch, *argv: str) -> int | None:
    monkeypatch.setattr(sys, "argv", ["evaluation", *argv])
    return cli._main_impl()


def test_the_edits_command_prints_writes_and_plots(tmp_path: pathlib.Path, monkeypatch, capsys):
    run = tmp_path / "bench"
    run.mkdir()
    _frame().write_parquet(run / "combined_metrics.parquet")
    split = tmp_path / "split.yaml"
    split.write_text(f"control_worlds:\n  {MOVE}: [{INERT}]\ncontest:\n  - {{name: mppi}}\n  - {{name: dwb}}\n")
    assert planners_from_split({"contest": [{"name": "mppi"}, {"name": "dwb"}]}) == ["mppi", "dwb"] and planners_from_split({}) == []
    assert _cli(monkeypatch, "edits", "--benchmark-dir", str(run), "--split", str(split), "--plot") == 0
    out = capsys.readouterr().out
    assert "personal_space_time: edited world minus base world" in out and "path_length:" in out
    pairs = pl.read_parquet(run / EDIT_PAIRS_FILE)
    assert set(pairs["measure"].to_list()) == {"personal_space_time", "time_in_intimate_zone", "path_length"}
    assert pl.read_csv(run / EDIT_TESTS_FILE)["role"].unique().sort().to_list() == [CONTRAST, CONTROL, EDIT]
    # the H2 contrasts print first
    assert out.index("function-carrying edit minus geometry-only control") < out.index("edited world minus base world")
    for name in ("edit_pairs", "edit_contrasts"):
        figure = run / "plots" / f"{name}_personal_space_time.png"
        assert figure.is_file() and figure.stat().st_size > 10_000


def test_the_edits_command_fails_without_pairs(tmp_path: pathlib.Path, monkeypatch, capsys):
    run = tmp_path / "bench"
    run.mkdir()
    _frame([_row("dwb", BASE, 1, 1.0, 2.0)]).write_parquet(run / "combined_metrics.parquet")
    assert _cli(monkeypatch, "edits", "--benchmark-dir", str(run)) == 1
    assert "No edit/base pairs" in capsys.readouterr().out and not (run / EDIT_PAIRS_FILE).exists()


LEAVE = "hospital_1__move_fountain_leave_demand__66723c25"
TWO_CONTROLS = {"control_worlds": {MOVE: [INERT, LEAVE]}}
# the model moved without its demand: +2 s at every seed (plus 0..1 s), next to ROWS' move (+11..15 s) and inert copy (+-1 s)
CONTRAST_ROWS = [*ROWS, *(_row("dwb", LEAVE, s, 1.0, 22.0 + s + 0.25 * s) for s in range(1, 6))]


def test_each_edit_minus_each_of_its_controls_at_the_same_seed_and_planner():
    episodes = episode_table(_frame(CONTRAST_ROWS), ["personal_space_time"])
    assert controls_from_split(TWO_CONTROLS) == {MOVE: [INERT, LEAVE]} and controls_from_split({}) == {}
    rows = contrast_table(episodes, controls_from_split(TWO_CONTROLS))
    assert dict(rows.schema) == PAIRS_SCHEMA and set(rows["role"].to_list()) == {CONTRAST}
    assert rows.select("world", "against").unique(maintain_order=True).rows() == [(MOVE, INERT), (MOVE, LEAVE)]
    assert set(rows["base"].to_list()) == {BASE}
    # the contrast is the edit's difference to the base minus the control's, the base cancels
    pairs = pair_table(episodes, roles_from_split(TWO_CONTROLS))
    for control in (INERT, LEAVE):
        got = rows.filter(pl.col("against") == control).sort("seed")["diff"].to_list()
        edit = pairs.filter(pl.col("world") == MOVE).sort("seed")["diff"].to_list()
        ctl = pairs.filter(pl.col("world") == control).sort("seed")["diff"].to_list()
        assert got == pytest.approx([e - c for e, c in zip(edit, ctl, strict=True)])
    assert pairs["against"].unique().to_list() == [BASE]
    # a control without a run at a seed, or no control at all, gives no row
    assert contrast_table(episodes.filter(~((pl.col("world") == LEAVE) & (pl.col("seed") == 3))), {MOVE: [LEAVE]})["seed"].to_list() == [1, 2, 4, 5]
    assert contrast_table(episodes, {}).is_empty()


def test_the_h2_contrasts_are_holm_adjusted_apart_from_the_rows_against_the_base():
    episodes = episode_table(_frame(CONTRAST_ROWS), ["personal_space_time"])
    pairs = pl.concat([contrast_table(episodes, controls_from_split(TWO_CONTROLS)), pair_table(episodes, roles_from_split(TWO_CONTROLS))])
    tests = edit_tests(pairs)
    assert dict(tests.schema) == TESTS_SCHEMA
    by = {(r["world"], r["against"]): r for r in tests.iter_rows(named=True)}
    contrasts = [by[(MOVE, INERT)], by[(MOVE, LEAVE)]]
    against_base = [by[(MOVE, BASE)], by[(INERT, BASE)], by[(LEAVE, BASE)]]
    assert all(r["role"] == CONTRAST for r in contrasts) and {r["role"] for r in against_base} == {EDIT, CONTROL}
    # two contrasts in their family, three rows against the base in theirs
    assert [r["p_t_holm"] for r in contrasts] == pytest.approx(holm([r["p_t"] for r in contrasts]))
    assert [r["p_t_holm"] for r in against_base] == pytest.approx(holm([r["p_t"] for r in against_base]))
    move_inert = by[(MOVE, INERT)]
    assert move_inert["mean_diff"] == pytest.approx(13.0 - (-0.2)) and move_inert["p_t"] < 0.001 and move_inert["base_mean"] == pytest.approx(by[(INERT, BASE)]["mean"])
    text = format_tests(tests)
    assert text.startswith("personal_space_time: function-carrying edit minus geometry-only control, paired by seed (H2)")
    assert "move_fountain_leave_demand" in text and "edited world minus base world" in text
