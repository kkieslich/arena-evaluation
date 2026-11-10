from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import polars as pl
import pytest

from arena_evaluation.processing import infra
from arena_evaluation.processing.ped_track_store import TRACK_SCHEMA

HORIZON = 1500.0


def _tracks(peds: list[tuple[str, str, str, float, float]], horizon: float = HORIZON) -> pl.DataFrame:
    """A track table: each (source_name, seq-key, agent_type, start, end) sampled every 5 s, and an anchor present the
    whole episode so every other one starts after the first frame and ends before the last."""
    rows = []
    everyone = [("", "anchor", "adult", 0.0, horizon), *peds]
    for i, (source, key, agent_type, start, end) in enumerate(everyone):
        for t in [*np.arange(start, end, 5.0), end]:
            rows.append({"episode_id": 0, "time_ns": int(t * 1e9), "t": float(t), "agent_id": i, "key": key if not source else f"{source}#{key}", "name": "", "source_name": source, "source_seq": i, "agent_type": agent_type, "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "radius": 0.25})
    return pl.DataFrame(rows, schema=TRACK_SCHEMA)


def _log(fields: dict[str, str]) -> pl.DataFrame:
    rows = [{"time_ns": i, "env_id": 0, "world": "w", "entity": "env_0/fleet", "kind": "fleet", "field": k, "field_kind": "discrete", "value_str": v, "value_num": None, "value_bool": None, "value_list": None} for i, (k, v) in enumerate(fields.items())]
    rows.append({"time_ns": 99, "env_id": 0, "world": "w", "entity": "env_0/jackal_0", "kind": "robot", "field": "job.0.kind", "field_kind": "discrete", "value_str": "x", "value_num": None, "value_bool": None, "value_list": None})
    return pl.DataFrame(rows)


LOG = {
    "t0": "100.000",
    # before the warm-up: not scored
    "job.0.kind": "delivery", "job.0.released": "100.000", "job.0.assigned": "100.000", "job.0.ended": "300.000", "job.0.outcome": "completed",
    "job.1.kind": "delivery", "job.1.released": "500.000", "job.1.assigned": "510.000", "job.1.loaded": "540.000", "job.1.ended": "700.000", "job.1.outcome": "completed", "job.1.wait_s": "40.000", "job.1.timeout_0": "330.000",
    "job.2.kind": "pickup", "job.2.released": "600.000", "job.2.assigned": "600.000", "job.2.ended": "1000.000", "job.2.outcome": "completed",
    "job.3.kind": "delivery", "job.3.released": "800.000", "job.3.assigned": "820.000", "job.3.ended": "900.000", "job.3.outcome": "failed", "job.3.reason": "jam",
    "job.4.kind": "delivery", "job.4.released": "900.000", "job.4.assigned": "900.000", "job.4.ended": "1400.000", "job.4.outcome": "abandoned", "job.4.reason": "timeout",
    "job.5.kind": "delivery", "job.5.released": "1450.000",
    "charge.0.robot": "r0", "charge.0.queued": "600.000", "charge.0.dispatched": "630.000", "charge.0.ended": "1100.000", "charge.0.outcome": "charged",
    "charge.1.robot": "r1", "charge.1.queued": "200.000", "charge.1.dispatched": "200.000",
    "empty.0.robot": "r1", "empty.0.at": "1200.000",
}


def test_the_fleet_log_is_the_last_value_of_each_field_of_the_fleet_entity() -> None:
    fields = {**LOG, "job.1.ended": "650.000"}
    snapshot = pl.concat([_log(LOG), _log({"job.1.ended": "700.000"}).with_columns(pl.col("time_ns") + 1000)])
    log = infra.log_from_snapshot(snapshot)
    assert log["job.1.ended"] == "700.000" and log["t0"] == "100.000"
    assert infra.log_from_snapshot(_log(fields))["job.1.ended"] == "650.000"
    assert infra.log_from_snapshot(pl.DataFrame()) == {}


def test_service_scores_the_jobs_released_after_the_warm_up() -> None:
    s = infra.service(LOG, warmup_s=420.0, episode_s=1500.0)
    assert (s["jobs_released"], s["jobs_completed"], s["jobs_failed"], s["jobs_abandoned"], s["jobs_open"], s["jobs_jammed"]) == (5, 2, 1, 1, 1, 1)
    assert s["task_time_s"] == pytest.approx((200.0 + 400.0) / 2)
    assert s["busy_s_per_job"] == pytest.approx((190.0 + 400.0) / 2)
    assert s["assign_wait_s"] == pytest.approx((10.0 + 0.0) / 2) and s["dock_wait_s"] == pytest.approx(40.0)
    assert infra.jobs(LOG)[1]["timeout_0"] == 330.0
    assert s["success_rate"] == pytest.approx(2 / 4) and s["jam_share"] == pytest.approx(1 / 4)
    assert (s["charges"], s["charger_queue_s"], s["charge_s"]) == (1, pytest.approx(30.0), pytest.approx(470.0))
    assert s["tasks_per_h"] == pytest.approx(2 / ((1500.0 - 420.0) / 3600.0))
    assert s["empty_batteries"] == 1
    assert infra.service({}, warmup_s=420.0)["task_time_s"] is None


SITE = [("south_adult", "1", "adult", 500.0, 800.0), ("south_wheelchair", "2", "wheelchair_manual", 500.0, 900.0), ("north_long_cane", "3", "blind_white_cane", 600.0, 1100.0), ("south_adult", "9", "adult", 100.0, 300.0)]


def test_cost_splits_the_extra_travel_time_into_element_and_traffic_parts() -> None:
    site = _tracks(SITE)
    # without robots: the adult 30 s and the wheelchair 40 s slower; with robots 20 s and 60 s more; the cane user does
    # not finish with robots and drops; a locker visitor walks without and with robots
    off = _tracks([("south_adult", "1", "adult", 500.0, 830.0), ("south_wheelchair", "2", "wheelchair_manual", 500.0, 940.0), ("north_long_cane", "3", "blind_white_cane", 600.0, 1100.0), ("south_locker_wheelchair", "7", "./types/locker_visitor__south_wheelchair.yaml", 450.0, 850.0)])
    on = _tracks([("south_adult", "1", "adult", 500.0, 850.0), ("south_wheelchair", "2", "wheelchair_manual", 500.0, 1000.0), ("north_long_cane", "3", "blind_white_cane", 600.0, 1500.0), ("south_locker_wheelchair", "7", "./types/locker_visitor__south_wheelchair.yaml", 450.0, 950.0)])
    c = infra.cost(site, off, on, visitor_sources=["south_locker_wheelchair"], warmup_s=420.0)
    adult, chair = c["groups"]["adult"], c["groups"]["wheelchair"]
    assert adult["peds"] == 1 and adult["total"] == pytest.approx(50.0 / 300.0) and adult["element"] == pytest.approx(30.0 / 300.0) and adult["traffic"] == pytest.approx(20.0 / 300.0)
    assert chair["total"] == pytest.approx(100.0 / 400.0) and chair["element"] + chair["traffic"] == pytest.approx(chair["total"])
    assert chair["delay_s"] == pytest.approx(100.0) and adult["site_travel_s"] == pytest.approx(300.0)
    assert c["groups"]["long_cane"]["peds"] == 0 and c["groups"]["long_cane"]["dropped"] == 1 and c["groups"]["long_cane"]["total"] is None
    assert (c["c_worst"], c["worst_group"]) == (pytest.approx(0.25), "wheelchair")
    # the pedestrian spawned before the warm-up is left out, the visitor is reported apart
    assert set(c["groups"]) == {"adult", "wheelchair", "long_cane"}
    assert c["visitors"] == {"wheelchair": {"peds": 1, "walk_s": pytest.approx(400.0), "paired": 1, "traffic": pytest.approx(100.0 / 400.0)}}
    # without the robot run, the total is the element part
    element = infra.cost(site, off, None, warmup_s=420.0)
    assert element["groups"]["adult"]["total"] == pytest.approx(30.0 / 300.0) and element["groups"]["adult"]["traffic"] is None


def _table(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def _row(layout: str, roles: str, seed: int, t: float, c: dict[str, float], ranking: float | None = None, jam: float = 0.0) -> dict:
    return {"layout_id": layout, "world": f"site__layout_{layout}", "seed": seed, "roles": roles, "ranking_score": ranking, "travel_m": 10.0, "complete": True, "task_time_s": t, "jam_share": jam, "success_rate": 1.0, **{f"C_{g}": v for g, v in c.items()}, "c_worst": max(c.values())}


def test_the_pick_is_the_lowest_c_worst_joint_layout_at_equal_service() -> None:
    rows = []
    for seed in (1, 2, 3):
        rows += [
            _row("robot", "robot_only", seed, 300.0 + seed, {"adult": 0.10, "wheelchair": 0.30, "long_cane": 0.20}, 9.0),
            _row("fast", "joint", seed, 310.0, {"adult": 0.05, "wheelchair": 0.20, "long_cane": 0.10}, 1.0),
            _row("best", "joint", seed, 312.0, {"adult": 0.05, "wheelchair": 0.08, "long_cane": 0.12}, 2.0),
            _row("slow", "joint", seed, 400.0, {"adult": 0.01, "wheelchair": 0.01, "long_cane": 0.01}, 3.0),
            _row("jammed", "joint", seed, 305.0, {"adult": 0.0, "wheelchair": 0.0, "long_cane": 0.0}, 4.0, jam=0.3),
            _row("rand", "random", seed, 500.0, {"adult": 0.4, "wheelchair": 0.5, "long_cane": 0.6}, 8.0),
        ]
    means = infra.layout_means(_table(rows))
    by = {r["layout_id"]: r for r in means.to_dicts()}
    assert by["robot"]["task_time_s"] == pytest.approx(302.0) and by["robot"]["seeds"] == 3
    assert by["best"]["c_worst_of_means"] == pytest.approx(0.12) and by["best"]["worst_group_of_means"] == "long_cane"
    assert by["robot"]["T_lo"] <= 302.0 <= by["robot"]["T_hi"]
    chosen = infra.pick(means)
    assert chosen["robot_only"] == "robot" and chosen["task_time_cap_s"] == pytest.approx(1.05 * 302.0)
    assert chosen["joint_pick"] == "best"
    excluded = {c["layout_id"]: c["excluded"] for c in chosen["joint"]}
    assert excluded["slow"].startswith("T 400.0 s over") and excluded["jammed"].startswith("jams 30%") and excluded["best"] is None
    check = infra.forecaster_check(means)
    assert check["layouts"] == 5 and check["spearman"] == pytest.approx(infra.spearman([1.0, 2.0, 3.0, 4.0, 8.0], [0.2, 0.12, 0.01, 0.0, 0.6]))


def test_spearman_handles_ties_and_too_few_pairs() -> None:
    assert infra.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert infra.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert infra.spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1.5, 1.5, 3, 4], [1, 2, 3, 4])[0, 1])
    assert math.isnan(infra.spearman([1, 2], [1, 2])) and math.isnan(infra.spearman([1, 1, 1], [1, 2, 3]))


def test_compare_takes_per_seed_differences_with_intervals_and_holm() -> None:
    rows = []
    rng = np.random.default_rng(1)
    for seed in range(11, 21):
        rows.append(_row("robot", "robot_only", seed, 300.0, {"adult": 0.10, "wheelchair": 0.40, "long_cane": 0.30}))
        rows.append(_row("joint", "joint", seed, 300.0 + rng.normal(0, 2), {"adult": 0.09 + rng.normal(0, 0.005), "wheelchair": 0.15 + rng.normal(0, 0.01), "long_cane": 0.29 + rng.normal(0, 0.01)}))
    result = infra.compare(_table(rows), {"joint": "joint", "robot_only": "robot"}, n_boot=2000)
    assert result["seeds"] == list(range(11, 21))
    d = result["delta_c_worst"]
    assert d["ci_high"] < 0 and d["seeds"] == 10 and result["h1_holds"]
    assert abs(result["delta_t_rel"]["estimate"]) < 0.05
    wheel, cane = result["delta_aid_minus_adult"]["wheelchair"], result["delta_aid_minus_adult"]["long_cane"]
    assert wheel["holds"] and wheel["ci_high"] < 0
    assert not cane["holds"]
    # worse service breaks H1 even with a lower cost
    slow = [{**r, "task_time_s": r["task_time_s"] * 1.2} if r["layout_id"] == "joint" else r for r in rows]
    assert not infra.compare(_table(slow), {"joint": "joint", "robot_only": "robot"}, n_boot=500)["h1_holds"]


def test_the_premise_check_compares_c_worst_with_the_seed_spread_and_measures_the_rate() -> None:
    rows = []
    for seed, (a, b) in enumerate([(0.10, 0.30), (0.12, 0.31), (0.11, 0.29)], start=1):
        rows.append({"layout_id": "narrow", "seed": seed, "c_worst": b, "jobs_completed": 10, "busy_s_per_job": 400.0, "charge_s": 600.0})
        rows.append({"layout_id": "wide", "seed": seed, "c_worst": a, "jobs_completed": 10, "busy_s_per_job": 200.0, "charge_s": 300.0})
    result = infra.premise(_table(rows))
    assert result["c_worst_gap"] == pytest.approx(0.30 - 0.11) and not result["raise_density"]
    # the slower layout: 400 s per job plus 1800 s of charging over 30 jobs
    assert result["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / (400.0 + 60.0), 2))
    close = [{**r, "c_worst": 0.2 + 0.05 * (r["seed"] % 2)} for r in rows]
    assert infra.premise(_table(close))["raise_density"]


def test_the_figure_draws_one_point_per_layout(tmp_path: pathlib.Path) -> None:
    rows = [_row(layout, roles, seed, t + seed, c) for seed in (1, 2) for layout, roles, t, c in (("robot", "robot_only", 300.0, {"adult": 0.1}), ("j", "joint", 310.0, {"adult": 0.05}), ("r", "random", 400.0, {"adult": 0.3}))]
    means = infra.layout_means(_table(rows))
    out = infra.figure(means, infra.pick(means), tmp_path / "plots" / infra.FIGURE_FILE)
    assert out.is_file() and out.stat().st_size > 10_000


def _benchmark(root: pathlib.Path, runs: list[tuple[str, str, int, pl.DataFrame | None]]) -> pathlib.Path:
    """A processed benchmark dir: combined metrics rows (planner, stage, episode, robot) and each episode's semantic
    snapshot; the tracks come from the monkeypatched track store."""
    root.mkdir(parents=True)
    rows = []
    for episode, (planner, stage, _, snapshot) in enumerate(runs):
        for robot in ("env_0_jackal_0", "env_0_jackal_1"):
            rows.append({"planner": planner, "stage": stage, "episode": episode, "robot": robot, "status": "evaluated", "is_reference": planner == "unhindered_peds"})
        topics = root / "episodes" / f"episode_{episode:03d}" / "topics"
        topics.mkdir(parents=True)
        if snapshot is not None:
            snapshot.write_parquet(topics / "semantic_snapshot.parquet")
    pl.DataFrame(rows).write_parquet(root / "combined_metrics.parquet")
    return root


def test_the_infra_command_tables_and_picks_a_selection(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    site = _benchmark(tmp_path / "site", [("peds_only", "site__s1", 1, None), ("peds_only", "site__s2", 2, None)])
    layout_runs = []
    tracks: dict[tuple[str, int], pl.DataFrame] = {}
    for layout, slow in (("aaaa", 30.0), ("bbbb", 10.0)):
        for seed in (1, 2):
            layout_runs.append(("fleet_dwb", f"site__layout_{layout}__s{seed}", seed, _log({**LOG, "job.3.reason": "no path"})))
            layout_runs.append(("unhindered_peds", f"site__layout_{layout}__s{seed}", seed, None))
    layouts = _benchmark(tmp_path / "sel", layout_runs)
    shift = {"aaaa": 30.0, "bbbb": 10.0}

    def fake(episode_dir: pathlib.Path, env: str) -> pl.DataFrame:
        bench = episode_dir.parent.parent
        episode = int(episode_dir.name.split("_")[1])
        if bench == site:
            return _tracks(SITE)
        planner, stage = layout_runs[episode][0], layout_runs[episode][1]
        extra = shift[stage.split("_layout_")[1][:4]] * (2 if planner == "fleet_dwb" else 1)
        return _tracks([(s, k, a, start, end + extra) for s, k, a, start, end in SITE])

    monkeypatch.setattr(infra, "tracks_for_episode", fake)
    manifest = tmp_path / "infra_h1.layouts.json"
    manifest.write_text(json.dumps({"site": "site", "seeds": [1, 2], "episode_s": 1500.0, "warmup_s": 420.0, "visitor_sources": [], "runs": [{"role": "robot_only", "layout_id": "aaaa", "travel_m": 10.0}, {"role": "joint", "layout_id": "aaaa", "ranking_score": 3.0}, {"role": "joint", "layout_id": "bbbb", "travel_m": 11.0, "ranking_score": 1.0}], "worlds": {"aaaa": "/x/site__layout_aaaa", "bbbb": "/x/site__layout_bbbb"}}))
    args = argparse.Namespace(infra="pick", site=site, layouts=layouts, manifest=manifest, warmup=None, output_dir=None, out=None, boot=200)
    assert infra.infra_command(args) == 0
    table = pl.read_parquet(layouts / infra.TABLE_FILE)
    assert len(table) == 4 and table["complete"].all()
    assert table.filter(pl.col("layout_id") == "aaaa")["C_adult"].to_list() == pytest.approx([60.0 / 300.0] * 2)
    assert table.filter(pl.col("layout_id") == "aaaa")["C_adult_element"].to_list() == pytest.approx([30.0 / 300.0] * 2)
    assert table["task_time_s"].to_list() == pytest.approx([300.0] * 4)
    chosen = json.loads((layouts / "infra_pick.json").read_text())
    assert (chosen["robot_only"], chosen["joint_pick"]) == ("aaaa", "bbbb")
    assert (layouts / "plots" / infra.FIGURE_FILE).is_file()
    printed = capsys.readouterr().out
    assert "joint pick: bbbb" in printed and "C_adult" in printed
    with pytest.raises(FileNotFoundError, match="no combined_metrics"):
        infra.runs(tmp_path)
