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


def test_the_fleet_log_leaves_out_the_previous_episodes_latched_log() -> None:
    # a recording starts with the log its lane's previous episode left latched (another t0, at time 0), then this
    # episode's reset publishes its own log: a field only the previous episode wrote must not survive
    previous = {"t0": "98.635", "job.0.kind": "delivery", "job.0.released": "134.657", "job.0.ended": "1015.317", "job.0.outcome": "abandoned", "charge.3.robot": "r0", "charge.3.started": "314.252", "charge.3.ended": "571.062", "charge.3.outcome": "charged"}
    own = {"t0": "1600.698", "job.0.kind": "delivery", "job.0.released": "28.179", "charge.3.robot": "r2", "charge.3.queued": "1006.060"}
    latched = pl.DataFrame([{**row, "time_ns": 0} for row in _log(previous).filter(pl.col("kind") == "fleet").to_dicts()])
    later = _log(own).filter(pl.col("kind") == "fleet").with_columns(pl.lit(1_600_764_300_000).alias("time_ns"))
    update = _log({"t0": "1600.698", "job.0.assigned": "28.238"}).filter(pl.col("kind") == "fleet").with_columns(pl.lit(1_628_969_400_000).alias("time_ns"))
    log = infra.log_from_snapshot(pl.concat([latched, later, update]))
    assert log == {**own, "job.0.assigned": "28.238"}
    assert infra.jobs(log)[0]["ended"] is None and infra.charges(log)[0]["started"] is None


def test_service_scores_the_jobs_released_after_the_warm_up() -> None:
    s = infra.service(LOG, warmup_s=420.0, episode_s=1500.0)
    assert (s["jobs_released"], s["jobs_completed"], s["jobs_failed"], s["jobs_abandoned"], s["jobs_open"], s["jobs_jammed"]) == (5, 2, 1, 1, 1, 1)
    # T is the restricted mean up to 1080 s: completions at 200 and 400 s, censored at 50 (open), 100 (jam) and 500 s (abandoned)
    assert s["task_time_s"] == pytest.approx(560.0)
    assert s["task_time_completed_s"] == pytest.approx((200.0 + 400.0) / 2)
    assert (s["task_time_tau_s"], s["completed_share"], s["censored_share"]) == (1080.0, pytest.approx(0.4), pytest.approx(0.6))
    assert s["busy_s_per_job"] == pytest.approx((190.0 + 400.0) / 2)
    assert s["assign_wait_s"] == pytest.approx((10.0 + 0.0) / 2) and s["dock_wait_s"] == pytest.approx(40.0)
    assert infra.jobs(LOG)[1]["timeout_0"] == 330.0
    assert s["success_rate"] == pytest.approx(2 / 4) and s["jam_share"] == pytest.approx(1 / 4)
    assert (s["charges"], s["charger_queue_s"], s["charge_s"]) == (1, pytest.approx(30.0), pytest.approx(470.0))
    # a log without top-ups (before they existed) has none
    assert (s["top_ups"], s["top_up_s"]) == (0, 0.0)
    assert s["tasks_per_h"] == pytest.approx(2 / ((1500.0 - 420.0) / 3600.0))
    assert s["empty_batteries"] == 1
    assert infra.service({}, warmup_s=420.0)["task_time_s"] is None


TOP_UPS = {
    # before the warm-up: not scored
    "top_up.0.robot": "r2", "top_up.0.charger": "charger_1", "top_up.0.dispatched": "300.000", "top_up.0.started": "500.000", "top_up.0.ended": "500.000", "top_up.0.outcome": "docked",
    # docked after a 360 s trip: free for jobs from then on, its charging there is no time out of service
    "top_up.1.robot": "r0", "top_up.1.soc_dispatched": "47.60", "top_up.1.charger": "charger_1", "top_up.1.dispatched": "700.000", "top_up.1.started": "1060.000", "top_up.1.soc_start": "29.00", "top_up.1.ended": "1060.000", "top_up.1.outcome": "docked", "top_up.1.soc_end": "29.00",
    # given up on its way after 100 s, to a robot queued to charge
    "top_up.2.robot": "r1", "top_up.2.charger": "charger_2", "top_up.2.dispatched": "800.000", "top_up.2.ended": "900.000", "top_up.2.outcome": "abandoned", "top_up.2.reason": "yielded", "top_up.2.yielded_to": "r3",
    # still on its way at the episode end
    "top_up.3.robot": "r2", "top_up.3.charger": "charger_1", "top_up.3.dispatched": "1450.000",
}


def test_service_counts_each_top_up_trip_as_time_out_of_service() -> None:
    s = infra.service({**LOG, **TOP_UPS}, warmup_s=420.0, episode_s=1500.0)
    assert (s["top_ups"], s["top_up_s"]) == (3, pytest.approx(360.0 + 100.0))
    assert [infra.top_up_trip_s(t) for t in infra.top_ups(TOP_UPS)] == [200.0, 360.0, 100.0, None]
    # everything else as without the top-ups
    plain = infra.service(LOG, warmup_s=420.0, episode_s=1500.0)
    assert {**s, "top_ups": 0, "top_up_s": 0.0} == plain


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


def test_the_pick_is_the_lowest_c_worst_joint_layout_and_flags_a_slow_one() -> None:
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
    # equal service is the search's travel pre-filter: a slow layout is flagged, not dropped; jams still drop one
    assert chosen["joint_pick"] == "slow" and chosen["joint_pick_t_over"] is True
    by_id = {c["layout_id"]: c for c in chosen["joint"]}
    assert by_id["jammed"]["excluded"].startswith("jams 30%") and by_id["slow"]["excluded"] is None and by_id["best"]["excluded"] is None
    assert by_id["slow"]["t_over"] and not by_id["best"]["t_over"] and not by_id["fast"]["t_over"]
    rows = [r for r in rows if r["layout_id"] != "slow"]
    assert infra.pick(infra.layout_means(_table(rows)))["joint_pick"] == "best"
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
    assert result["verdict"] == "lower" and result["delta"] == infra.DELTA
    # equal service is the search's travel pre-filter: measured T is reported, it does not decide H1
    slow = [{**r, "task_time_s": r["task_time_s"] * 1.2} if r["layout_id"] == "joint" else r for r in rows]
    slowed = infra.compare(_table(slow), {"joint": "joint", "robot_only": "robot"}, n_boot=500)
    assert slowed["h1_holds"] and slowed["delta_t_rel"]["estimate"] == pytest.approx(0.2, abs=0.02)


def test_compare_calls_a_difference_within_the_margin_equivalent_and_a_wide_one_inconclusive() -> None:
    rng = np.random.default_rng(2)
    rows = []
    for seed in range(11, 21):
        base = {"adult": 0.10, "wheelchair": 0.20, "long_cane": 0.15}
        rows.append(_row("robot", "robot_only", seed, 300.0, base))
        rows.append(_row("joint", "joint", seed, 300.0, {g: v + rng.normal(0, 0.002) for g, v in base.items()}))
        rows.append(_row("guide", "guideline", seed, 330.0, {g: v + 0.05 + rng.normal(0, 0.002) for g, v in base.items()}))
        rows.append(_row("noisy", "joint", seed, 300.0, {g: v + rng.normal(0, 0.05) for g, v in base.items()}))
    table = _table(rows)
    same = infra.compare(table, {"joint": "joint", "robot_only": "robot", "guideline": "guide"}, n_boot=2000)
    assert same["verdict"] == "equivalent" and not same["h1_holds"]
    d90 = same["delta_c_worst_90"]
    assert -infra.DELTA < d90["ci_low"] <= d90["ci_high"] < infra.DELTA
    assert same["delta_c_worst"]["ci_low"] <= d90["ci_low"] and d90["ci_high"] <= same["delta_c_worst"]["ci_high"]  # 90 % inside 95 %
    guide = same["guideline"]
    assert set(guide) == {"joint_minus_guideline", "robot_only_minus_guideline"}
    for c in guide.values():
        assert c["verdict"] == "lower" and c["delta_c_worst"]["ci_high"] < 0 and c["p_holm"] < 0.05 and c["b"] == "guide"
    assert guide["joint_minus_guideline"]["delta_t_rel"]["estimate"] == pytest.approx(-30.0 / 330.0, abs=1e-6)
    wide = infra.compare(table, {"joint": "noisy", "robot_only": "robot"}, n_boot=2000)
    assert wide["verdict"] == "inconclusive" and "guideline" not in wide


def test_the_score_check_ranks_the_selection_layouts_by_predicted_exposure() -> None:
    rows = []
    for seed in (1, 2):
        for i, (layout, roles) in enumerate([("robot", "robot_only"), *((f"j{k}", "joint") for k in range(10)), *((f"r{k}", "random") for k in range(5))]):
            c = 0.01 * (i + 1)
            row = _row(layout, roles, seed, 300.0, {"adult": c / 2, "wheelchair": c, "long_cane": c / 3}, ranking=float(i))
            row |= {"pred_adult": 99.0, "pred_traffic_adult": float(i), "pred_traffic_wheelchair": float(-i), "pred_traffic_long_cane": float(i), **{f"C_{g}_traffic": c for g in ("adult", "wheelchair", "long_cane")}}
            rows.append(row)
    means = infra.layout_means(_table(rows))
    check = infra.score_check(means, n_perm=2000)
    assert check["layouts"] == 16 and check["spearman"] == pytest.approx(1.0) and check["p"] < 0.01 and check["holds"]
    assert check["groups"]["adult"]["spearman"] == pytest.approx(1.0) and check["groups"]["wheelchair"]["spearman"] == pytest.approx(-1.0)
    assert check["groups"]["wheelchair"]["p"] > 0.9 and check["groups"]["adult"]["predicted"] == "pred_traffic_adult"  # the traffic term, not the total
    rho, p, n = infra.permutation_p([1.0, 2.0], [1.0, 2.0])
    assert math.isnan(rho) and math.isnan(p) and n == 2
    # one prediction of all walkers serves every group
    rows = [{k: v for k, v in r.items() if not k.startswith("pred_")} | {"pred_traffic_all": r["ranking_score"]} for r in rows]
    assert infra.score_check(infra.layout_means(_table(rows)), n_perm=200)["groups"]["long_cane"]["predicted"] == "pred_traffic_all"


def test_conflict_exposure_counts_walkers_near_a_driving_robot() -> None:
    # a robot drives along y = 0 at 1 m/s from t 400 to 600 s; walker a stands on its line at x = 100 (passed at 500 s),
    # walker b stands 5 m off it, walker c (wheelchair) on the line at x = 150, spawned after the warm-up
    rows = []
    for t in np.arange(0.0, 1000.0, 0.5):
        for i, (key, agent, x, y, start) in enumerate((("a", "adult", 100.0, 0.0, 0.0), ("b", "adult", 100.0, 5.0, 0.0), ("c", "wheelchair_manual", 150.0, 0.0, 450.0))):
            if t >= start:
                rows.append({"episode_id": 0, "time_ns": int((t + 50.0) * 1e9), "t": float(t), "agent_id": i, "key": key, "name": "", "source_name": "s", "source_seq": i, "agent_type": agent, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "radius": 0.25})
    tracks = pl.DataFrame(rows, schema=TRACK_SCHEMA)
    times = np.arange(0.0, 1000.0, 0.1)
    rx = np.clip(times - 400.0, 0.0, 200.0)
    robot = pl.DataFrame({"time_ns": ((times + 50.0) * 1e9).astype(np.int64), "x": rx, "y": np.zeros_like(times)})
    out = {r["group"]: r for r in infra.conflict_exposure(tracks, {"jackal_0": robot}, 420.0).iter_rows(named=True)}
    # within 1 m of a robot at 1 m/s: 2 s for a, none for b (5 m off), 2 s for c
    assert out["adult"]["walkers"] == 2 and out["adult"]["exposed_s"] == pytest.approx(2.0, abs=0.6) and out["adult"]["exposure_s"] == pytest.approx(1.0, abs=0.3)
    assert out["wheelchair"]["walkers"] == 1 and out["wheelchair"]["exposure_s"] == pytest.approx(2.0, abs=0.6)
    # a parked robot exposes nobody, and before the warm-up nothing counts
    parked = robot.with_columns(pl.lit(100.0).alias("x"))
    assert infra.conflict_exposure(tracks, {"jackal_0": parked}, 420.0)["exposed_s"].sum() == 0.0
    assert infra.conflict_exposure(tracks, {"jackal_0": robot}, 700.0)["exposed_s"].sum() == 0.0
    assert infra.conflict_exposure(tracks.head(0), {"jackal_0": robot}, 420.0).is_empty()
    # a repeated stamp in a robot track changes nothing (deduplicated before the speed)
    repeated = robot.with_columns(pl.when(pl.int_range(pl.len()) == 5001).then(pl.col("time_ns").shift(1)).otherwise(pl.col("time_ns")).alias("time_ns"))
    assert repeated["time_ns"].n_unique() == len(repeated) - 1
    again = {r["group"]: r for r in infra.conflict_exposure(tracks, {"jackal_0": repeated}, 420.0).iter_rows(named=True)}
    assert again["adult"]["exposed_s"] == pytest.approx(out["adult"]["exposed_s"], abs=0.6)
    # without robot tracks nothing is measured: None, not 0
    none = infra.conflict_exposure(tracks, {}, 420.0)
    assert none["exposed_s"].null_count() == len(none) and none["exposure_s"].null_count() == len(none) and none["walkers"].to_list() == [2, 1]


def test_group_cells_sum_the_walker_seconds_per_group_and_cell(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = []
    for t in np.arange(0.0, 10.0, 0.5):
        rows.append({"episode_id": 0, "time_ns": int(t * 1e9), "t": float(t), "agent_id": 0, "key": "a", "name": "", "source_name": "s", "source_seq": 0, "agent_type": "adult", "x": 1.1, "y": 2.2, "vx": 0.0, "vy": 0.0, "radius": 0.25})
        rows.append({"episode_id": 0, "time_ns": int(t * 1e9), "t": float(t), "agent_id": 1, "key": "w", "name": "", "source_name": "s", "source_seq": 1, "agent_type": "wheelchair_manual", "x": 1.4, "y": 2.4, "vx": 0.0, "vy": 0.0, "radius": 0.3})
    tracks = pl.DataFrame(rows, schema=TRACK_SCHEMA)
    cells = infra.group_cells(tracks, 0.5)
    assert cells.sort("group").to_dicts() == [
        {"group": "adult", "x": 1.25, "y": 2.25, "cell_m": 0.5, "ped_seconds": pytest.approx(10.0)},
        {"group": "wheelchair", "x": 1.25, "y": 2.25, "cell_m": 0.5, "ped_seconds": pytest.approx(10.0)},
    ]
    site = _benchmark(tmp_path / "site", [("peds_only", "site__s1", 1, None), ("peds_only", "site__s2", 2, None)])
    monkeypatch.setattr(infra, "tracks_for_episode", lambda episode_dir, env: tracks)
    monkeypatch.setattr(infra, "has_tracks", lambda episode_dir, env: True)
    occ = infra.occupancy(site)
    assert occ.columns == ["world", "stage", "planner", "group", "x", "y", "cell_m", "ped_seconds"]
    assert sorted(set(occ["stage"])) == ["site__s1", "site__s2"] and set(occ["world"]) == {"site"} and occ["ped_seconds"].sum() == pytest.approx(40.0)
    monkeypatch.setattr(infra, "has_tracks", lambda episode_dir, env: False)
    monkeypatch.setattr(infra, "track_problem", lambda episode_dir, env: "a copy")
    with pytest.raises(FileNotFoundError, match="no walker tracks"):
        infra.occupancy(site)


def _premise_rows(narrow: dict, wide: dict) -> list[dict]:
    """Premise rows at seeds 1-3: the narrow layout `n1` and the wide `w1` by their premise roles, with C_worst per seed
    and the service fields in `narrow` and `wide`."""
    rows = []
    for seed, (a, b) in enumerate([(0.10, 0.30), (0.12, 0.31), (0.11, 0.29)], start=1):
        rows.append({"layout_id": "n1", "roles": "narrow", "seed": seed, "c_worst": b, **narrow})
        rows.append({"layout_id": "w1", "roles": "wide", "seed": seed, "c_worst": a, **wide})
    return rows


def test_the_premise_check_compares_c_worst_with_the_seed_spread_and_measures_the_rate_in_the_wide_layout() -> None:
    rows = _premise_rows({"jobs_completed": 10, "busy_s_per_job": 400.0, "charge_s": 600.0}, {"jobs_completed": 10, "busy_s_per_job": 200.0, "charge_s": 300.0})
    result = infra.premise(_table(rows))
    assert result["c_worst_gap"] == pytest.approx(0.30 - 0.11) and not result["raise_density"]
    # the wide layout: 200 s per job plus 900 s of charging over 30 jobs; the narrow one's capacity is reported beside it
    assert result["rate_layout"] == "w1" and result["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / (200.0 + 30.0), 2))
    assert result["layouts"]["n1"]["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / (400.0 + 60.0), 2))
    assert result["layouts"]["w1"]["roles"] == ["wide"] and result["layouts"]["n1"]["roles"] == ["narrow"]
    close = [{**r, "c_worst": 0.2 + 0.05 * (r["seed"] % 2)} for r in rows]
    assert infra.premise(_table(close))["raise_density"]


def test_the_premise_capacity_counts_top_up_trips_as_time_out_of_service() -> None:
    narrow, wide = {"jobs_completed": 10, "busy_s_per_job": 400.0, "charge_s": 600.0}, {"jobs_completed": 10, "busy_s_per_job": 200.0, "charge_s": 300.0}
    result = infra.premise(_table(_premise_rows({**narrow, "top_up_s": 0.0}, {**wide, "top_up_s": 450.0})))
    # the wide layout: 200 s per job plus 900 s of charging and 1350 s of top-up trips over 30 jobs
    assert result["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / (200.0 + (900.0 + 1350.0) / 30), 2))
    assert (result["layouts"]["w1"]["charge_s"], result["layouts"]["w1"]["top_up_s"]) == (900.0, 1350.0)
    # without top-ups, or on a table from before they existed, the rate is the old one to the bit
    old = infra.premise(_table(_premise_rows(narrow, wide)))
    zero = infra.premise(_table(_premise_rows({**narrow, "top_up_s": 0.0}, {**wide, "top_up_s": 0.0})))
    assert old["rate_per_h"] == zero["rate_per_h"] == round(0.75 * 4 * 3600 / (200.0 + 900.0 / 30), 2)
    assert old["layouts"]["w1"]["busy_s_per_job"] == zero["layouts"]["w1"]["busy_s_per_job"]
    assert old["layouts"]["n1"]["top_up_s"] == 0.0


def test_the_premise_rate_comes_from_the_wide_layout_by_its_role_whichever_is_faster() -> None:
    # the narrow layout measured faster here: the rate still comes from the wide one, not from the slower or the faster
    rows = _premise_rows({"jobs_completed": 10, "busy_s_per_job": 150.0, "charge_s": 0.0}, {"jobs_completed": 10, "busy_s_per_job": 300.0, "charge_s": 0.0})
    result = infra.premise(_table(rows))
    assert result["rate_layout"] == "w1" and result["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / 300.0, 2))


def test_a_narrow_layout_without_a_completed_job_keeps_its_cost_in_the_premise_check() -> None:
    rows = _premise_rows({"jobs_completed": 0, "busy_s_per_job": None, "charge_s": 1200.0}, {"jobs_completed": 9, "busy_s_per_job": 250.0, "charge_s": 900.0})
    result = infra.premise(_table(rows))
    assert result["layouts"]["n1"]["rate_per_h"] is None and result["layouts"]["n1"]["jobs_completed"] == 0
    # its C_worst still counts in the gap
    assert result["c_worst_gap"] == pytest.approx(0.30 - 0.11) and result["layouts"]["n1"]["c_worst"] == pytest.approx(0.30)
    assert result["rate_layout"] == "w1" and result["rate_per_h"] == pytest.approx(round(0.75 * 4 * 3600 / (250.0 + 2700.0 / 27), 2))


def test_the_premise_check_needs_exactly_one_wide_layout() -> None:
    rows = _premise_rows({"jobs_completed": 10, "busy_s_per_job": 400.0, "charge_s": 0.0}, {"jobs_completed": 10, "busy_s_per_job": 200.0, "charge_s": 0.0})
    with pytest.raises(ValueError, match="role 'wide', and 0 layouts have it"):
        infra.premise(_table([{**r, "roles": "narrow"} for r in rows]))
    with pytest.raises(ValueError, match="role 'wide', and 0 layouts have it"):
        infra.premise(_table([{k: v for k, v in r.items() if k != "roles"} for r in rows]))
    with pytest.raises(ValueError, match="role 'wide', and 2 layouts have it"):
        infra.premise(_table([{**r, "roles": "wide"} for r in rows]))


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
    monkeypatch.setattr(infra, "has_tracks", lambda episode_dir, env: True)
    manifest = tmp_path / "infra_h1.layouts.json"
    manifest.write_text(json.dumps({"site": "site", "seeds": [1, 2], "episode_s": 1500.0, "warmup_s": 420.0, "visitor_sources": [], "runs": [{"role": "robot_only", "layout_id": "aaaa", "travel_m": 10.0}, {"role": "joint", "layout_id": "aaaa", "ranking_score": 3.0}, {"role": "joint", "layout_id": "bbbb", "travel_m": 11.0, "ranking_score": 1.0}], "worlds": {"aaaa": "/x/site__layout_aaaa", "bbbb": "/x/site__layout_bbbb"}}))
    args = argparse.Namespace(infra="pick", site=site, layouts=layouts, manifest=manifest, warmup=None, tau=None, output_dir=None, out=None, boot=200)
    assert infra.infra_command(args) == 0
    table = pl.read_parquet(layouts / infra.TABLE_FILE)
    assert len(table) == 4 and table["complete"].all()
    assert table.filter(pl.col("layout_id") == "aaaa")["C_adult"].to_list() == pytest.approx([60.0 / 300.0] * 2)
    assert table.filter(pl.col("layout_id") == "aaaa")["C_adult_element"].to_list() == pytest.approx([30.0 / 300.0] * 2)
    assert table["task_time_completed_s"].to_list() == pytest.approx([300.0] * 4)
    assert table["task_time_s"].to_list() == pytest.approx([560.0] * 4)
    chosen = json.loads((layouts / "infra_pick.json").read_text())
    assert (chosen["robot_only"], chosen["joint_pick"]) == ("aaaa", "bbbb")
    assert chosen["score_check"]["layouts"] <= 2 and "groups" in chosen["score_check"]
    assert table.filter(pl.col("layout_id") == "aaaa")["exposure_adult_s"].to_list() == [None, None]  # no robot tracks: not measured
    assert (layouts / "plots" / infra.FIGURE_FILE).is_file()
    printed = capsys.readouterr().out
    assert "joint pick: bbbb" in printed and "C_adult" in printed
    with pytest.raises(FileNotFoundError, match="no combined_metrics"):
        infra.runs(tmp_path)
    # a stage the runner ran twice (the failed attempt kept its recording) is refused, not read silently
    metrics = pl.read_parquet(layouts / "combined_metrics.parquet")
    extra = metrics.filter((pl.col("stage") == "site__layout_aaaa__s1") & (pl.col("planner") == "fleet_dwb")).with_columns(pl.lit(99, dtype=metrics.schema["episode"]).alias("episode"))
    pl.concat([metrics, extra]).write_parquet(layouts / "combined_metrics.parquet")
    with pytest.raises(ValueError, match="stage site__layout_aaaa__s1 has more than one usable run"):
        infra.infra_command(args)


def test_the_restricted_mean_counts_censored_jobs_up_to_the_horizon() -> None:
    # no censoring: the mean of the times cut at the horizon
    assert infra.restricted_mean([100.0, 300.0, 2000.0], [True, True, True], 1000.0) == pytest.approx((100.0 + 300.0 + 1000.0) / 3)
    # a job censored at 500 s keeps the curve at its level beyond its last completion
    assert infra.restricted_mean([100.0, 500.0], [True, False], 1000.0) == pytest.approx(100.0 + 0.5 * 900.0)
    # dropping the open jobs (the plain mean) would read 100 s; ties put completions before censorings
    assert infra.restricted_mean([200.0, 200.0], [True, False], 1000.0) == pytest.approx(200.0 + 0.5 * 800.0)
    assert infra.restricted_mean([], [], 1000.0) is None


def test_runs_leave_out_failed_attempts_and_robot_references_and_a_stage_run_twice_is_refused(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "bench"
    root.mkdir()
    rows = [
        {"planner": "fleet_dwb", "stage": "w__s1", "episode": 0, "robot": "env_0_jackal_0", "status": "error", "is_reference": False, "reference_type": None},
        {"planner": "fleet_dwb", "stage": "w__s1", "episode": 1, "robot": "env_0_jackal_0", "status": "evaluated", "is_reference": False, "reference_type": None},
        {"planner": "fleet_dwb_unobstructed_robot", "stage": "w__s1", "episode": 2, "robot": "env_0_jackal_0", "status": "evaluated", "is_reference": True, "reference_type": "unobstructed_robot"},
        {"planner": "unhindered_peds", "stage": "w__s1", "episode": 3, "robot": "env_0_jackal_0", "status": "no_trajectory", "is_reference": True, "reference_type": "unhindered_peds"},
        {"planner": "unhindered_peds", "stage": "w__s2", "episode": 4, "robot": "env_0_jackal_0", "status": "path_only", "is_reference": True, "reference_type": "unhindered_peds"},
        {"planner": "unhindered_peds", "stage": "w__s2", "episode": 5, "robot": "env_0_jackal_0", "status": "evaluated", "is_reference": True, "reference_type": "unhindered_peds"},
    ]
    pl.DataFrame(rows).write_parquet(root / "combined_metrics.parquet")
    found = infra.runs(root)
    assert [(r.planner, r.stage, r.episode) for r in found] == [("fleet_dwb", "w__s1", 1), ("unhindered_peds", "w__s1", 3), ("unhindered_peds", "w__s2", 4), ("unhindered_peds", "w__s2", 5)]
    assert infra.unique([r for r in found if r.stage == "w__s1" and r.planner == "fleet_dwb"], "here").episode == 1
    assert infra.unique([], "here") is None
    with pytest.raises(ValueError, match=r"stage w__s2 has more than one usable run \(episode_004 \(env_0\), episode_005 \(env_0\)\)"):
        infra.unique([r for r in found if r.stage == "w__s2"], "here")


def _selection(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, parts: dict[str, tuple[str, ...]]) -> tuple[pathlib.Path, dict[str, pathlib.Path], pathlib.Path]:
    """The site as is and a selection of layouts aaaa and bbbb, its runs split into run dirs by `parts` (dir name ->
    layouts), with tracks from a monkeypatched store: a layout's walkers are slower by its shift, twice with the fleet."""
    site = _benchmark(tmp_path / "site", [("peds_only", "site__s1", 1, None), ("peds_only", "site__s2", 2, None)])
    shift = {"aaaa": 30.0, "bbbb": 10.0}
    dirs: dict[str, pathlib.Path] = {}
    layout_runs: dict[pathlib.Path, list[tuple[str, str, int, pl.DataFrame | None]]] = {}
    for part, layouts in parts.items():
        runs_here = []
        for layout in layouts:
            for seed in (1, 2):
                runs_here.append(("fleet_dwb", f"site__layout_{layout}__s{seed}", seed, _log({**LOG, "job.3.reason": "no path"})))
                runs_here.append(("unhindered_peds", f"site__layout_{layout}__s{seed}", seed, None))
        dirs[part] = _benchmark(tmp_path / part, runs_here)
        layout_runs[dirs[part]] = runs_here

    def fake(episode_dir: pathlib.Path, env: str) -> pl.DataFrame:
        bench = episode_dir.parent.parent
        if bench == site:
            return _tracks(SITE)
        planner, stage = layout_runs[bench][int(episode_dir.name.split("_")[1])][:2]
        extra = shift[stage.split("_layout_")[1][:4]] * (2 if planner == "fleet_dwb" else 1)
        return _tracks([(s, k, a, start, end + extra) for s, k, a, start, end in SITE])

    monkeypatch.setattr(infra, "tracks_for_episode", fake)
    monkeypatch.setattr(infra, "has_tracks", lambda episode_dir, env: True)  # the fake store holds every episode
    manifest = tmp_path / "infra_h1.layouts.json"
    manifest.write_text(json.dumps({"site": "site", "seeds": [1, 2], "episode_s": 1500.0, "warmup_s": 420.0, "visitor_sources": [], "runs": [{"role": "robot_only", "layout_id": "aaaa", "travel_m": 10.0}, {"role": "joint", "layout_id": "aaaa", "ranking_score": 3.0}, {"role": "joint", "layout_id": "bbbb", "travel_m": 11.0, "ranking_score": 1.0}], "worlds": {"aaaa": "/x/site__layout_aaaa", "bbbb": "/x/site__layout_bbbb"}}))
    return site, dirs, manifest


def _commits(run_dir: pathlib.Path, arena: str, humansim: str = "h" * 40, evaluation: str = "e" * 40) -> None:
    import yaml

    commits = {"arena": {"path": "/a", "commit": arena, "dirty": False}, "humansim": {"path": "/h", "commit": humansim, "dirty": False}, "arena_evaluation": {"path": "/e", "commit": evaluation, "dirty": True}}
    (run_dir / "manifest.yaml").write_text(yaml.safe_dump({"run_id": run_dir.name, "commits": commits}))


def test_a_suite_split_over_two_run_dirs_reads_as_one(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, one, manifest = _selection(tmp_path / "one", monkeypatch, {"sel": ("aaaa", "bbbb")})
    whole = infra.layout_table(site, one["sel"], json.loads(manifest.read_text()))
    site2, two, manifest2 = _selection(tmp_path / "two", monkeypatch, {"part1": ("aaaa",), "part2": ("bbbb",)})
    for run_dir in (site2, *two.values()):
        _commits(run_dir, "a" * 40)
    split = infra.layout_table([site2], [two["part1"], two["part2"]], json.loads(manifest2.read_text()))
    assert split.drop("world").equals(whole.drop("world")) and split["complete"].all()
    # the command takes them as repeated --layouts, the outputs go to the first
    import argparse

    args = argparse.Namespace(infra="pick", site=[site2], layouts=[two["part1"], two["part2"]], manifest=manifest2, warmup=None, tau=None, output_dir=None, out=None, boot=200)
    assert infra.infra_command(args) == 0
    assert json.loads((two["part1"] / "infra_pick.json").read_text())["joint_pick"] == "bbbb"


def test_run_dirs_of_different_commits_or_sharing_a_stage_are_refused(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"part1": ("aaaa",), "part2": ("aaaa", "bbbb")})
    doc = json.loads(manifest.read_text())
    with pytest.raises(ValueError, match=r"stage site__layout_aaaa__s1 has more than one usable run .* in .*part1.*; run dirs read together must not share a stage"):
        infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)
    for run_dir in (site, dirs["part1"]):
        _commits(run_dir, "a" * 40)
    _commits(dirs["part2"], "b" * 40)
    with pytest.raises(ValueError, match="the run dirs ran different arena commits"):
        infra.check_commits([site, dirs["part1"], dirs["part2"]])
    _commits(dirs["part2"], "a" * 40, humansim="c" * 40)
    with pytest.raises(ValueError, match="different humansim commits"):
        infra.check_commits([site, dirs["part2"]])
    # the runner (arena_evaluation) records the episodes: its commit is compared too
    _commits(dirs["part2"], "a" * 40, evaluation="f" * 40)
    with pytest.raises(ValueError, match=r"different arena_evaluation commits.*--allow-mixed-commits"):
        infra.check_commits([site, dirs["part2"]])
    infra.check_commits([site, dirs["part2"]], allow_mixed=True)  # warns
    # a dirty checkout is compared by its commit (the manifests above record a dirty arena_evaluation)
    infra.check_commits([site, dirs["part1"]])


def test_kept_walker_times_let_a_copy_without_tracks_read_the_same(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse
    import shutil

    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"sel": ("aaaa", "bbbb")})
    doc = json.loads(manifest.read_text())
    assert infra.infra_command(argparse.Namespace(infra="times", benchmark=[site, dirs["sel"]], manifest=manifest, warmup=None)) == 0
    assert "kept the walker travel times of 10 runs (warm-up 420 s)" in capsys.readouterr().out
    original = infra.layout_table(site, dirs["sel"], doc)
    # copy only the small files, as from another machine
    copies = []
    for run_dir in (site, dirs["sel"]):
        copy = tmp_path / "copy" / run_dir.name
        for path in run_dir.rglob("*"):
            rel = path.relative_to(run_dir)
            if path.is_file() and (rel.name in ("combined_metrics.parquet", "semantic_snapshot.parquet") or "infra_times." in str(rel)):
                (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy / rel)
        copies.append(copy)

    def no_tracks(episode_dir: pathlib.Path, env: str) -> pl.DataFrame:
        raise AssertionError(f"read the tracks of {episode_dir}")

    monkeypatch.setattr(infra, "tracks_for_episode", no_tracks)
    monkeypatch.setattr(infra, "has_tracks", _real_has_tracks)  # the copy has no extraction
    copied = infra.layout_table(copies[0], copies[1], doc)
    assert copied.equals(original)
    # another warm-up needs the tracks, which the copy does not have: it says so
    with pytest.raises(FileNotFoundError, match="no walker times of env_0 kept for a warm-up of 300 s"):
        infra.layout_table(copies[0], copies[1], doc, warmup_s=300.0)


_real_has_tracks = infra.has_tracks


def _small_copy(tmp_path: pathlib.Path, run_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    """The small files of each run dir, and every episode's (empty) topics dir, as a copy made elsewhere."""
    import shutil

    copies = []
    for run_dir in run_dirs:
        copy = tmp_path / "copy" / run_dir.name
        for path in run_dir.rglob("*"):
            rel = path.relative_to(run_dir)
            if path.is_file() and (rel.name in ("combined_metrics.parquet", "semantic_snapshot.parquet", "manifest.yaml") or "infra_times." in str(rel)):
                (copy / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy / rel)
        for episode in (run_dir / "episodes").iterdir():
            (copy / "episodes" / episode.name / "topics").mkdir(parents=True, exist_ok=True)
        copies.append(copy)
    return copies


def test_a_copy_without_times_for_the_warm_up_raises_with_the_real_track_store(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_evaluation.processing import ped_track_store

    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"sel": ("aaaa", "bbbb")})
    doc = json.loads(manifest.read_text())
    assert infra.keep_times([site, dirs["sel"]], 420.0) == 10
    original = infra.layout_table(site, dirs["sel"], doc)
    copies = _small_copy(tmp_path, [site, dirs["sel"]])
    # the reading machine's real track store: it reads a topics dir without an extraction as an episode without walkers
    monkeypatch.setattr(infra, "tracks_for_episode", ped_track_store.tracks_for_episode)
    monkeypatch.setattr(infra, "has_tracks", _real_has_tracks)
    assert infra.layout_table(copies[0], copies[1], doc).equals(original)
    with pytest.raises(FileNotFoundError, match=r"episode_000: no walker times of env_0 kept for a warm-up of 300 s .*no extraction of its recording .*evaluation infra times --benchmark <its run dir> --warmup 300"):
        infra.layout_table(copies[0], copies[1], doc, warmup_s=300.0)
    # nothing empty is kept for that warm-up
    assert not list(copies[1].rglob(f"*.w{300.0!r}.parquet")) and not list(copies[0].rglob(f"*.w{300.0!r}.parquet"))
    with pytest.raises(FileNotFoundError, match=r"kept the walker times of 0 runs, but could not keep 8: .*episode_000 \(env_0\): no walker times of env_0 kept for a warm-up of 300 s"):
        infra.keep_times(copies[1], 300.0)


def test_kept_times_name_the_track_store_and_their_code_and_record_the_commit(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_evaluation.processing import ped_track_store

    _, dirs, _ = _selection(tmp_path, monkeypatch, {"sel": ("aaaa",)})
    episode = dirs["sel"] / "episodes" / "episode_000"
    first = infra.walker_times(episode, "env_0", 420.0)
    path = infra.times_path(episode, "env_0", 420.0)
    assert path.is_file() and f".t{ped_track_store.TRACKS_VERSION}." in path.parent.name and path.name == "env_0.w420.0.parquet"
    provenance = infra.times_provenance(path)
    assert provenance["tracks_version"] == ped_track_store.TRACKS_VERSION and provenance["times_version"] == infra.TIMES_VERSION and provenance["warmup_s"] == 420.0
    assert set(provenance["arena_evaluation"]) >= {"path", "commit", "dirty"} and provenance["code"] == infra._times_code()
    # new tracks after a track store bump are never read through the old kept times
    monkeypatch.setattr(ped_track_store, "TRACKS_VERSION", ped_track_store.TRACKS_VERSION + 1)
    monkeypatch.setattr(infra, "tracks_for_episode", lambda d, e: _tracks([("south_adult", "1", "adult", 500.0, 1000.0)]))
    assert infra.times_path(episode, "env_0", 420.0) != path
    again = infra.walker_times(episode, "env_0", 420.0)
    assert not again.equals(first) and again["travel_s"].to_list() == [500.0]


def test_walker_times_kept_by_different_commits_are_refused(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"part1": ("aaaa",), "part2": ("bbbb",)})
    doc = json.loads(manifest.read_text())
    for run_dir, commit in ((site, "a" * 40), (dirs["part1"], "a" * 40), (dirs["part2"], "b" * 40)):
        monkeypatch.setattr(infra, "_processing", lambda c=commit: {"path": "/e", "commit": c, "dirty": False})
        infra.keep_times(run_dir, 420.0)
    with pytest.raises(ValueError, match=r"walker times kept by different arena_evaluation commits: aaaaaaaaaaaa in .*bbbbbbbbbbbb in .*part2"):
        infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)
    assert infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc, allow_mixed_commits=True)["complete"].all()


def test_walker_times_and_the_cost_are_the_same_bits_in_any_row_order() -> None:
    rng = np.random.default_rng(0)
    peds = [("south_adult" if i % 3 else "north_wheel", str(i), "adult" if i % 3 else "wheelchair_manual", 430.0 + float(rng.uniform(0, 50)), 430.0 + float(rng.uniform(100, 900))) for i in range(1500)]
    tracks = _tracks(peds)
    orders = {tuple(infra._times(tracks, 420.0)["key"].to_list()) for _ in range(10)}
    assert len(orders) == 1 and list(next(iter(orders))) == sorted(next(iter(orders)))
    site = infra._times(tracks, 420.0)
    off = site.with_columns(pl.col("travel_s") * 1.0373)
    on = site.with_columns(pl.col("travel_s") * 1.0911)
    base = infra.cost_from_times(site, off, on)
    for k in range(10):
        shuffled = infra.cost_from_times(site.sample(fraction=1.0, shuffle=True, seed=k), off.sample(fraction=1.0, shuffle=True, seed=k + 100), on)
        assert shuffled == base


def _extraction(episode_dir: pathlib.Path, tables: dict[str, pl.DataFrame]) -> pathlib.Path:
    """A current extraction of an episode holding `tables` for robot env_0_jackal_0, as `evaluation process` writes it."""
    from arena_evaluation.processing.parquet_store import TopicParquetStore
    from arena_evaluation.storage.schemas import TopicBundle

    bundle = TopicBundle()
    for name, frame in tables.items():
        setattr(bundle, name, frame)
    topics = episode_dir / "topics"
    TopicParquetStore.write({"env_0_jackal_0": bundle}, topics, tables=frozenset(tables))
    return topics


def test_an_extraction_without_pedestrian_tables_or_of_an_older_format_gives_no_times(tmp_path: pathlib.Path) -> None:
    from arena_evaluation.processing.parquet_store import TopicParquetStore

    episode = tmp_path / "run" / "episodes" / "episode_000"
    topics = _extraction(episode, {"odom": pl.DataFrame({"time_ns": [1, 2], "pos_x": [0.0, 1.0]})})  # e.g. --measures with robot tables only
    assert TopicParquetStore.is_current(topics, frozenset()) and not infra.has_tracks(episode, "env_0")
    assert "no pedestrian table (odom)" in infra.track_problem(episode, "env_0")
    with pytest.raises(FileNotFoundError, match=r"no tracks to compute them from: its extraction holds no pedestrian table .*re-run `evaluation process`"):
        infra.walker_times(episode, "env_0", 420.0)
    assert not infra.times_path(episode, "env_0", 420.0).exists()
    (topics / TopicParquetStore.VERSION_FILE).write_text("0\n")
    assert "older format; re-run `evaluation process`" in infra.track_problem(episode, "env_0")


def test_empty_tracks_are_refused_and_not_kept(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_evaluation.processing.ped_track_store import empty_tracks

    episode = tmp_path / "run" / "episodes" / "episode_000"
    (episode / "topics").mkdir(parents=True)
    monkeypatch.setattr(infra, "has_tracks", lambda d, e: True)
    monkeypatch.setattr(infra, "tracks_for_episode", lambda d, e: empty_tracks())
    with pytest.raises(ValueError, match="the tracks of env_0 hold no walker"):
        infra.walker_times(episode, "env_0", 420.0)
    assert not infra.times_path(episode, "env_0", 420.0).exists()


def test_recompute_keeps_the_times_again_and_mends_mixed_commits(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"part1": ("aaaa",), "part2": ("bbbb",)})
    doc = json.loads(manifest.read_text())

    def at(commit: str | None) -> None:
        monkeypatch.setattr(infra, "_processing", lambda: {"path": "/e", "commit": commit, "dirty": False})

    at("a" * 40)
    infra.keep_times([site, dirs["part1"]], 420.0)
    at("b" * 40)
    infra.keep_times(dirs["part2"], 420.0)
    with pytest.raises(ValueError, match=r"different arena_evaluation commits.*evaluation infra times --recompute --benchmark <run dir>"):
        infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)
    infra.keep_times([site, dirs["part1"]], 420.0)  # without --recompute the kept times stay as they are
    with pytest.raises(ValueError, match="different arena_evaluation commits"):
        infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)
    assert infra.keep_times([site, dirs["part1"]], 420.0, recompute=True) == 6
    path = infra.times_path(dirs["part1"] / "episodes" / "episode_000", "env_0", 420.0)
    assert infra.times_provenance(path)["arena_evaluation"]["commit"] == "b" * 40
    assert infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)["complete"].all()
    # times whose commit git could not read are not compared, with a warning
    at(None)
    infra.keep_times([site, dirs["part1"], dirs["part2"]], 420.0, recompute=True)
    capsys.readouterr()
    assert infra.layout_table(site, [dirs["part1"], dirs["part2"]], doc)["complete"].all()
    assert "walker times kept without a readable arena_evaluation commit" in capsys.readouterr().err


def test_an_unreadable_code_source_warns(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import inspect

    def unreadable(obj: object) -> str:
        raise OSError("no source")

    monkeypatch.setattr(inspect, "getsource", unreadable)
    assert infra._times_code.__wrapped__() == "nosource"
    assert "kept walker times are not re-keyed when that code changes" in capsys.readouterr().err


def test_the_visitors_are_the_same_bits_in_any_row_order() -> None:
    rng = np.random.default_rng(1)
    peds = [("south_locker_wheelchair" if i < 600 else ("south_adult" if i % 3 else "north_wheel"), str(i), "adult" if i % 3 else "wheelchair_manual", 430.0 + float(rng.uniform(0, 50)), 430.0 + float(rng.uniform(100, 900))) for i in range(1500)]
    site = infra._times(_tracks(peds), 420.0)
    off = site.with_columns(pl.col("travel_s") * 1.0373)
    on = site.with_columns(pl.col("travel_s") * 1.0911)
    base = infra.cost_from_times(site, off, on, visitor_sources=["south_locker_wheelchair"])["visitors"]
    for k in range(10):
        shuffled = infra.cost_from_times(site, off.sample(fraction=1.0, shuffle=True, seed=k), on.sample(fraction=1.0, shuffle=True, seed=k + 1), visitor_sources=["south_locker_wheelchair"])["visitors"]
        assert shuffled == base


def test_a_row_without_a_completing_walker_is_not_complete(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import argparse

    site, dirs, manifest = _selection(tmp_path, monkeypatch, {"sel": ("aaaa", "bbbb")})
    doc = json.loads(manifest.read_text())
    assert infra.layout_table(site, dirs["sel"], doc)["complete"].all()
    # every walker spawns before a 1,500 s warm-up: tracks, but no walker times
    late = infra.layout_table(site, dirs["sel"], doc, warmup_s=1500.0)
    assert not late["complete"].any() and late["c_worst"].is_null().all()
    assert set(late["incomplete"].to_list()) == {"no walker spawned after the 1500 s warm-up and finished: the site as is, without robots, with the fleet"}
    # the command says why
    args = argparse.Namespace(infra="table", site=site, layouts=dirs["sel"], manifest=manifest, warmup=1500.0, tau=None, output_dir=None, out=None)
    assert infra.infra_command(args) == 0
    assert "4 (layout, seed) rows not complete (no walker spawned after the 1500 s warm-up and finished" in capsys.readouterr().out
    # a missing run says so too
    rows = infra.layout_table(site, dirs["sel"], {**doc, "seeds": [1, 2, 3]})
    assert rows.filter(pl.col("seed") == 3)["incomplete"].to_list() == ["missing runs: the site as is, without robots, with the fleet"] * 2


def test_kept_times_of_the_layout_before_this_version_are_never_read() -> None:
    assert infra.TIMES_VERSION == 3 and infra.times_dir().startswith("infra_times.v3.")


def test_keep_times_keeps_going_past_a_run_without_walkers_and_names_it(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_evaluation.processing.ped_track_store import empty_tracks

    _, dirs, _ = _selection(tmp_path, monkeypatch, {"sel": ("aaaa",)})
    real = infra.tracks_for_episode

    def one_empty(episode_dir: pathlib.Path, env: str) -> pl.DataFrame:
        return empty_tracks() if episode_dir.name == "episode_001" else real(episode_dir, env)

    monkeypatch.setattr(infra, "tracks_for_episode", one_empty)
    with pytest.raises(ValueError, match=r"kept the walker times of 3 runs, but could not keep 1: .*episode_001 \(env_0\): the tracks of env_0 hold no walker"):
        infra.keep_times(dirs["sel"], 420.0)
    kept = sorted(p.parent.parent.parent.parent.name for p in dirs["sel"].rglob("env_0.w420.0.parquet"))
    assert kept == ["episode_000", "episode_002", "episode_003"]
