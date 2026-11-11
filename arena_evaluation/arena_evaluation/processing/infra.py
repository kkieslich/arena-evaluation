"""INFRA measures (study design §6, §8, §10, §12) over processed benchmark runs of `flowedit infra`'s suites.

Inputs are benchmark directories `evaluation process` has run on (their `combined_metrics.parquet`, each episode's
extracted `topics/`) and the `<suite>.layouts.json` manifest `flowedit infra` writes beside a layout suite:

- **Robot service** (§6), from the fleet mode's task log (the `fleet` semantic entity, schema in kkieslich/Arena's
  `task_generator/tasks/robots/README.md#fleet-task-log`): `T` = the restricted mean task time (release to
  completion) of the jobs released after the warm-up, up to a horizon fixed per study (default the episode minus the
  warm-up), from the Kaplan-Meier estimate with the jobs not completed censored at their elapsed time, so the slow jobs
  an episode cuts off still count (`restricted_mean`); the plain mean over completed jobs, the completed and censored
  shares are reported beside it; success rate = completed over released after the warm-up (jobs still open at the
  episode end apart); charger queue time (`dispatched - queued`); tasks per hour; failed and abandoned jobs, and the
  jobs failed by jams (`reason: jam`, §8); the time a job waited for a robot (`assigned - released`) and the time its
  robot held in place before busy docks (`wait_s`), both part of T.
- **Pedestrian cost** (§6): per group, `C_g` = mean extra travel time per pedestrian over the group's mean travel time
  in the site as is (a ratio of means), through-walkers only, paired by `(source_name, source_seq)`, counted when
  spawned after the warm-up and finished in all three runs: the site as is, the layout without robots and the layout
  with the fleet. The element part pairs the layout without robots with the site, the traffic part the layout with
  robots with the one without; all divide by the same reference travel time, so element + traffic = total.
  `C_worst` = max over groups. Locker visitors (the layout's own sources) are reported apart: their walk in the layout
  without robots and their traffic part.
- **The premise check** (§10) on the narrowest and the widest layout: their `C_worst` gap against the seed spread,
  over both layouts, and the task rate the selection runs at, 75 % of the fleet's measured capacity in the wide layout
  (`RATE_ROLE`); the narrow layout's capacity is reported, its service and cost count as layout cost.
- **The pick** (§5, §6): among the joint top 10, the layout with the lowest `C_worst` whose `T` is at most 1.05 times
  the robot-only pick's, both as means over the selection seeds; a layout with more than 20 % of its tasks failed by
  jams is dropped (§8).
- **The comparison** (§12) on the evaluation seeds: per seed the difference joint pick - robot-only pick of `C_worst`,
  `T` (relative to the robot-only pick's) and `C_aid - C_adult` per aid group, with 95 % bootstrap intervals over
  seeds (`experiments.stats.cluster_bootstrap`, a seed per cluster) and Holm across the two aid groups.
- **The forecaster check** (§12): the rank correlation (Spearman) between the ranking score and the full-episode
  `C_worst` over the joint top 10 and the random layouts.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import re
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.social.paired_peds import _travel_times
from arena_evaluation.processing.metrics.social.pedestrian_groups import agent_group
from arena_evaluation.processing.ped_track_store import tracks_for_episode

#: The warm-up the analysis leaves out and the episode length (Kolja, 9 Nov: 25 min with 7 min warm-up)
WARMUP_S = 420.0
#: §6: equal robot service is a mean task time within 5 % of the robot-only pick's
SERVICE_SLACK = 1.05
#: §8: a layout with more than this share of its tasks failed by jams is dropped
JAM_DROP = 0.20
#: The groups (§9) and the aid groups the comparison contrasts with adults (§12)
GROUPS = ("adult", "wheelchair", "long_cane")
AID_GROUPS = ("wheelchair", "long_cane")
#: The runner's contestant of the runs without robots
NO_ROBOTS = "unhindered_peds"
FLEET_KIND = "fleet"
TABLE_FILE = "infra_layouts.parquet"
FIGURE_FILE = "infra_service_cost.png"
_STAGE = re.compile(r"^(?P<world>.+)__s(?P<seed>\d+)$")
_VISITOR = re.compile(r"_locker_(?P<group>.+)$")


# --- runs ------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Run:
    """One recorded run of a stage: its contestant, world and seed, and where its episode's tracks are."""

    planner: str
    stage: str
    world: str
    seed: int
    episode: int
    env: str
    episode_dir: pathlib.Path


#: Processing statuses of a row whose episode has no usable recording (a failed attempt the runner retried)
_UNUSABLE = frozenset({"error", "no_recording"})


def runs(benchmark_dir: pathlib.Path) -> list[Run]:
    """Every usable run of a processed benchmark dir whose stage is `<world>__s<seed>`, one per (planner, stage,
    episode, env): rows of status `error` or `no_recording` are left out (a failed attempt the runner retried), and of
    the reference rows only the runs without robots (`unhindered_peds`). Check for a stage run twice with `unique`."""
    benchmark_dir = pathlib.Path(benchmark_dir)
    path = benchmark_dir / "combined_metrics.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{benchmark_dir}: no combined_metrics.parquet, run `evaluation process --benchmark-dir {benchmark_dir}` first")
    metrics = pl.read_parquet(path, columns=None)
    out: dict[tuple[str, str, int, str], Run] = {}
    columns = [c for c in ("planner", "stage", "episode", "robot", "status", "is_reference", "reference_type") if c in metrics.columns]
    for row in metrics.select(columns).iter_rows(named=True):
        match = _STAGE.match(str(row.get("stage") or ""))
        if match is None or row.get("episode") is None:
            continue
        if str(row.get("status") or "") in _UNUSABLE:
            continue
        if row.get("is_reference") and str(row.get("reference_type") or row.get("planner")) != NO_ROBOTS:
            continue
        env_match = re.search(r"(env_\d+)", str(row.get("robot") or ""))
        key = (str(row["planner"]), str(row["stage"]), int(row["episode"]), env_match.group(1) if env_match else "env_0")
        if key not in out:
            out[key] = Run(key[0], key[1], match["world"], int(match["seed"]), key[2], key[3], benchmark_dir / "episodes" / f"episode_{key[2]:03d}")
    return sorted(out.values(), key=lambda r: (r.world, r.seed, r.planner, r.episode, r.env))


def unique(candidates: typing.Sequence[Run], what: str) -> Run | None:
    """The one run of `candidates`, None without any; ValueError naming the stage and its episodes when a stage was
    run more than once (a retried step whose failed attempt still has a recording): the analysis never guesses."""
    if not candidates:
        return None
    if len({(r.episode, r.env) for r in candidates}) > 1:
        episodes = ", ".join(f"episode_{r.episode:03d} ({r.env})" for r in candidates)
        raise ValueError(f"{what}: stage {candidates[0].stage} has more than one usable run ({episodes}); remove the failed attempt or re-process")
    return candidates[0]


def restricted_mean(times: typing.Sequence[float], completed: typing.Sequence[bool], tau_s: float) -> float | None:
    """The restricted mean of a time to completion up to `tau_s` from the Kaplan-Meier estimate: the area under the
    survival curve S(t) (the share not yet completed) from 0 to `tau_s`. A time with `completed` False is right-censored
    there (a job still open at the episode end, or one that failed or was abandoned: its task time exceeds what was
    seen). At a tie, completions go before censorings; S stays at its last value up to `tau_s`. None without times."""
    pairs = sorted((float(t), bool(c)) for t, c in zip(times, completed, strict=True) if t is not None and math.isfinite(t))
    if not pairs:
        return None
    at_risk, survival, area, last = len(pairs), 1.0, 0.0, 0.0
    i = 0
    while i < len(pairs) and pairs[i][0] < tau_s:
        t = max(pairs[i][0], 0.0)
        area += survival * (t - last)
        last = t
        j, events = i, 0
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            events += pairs[j][1]
            j += 1
        if events:
            survival *= 1.0 - events / at_risk
        at_risk -= j - i
        i = j
    return area + survival * (tau_s - last)


# --- robot service ---------------------------------------------------------------------------------------------------


def fleet_log(episode_dir: pathlib.Path, env: str = "env_0") -> dict[str, str]:
    """The fleet mode's task log of `env` in an extracted episode: the last value of each field of its `fleet` entity,
    {} without one."""
    path = pathlib.Path(episode_dir) / "topics" / "semantic_snapshot.parquet"
    if not path.is_file():
        return {}
    snapshot = pl.read_parquet(path)
    return log_from_snapshot(snapshot, env)


def log_from_snapshot(snapshot: pl.DataFrame, env: str = "env_0") -> dict[str, str]:
    """The task log of `env`'s `fleet` entity in a semantic snapshot table (`mcap_reader` layout)."""
    if len(snapshot) == 0 or "kind" not in snapshot.columns:
        return {}
    rows = snapshot.filter(pl.col("kind") == FLEET_KIND)
    if "env_id" in rows.columns and rows["env_id"].n_unique() > 1:
        rows = rows.filter(pl.col("env_id") == int(env.rsplit("_", 1)[-1]))
    rows = rows.filter(pl.col("field") != "members").sort("time_ns")
    return {field: value for field, value in rows.select("field", pl.col("value_str").fill_null("")).iter_rows()}


def _table(log: typing.Mapping[str, str], prefix: str) -> list[dict[str, typing.Any]]:
    """The `<prefix>.<i>.<field>` records of a task log, by index."""
    out: dict[int, dict[str, typing.Any]] = {}
    for key, value in log.items():
        head, _, rest = key.partition(".")
        if head != prefix:
            continue
        index, _, field = rest.partition(".")
        out.setdefault(int(index), {"index": int(index)})[field] = value
    return [out[i] for i in sorted(out)]


def _f(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def jobs(log: typing.Mapping[str, str]) -> list[dict[str, typing.Any]]:
    """The jobs of a task log: index, kind, release, robot, assignment, load, end, outcome and reason, times as floats."""
    return [{**job, **{k: _f(job.get(k)) for k in ("released", "assigned", "loaded", "ended", "energy_wh", "timeout_0", "timeout_1", "wait_s")}} for job in _table(log, "job")]


def charges(log: typing.Mapping[str, str]) -> list[dict[str, typing.Any]]:
    """The charges of a task log, times and charges as floats."""
    return [{**c, **{k: _f(c.get(k)) for k in ("queued", "dispatched", "started", "ended", "soc_start", "soc_end")}} for c in _table(log, "charge")]


def service(log: typing.Mapping[str, str], *, warmup_s: float = WARMUP_S, episode_s: float | None = None, tau_s: float | None = None) -> dict[str, typing.Any]:
    """Robot service of one run (§6) over the jobs released after `warmup_s`.

    T (`task_time_s`) is the restricted mean task time up to `tau_s` (default `episode_s` - `warmup_s`) from the
    Kaplan-Meier estimate of the time from release to completion (`restricted_mean`): a completed job counts at
    `ended - released`, a job that failed or was abandoned is censored at `ended - released`, and a job still open at the
    episode end at `episode_s - released`. So the slow jobs an episode cuts off still count. The plain mean over the
    completed jobs is `task_time_completed_s`, and `completed_share` and `censored_share` say how many there were. Also:
    success rate (completed over those that ended), counts by outcome and of jam failures, the time a job waited for a
    robot and for docks, the charger queue time of charges queued after the warm-up, and completed tasks per hour of the
    scored time (`episode_s` - `warmup_s`). Without `episode_s` the open jobs are left out of T."""
    scored = [j for j in jobs(log) if j["released"] is not None and j["released"] >= warmup_s]
    ended = [j for j in scored if j["ended"] is not None]
    completed = [j for j in ended if j.get("outcome") == "completed"]
    jam = [j for j in ended if j.get("outcome") != "completed" and j.get("reason") == "jam"]
    queued = [c for c in charges(log) if c["queued"] is not None and c["queued"] >= warmup_s]
    waits = [c["dispatched"] - c["queued"] for c in queued if c["dispatched"] is not None]
    busy = [j["ended"] - j["assigned"] for j in completed if j["assigned"] is not None]
    assign_waits = [j["assigned"] - j["released"] for j in completed if j["assigned"] is not None]
    dock_waits = [j["wait_s"] for j in completed if j["wait_s"] is not None]
    charging = [c["ended"] - c["dispatched"] for c in queued if c["dispatched"] is not None and c["ended"] is not None and c.get("outcome") == "charged"]
    hours = None if episode_s is None or episode_s <= warmup_s else (episode_s - warmup_s) / 3600.0
    tau = tau_s if tau_s is not None else (episode_s - warmup_s if episode_s is not None else None)
    observed = [(j["ended"] - j["released"], j.get("outcome") == "completed") for j in ended]
    if episode_s is not None:
        observed += [(episode_s - j["released"], False) for j in scored if j["ended"] is None]
    rmst = restricted_mean([t for t, _ in observed], [c for _, c in observed], tau) if tau is not None and observed else None
    return {
        "task_time_s": rmst,
        "task_time_completed_s": float(np.mean([j["ended"] - j["released"] for j in completed])) if completed else None,
        "task_time_tau_s": tau,
        "completed_share": len(completed) / len(scored) if scored else None,
        "censored_share": (len(scored) - len(completed)) / len(scored) if scored else None,
        "jobs_released": len(scored),
        "jobs_completed": len(completed),
        "jobs_failed": sum(j.get("outcome") == "failed" for j in ended),
        "jobs_abandoned": sum(j.get("outcome") == "abandoned" for j in ended),
        "jobs_open": len(scored) - len(ended),
        "jobs_jammed": len(jam),
        "busy_s_per_job": float(np.mean(busy)) if busy else None,
        "assign_wait_s": float(np.mean(assign_waits)) if assign_waits else None,
        "dock_wait_s": float(np.mean(dock_waits)) if dock_waits else None,
        "success_rate": len(completed) / len(ended) if ended else None,
        "jam_share": len(jam) / len(ended) if ended else None,
        "charges": len(queued),
        "charger_queue_s": float(np.mean(waits)) if waits else None,
        "charge_s": float(np.sum(charging)) if charging else 0.0,
        "tasks_per_h": len(completed) / hours if hours else None,
        "empty_batteries": len(_table(log, "empty")),
    }


# --- pedestrian cost -------------------------------------------------------------------------------------------------


def _times(tracks: pl.DataFrame, warmup_s: float) -> pl.DataFrame:
    """key, source_name, agent_type and travel time of each pedestrian spawned after the warm-up that finished."""
    cut = tracks.filter(pl.col("t") >= warmup_s) if len(tracks) else tracks
    done = _travel_times(cut)
    sources = cut.group_by("key").agg(pl.col("source_name").first()) if len(cut) else pl.DataFrame(schema={"key": pl.String, "source_name": pl.String})
    return done.join(sources, on="key", how="left")


def _group(agent_type: str, source_name: str | None) -> str:
    """A pedestrian's group: a locker visitor's from its source name (`<end>_locker_<group>`), else its agent type's."""
    match = _VISITOR.search(source_name or "")
    return match["group"] if match else agent_group(agent_type)


def cost(site: pl.DataFrame, without: pl.DataFrame, robots: pl.DataFrame | None, *, visitor_sources: typing.Collection[str] = (), warmup_s: float = WARMUP_S) -> dict[str, typing.Any]:
    """The pedestrian cost of one layout at one seed (§6) from the track tables of the site as is, the layout without
    robots and (None for a premise or element-only check) the layout with robots: per group the paired through-walkers,
    the total, element and traffic parts of `C_g`, the absolute delay, and the drops; `C_worst` and its group; the locker
    visitors' walk without robots and their traffic part."""
    visitors = set(visitor_sources)
    ref = _times(site, warmup_s).filter(~pl.col("source_name").is_in(list(visitors)) & (pl.col("source_name") != ""))
    off = _times(without, warmup_s)
    on = _times(robots, warmup_s) if robots is not None else None
    walkers = ref.select("key", "agent_type", "source_name", pl.col("travel_s").alias("site_s"))
    joined = walkers.join(off.select("key", pl.col("travel_s").alias("off_s")), on="key", how="left")
    if on is not None:
        joined = joined.join(on.select("key", pl.col("travel_s").alias("on_s")), on="key", how="left")
    groups: dict[str, dict[str, typing.Any]] = {}
    for row in joined.iter_rows(named=True):
        group = _group(row["agent_type"], row["source_name"])
        g = groups.setdefault(group, {"peds": 0, "dropped": 0, "site_s": 0.0, "off_s": 0.0, "on_s": 0.0})
        # a through-walker that finished the site as is but not a layout run drops from the pair (§6), counted per group
        if row["off_s"] is None or (on is not None and row["on_s"] is None):
            g["dropped"] += 1
            continue
        g["peds"] += 1
        g["site_s"] += row["site_s"]
        g["off_s"] += row["off_s"]
        g["on_s"] += row["on_s"] if on is not None else row["off_s"]
    out: dict[str, typing.Any] = {"groups": {}}
    for group, g in sorted(groups.items()):
        ref_s = g["site_s"]
        record = {"peds": g["peds"], "dropped": g["dropped"], "site_travel_s": ref_s / g["peds"] if g["peds"] else None}
        if ref_s > 0:
            record["element"] = (g["off_s"] - ref_s) / ref_s
            record["traffic"] = (g["on_s"] - g["off_s"]) / ref_s if on is not None else None
            record["total"] = (g["on_s"] - ref_s) / ref_s if on is not None else record["element"]
            record["delay_s"] = ((g["on_s"] if on is not None else g["off_s"]) - ref_s) / g["peds"]
        else:
            record.update(element=None, traffic=None, total=None, delay_s=None)
        out["groups"][group] = record
    totals = {group: r["total"] for group, r in out["groups"].items() if r["total"] is not None}
    out["c_worst"] = max(totals.values()) if totals else None
    out["worst_group"] = max(totals, key=lambda g: totals[g]) if totals else None
    delays = {group: r["delay_s"] for group, r in out["groups"].items() if r["delay_s"] is not None}
    out["worst_group_delay_s"] = max(delays, key=lambda g: delays[g]) if delays else None
    out["visitors"] = _visitors(off, on, visitors)
    return out


def _visitors(off: pl.DataFrame, on: pl.DataFrame | None, sources: set[str]) -> dict[str, typing.Any]:
    """The locker visitors who finished without robots: their count and mean walk per group, and their traffic part
    (with robots against without, over their own walk) where they finished both."""
    if not sources:
        return {}
    mine = off.filter(pl.col("source_name").is_in(list(sources)))
    out: dict[str, typing.Any] = {}
    for row in mine.iter_rows(named=True):
        g = out.setdefault(_group(row["agent_type"], row["source_name"]), {"peds": 0, "walk_s": 0.0, "paired": 0, "off_s": 0.0, "on_s": 0.0})
        g["peds"] += 1
        g["walk_s"] += row["travel_s"]
    if on is not None and len(mine):
        both = mine.select("key", "agent_type", "source_name", pl.col("travel_s").alias("off_s")).join(on.select("key", pl.col("travel_s").alias("on_s")), on="key")
        for row in both.iter_rows(named=True):
            g = out[_group(row["agent_type"], row["source_name"])]
            g["paired"] += 1
            g["off_s"] += row["off_s"]
            g["on_s"] += row["on_s"]
    return {group: {"peds": g["peds"], "walk_s": g["walk_s"] / g["peds"], "paired": g["paired"], "traffic": (g["on_s"] - g["off_s"]) / g["off_s"] if g["off_s"] > 0 else None} for group, g in sorted(out.items())}


# --- tables ----------------------------------------------------------------------------------------------------------


def layout_table(site_dir: pathlib.Path, layouts_dir: pathlib.Path, manifest: typing.Mapping[str, typing.Any], *, warmup_s: float | None = None, fleet: str | None = None, tau_s: float | None = None) -> pl.DataFrame:
    """One row per layout of `manifest` and seed: its roles, the service of its fleet run, its pedestrian cost against
    the site as is at that seed (`cost`), flattened (`C_<group>`, `C_<group>_element`, `C_<group>_traffic`,
    `delay_<group>_s`, `peds_<group>`, `dropped_<group>`, `c_worst`, `worst_group`, `visitor_*`)."""
    warmup = float(manifest.get("warmup_s", WARMUP_S) if warmup_s is None else warmup_s)
    episode_s = float(manifest.get("episode_s") or 0.0) or None
    visitor_sources = list(manifest.get("visitor_sources") or [])
    site_runs = runs(site_dir)
    layout_runs = runs(layouts_dir)
    roles: dict[str, list[str]] = {}
    ranking: dict[str, float | None] = {}
    travel: dict[str, float | None] = {}
    for record in manifest.get("runs") or []:
        roles.setdefault(record["layout_id"], []).append(record["role"])
        ranking.setdefault(record["layout_id"], record.get("ranking_score"))
        travel.setdefault(record["layout_id"], record.get("travel_m"))
    world_of = {layout_id: pathlib.Path(path).name for layout_id, path in (manifest.get("worlds") or {}).items()}
    rows = []
    site_name = str(manifest["site"])
    for layout_id, world in world_of.items():
        for seed in manifest.get("seeds") or []:
            off = unique([r for r in layout_runs if r.world == world and r.seed == seed and r.planner == NO_ROBOTS], f"{layouts_dir} (without robots)")
            on = unique([r for r in layout_runs if r.world == world and r.seed == seed and r.planner != NO_ROBOTS and (fleet is None or r.planner == fleet)], f"{layouts_dir} (with the fleet)")
            ref = unique([r for r in site_runs if r.world == site_name and r.seed == seed], f"{site_dir} (the site as is)")
            row: dict[str, typing.Any] = {"layout_id": layout_id, "world": world, "seed": int(seed), "roles": ",".join(roles.get(layout_id, [])), "ranking_score": ranking.get(layout_id), "travel_m": travel.get(layout_id), "complete": off is not None and on is not None and ref is not None}
            if on is not None:
                row.update(service(fleet_log(on.episode_dir, on.env), warmup_s=warmup, episode_s=episode_s, tau_s=tau_s))
            if off is not None and ref is not None:
                measured = cost(tracks_for_episode(ref.episode_dir, ref.env), tracks_for_episode(off.episode_dir, off.env), None if on is None else tracks_for_episode(on.episode_dir, on.env), visitor_sources=visitor_sources, warmup_s=warmup)
                row.update(_flatten(measured))
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _flatten(measured: typing.Mapping[str, typing.Any]) -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {"c_worst": measured["c_worst"], "worst_group": measured["worst_group"], "worst_group_delay_s": measured["worst_group_delay_s"]}
    for group, r in measured["groups"].items():
        out |= {f"C_{group}": r["total"], f"C_{group}_element": r["element"], f"C_{group}_traffic": r["traffic"], f"delay_{group}_s": r["delay_s"], f"peds_{group}": r["peds"], f"dropped_{group}": r["dropped"], f"site_travel_{group}_s": r["site_travel_s"]}
    for group, r in (measured.get("visitors") or {}).items():
        out |= {f"visitor_{group}_peds": r["peds"], f"visitor_{group}_walk_s": r["walk_s"], f"visitor_{group}_traffic": r["traffic"]}
    return out


def _interval(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """The 95 % percentile bootstrap interval of the mean over seeds, (nan, nan) for fewer than 2 values."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def layout_means(table: pl.DataFrame) -> pl.DataFrame:
    """One row per layout: the equal-weight mean over its seeds of `T` (`task_time_s`), every `C_<group>` and the
    service counts, `C_worst` as the max over the groups' means, the seeds counted, and 95 % seed-bootstrap intervals of
    `T` and of the per-seed `c_worst` (the figure's error bars)."""
    if len(table) == 0:
        return table
    numeric = [c for c, dtype in table.schema.items() if dtype.is_numeric() and c not in ("seed",)]
    means = table.group_by("layout_id", "world", "roles", maintain_order=True).agg(*(pl.col(c).mean() for c in numeric), pl.len().alias("seeds"), pl.col("complete").all())
    groups = [g for g in GROUPS if f"C_{g}" in means.columns]
    means = means.with_columns(pl.max_horizontal([pl.col(f"C_{g}") for g in groups]).alias("c_worst_of_means") if groups else pl.lit(None).alias("c_worst_of_means"))
    if groups:
        means = means.with_columns(pl.concat_list([pl.col(f"C_{g}") for g in groups]).list.arg_max().map_elements(lambda i: groups[i] if i is not None else None, return_dtype=pl.String).alias("worst_group_of_means"))
    intervals = []
    for layout_id in means["layout_id"].to_list():
        sub = table.filter(pl.col("layout_id") == layout_id)
        t = _interval(sub["task_time_s"].cast(pl.Float64).to_numpy() if "task_time_s" in sub.columns else np.array([]))
        c = _interval(sub["c_worst"].cast(pl.Float64).to_numpy() if "c_worst" in sub.columns else np.array([]))
        intervals.append({"layout_id": layout_id, "T_lo": t[0], "T_hi": t[1], "c_worst_lo": c[0], "c_worst_hi": c[1]})
    return means.join(pl.DataFrame(intervals), on="layout_id", how="left")


def pick(means: pl.DataFrame, *, slack: float = SERVICE_SLACK, jam_drop: float = JAM_DROP) -> dict[str, typing.Any]:
    """The selection pick (§5, §6): the robot-only pick's mean `T`, and among the joint top 10 that are not dropped for
    jams and whose mean `T` is within `slack` of it, the one with the lowest `C_worst` (the max over the groups' means).
    ValueError without a robot-only run or a joint layout."""
    rows = means.to_dicts()
    robot = next((r for r in rows if "robot_only" in str(r["roles"]).split(",")), None)
    if robot is None or robot.get("task_time_s") is None:
        raise ValueError("no robot-only layout with a task time among the runs")
    t_cap = slack * float(robot["task_time_s"])
    considered = []
    for r in rows:
        if "joint" not in str(r["roles"]).split(","):
            continue
        reasons = []
        if r.get("task_time_s") is None or r.get("c_worst_of_means") is None:
            reasons.append("not measured")
        else:
            if r["task_time_s"] > t_cap:
                reasons.append(f"T {r['task_time_s']:.1f} s over {t_cap:.1f} s")
            if (r.get("jam_share") or 0.0) > jam_drop:
                reasons.append(f"jams {r['jam_share']:.0%} of tasks")
        considered.append({"layout_id": r["layout_id"], "task_time_s": r.get("task_time_s"), "c_worst": r.get("c_worst_of_means"), "worst_group": r.get("worst_group_of_means"), "excluded": "; ".join(reasons) or None})
    eligible = [c for c in considered if c["excluded"] is None]
    if not considered:
        raise ValueError("no joint layout among the runs")
    best = min(eligible, key=lambda c: (c["c_worst"], c["layout_id"])) if eligible else None
    return {
        "robot_only": robot["layout_id"],
        "robot_only_task_time_s": robot["task_time_s"],
        "robot_only_c_worst": robot.get("c_worst_of_means"),
        "task_time_cap_s": t_cap,
        "joint_pick": None if best is None else best["layout_id"],
        "joint_pick_task_time_s": None if best is None else best["task_time_s"],
        "joint_pick_c_worst": None if best is None else best["c_worst"],
        "joint": considered,
    }


def spearman(x: typing.Sequence[float], y: typing.Sequence[float]) -> float:
    """Spearman's rank correlation with average ranks for ties, nan for fewer than 3 pairs or a constant side."""
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return math.nan

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        out = np.empty(len(values))
        sorted_values = np.asarray(values)[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
                j += 1
            out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def forecaster_check(means: pl.DataFrame) -> dict[str, typing.Any]:
    """§12's forecaster check: Spearman between the ranking score and the full-episode `C_worst` over the joint top 10
    and the random layouts."""
    rows = [r for r in means.to_dicts() if {"joint", "random"} & set(str(r["roles"]).split(","))]
    return {"layouts": len(rows), "spearman": spearman([r.get("ranking_score") for r in rows], [r.get("c_worst_of_means") for r in rows])}


def compare(table: pl.DataFrame, picks: typing.Mapping[str, str], *, slack: float = SERVICE_SLACK, n_boot: int = 10_000, seed: int = 0) -> dict[str, typing.Any]:
    """§12 on the evaluation seeds: per seed joint pick - robot-only pick of `c_worst`, of `T` over the robot-only
    pick's `T`, and of `C_aid - C_adult` per aid group; 95 % bootstrap intervals over seeds, Holm across the aid groups;
    H1 holds when the interval of the `C_worst` difference lies below 0 and the relative `T` difference stays within
    `slack` - 1, "most for aid users" when an aid group's Holm-adjusted p < 0.05 with its interval below 0."""
    from arena_evaluation.experiments.stats import cluster_bootstrap, holm

    joint, robot = table.filter(pl.col("layout_id") == picks["joint"]), table.filter(pl.col("layout_id") == picks["robot_only"])
    pairs = joint.join(robot, on="seed", suffix="_ro")
    rng = np.random.default_rng(seed)

    def boot(values: np.ndarray) -> dict[str, typing.Any]:
        values = values[np.isfinite(values)]
        res = cluster_bootstrap([np.array([v]) for v in values], n_boot=n_boot, rng=rng)
        return {"estimate": res.estimate, "ci_low": res.ci_low, "ci_high": res.ci_high, "p": res.p, "seeds": res.n_pairs}

    def col(name: str) -> np.ndarray:
        return pairs[name].cast(pl.Float64).fill_null(float("nan")).to_numpy() if name in pairs.columns else np.full(len(pairs), np.nan)

    out: dict[str, typing.Any] = {"picks": dict(picks), "seeds": pairs["seed"].to_list()}
    out["delta_c_worst"] = boot(col("c_worst") - col("c_worst_ro"))
    out["delta_t_rel"] = boot((col("task_time_s") - col("task_time_s_ro")) / col("task_time_s_ro"))
    aid = {}
    for group in AID_GROUPS:
        aid[group] = boot((col(f"C_{group}") - col(f"C_{group}_ro")) - (col("C_adult") - col("C_adult_ro")))
    adjusted = holm([aid[g]["p"] for g in AID_GROUPS])
    for group, p in zip(AID_GROUPS, adjusted, strict=True):
        aid[group]["p_holm"] = p
        aid[group]["holds"] = bool(aid[group]["ci_high"] < 0 and p < 0.05) if math.isfinite(aid[group]["ci_high"]) and math.isfinite(p) else False
    out["delta_aid_minus_adult"] = aid
    d, t = out["delta_c_worst"], out["delta_t_rel"]
    out["h1_holds"] = bool(math.isfinite(d["ci_high"]) and d["ci_high"] < 0 and math.isfinite(t["estimate"]) and abs(t["estimate"]) <= slack - 1.0)
    return out


def figure(means: pl.DataFrame, picks: typing.Mapping[str, typing.Any], out: pathlib.Path) -> pathlib.Path:
    """§12's figure: robot task time T against C_worst, one point per selection layout with 95 % seed intervals, the
    robot-only and the joint pick marked, the site as is the line C = 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.axhline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
    ax.text(0.99, 0.0, "site as is (C = 0)", color="0.35", fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
    styles = {"random": ("o", "#9aa0a6", "random valid"), "joint": ("s", "#4c78a8", "joint top 10"), "robot_only": ("D", "#e45756", "robot-only pick")}
    seen: set[str] = set()
    for r in means.to_dicts():
        if r.get("task_time_s") is None or r.get("c_worst_of_means") is None:
            continue
        roles = str(r["roles"]).split(",")
        role = "robot_only" if "robot_only" in roles else ("joint" if "joint" in roles else "random")
        marker, colour, label = styles[role]
        x, y = r["task_time_s"] / 60.0, r["c_worst_of_means"]
        xerr = None if r.get("T_lo") is None or not math.isfinite(r["T_lo"]) else [[x - r["T_lo"] / 60.0], [r["T_hi"] / 60.0 - x]]
        yerr = None if r.get("c_worst_lo") is None or not math.isfinite(r["c_worst_lo"]) else [[max(0.0, y - r["c_worst_lo"])], [max(0.0, r["c_worst_hi"] - y)]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, color=colour, ecolor=colour, elinewidth=0.8, capsize=2, ms=6, label=None if role in seen else label, zorder=3)
        seen.add(role)
        if r["layout_id"] == picks.get("joint_pick"):
            ax.scatter([x], [y], s=180, facecolors="none", edgecolors="#f58518", linewidths=1.8, zorder=4, label="joint pick")
    if picks.get("task_time_cap_s"):
        ax.axvline(picks["task_time_cap_s"] / 60.0, color="#e45756", lw=0.8, ls=":", zorder=1)
        ax.text(picks["task_time_cap_s"] / 60.0, 0.97, " 1.05 T(robot-only)", color="#e45756", fontsize=8, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("robot task time T (min, mean over selection seeds)")
    ax.set_ylabel("C_worst (extra travel time / reference)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


#: The premise layout whose measured fleet capacity sets the task rate: `flowedit infra premise`'s layout with every
#: element at the widest section (its roles are `narrow` and `wide`)
RATE_ROLE = "wide"


def premise(table: pl.DataFrame, *, utilisation: float = 0.75, robots: int = 4, rate_role: str = RATE_ROLE) -> dict[str, typing.Any]:
    """§10's premise check: the two layouts' mean `C_worst` against the seed spread (the larger of their per-seed
    standard deviations), and the task rate at `utilisation` of the fleet's measured capacity in the layout with role
    `rate_role`, the wide one: robots x 3600 over (busy seconds per completed job + charging seconds per completed job).

    Both layouts stay in the `C_worst` comparison, and each reports its own measured capacity; only the wide one sets the
    rate. The narrow one's crowd cost and failed jobs are layout cost (at the study density it may complete no job at
    all, which would give a rate of about 0). ValueError unless exactly one layout has role `rate_role`."""
    out: dict[str, typing.Any] = {"layouts": {}}
    for layout_id, sub in table.group_by("layout_id", maintain_order=True):
        c = sub["c_worst"].cast(pl.Float64).drop_nulls().to_numpy() if "c_worst" in sub.columns else np.array([])
        completed = float(sub["jobs_completed"].sum()) if "jobs_completed" in sub.columns else 0.0
        busy = sub["busy_s_per_job"].cast(pl.Float64).drop_nulls().to_numpy() if "busy_s_per_job" in sub.columns else np.array([])
        charge = float(sub["charge_s"].sum()) if "charge_s" in sub.columns else 0.0
        per_job = (float(np.mean(busy)) + (charge / completed if completed else 0.0)) if len(busy) else None
        roles = sorted({role for value in sub["roles"].drop_nulls().to_list() for role in str(value).split(",") if role}) if "roles" in sub.columns else []
        out["layouts"][layout_id[0] if isinstance(layout_id, tuple) else layout_id] = {
            "roles": roles,
            "c_worst": float(c.mean()) if len(c) else None,
            "c_worst_sd": float(c.std(ddof=1)) if len(c) > 1 else None,
            "seeds": len(c),
            "jobs_completed": int(completed),
            "busy_s_per_job": per_job,
            "rate_per_h": round(utilisation * robots * 3600.0 / per_job, 2) if per_job else None,
        }
    layouts = list(out["layouts"].values())
    measured = [r for r in layouts if r["c_worst"] is not None]
    if len(measured) >= 2:
        spread = max((r["c_worst_sd"] or 0.0) for r in measured)
        gap = max(r["c_worst"] for r in measured) - min(r["c_worst"] for r in measured)
        out |= {"c_worst_gap": gap, "seed_spread": spread, "raise_density": bool(gap < spread)}
    sets_rate = [layout_id for layout_id, r in out["layouts"].items() if rate_role in r["roles"]]
    if len(sets_rate) != 1:
        found = ", ".join(f"{layout_id} ({','.join(r['roles']) or 'no role'})" for layout_id, r in out["layouts"].items())
        raise ValueError(f"the task rate comes from the premise layout with role {rate_role!r}, and {len(sets_rate)} layouts have it: {found or 'no layouts'}; read the premise suite's manifest from `flowedit infra premise`")
    out["rate_layout"] = sets_rate[0]
    out["rate_per_h"] = out["layouts"][sets_rate[0]]["rate_per_h"]
    return out


def _plain(value: typing.Any) -> typing.Any:
    """`value` with non-finite floats as None and numpy scalars as Python ones, for JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(data: typing.Any, path: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_text(json.dumps(_plain(data), indent=1, default=str) + "\n")
    return path


# --- the command -----------------------------------------------------------------------------------------------------


def _manifest(path: pathlib.Path) -> dict[str, typing.Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no manifest {path}: `flowedit infra` writes <suite>.layouts.json beside the suite")
    return json.loads(path.read_text())


def _fmt(value: typing.Any, spec: str = ".3f") -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else format(value, spec)


def _print_layouts(means: pl.DataFrame) -> None:
    groups = [g for g in GROUPS if f"C_{g}" in means.columns]
    print(f"{'layout':10s} {'roles':16s} {'seeds':>5s} {'T s':>7s} {'done':>5s} {'success':>7s} {'queue s':>7s} {'tasks/h':>7s} {'jams':>5s} " + " ".join(f"{'C_' + g:>12s}" for g in groups) + f" {'C_worst':>8s} worst")
    for r in means.to_dicts():
        print(
            f"{r['layout_id']:10s} {str(r['roles'])[:16]:16s} {r['seeds']:5d} {_fmt(r.get('task_time_s'), '.1f'):>7s} {_fmt(r.get('completed_share'), '.2f'):>5s} {_fmt(r.get('success_rate'), '.2f'):>7s} {_fmt(r.get('charger_queue_s'), '.1f'):>7s} {_fmt(r.get('tasks_per_h'), '.1f'):>7s} {_fmt(r.get('jam_share'), '.2f'):>5s} "
            + " ".join(f"{_fmt(r.get(f'C_{g}')):>12s}" for g in groups)
            + f" {_fmt(r.get('c_worst_of_means')):>8s} {r.get('worst_group_of_means') or '-'}"
        )
        parts = ", ".join(f"{g} element {_fmt(r.get(f'C_{g}_element'))} traffic {_fmt(r.get(f'C_{g}_traffic'))} delay {_fmt(r.get(f'delay_{g}_s'), '.1f')} s" for g in groups)
        if parts:
            print(f"{'':10s} {parts}")


def infra_command(args: typing.Any) -> int:
    """`evaluation infra {table,premise,pick,compare}`; see `add_infra_parser`."""
    manifest = _manifest(args.manifest)
    table = layout_table(args.site, args.layouts, manifest, warmup_s=args.warmup, tau_s=args.tau)
    out_dir = pathlib.Path(args.output_dir or args.layouts)
    table.write_parquet(out_dir / TABLE_FILE)
    means = layout_means(table)
    print(f"{len(table)} (layout, seed) rows from {args.layouts} against the site as is in {args.site}; warm-up {args.warmup if args.warmup is not None else manifest.get('warmup_s', WARMUP_S):g} s; per seed in {out_dir / TABLE_FILE}")
    incomplete = table.filter(~pl.col("complete")) if "complete" in table.columns else table.head(0)
    if len(incomplete):
        print(f"missing runs for {len(incomplete)} (layout, seed) rows: " + ", ".join(f"{r['layout_id']}@s{r['seed']}" for r in incomplete.to_dicts()))
    _print_layouts(means)
    if args.infra == "premise":
        result = premise(table)
        print(json.dumps(_plain(result), indent=1))
        if result.get("raise_density"):
            print("C_worst of the two layouts differs by less than the seed spread: raise the density and rerun the premise check (§10)")
        if result.get("rate_per_h"):
            print(f"task rate at 75 % of the measured capacity of the wide layout {result['rate_layout']}: {result['rate_per_h']:g} jobs/h (pass it as `flowedit infra layouts --rate`)")
        else:
            print(f"the wide layout {result['rate_layout']} completed no job: no measured task rate")
        write_json(result, out_dir / "infra_premise.json")
    elif args.infra == "pick":
        chosen = pick(means)
        chosen["forecaster_check"] = forecaster_check(means)
        out = pathlib.Path(args.out or out_dir / "infra_pick.json")
        write_json(chosen, out)
        png = figure(means, chosen, out_dir / "plots" / FIGURE_FILE)
        print(f"robot-only {chosen['robot_only']}: T {_fmt(chosen['robot_only_task_time_s'], '.1f')} s (cap {_fmt(chosen['task_time_cap_s'], '.1f')} s), C_worst {_fmt(chosen['robot_only_c_worst'])}")
        for c in chosen["joint"]:
            print(f"  joint {c['layout_id']}: T {_fmt(c['task_time_s'], '.1f')} s, C_worst {_fmt(c['c_worst'])} ({c['worst_group'] or '-'}){'' if c['excluded'] is None else ' - excluded: ' + c['excluded']}")
        print(f"joint pick: {chosen['joint_pick'] or 'none (no joint layout meets the service constraint)'}; forecaster check: Spearman {_fmt(chosen['forecaster_check']['spearman'])} over {chosen['forecaster_check']['layouts']} layouts")
        print(f"wrote {out} and {png}; next: flowedit infra evaluation --pick {out}")
        return 0 if chosen["joint_pick"] else 1
    elif args.infra == "compare":
        picks = manifest.get("picks") or {}
        if not picks.get("joint") or not picks.get("robot_only"):
            raise ValueError(f"{args.manifest}: no picks; compare reads the evaluation manifest `flowedit infra evaluation` writes")
        result = compare(table, {"joint": picks["joint"], "robot_only": picks["robot_only"]}, n_boot=args.boot)
        out = write_json(result, pathlib.Path(args.out or out_dir / "infra_compare.json"))
        d, t = result["delta_c_worst"], result["delta_t_rel"]
        print(f"joint {picks['joint']} - robot-only {picks['robot_only']} over {d['seeds']} seeds: dC_worst {_fmt(d['estimate'])} [{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}], dT/T {_fmt(t['estimate'])} [{_fmt(t['ci_low'])}, {_fmt(t['ci_high'])}] -> H1 {'holds' if result['h1_holds'] else 'does not hold'}")
        for group, r in result["delta_aid_minus_adult"].items():
            print(f"  d(C_{group} - C_adult) {_fmt(r['estimate'])} [{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}], Holm p {_fmt(r['p_holm'])} -> {'most for ' + group if r['holds'] else 'not shown'}")
        print(f"wrote {out}")
    return 0


def add_infra_parser(subparsers: typing.Any) -> None:
    """The `infra` command: INFRA's robot service, pedestrian cost, premise check, pick and comparison."""
    import argparse

    p = subparsers.add_parser("infra", help="INFRA (study design §6, §10, §12): robot service and pedestrian cost per layout, the premise check, the selection pick, the evaluation comparison")
    sub = p.add_subparsers(dest="infra", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", type=pathlib.Path, required=True, help="processed benchmark dir of the site as is (flowedit infra reference) on the same seeds")
    common.add_argument("--layouts", type=pathlib.Path, required=True, help="processed benchmark dir of a layout suite (premise, selection or evaluation)")
    common.add_argument("--manifest", type=pathlib.Path, required=True, help="the suite's <suite>.layouts.json from flowedit infra")
    common.add_argument("--warmup", type=float, default=None, help=f"seconds left out at the start (default the manifest's, else {WARMUP_S:g})")
    common.add_argument("--tau", type=float, default=None, help="horizon of the restricted mean task time T, s (default the episode minus the warm-up; one value for a whole study)")
    common.add_argument("--output-dir", type=pathlib.Path, default=None, help="where the table, JSON and plots go (default --layouts)")
    common.add_argument("--out", type=pathlib.Path, default=None, help="the pick or comparison JSON (default <output dir>/infra_pick.json, infra_compare.json)")
    sub.add_parser("table", parents=[common], help="print service and cost per layout, write infra_layouts.parquet (per seed)")
    sub.add_parser("premise", parents=[common], help="§10: C_worst of the two layouts against the seed spread, the task rate at 75 %% of the wide layout's measured capacity")
    sub.add_parser("pick", parents=[common], help="§6: the joint pick among the joint top 10 at equal service, the forecaster check, the T vs C_worst figure")
    q = sub.add_parser("compare", parents=[common], help="§12: joint pick - robot-only pick on the evaluation seeds, with bootstrap intervals and Holm")
    q.add_argument("--boot", type=int, default=10_000, help="bootstrap draws (default 10000)")
