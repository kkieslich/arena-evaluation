from __future__ import annotations

import pathlib
import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import ped_keys, strip_env
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

# arena_people_msgs/Pedestrian: an empty agent_type is an adult.
DEFAULT_AGENT_TYPE = "adult"

# Agent types scored together as one group, every other type is a group of its own.
TYPE_GROUPS = {
    "wheelchair_manual": "wheelchair",
    "wheelchair_electric": "wheelchair",
    "blind_white_cane": "long_cane",
    "blind_rolling_cane": "long_cane",
}

# Measures with a `<measure>_per_ped` output and the direction that is better, the other end names the worst-served group.
PED_MEASURES = {
    "ped_path_deflection_m": "lower",
    "ped_velocity_delay_ratio": "lower",
    "ped_round_trips_completed": "higher",
    "ped_mechanism_delay_seconds": "lower",
    "ped_max_exposure_dba": "lower",
    "ped_leq_exposure_dba": "lower",
    "ped_max_startle_factor": "lower",
}

# Fixed column types, so an episode without a value writes the same schema as one with values.
PED_MEASURE_DTYPES = {
    "ped_agent_types": pl.List(pl.Struct({"ped": pl.String, "agent_type": pl.String})),
    **{f"{measure}_per_ped": pl.List(pl.Struct({"ped": pl.String, "value": pl.Float64})) for measure in PED_MEASURES},
    **{f"{measure}_by_group": pl.List(pl.Struct({"group": pl.String, "value": pl.Float64, "peds": pl.Int64})) for measure in PED_MEASURES},
    **{f"{measure}_worst_group": pl.String for measure in PED_MEASURES},
}


def ped_agent_types(data: pl.DataFrame | None) -> dict[str, str]:
    """Agent type by pedestrian key without the env prefix, from the `peds_agent_types` recorded beside `peds_names`, {} without them.

    A pedestrian takes the first non-empty type recorded for it, `adult` when every one is empty: its first
    frames on `arena_peds` can precede its type.
    """
    if data is None or "peds_names" not in data.columns or "peds_agent_types" not in data.columns:
        return {}
    rows = data.select("peds_names", "peds_agent_types").filter(pl.col("peds_names").list.len() == pl.col("peds_agent_types").list.len()).unique(maintain_order=True)
    types: dict[str, str] = {}
    for names, agent_types in rows.iter_rows():
        for key, agent_type in zip(map(strip_env, ped_keys(names, len(agent_types))), agent_types, strict=True):
            if not types.get(key):
                types[key] = agent_type
    return {key: agent_type or DEFAULT_AGENT_TYPE for key, agent_type in types.items()}


def agent_group(agent_type: str) -> str:
    """The group an agent type is scored in, a type recorded as a yaml path by its file stem."""
    name = pathlib.PurePosixPath(agent_type).stem if "/" in agent_type or agent_type.endswith(".yaml") else agent_type
    return TYPE_GROUPS.get(name, name)


def by_group(records: list[dict[str, typing.Any]] | None, groups: dict[str, str]) -> list[dict[str, typing.Any]] | None:
    """Mean per-pedestrian value of each group as `{group, value, peds}` records by group name, None without a grouped value."""
    values: dict[str, list[float]] = {}
    for record in records or ():
        group = groups.get(record["ped"])
        if group is not None and record["value"] is not None:
            values.setdefault(group, []).append(float(record["value"]))
    return [{"group": group, "value": float(np.mean(v)), "peds": len(v)} for group, v in sorted(values.items())] or None


def worst_group(groups: list[dict[str, typing.Any]] | None, direction: str) -> str | None:
    """The group whose mean is worst for a measure where `direction` is better, None without groups or when several share it."""
    if not groups:
        return None
    worst = (max if direction == "lower" else min)(group["value"] for group in groups)
    named = [group["group"] for group in groups if group["value"] == worst]
    return named[0] if len(named) == 1 else None


class PedestrianGroupsCalculator(BaseMetricCalculator):
    """Every pedestrian measure per group of agent types, with the worst-served group."""

    NAME = "pedestrian_groups"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["pedestrian_disturbance", "mechanism_metrics", "acoustic_exposure"]
    REQUIRED_TOPICS = ["peds"]

    @classmethod
    def output_keys(cls) -> list[str]:
        return ["ped_agent_types", *(key for measure in PED_MEASURES for key in (f"{measure}_by_group", f"{measure}_worst_group"))]

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:
        agent_types = ped_agent_types(episode.data)
        results: dict[str, typing.Any] = {"ped_agent_types": [{"ped": key, "agent_type": agent_type} for key, agent_type in agent_types.items()] or None}
        ped_groups = {key: agent_group(agent_type) for key, agent_type in agent_types.items()}
        for measure, direction in PED_MEASURES.items():
            groups = by_group(prior_results.get(f"{measure}_per_ped"), ped_groups)
            results[f"{measure}_by_group"] = groups
            results[f"{measure}_worst_group"] = worst_group(groups, direction)
        return results
