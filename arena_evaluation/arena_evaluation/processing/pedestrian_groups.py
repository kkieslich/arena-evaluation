"""Pedestrian groups: the group each agent type is scored in, and a measure's mean and worst group across them."""

from __future__ import annotations

import pathlib
import typing

import numpy as np

# Agent types scored together as one group, every other type is a group of its own.
TYPE_GROUPS = {
    "wheelchair_manual": "wheelchair",
    "wheelchair_electric": "wheelchair",
    "blind_white_cane": "long_cane",
    "blind_rolling_cane": "long_cane",
}

# The per-pedestrian measures (columns of fact_pedestrians) scored per group, with the direction that is better: the
# other end names the worst-served group
PED_MEASURES = {
    "ped_path_deflection_m": "lower",
    "ped_velocity_delay_ratio": "lower",
    "round_trips_completed": "higher",
    "ped_max_exposure_dba": "lower",
    "ped_leq_exposure_dba": "lower",
    "ped_max_startle_factor": "lower",
    "ped_robot_contacts": "lower",
}

# The fleet measures, every robot of an env together, scored per group, with the direction that is better
FLEET_MEASURES = {
    "ped_fleet_contacts": "lower",
    "ped_contact_free_rate": "higher",
}


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
