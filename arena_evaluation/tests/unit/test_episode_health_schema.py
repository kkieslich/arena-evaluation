"""The native EpisodeRecord's health field is optional: a recording from before it (Stage 1, set A, set B) is still
native and has no health, one with it carries its entries into the episode_health table."""

from __future__ import annotations

import dataclasses
import types

import pytest

from arena_evaluation.processing import legacy_schemas

_NATIVE_BEFORE_HEALTH = (
    "uint32 episode_id\n"
    "string world\n"
    "int64 seed\n"
    "string task_layer\n"
    "diagnostic_msgs/KeyValue[] modes\n"
    "string[] robots\n"
    "uint8 RUNNING = 1\n"
    "uint8 outcome_state\n"
    "string outcome_info\n"
    "string goal_uuid\n"
    "float32 goal_dist_start\n"
    "float32 goal_dist_min\n"
    "float32 path_length\n"
    "builtin_interfaces/Time start_time\n"
    "string[] integrity\n"
    "rcl_interfaces/Parameter[] params\n"
    "rcl_interfaces/Parameter[] human_params\n"
    "string conditions\n"
    "string phases\n"
)
_NATIVE_WITH_HEALTH = _NATIVE_BEFORE_HEALTH + "# how the env ran\ndiagnostic_msgs/KeyValue[] health\n"


@dataclasses.dataclass(frozen=True)
class _Schema:
    name: str
    definition: str

    @property
    def data(self) -> bytes:
        return self.definition.encode("utf-8")


def _record(**overrides: object) -> types.SimpleNamespace:
    values: dict[str, object] = {
        "episode_id": 1,
        "world": "smu_link_full_v2__layout_7096a57e",
        "seed": 308,
        "task_layer": "task_generator",
        "modes": [],
        "robots": ["jackal"],
        "outcome_state": 3,
        "outcome_info": "timeout",
        "goal_uuid": "",
        "goal_dist_start": 0.0,
        "goal_dist_min": 0.0,
        "path_length": 0.0,
        "start_time": types.SimpleNamespace(sec=95, nanosec=0),
        "integrity": [],
        "params": [],
        "human_params": [],
        "conditions": "",
        "phases": "",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _kv(key: str, value: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(key=key, value=value)


def test_a_native_recording_from_before_the_health_field_is_native_without_health() -> None:
    schema = legacy_schemas.episode_schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_BEFORE_HEALTH)
    assert not schema.carries("health")
    fields = legacy_schemas.episode_fields(_record(), _Schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_BEFORE_HEALTH))
    assert fields.health == () and fields.not_recorded == ()


def test_a_native_recording_with_health_carries_its_entries_as_numbers() -> None:
    schema = legacy_schemas.episode_schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_WITH_HEALTH)
    assert schema.carries("health")
    record = _record(health=[_kv("setup_loop_lag_peak_s", "2.351"), _kv("episode_relay_ticks_coalesced", "25463.000"), _kv("broken", "n/a")])
    fields = legacy_schemas.episode_fields(record, _Schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_WITH_HEALTH))
    assert fields.health == (("setup_loop_lag_peak_s", 2.351), ("episode_relay_ticks_coalesced", 25463.0))


def test_any_other_unknown_native_field_is_still_loud() -> None:
    with pytest.raises(legacy_schemas.LegacySchemaError, match="surprise"):
        legacy_schemas.episode_schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_WITH_HEALTH + "string surprise\n")


def test_the_health_entries_become_episode_health_rows() -> None:
    from arena_evaluation.processing.telemetry.context import record_episode

    rows: dict[str, list[dict[str, object]]] = {}
    sink = types.SimpleNamespace(append=lambda table, row: rows.setdefault(table, []).append(row))
    envelope = types.SimpleNamespace(t_ns=95_000_000_000, ref=types.SimpleNamespace(env_id="env_0"))
    fields = legacy_schemas.episode_fields(_record(health=[_kv("episode_loop_lag_peak_s", "1.011")]), _Schema(legacy_schemas.NATIVE_EPISODE_SCHEMA, _NATIVE_WITH_HEALTH))
    record_episode(sink, envelope, fields)  # type: ignore[arg-type]
    assert [(row["measure"], row["value"]) for row in rows["episode_health"]] == [("episode_loop_lag_peak_s", 1.011)]
    assert len(rows["episode_lifecycle"]) == 1
