import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import per_ped
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.social.pedestrian_groups import FLEET_DTYPES, PED_MEASURES, PedestrianGroupsCalculator, add_fleet_contacts, agent_group, by_group, ped_agent_types, record_keys, rekey_records, worst_group
from arena_evaluation.storage.schemas import AlignedEpisodeBundle, RobotParams


def _types(*rows: tuple[list[str], list[str]]) -> pl.DataFrame:
    return pl.DataFrame({"peds_names": [names for names, _ in rows], "peds_agent_types": [types for _, types in rows]}, schema={"peds_names": pl.List(pl.Utf8), "peds_agent_types": pl.List(pl.Utf8)})


def _records(**values: float | None) -> list[dict]:
    return [{"ped": ped, "value": value} for ped, value in values.items()]


def _calc(data: pl.DataFrame | None, prior: dict) -> dict:
    episode = AlignedEpisodeBundle(episode_id=1, data=data, start_pos=[], goal_pos=[])
    return PedestrianGroupsCalculator(RobotParams()).calculate(episode, prior)


def test_every_pedestrian_measure_has_a_per_ped_output():
    MetricRegistry.discover_calculators_cls()
    outputs = {key for cls in BaseMetricCalculator.__subclasses__() for key in cls.output_keys()}
    assert {f"{measure}_per_ped" for measure in PED_MEASURES} <= outputs


def test_output_keys_name_a_group_breakdown_and_worst_group_per_measure():
    keys = PedestrianGroupsCalculator.output_keys()
    assert keys[0] == "ped_agent_types"
    assert keys[1:] == [key for measure in PED_MEASURES for key in (f"{measure}_by_group", f"{measure}_worst_group")]


def test_pedestrian_takes_its_first_non_empty_type_and_is_an_adult_without_one():
    data = _types(
        (["/env_0/a", "/env_0/b", "/env_0/d"], ["", "wheelchair_manual", ""]),
        (["/env_0/a", "/env_0/c"], ["elder", "blind_white_cane"]),
        (["/env_0/a", "/env_0/c"], ["adult", "blind_white_cane"]),
    )
    assert ped_agent_types(data) == {"a": "elder", "b": "wheelchair_manual", "d": "adult", "c": "blind_white_cane"}


def test_pedestrians_are_keyed_without_their_env_prefix():
    data = _types((["/env_1/flow_3", "env_12/lobby/visitor", "kiosk_user"], ["elder", "adult", "adult"]))
    assert ped_agent_types(data) == {"flow_3": "elder", "lobby/visitor": "adult", "kiosk_user": "adult"}
    assert per_ped({"env_2/flow_3": 1.0, "flow_4": 2.0}) == [{"ped": "flow_3", "value": 1.0}, {"ped": "flow_4", "value": 2.0}]


def test_rows_whose_types_do_not_match_their_names_are_skipped():
    data = _types((["a", "b"], ["elder"]), (["a"], []), (["b"], ["wheelchair_manual"]))
    assert ped_agent_types(data) == {"b": "wheelchair_manual"}


def test_null_rows_are_skipped():
    data = pl.DataFrame({"peds_names": [None, ["a"]], "peds_agent_types": [None, ["elder"]]}, schema={"peds_names": pl.List(pl.Utf8), "peds_agent_types": pl.List(pl.Utf8)})
    assert ped_agent_types(data) == {"a": "elder"}


@pytest.mark.parametrize("data", [None, pl.DataFrame({"peds_names": [["a"]]}), pl.DataFrame({"peds_agent_types": [["elder"]]})], ids=["no frame", "no types", "no names"])
def test_recordings_without_types_type_nobody(data: pl.DataFrame | None):
    assert ped_agent_types(data) == {}


def test_group_value_is_the_mean_over_its_pedestrians():
    records = _records(a=1.0, b=3.0, c=4.0, d=None, stranger=9.0)
    groups = {"a": "adult", "b": "adult", "c": "wheelchair", "d": "wheelchair"}
    assert by_group(records, groups) == [{"group": "adult", "value": 2.0, "peds": 2}, {"group": "wheelchair", "value": 4.0, "peds": 1}]


@pytest.mark.parametrize(("records", "types"), [(None, {"a": "adult"}), ([], {"a": "adult"}), (_records(a=1.0), {}), (_records(a=None), {"a": "adult"})], ids=["no measure", "no pedestrians", "untyped", "no values"])
def test_group_breakdown_is_none_without_typed_values(records: list | None, types: dict):
    assert by_group(records, types) is None


def test_worst_group_is_the_far_end_of_the_better_direction():
    groups = [{"group": "adult", "value": 1.0, "peds": 3}, {"group": "elder", "value": 2.0, "peds": 1}, {"group": "wheelchair", "value": 0.5, "peds": 1}]
    assert worst_group(groups, "lower") == "elder"
    assert worst_group(groups, "higher") == "wheelchair"


def test_a_tie_for_worst_names_no_group():
    groups = [{"group": "adult", "value": 1.0, "peds": 3}, {"group": "elder", "value": 1.0, "peds": 1}, {"group": "wheelchair", "value": 0.5, "peds": 1}]
    assert worst_group(groups, "lower") is None
    assert worst_group(groups, "higher") == "wheelchair"


def test_a_single_group_is_the_worst_served():
    assert worst_group([{"group": "adult", "value": 0.0, "peds": 3}], "lower") == "adult"


def test_worst_group_is_none_without_groups():
    assert worst_group(None, "lower") is None


def test_calculator_reports_each_measure_per_group():
    data = _types((["/env_0/a", "/env_0/b", "/env_0/c"], ["adult", "wheelchair_manual", "adult"]))
    prior = {
        "ped_mechanism_delay_seconds_per_ped": _records(a=0.0, b=6.0, c=2.0),
        "ped_round_trips_completed_per_ped": _records(a=2.0, b=1.0, c=2.0),
        "ped_max_exposure_dba_per_ped": None,
    }
    result = _calc(data, prior)

    assert result["ped_agent_types"] == [{"ped": "a", "agent_type": "adult"}, {"ped": "b", "agent_type": "wheelchair_manual"}, {"ped": "c", "agent_type": "adult"}]
    assert result["ped_mechanism_delay_seconds_by_group"] == [{"group": "adult", "value": 1.0, "peds": 2}, {"group": "wheelchair", "value": 6.0, "peds": 1}]
    assert result["ped_mechanism_delay_seconds_worst_group"] == "wheelchair"
    assert result["ped_round_trips_completed_worst_group"] == "wheelchair"
    assert result["ped_max_exposure_dba_by_group"] is None
    assert result["ped_max_exposure_dba_worst_group"] is None
    assert result["ped_path_deflection_m_by_group"] is None


@pytest.mark.parametrize(
    ("agent_type", "group"),
    [
        ("wheelchair_manual", "wheelchair"),
        ("wheelchair_electric", "wheelchair"),
        ("blind_white_cane", "long_cane"),
        ("blind_rolling_cane", "long_cane"),
        ("blind", "blind"),
        ("elder", "elder"),
        ("/opt/arena_ws/scenarios/caller.yaml", "caller"),
        ("agent_types/wheelchair_manual.yaml", "wheelchair"),
        ("visitor.yaml", "visitor"),
    ],
)
def test_agent_types_fall_into_their_study_group(agent_type: str, group: str):
    assert agent_group(agent_type) == group


def test_calculator_groups_types_together_and_keeps_the_raw_type():
    data = _types((["a", "b", "c"], ["wheelchair_manual", "wheelchair_electric", "blind_white_cane"]))
    result = _calc(data, {"ped_mechanism_delay_seconds_per_ped": _records(a=2.0, b=4.0, c=1.0)})

    assert result["ped_agent_types"] == [{"ped": "a", "agent_type": "wheelchair_manual"}, {"ped": "b", "agent_type": "wheelchair_electric"}, {"ped": "c", "agent_type": "blind_white_cane"}]
    assert result["ped_mechanism_delay_seconds_by_group"] == [{"group": "long_cane", "value": 1.0, "peds": 1}, {"group": "wheelchair", "value": 3.0, "peds": 2}]
    assert result["ped_mechanism_delay_seconds_worst_group"] == "wheelchair"


def test_calculator_without_agent_types_reports_nothing():
    result = _calc(pl.DataFrame({"peds_names": [["a"]]}), {"ped_mechanism_delay_seconds_per_ped": _records(a=1.0)})
    assert result == dict.fromkeys(PedestrianGroupsCalculator.output_keys())


TYPES = [{"ped": "a", "agent_type": "adult"}, {"ped": "b", "agent_type": "wheelchair_manual"}, {"ped": "c", "agent_type": "wheelchair_electric"}]


def _robot(name: str, **contacts: float) -> dict:
    return {"robot": name, "ped_agent_types": TYPES, "ped_robot_contacts_per_ped": _records(**contacts)}


def test_fleet_contacts_sum_each_pedestrian_over_the_robots_of_an_env():
    rows = [_robot("env_0_jackal", a=1.0, b=0.0, c=0.0), _robot("env_0_burger", a=1.0, b=2.0, c=0.0)]
    add_fleet_contacts(rows)

    assert rows[0]["ped_fleet_contacts"] == 4
    assert rows[0]["ped_fleet_contacts_per_ped"] == _records(a=2.0, b=2.0, c=0.0)
    assert rows[0]["ped_fleet_contacts_by_group"] == [{"group": "adult", "value": 2.0, "peds": 1}, {"group": "wheelchair", "value": 1.0, "peds": 2}]
    assert rows[0]["ped_fleet_contacts_worst_group"] == "adult"
    assert rows[0]["ped_contact_free_rate"] == pytest.approx(1.0 / 3.0)
    assert rows[0]["ped_contact_free_rate_by_group"] == [{"group": "adult", "value": 0.0, "peds": 1}, {"group": "wheelchair", "value": 0.5, "peds": 2}]
    assert rows[0]["ped_contact_free_rate_worst_group"] == "adult"
    assert {key: rows[1][key] for key in FLEET_DTYPES} == {key: rows[0][key] for key in FLEET_DTYPES}


def test_fleet_contacts_keep_envs_apart():
    rows = [_robot("env_0_jackal", a=1.0), _robot("env_1_jackal", a=0.0)]
    add_fleet_contacts(rows)
    assert [(row["ped_fleet_contacts"], row["ped_contact_free_rate"]) for row in rows] == [(1, 0.0), (0, 1.0)]


def test_fleet_contacts_need_every_robots_contacts():
    rows = [_robot("env_0_jackal", a=1.0), {"robot": "env_0_burger", "status": "no_trajectory"}]
    add_fleet_contacts(rows)
    assert all(row[key] is None for row in rows for key in FLEET_DTYPES)


def test_fleet_without_pedestrians_has_no_contact_free_rate():
    rows = [_robot("env_0_jackal")]
    add_fleet_contacts(rows)
    assert (rows[0]["ped_fleet_contacts"], rows[0]["ped_contact_free_rate"], rows[0]["ped_contact_free_rate_by_group"]) == (0, None, None)


def _peds(names: list[list[str]], ids: list[list[int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time_ns": [10 * i for i in range(len(ids))],
            "peds_ids": ids,
            "peds_positions": [[0.0] * (3 * len(row)) for row in ids],
            "peds_names": names,
        },
        schema={"time_ns": pl.Int64, "peds_ids": pl.List(pl.Int64), "peds_positions": pl.List(pl.Float64), "peds_names": pl.List(pl.String)},
    )


def test_record_keys_follow_the_agent_id_to_its_track_key():
    peds = _peds([["/env_0/17", "env_0/Pedestrian_1", "env_0/stranger"]], [[17, 1, 5]])
    tracks = pl.DataFrame({"agent_id": [17, 17, 1], "key": ["north#3", "north#3", "Pedestrian_1"]})
    assert record_keys(peds, tracks) == {"17": "north#3", "Pedestrian_1": "Pedestrian_1"}


def test_records_without_tracks_keep_their_keys():
    assert record_keys(_peds([["env_0/17"]], [[17]]), None) == {}


def test_rekey_moves_records_and_types_onto_track_keys():
    row = {
        "ped_robot_contacts_per_ped": _records(**{"17": 1.0, "unknown": 2.0}),
        "ped_agent_types": [{"ped": "17", "agent_type": "elder"}],
        "ped_mechanism_delay_seconds_per_ped": None,
        "ped_robot_contacts": 3,
    }
    rekey_records(row, {"17": "north#3"})
    assert row == {
        "ped_robot_contacts_per_ped": _records(**{"north#3": 1.0, "unknown": 2.0}),
        "ped_agent_types": [{"ped": "north#3", "agent_type": "elder"}],
        "ped_mechanism_delay_seconds_per_ped": None,
        "ped_robot_contacts": 3,
    }
