import polars as pl
import pytest

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.registry import MetricRegistry
from arena_evaluation.processing.metrics.social.pedestrian_groups import PED_MEASURES, PedestrianGroupsCalculator, by_group, ped_agent_types, worst_group
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
    assert ped_agent_types(data) == {"env_0/a": "elder", "env_0/b": "wheelchair_manual", "env_0/d": "adult", "env_0/c": "blind_white_cane"}


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
    types = {"a": "adult", "b": "adult", "c": "wheelchair_manual", "d": "wheelchair_manual"}
    assert by_group(records, types) == [{"group": "adult", "value": 2.0, "peds": 2}, {"group": "wheelchair_manual", "value": 4.0, "peds": 1}]


@pytest.mark.parametrize(("records", "types"), [(None, {"a": "adult"}), ([], {"a": "adult"}), (_records(a=1.0), {}), (_records(a=None), {"a": "adult"})], ids=["no measure", "no pedestrians", "untyped", "no values"])
def test_group_breakdown_is_none_without_typed_values(records: list | None, types: dict):
    assert by_group(records, types) is None


def test_worst_group_is_the_far_end_of_the_better_direction():
    groups = [{"group": "adult", "value": 1.0, "peds": 3}, {"group": "elder", "value": 2.0, "peds": 1}, {"group": "wheelchair_manual", "value": 0.5, "peds": 1}]
    assert worst_group(groups, "lower") == "elder"
    assert worst_group(groups, "higher") == "wheelchair_manual"


def test_a_tie_for_worst_names_no_group():
    groups = [{"group": "adult", "value": 1.0, "peds": 3}, {"group": "elder", "value": 1.0, "peds": 1}, {"group": "wheelchair_manual", "value": 0.5, "peds": 1}]
    assert worst_group(groups, "lower") is None
    assert worst_group(groups, "higher") == "wheelchair_manual"


def test_a_single_group_is_the_worst_served():
    assert worst_group([{"group": "adult", "value": 0.0, "peds": 3}], "lower") == "adult"


def test_worst_group_is_none_without_groups():
    assert worst_group(None, "lower") is None


def test_calculator_reports_each_measure_per_agent_type():
    data = _types((["/env_0/a", "/env_0/b", "/env_0/c"], ["adult", "wheelchair_manual", "adult"]))
    prior = {
        "ped_mechanism_delay_seconds_per_ped": _records(**{"env_0/a": 0.0, "env_0/b": 6.0, "env_0/c": 2.0}),
        "ped_round_trips_completed_per_ped": _records(**{"env_0/a": 2.0, "env_0/b": 1.0, "env_0/c": 2.0}),
        "ped_max_exposure_dba_per_ped": None,
    }
    result = _calc(data, prior)

    assert result["ped_agent_types"] == [{"ped": "env_0/a", "agent_type": "adult"}, {"ped": "env_0/b", "agent_type": "wheelchair_manual"}, {"ped": "env_0/c", "agent_type": "adult"}]
    assert result["ped_mechanism_delay_seconds_by_group"] == [{"group": "adult", "value": 1.0, "peds": 2}, {"group": "wheelchair_manual", "value": 6.0, "peds": 1}]
    assert result["ped_mechanism_delay_seconds_worst_group"] == "wheelchair_manual"
    assert result["ped_round_trips_completed_worst_group"] == "wheelchair_manual"
    assert result["ped_max_exposure_dba_by_group"] is None
    assert result["ped_max_exposure_dba_worst_group"] is None
    assert result["ped_path_deflection_m_by_group"] is None


def test_calculator_without_agent_types_reports_nothing():
    result = _calc(pl.DataFrame({"peds_names": [["a"]]}), {"ped_mechanism_delay_seconds_per_ped": _records(a=1.0)})
    assert result == dict.fromkeys(PedestrianGroupsCalculator.output_keys())
