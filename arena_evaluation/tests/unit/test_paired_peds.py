import pytest

pl = pytest.importorskip("polars")

from arena_evaluation.processing.metrics.social.paired_peds import PAIRED_DTYPES, add_paired_ped_measures, paired_deflection, paired_ped_measures, paired_travel_time_delay
from arena_evaluation.processing.ped_track_store import TRACK_SCHEMA, pair

S = 1_000_000_000


def _tracks(*peds: tuple[str, str, list[tuple[float, float, float]]]) -> pl.DataFrame:
    """Track rows of (key, agent_type, [(t, x, y), ...]) pedestrians."""
    rows = [
        {"time_ns": int(t * S), "t": t, "agent_id": i, "key": key, "name": "", "source_name": "", "source_seq": 0, "agent_type": agent_type, "x": x, "y": y, "vx": 0.0, "vy": 0.0}
        for i, (key, agent_type, points) in enumerate(peds)
        for t, x, y in points
    ]
    return pl.DataFrame(rows, schema=TRACK_SCHEMA)


CLOCK = [(t, 0.0, 0.0) for t in range(11)]
# a clock pedestrian spans both tables, so the others' first and last frames lie inside them
REF = _tracks(
    ("clock", "adult", CLOCK),
    ("door#1", "adult", [(t, t - 1.0, 0.0) for t in range(1, 6)]),
    ("door#2", "wheelchair_manual", [(t, t - 2.0, 0.0) for t in range(2, 7)]),
    ("door#3", "blind_white_cane", [(t, t - 2.0, 0.0) for t in range(2, 5)]),
)
RUN = _tracks(
    ("clock", "adult", CLOCK),
    ("door#1", "adult", [(t, min(t - 1.0, 4.0), 1.0) for t in range(1, 9)]),
    ("door#2", "wheelchair_manual", [(t, min(t - 2.0, 4.0), 0.0) for t in range(2, 11)]),
    ("door#3", "blind_white_cane", [(t, min(t - 2.0, 2.0), 0.25) for t in range(2, 6)]),
    ("door#9", "adult", [(t, 0.0, 3.0) for t in range(3, 6)]),
)


def test_deflection_measures_each_pedestrian_against_its_own_reference_track():
    assert paired_deflection(REF, RUN, pair(REF, RUN)) == [{"ped": "clock", "value": 0.0}, {"ped": "door#1", "value": 1.0}, {"ped": "door#2", "value": 0.0}, {"ped": "door#3", "value": 0.25}]


def test_travel_time_delay_counts_pedestrians_that_complete_both_runs():
    # door#2 is still walking at the run's last frame, the clock was there from the first, door#9 has no reference
    assert paired_travel_time_delay(REF, RUN, pair(REF, RUN)) == [{"ped": "door#1", "value": 3.0}, {"ped": "door#3", "value": 1.0}]


def test_measures_report_each_group_and_the_worst_served_one():
    result = paired_ped_measures(REF, RUN)

    assert result["ped_paired_deflection_m"] == pytest.approx(0.3125, abs=1e-3)
    assert result["ped_paired_deflection_m_by_group"] == [{"group": "adult", "value": 0.5, "peds": 2}, {"group": "long_cane", "value": 0.25, "peds": 1}, {"group": "wheelchair", "value": 0.0, "peds": 1}]
    assert result["ped_paired_deflection_m_worst_group"] == "adult"
    assert result["ped_paired_delay_s"] == 2.0
    assert result["ped_paired_delay_s_by_group"] == [{"group": "adult", "value": 3.0, "peds": 1}, {"group": "long_cane", "value": 1.0, "peds": 1}]
    assert result["ped_paired_delay_s_worst_group"] == "adult"
    assert result["ped_paired_relative_delay"] == pytest.approx(4.0 / 6.0)
    assert result["ped_paired_relative_delay_by_group"] == [{"group": "adult", "value": 0.75, "peds": 1}, {"group": "long_cane", "value": 0.5, "peds": 1}]
    assert result["ped_paired_relative_delay_worst_group"] == "adult"
    assert result["ped_paired_dropped"] == 2
    assert result["ped_paired_dropped_by_group"] == [{"group": "adult", "value": 1.0, "peds": 2}, {"group": "long_cane", "value": 0.0, "peds": 1}, {"group": "wheelchair", "value": 1.0, "peds": 1}]
    assert set(result) == set(PAIRED_DTYPES)


def test_relative_delay_is_a_ratio_of_means():
    ref = _tracks(("clock", "adult", CLOCK), ("a", "adult", [(1, 0.0, 0.0), (3, 0.0, 0.0)]), ("b", "adult", [(1, 0.0, 0.0), (9, 0.0, 0.0)]))
    run = _tracks(("clock", "adult", CLOCK), ("a", "adult", [(1, 0.0, 0.0), (5, 0.0, 0.0)]), ("b", "adult", [(1, 0.0, 0.0), (9, 0.0, 0.0)]))
    # mean extra time 1 s over mean reference time 5 s, not the mean of 2/2 and 0/8
    assert paired_ped_measures(ref, run)["ped_paired_relative_delay_by_group"] == [{"group": "adult", "value": 0.2, "peds": 2}]


@pytest.mark.parametrize("missing", ["ref", "run"])
def test_measures_are_none_without_both_tracks(missing: str):
    empty = pl.DataFrame(schema=TRACK_SCHEMA)
    result = paired_ped_measures(empty if missing == "ref" else REF, empty if missing == "run" else RUN)
    assert result == dict.fromkeys(PAIRED_DTYPES)


def test_rows_get_the_measures_of_their_stage_reference_reading_each_episode_once():
    tracks = {1: RUN, 9: REF}
    reads: list[int] = []

    def tracks_of(row: dict) -> pl.DataFrame:
        reads.append(row["episode"])
        return tracks[row["episode"]]

    jackal = {"episode": 1, "stage": "s", "robot": "env_0_jackal"}
    burger = {"episode": 1, "stage": "s", "robot": "env_0_burger"}
    unreferenced = {"episode": 2, "stage": "t", "robot": "env_0_jackal"}
    reference = {"episode": 9, "stage": "s", "is_reference": True, "status": "evaluated"}
    rows = [jackal, burger, unreferenced, reference]
    add_paired_ped_measures(rows, {"s": [reference]}, tracks_of)

    assert sorted(reads) == [1, 9]
    assert jackal["ped_paired_delay_s"] == burger["ped_paired_delay_s"] == 2.0
    assert "ped_paired_delay_s" not in unreferenced and "ped_paired_delay_s" not in reference
