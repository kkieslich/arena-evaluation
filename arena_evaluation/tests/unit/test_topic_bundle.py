"""Unit tests for TopicBundle.available() and the native-topic collection of an episode."""

import pytest

pytest.importorskip("polars")

import polars as pl
from arena_evaluation.processing.episode_topics import native_topics
from arena_evaluation.storage.schemas import TopicBundle


def test_topic_bundle_available_reports_only_populated_fields():
    odom = pl.DataFrame({"time_ns": [0, 1], "pos_x": [0.0, 1.0]})
    schedule = pl.DataFrame({"phase_label": ["a"], "phase_kind": ["linear"]})
    bundle = TopicBundle(odom=odom, characterization_schedule=schedule)
    assert bundle.available() == {"odom", "characterization_schedule"}


def test_native_topics_excludes_shared_frames_but_keeps_schedule():
    odom = pl.DataFrame({"time_ns": [0, 1], "pos_x": [0.0, 1.0]})
    # characterization_schedule carries no time_ns column, unlike per-sample topics.
    schedule = pl.DataFrame({"phase_label": ["a"], "phase_kind": ["linear"], "vx_target": [1.0]})
    tf = pl.DataFrame({"time_ns": [0], "frame_id": ["map"]})
    tf_static = pl.DataFrame({"time_ns": [0], "frame_id": ["map"]})
    semantic_snapshot = pl.DataFrame({"time_ns": [0], "entity": ["x"]})
    episode_record = pl.DataFrame({"time_ns": [0], "episode_id": [0]})

    bundle = TopicBundle(
        odom=odom,
        characterization_schedule=schedule,
        tf=tf,
        tf_static=tf_static,
        semantic_snapshot=semantic_snapshot,
        episode_record=episode_record,
    )

    topics = native_topics(bundle)
    assert "tf" not in topics
    assert "tf_static" not in topics
    assert "semantic_snapshot" not in topics
    assert "episode_record" not in topics
    assert "odom" in topics
    assert "characterization_schedule" in topics


def test_native_topics_keeps_an_empty_gesture_table():
    odom = pl.DataFrame({"time_ns": [0, 1], "pos_x": [0.0, 1.0]})
    gestures = pl.DataFrame(schema={"time_ns": pl.Int64, "ped_id": pl.Int64, "clip": pl.String})
    interactions = pl.DataFrame(schema={"time_ns": pl.Int64, "interaction_id": pl.Int64})

    topics = native_topics(TopicBundle(odom=odom, ped_gestures=gestures, interactions=interactions))
    assert "ped_gestures" in topics
    assert len(topics["ped_gestures"]) == 0
    assert "interactions" not in topics


def test_native_topics_by_name_collects_only_those_the_bundle_carries():
    odom = pl.DataFrame({"time_ns": [0, 1], "pos_x": [0.0, 1.0]})
    peds = pl.DataFrame({"time_ns": [0], "peds_ids": [[4]]})

    topics = native_topics(TopicBundle(odom=odom, peds_physics=peds.lazy()), ["peds_physics", "agent_meta"])
    assert list(topics) == ["peds_physics"]
    assert isinstance(topics["peds_physics"], pl.DataFrame)
