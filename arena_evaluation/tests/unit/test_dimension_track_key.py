"""dim_pedestrians.track_key pairs a pedestrian across runs as the track store keys its track."""

import polars as pl

from arena_evaluation.processing.dimension_rows import _track_keys


def test_the_track_key_is_the_source_else_the_meta_name_else_the_recorded_name_else_the_id():
    rows = pl.DataFrame({"ped_id": ["Pedestrian_4", "Pedestrian_11", "Pedestrian_12", "Pedestrian_13"], "ped_name": ["4", "11", "env_2/scene_lady", "13"]})
    meta = pl.DataFrame(
        {
            "t_ns": [9, 11, 12],
            "agent_id": [4, 11, 11],
            "name": ["env_2/flow_1", "", "env_2/late"],
            "source_name": ["", "door_a", "door_b"],
            "source_seq": [0, 7, 8],
        }
    )
    keys = rows.select(_track_keys(rows, meta).alias("key"))["key"].to_list()
    assert keys == ["flow_1", "door_a#7", "scene_lady", "id13"]
    assert rows.select(_track_keys(rows, None).alias("key"))["key"].to_list() == ["id4", "id11", "scene_lady", "id13"]
