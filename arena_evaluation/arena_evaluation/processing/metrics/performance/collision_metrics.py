from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import per_ped, ped_keys
from arena_evaluation.storage.schemas import AlignedEpisodeBundle

# arena_runtime_msgs/msg/EpisodeRecord outcome_state values.
_OUTCOME_SUCCESS = 2
_OUTCOME_RESULTS = {3: "FAILED", 4: "CANCELLED", 5: "FATAL"}
# QUEUED / RUNNING: never resolved by the runtime.
_OUTCOME_UNRESOLVED = (0, 1)


class CollisionMetricsCalculator(BaseMetricCalculator):
    """Calculates collision counts, determines episode success status, and computes SPL."""

    NAME = "collision_metrics"
    CATEGORY = "performance"
    DEPENDS_ON = ["time_metrics", "path_metrics", "trajectory_naturalness"]
    REQUIRED_TOPICS = ["collision_events", ("tf_gt", "odom")]

    UNITS = {
        "collision_amount": "",
        "collision_amount_wall": "",
        "collision_amount_static": "",
        "collision_amount_pedestrian": "",
        "collision_obstacles": "",
        "collisions": "",
        "result": "",
        "success": "",
        "spl": "",
        "ped_robot_contacts": "",
        "ped_robot_contacts_per_ped": "",
    }

    PRIMARY_OUTPUTS = ["success", "collision_amount", "spl"]
    OUTPUT_DIRECTIONS = {"success": "higher", "spl": "higher", "collision_amount": "lower", "ped_robot_contacts": "lower"}

    TIMEOUT_THRESHOLD_S = 180.0
    MAX_COLLISIONS = 3

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "collision_amount",
            "collision_amount_wall",
            "collision_amount_static",
            "collision_amount_pedestrian",
            "collision_obstacles",
            "collisions",
            "result",
            "success",
            "spl",
            "ped_robot_contacts",
            "ped_robot_contacts_per_ped",
        ]

    @staticmethod
    def _count_rising_edges(is_active: np.ndarray) -> int:
        count = 0
        if len(is_active) > 0 and is_active[0]:
            count += 1
        for i in range(1, len(is_active)):
            if is_active[i] and not is_active[i - 1]:
                count += 1
        return count

    @staticmethod
    def _collision_frame(episode: AlignedEpisodeBundle) -> pl.DataFrame | None:
        """The native collision_events stream when present, else the aligned data."""
        if episode.topics is not None and "collision_events" in episode.topics:
            return episode.topics["collision_events"].sort("time_ns")
        return episode.data

    @classmethod
    def _column_rising_edges(cls, episode: AlignedEpisodeBundle, column: str) -> int | None:
        frame = cls._collision_frame(episode)
        if frame is None or column not in frame.columns:
            return None
        counts = frame[column].to_numpy().astype(float)
        if np.isnan(counts).all():
            return None
        return cls._count_rising_edges(np.nan_to_num(counts, nan=0.0) > 0)

    @classmethod
    def _hit_obstacles(cls, episode: AlignedEpisodeBundle) -> list[str]:
        frame = cls._collision_frame(episode)
        if frame is None or "collision_obstacle_ids" not in frame.columns:
            return []
        hit: set[str] = set()
        for ids in frame["collision_obstacle_ids"].to_list():
            if ids is not None:
                hit.update(i for i in ids if i)
        return sorted(hit)

    @classmethod
    def _pedestrian_contacts(cls, episode: AlignedEpisodeBundle) -> dict[str, float] | None:
        """Contacts by pedestrian key, each start of a pedestrian touching the footprint one, 0 for every recorded pedestrian it never touched.

        None when the recording names no touched pedestrians (a schema without event kinds) or no pedestrians.
        """
        frame = cls._collision_frame(episode)
        if frame is None or "collision_pedestrian_ids" not in frame.columns or episode.data is None or "peds_names" not in episode.data.columns:
            return None
        touched = frame["collision_pedestrian_ids"].to_list()
        if all(ids is None for ids in touched):
            return None
        contacts = dict.fromkeys((key for names in episode.data["peds_names"].unique(maintain_order=True).to_list() if names is not None for key in ped_keys(names, len(names))), 0.0)
        touching: set[str] = set()
        for ids in touched:
            now = set(ped_keys(ids, len(ids))) if ids is not None else set()
            for key in now - touching:
                contacts[key] = contacts.get(key, 0.0) + 1.0
            touching = now
        return contacts

    def calculate(self, episode: AlignedEpisodeBundle, prior_results: dict[str, typing.Any]) -> dict[str, typing.Any]:

        collisions = []
        result = "GOAL_REACHED"
        success = True

        collision_amount = self._column_rising_edges(episode, "collision_event") or 0
        collision_amount_wall = self._column_rising_edges(episode, "collision_wall")
        collision_amount_static = self._column_rising_edges(episode, "collision_static")
        collision_amount_pedestrian = self._column_rising_edges(episode, "collision_pedestrian")
        collision_obstacles = self._hit_obstacles(episode)
        contacts = self._pedestrian_contacts(episode)

        time_to_goal = prior_results.get("time_to_goal")

        max_collisions = self.metrics_config.get("max_collisions", self.MAX_COLLISIONS)

        if time_to_goal is not None and float(time_to_goal) >= self.TIMEOUT_THRESHOLD_S:
            result = "TIMEOUT"
            success = False
        elif collision_amount >= max_collisions:
            result = "COLLISION"
            success = False
        else:
            result = "GOAL_REACHED"
            success = True

        # The runtime's verdict wins over anything derived from the trace.
        if episode.outcome_info == "collision":
            result = "COLLISION"
            success = False
        elif episode.outcome_state in _OUTCOME_UNRESOLVED:
            result = "UNRESOLVED"
            success = False
        elif episode.outcome_state is not None and episode.outcome_state != _OUTCOME_SUCCESS:
            success = False
            if result == "GOAL_REACHED":
                result = _OUTCOME_RESULTS.get(episode.outcome_state, "FAILED")

        # Calculate Success weighted by Path Length (SPL)
        # SPL = Success * (L_0 / max(L_actual, L_0))
        actual_length = prior_results.get("path_length", 0.0) or 0.0
        l0 = prior_results.get("theta_star_length")
        if l0 is None or l0 <= 0:
            if episode.start_pos and episode.goal_pos and len(episode.start_pos) >= 2 and len(episode.goal_pos) >= 2:
                l0 = float(np.linalg.norm(np.array(episode.goal_pos[:2]) - np.array(episode.start_pos[:2])))
            else:
                l0 = 0.0

        if success:
            denom = max(float(actual_length), float(l0))
            if denom > 0:
                spl = float(l0 / denom)
                spl = min(max(spl, 0.0), 1.0)
            else:
                spl = 1.0
        else:
            spl = 0.0

        return {
            "collision_amount": collision_amount,
            "collision_amount_wall": collision_amount_wall,
            "collision_amount_static": collision_amount_static,
            "collision_amount_pedestrian": collision_amount_pedestrian,
            "collision_obstacles": collision_obstacles,
            "collisions": collisions,
            "result": result,
            "success": success,
            "spl": float(spl),
            "ped_robot_contacts": int(sum(contacts.values())) if contacts is not None else None,
            "ped_robot_contacts_per_ped": per_ped(contacts) if contacts is not None else None,
        }
