from __future__ import annotations

import typing

import numpy as np
import polars as pl

from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.processing.metrics.ecological.mechanism_common import per_ped, ped_keys
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class PedestrianDisturbanceCalculator(BaseMetricCalculator):
    """Computes pedestrian deflection and slowdown metrics against baseline trajectories."""

    NAME = "pedestrian_disturbance"
    CATEGORY = "social"
    REQUIRES_PEDSIM = True
    DEPENDS_ON = ["proxemics"]
    REQUIRED_TOPICS = ["peds"]

    UNITS = {
        "ped_path_deflection_m": "m",
        "ped_velocity_delay_ratio": "",
        "ped_round_trips_completed": "",
        "ped_path_deflection_m_per_ped": "m",
        "ped_velocity_delay_ratio_per_ped": "",
        "ped_round_trips_completed_per_ped": "",
    }

    PRIMARY_OUTPUTS = ["ped_path_deflection_m"]

    OUTPUT_DIRECTIONS = {"ped_path_deflection_m": "lower"}

    @classmethod
    def output_keys(cls) -> list[str]:
        return [
            "ped_path_deflection_m",
            "ped_velocity_delay_ratio",
            "ped_round_trips_completed",
            "ped_path_deflection_m_per_ped",
            "ped_velocity_delay_ratio_per_ped",
            "ped_round_trips_completed_per_ped",
        ]

    @staticmethod
    def _velocity_delay(speed: float, desired: float) -> float:
        """Shortfall of a mean speed against a desired speed, 0 for no or an unusable speed."""
        return max(0.0, float(1.0 - speed / desired)) if speed > 0 and np.isfinite(speed) else 0.0

    def _desired_speeds(self, episode: AlignedEpisodeBundle) -> dict[str, float]:
        """Desired speed by pedestrian key: the median of the positive desired speeds the crowd engine recorded for its id (`agent_states`)."""
        physics = self.native_topics(episode).get("peds_physics")
        if physics is None or "peds_desired_velocities" not in physics.columns or "peds_ids" not in episode.data.columns:
            return {}
        by_id = dict(
            physics.select("peds_ids", "peds_desired_velocities")
            .filter(pl.col("peds_ids").list.len() == pl.col("peds_desired_velocities").list.len())
            .explode("peds_ids", "peds_desired_velocities")
            .filter(pl.col("peds_desired_velocities") > 0)
            .group_by("peds_ids")
            .agg(pl.col("peds_desired_velocities").median())
            .iter_rows()
        )
        names = pl.col("peds_names") if "peds_names" in episode.data.columns else pl.lit(None, dtype=pl.List(pl.String)).alias("peds_names")
        speeds: dict[str, float] = {}
        for row_names, row_ids in episode.data.select(names, "peds_ids").drop_nulls("peds_ids").unique().iter_rows():
            for key, agent_id in zip(ped_keys(row_names, len(row_ids)), row_ids, strict=True):
                if agent_id in by_id:
                    speeds[key] = float(by_id[agent_id])
        return speeds

    @staticmethod
    def _compute_trajectory_deflection(actual_coords: np.ndarray, reference_coords: np.ndarray) -> float:
        """
        Compute mean distance between actual trajectory points and nearest reference trajectory points.
        actual_coords: (N, 2) or (N, 3)
        reference_coords: (M, 2) or (M, 3)
        """
        if len(actual_coords) == 0 or len(reference_coords) == 0:
            return 0.0

        valid_act = actual_coords[~np.isnan(actual_coords).any(axis=1)]
        valid_ref = reference_coords[~np.isnan(reference_coords).any(axis=1)]
        if len(valid_act) == 0 or len(valid_ref) == 0:
            return 0.0

        valid_act = np.ascontiguousarray(valid_act[:, :2], dtype=np.float64)
        valid_ref = np.ascontiguousarray(valid_ref[:, :2], dtype=np.float64)

        try:
            from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

            tree = cKDTree(valid_ref)
            dists, _ = tree.query(valid_act, k=1)
            return float(np.mean(dists))
        except Exception:
            from scipy.spatial.distance import cdist

            dists = cdist(valid_act, valid_ref)
            return float(np.mean(np.min(dists, axis=1)))

    @staticmethod
    def _count_round_trips(dists_from_start: np.ndarray, max_disp: float) -> int:
        """Counts returns to within a quarter of the farthest reach, each after passing three quarters of it."""
        trips = 0
        away = False
        for dist in dists_from_start:
            if not away and dist >= 0.75 * max_disp:
                away = True
            elif away and dist <= 0.25 * max_disp:
                trips += 1
                away = False
        return trips

    def calculate(
        self,
        episode: AlignedEpisodeBundle,
        prior_results: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        if episode.data is None or "peds_positions" not in episode.data.columns:
            return {k: None for k in self.output_keys()}

        peds_pos = episode.data["peds_positions"].to_list()
        if not peds_pos or len(peds_pos) == 0:
            return {k: None for k in self.output_keys()}

        # Extract pedestrian positions per agent, keyed by recorded name so a pedestrian keeps its track when others leave
        names = episode.data["peds_names"].to_list() if "peds_names" in episode.data.columns else [None] * len(peds_pos)
        time_ns = episode.data["time_ns"].to_numpy() if "time_ns" in episode.data.columns else None
        agent_paths: dict[str, list[tuple[float, float]]] = {}
        agent_times: dict[str, list[int]] = {}
        for i, row in enumerate(peds_pos):
            arr = self._parse_peds(row)
            for key, item in zip(ped_keys(names[i], len(arr)), arr, strict=True):
                if not (np.isfinite(item[0]) and np.isfinite(item[1]) and abs(item[0]) < 1e6 and abs(item[1]) < 1e6):
                    continue
                agent_paths.setdefault(key, []).append((float(item[0]), float(item[1])))
                if time_ns is not None:
                    agent_times.setdefault(key, []).append(int(time_ns[i]))

        if not agent_paths:
            return {
                "ped_path_deflection_m": 0.0,
                "ped_velocity_delay_ratio": 0.0,
                "ped_round_trips_completed": 0,
                "ped_path_deflection_m_per_ped": [],
                "ped_velocity_delay_ratio_per_ped": [],
                "ped_round_trips_completed_per_ped": [],
            }

        round_trips = 0
        total_deflection = 0.0
        num_agents = len(agent_paths)
        deflections: dict[str, float] = {}
        trips: dict[str, float] = {}

        for key, coords in agent_paths.items():
            arr = np.array(coords)
            if len(arr) < 5:
                continue

            diffs = arr - arr[0]
            dists_from_start = np.hypot(diffs[:, 0], diffs[:, 1])
            max_disp = float(np.max(dists_from_start))

            agent_trips = self._count_round_trips(dists_from_start, max_disp) if max_disp > 0.5 else 0
            round_trips += agent_trips
            trips[key] = float(agent_trips)

            # Standalone geometric lateral deflection against straight-line start-to-end vector
            p_start, p_end = arr[0], arr[-1]
            seg_vec = p_end - p_start
            seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
            agent_deflection = 0.0
            if np.isfinite(seg_len) and seg_len > 0.5:
                u_vec = seg_vec / seg_len
                # Cross-product magnitude gives perpendicular distance in 2D
                rel_pos = arr - p_start
                lateral_dists = np.abs(rel_pos[:, 0] * u_vec[1] - rel_pos[:, 1] * u_vec[0])
                finite_lat = lateral_dists[np.isfinite(lateral_dists)]
                agent_deflection = float(np.mean(finite_lat))
                total_deflection += agent_deflection
            deflections[key] = round(agent_deflection, 3)

        # Velocity delay: compare actual mean speed to each pedestrian's own desired speed, a pedestrian over the time it was recorded
        desired = self._desired_speeds(episode)
        velocity_delay_ratio = 0.0
        delays: dict[str, float] = {}
        if desired and time_ns is not None and len(time_ns) > 1:
            dt_s = (time_ns[-1] - time_ns[0]) / 1e9
            if dt_s > 0:
                all_dists = []
                all_desired = []
                for key, pts in agent_paths.items():
                    if len(pts) > 1 and key in desired:
                        p_arr = np.array(pts)
                        step_dists = np.hypot(np.diff(p_arr[:, 0]), np.diff(p_arr[:, 1]))
                        finite_d = step_dists[np.isfinite(step_dists)]
                        all_dists.append(float(np.sum(finite_d)))
                        all_desired.append(desired[key])
                        span_s = (agent_times[key][-1] - agent_times[key][0]) / 1e9
                        if span_s > 0:
                            delays[key] = round(self._velocity_delay(all_dists[-1] / span_s, desired[key]), 3)
                if all_dists:
                    velocity_delay_ratio = self._velocity_delay(float(np.mean(all_dists)) / dt_s, float(np.mean(all_desired)))

        deflection_val = round(total_deflection / num_agents, 3) if num_agents > 0 and np.isfinite(total_deflection) else 0.0
        delay_val = (round(velocity_delay_ratio, 3) if np.isfinite(velocity_delay_ratio) else 0.0) if desired else None
        trips_val = round_trips / num_agents if num_agents > 0 and np.isfinite(round_trips) else 0

        return {
            "ped_path_deflection_m": deflection_val,
            "ped_velocity_delay_ratio": delay_val,
            "ped_round_trips_completed": trips_val,
            "ped_path_deflection_m_per_ped": per_ped(deflections),
            "ped_velocity_delay_ratio_per_ped": per_ped(delays) if desired else None,
            "ped_round_trips_completed_per_ped": per_ped(trips),
        }
