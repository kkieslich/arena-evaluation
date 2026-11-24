"""The usable runs of a processed benchmark dir, one per episode and env, as the cross-run measures read them."""

from __future__ import annotations

import dataclasses
import pathlib
import re

import polars as pl

from arena_evaluation.processing.parquet_store import TopicParquetStore
from arena_evaluation.processing.ped_track_store import ARENA_TABLE, ENGINE_TABLE, TOPICS_DIR, topics_envs

#: An episode's fact rows: an episode `evaluation process` could not evaluate (no recording, or an error: a failed
#: attempt the runner retried) has none
FACTS_FILE = pathlib.PurePosixPath("metrics") / "fact_episodes.parquet"
#: The combined metrics older arena_evaluation releases wrote; the native layout has none
LEGACY_METRICS_FILE = "combined_metrics.parquet"
_ENV = re.compile(r"(env_\d+)")


@dataclasses.dataclass(frozen=True)
class EpisodeRun:
    """One env of a usable episode: its contestant (`planner`), stage and reference type from its `episode_NNN.yaml`,
    and the robots of its fact rows in that env (none for a crowd-only run)."""

    planner: str
    stage: str
    episode: int
    env: str
    is_reference: bool
    reference_type: str | None
    episode_dir: pathlib.Path
    robots: tuple[str, ...]


def _env_has_rows(topics: pathlib.Path, table: str, env: str) -> bool:
    path = topics / f"{table}.parquet"
    return path.is_file() and pl.scan_parquet(path).filter(pl.col("env_id") == env).head(1).collect().height > 0


def run_envs(episode_dir: pathlib.Path) -> dict[str, tuple[str, ...]] | None:
    """The envs an episode is a usable run of, with the robots of each, None when it is none: the envs of the robots of
    its fact rows, or, for a finished extraction without any robot and with pedestrians (a crowd-only run, which
    `evaluation process` has no robot to evaluate and writes no fact rows for), its envs with pedestrians. An episode
    with robots and without fact rows has no recording or failed to process (a failed attempt the runner retried)."""
    episode_dir = pathlib.Path(episode_dir)
    facts = episode_dir / FACTS_FILE
    if facts.is_file():
        envs: dict[str, list[str]] = {}
        for robot in pl.read_parquet(facts, columns=["robot_id"])["robot_id"].to_list():
            match = _ENV.search(str(robot or ""))
            envs.setdefault(match.group(1) if match else "env_0", []).append(str(robot))
        return {env: tuple(dict.fromkeys(robots)) for env, robots in envs.items()}
    topics = episode_dir / TOPICS_DIR
    if not (topics / TopicParquetStore.VERSION_FILE).is_file() or any(topics.glob("robot_*.parquet")):
        return None
    crowd = {env: () for env in topics_envs(topics) if any(_env_has_rows(topics, table, env) for table in (ENGINE_TABLE, ARENA_TABLE))}
    return crowd or None


def episode_runs(benchmark_dir: pathlib.Path) -> list[EpisodeRun]:
    """Every usable run of a processed benchmark dir, one per (episode, env), sorted by episode and env (`run_envs`).

    Raises FileNotFoundError for a dir without `episodes/`, and ValueError for one an older arena_evaluation processed
    (a `combined_metrics.parquet` and no episode with fact rows): its outputs are of another layout."""
    from arena_evaluation.storage.folder_manager import FolderManager

    benchmark_dir = pathlib.Path(benchmark_dir)
    episodes_dir = benchmark_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"{benchmark_dir}: no episodes/, not a benchmark run dir")
    if (benchmark_dir / LEGACY_METRICS_FILE).is_file() and not any(episodes_dir.glob(f"episode_*/{FACTS_FILE}")):
        raise ValueError(f"{benchmark_dir}: processed by an older arena_evaluation ({LEGACY_METRICS_FILE}); process it again with `evaluation process --force-process --benchmark-dir {benchmark_dir}`, or read it with the release that processed it")
    out: list[EpisodeRun] = []
    for episode_dir in sorted(episodes_dir.glob("episode_*")):
        envs = run_envs(episode_dir)
        if envs is None:
            continue
        ep = FolderManager.describe_episode(episode_dir)
        if ep is None:
            continue
        for env, robots in sorted(envs.items()):
            out.append(EpisodeRun(str(ep.planner), str(ep.stage), int(ep.episode_id), env, bool(ep.is_reference), ep.reference_type, episode_dir, robots))
    return out


__all__ = ["FACTS_FILE", "LEGACY_METRICS_FILE", "EpisodeRun", "episode_runs", "run_envs"]
