"""Per-pedestrian tracks of an episode env in the world frame, keyed so the same pedestrian pairs across runs."""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from collections.abc import Mapping

import polars as pl

from arena_evaluation.processing.episode_topics import episode_window, native_topics
from arena_evaluation.processing.mcap_reader import bundle_env
from arena_evaluation.processing.parquet_store import TopicParquetStore

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import TopicBundle

DEFAULT_AGENT_TYPE = "adult"
TOPICS_DIR = "topics"
TRACK_TOPICS = ("peds_physics", "peds", "agent_meta")
TRACKS_VERSION = 1
_META_COLUMNS = ("meta_name", "meta_type", "source_name", "source_seq")
_ENV_PREFIX = re.compile(r"^/?env_\d+/")
_EPISODE_DIR = re.compile(r"^episode_(\d+)$")

TRACK_SCHEMA = {
    "episode_id": pl.Int64,
    "time_ns": pl.Int64,
    "t": pl.Float64,
    "agent_id": pl.Int64,
    "key": pl.String,
    "name": pl.String,
    "source_name": pl.String,
    "source_seq": pl.Int64,
    "agent_type": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "vx": pl.Float64,
    "vy": pl.Float64,
    "radius": pl.Float64,
}


def empty_tracks() -> pl.DataFrame:
    return pl.DataFrame(schema=TRACK_SCHEMA)


def strip_env(name: str) -> str:
    """A name without its `env_<n>/` prefix: env ids count up per spawn, so the same pedestrian is prefixed differently in another env."""
    return _ENV_PREFIX.sub("", name, count=1)


def _frames(peds: pl.DataFrame | None) -> pl.DataFrame:
    """One row per agent per frame from a `peds_*` table: time_ns, agent_id, recorded name, x, y, vx, vy, radius."""
    need = {"time_ns", "peds_ids", "peds_positions"}
    if peds is None or len(peds) == 0 or not need <= set(peds.columns):
        return pl.DataFrame(schema={"time_ns": pl.Int64, "agent_id": pl.Int64, "recorded": pl.String, "x": pl.Float64, "y": pl.Float64, "vx": pl.Float64, "vy": pl.Float64, "radius": pl.Float64})
    optional = {"peds_twists": pl.Float64, "peds_names": pl.String, "peds_radii": pl.Float64}
    cols = [pl.col("time_ns"), pl.col("peds_ids"), pl.col("peds_positions")]
    cols.extend(pl.col(c).cast(pl.List(dtype)) if c in peds.columns else pl.lit(None, dtype=pl.List(dtype)).alias(c) for c, dtype in optional.items())
    df = peds.select(cols).filter(pl.col("peds_ids").list.len() > 0)
    n = pl.col("peds_ids").list.len()
    # positions and twists are flat [x, y, z, ...]; a row whose lists disagree with its ids is dropped, not misread
    df = df.filter(pl.col("peds_positions").list.len() == 3 * n)
    df = df.with_columns(pl.col("peds_positions").list.gather_every(3, 0).alias("x"), pl.col("peds_positions").list.gather_every(3, 1).alias("y"))
    # a frame without matching twists gets NaN velocities (derived from the positions downstream), one without matching
    # radii NaN radii, one without matching names the ids as names, which track_table reads as unnamed
    twists = pl.col("peds_twists").list.len() == 3 * n
    nan = pl.col("x").list.eval(pl.element() * float("nan"))
    df = df.with_columns(
        pl.when(twists).then(pl.col("peds_twists").list.gather_every(3, 0)).otherwise(nan).alias("vx"),
        pl.when(twists).then(pl.col("peds_twists").list.gather_every(3, 1)).otherwise(nan).alias("vy"),
        pl.when(pl.col("peds_radii").list.len() == n).then(pl.col("peds_radii")).otherwise(nan).alias("radius"),
        pl.when(pl.col("peds_names").list.len() == n).then(pl.col("peds_names")).otherwise(pl.col("peds_ids").list.eval(pl.element().cast(pl.String))).alias("recorded"),
    )
    out = df.select("time_ns", pl.col("peds_ids").alias("agent_id"), "recorded", "x", "y", "vx", "vy", "radius").explode("agent_id", "recorded", "x", "y", "vx", "vy", "radius")
    # a radius of 0 is the arena_peds sentinel for unknown
    radius = pl.col("radius").fill_null(float("nan"))
    return out.with_columns(pl.col("agent_id").cast(pl.Int64), pl.col("recorded").fill_null("").cast(pl.String), pl.when(radius.is_nan() | (radius <= 0)).then(float("nan")).otherwise(radius).alias("radius"))


def _meta(agent_meta: pl.DataFrame | None) -> pl.DataFrame:
    schema = {"time_ns": pl.Int64, "agent_id": pl.Int64, "meta_name": pl.String, "meta_type": pl.String, "source_name": pl.String, "source_seq": pl.Int64}
    if agent_meta is None or len(agent_meta) == 0 or "agent_id" not in agent_meta.columns:
        return pl.DataFrame(schema=schema)
    return agent_meta.select(
        pl.col("time_ns").cast(pl.Int64),
        pl.col("agent_id").cast(pl.Int64),
        pl.col("name").cast(pl.String).alias("meta_name") if "name" in agent_meta.columns else pl.lit("").alias("meta_name"),
        pl.col("agent_type").cast(pl.String).alias("meta_type") if "agent_type" in agent_meta.columns else pl.lit("").alias("meta_type"),
        pl.col("source_name").cast(pl.String) if "source_name" in agent_meta.columns else pl.lit("").alias("source_name"),
        pl.col("source_seq").cast(pl.Int64) if "source_seq" in agent_meta.columns else pl.lit(0, dtype=pl.Int64).alias("source_seq"),
    ).sort("time_ns")


def track_table(topics: Mapping[str, pl.DataFrame | None], start_ns: int | None = None, end_ns: int | None = None, episode_id: int | None = None) -> pl.DataFrame:
    """One env's pedestrian tracks (TRACK_SCHEMA) from its native topic frames, cut to [start_ns, end_ns]; the metrics README defines the columns."""
    peds = topics.get("peds_physics")
    if peds is None or len(peds) == 0:
        peds = topics.get("peds")
    frames = _frames(peds)
    if start_ns is not None:
        frames = frames.filter(pl.col("time_ns") >= start_ns)
    if end_ns is not None:
        frames = frames.filter(pl.col("time_ns") <= end_ns)
    if len(frames) == 0:
        return empty_tracks()
    # as-of joins need both sides sorted by time; `by` matches each agent's own meta rows
    frames = frames.sort("time_ns")
    meta = _meta(topics.get("agent_meta"))
    if len(meta) > 0:
        before = frames.join_asof(meta, on="time_ns", by="agent_id", strategy="backward", check_sortedness=False)
        after = frames.join_asof(meta, on="time_ns", by="agent_id", strategy="forward", check_sortedness=False)
        later = after.select(pl.col(c).alias(f"{c}_after") for c in _META_COLUMNS)
        frames = pl.concat([before, later], how="horizontal").with_columns(pl.coalesce(pl.col(c), pl.col(f"{c}_after")).alias(c) for c in _META_COLUMNS).drop(f"{c}_after" for c in _META_COLUMNS)
    else:
        frames = frames.with_columns(pl.lit("").alias("meta_name"), pl.lit("").alias("meta_type"), pl.lit("").alias("source_name"), pl.lit(0, dtype=pl.Int64).alias("source_seq"))
    frames = frames.with_columns(pl.col("meta_name").fill_null(""), pl.col("meta_type").fill_null(""), pl.col("source_name").fill_null(""), pl.col("source_seq").fill_null(0))
    # the engine table names agents by their id; a recorded name equal to the id is no name
    recorded = pl.when(pl.col("recorded") == pl.col("agent_id").cast(pl.String)).then(pl.lit("")).otherwise(pl.col("recorded"))
    name = pl.when(pl.col("meta_name") != "").then(pl.col("meta_name")).otherwise(recorded).str.replace(_ENV_PREFIX.pattern, "")
    frames = frames.with_columns(name.alias("name"))
    released = pl.col("source_name") + "#" + pl.col("source_seq").cast(pl.String)
    key = pl.when(pl.col("source_name") != "").then(released).when(pl.col("name") != "").then(pl.col("name")).otherwise(pl.lit("id") + pl.col("agent_id").cast(pl.String))
    t0 = start_ns if start_ns is not None else int(frames["time_ns"].min())  # pyright: ignore[reportArgumentType]
    out = frames.with_columns(
        key.alias("key"),
        ((pl.col("time_ns") - t0) / 1e9).alias("t"),
        pl.when(pl.col("meta_type") != "").then(pl.col("meta_type")).otherwise(pl.lit(DEFAULT_AGENT_TYPE)).alias("agent_type"),
        pl.lit(episode_id, dtype=pl.Int64).alias("episode_id"),
    )
    return out.select([pl.col(c).cast(dtype) for c, dtype in TRACK_SCHEMA.items()]).sort("time_ns", "key")


def episode_envs(bundles: Mapping[str, TopicBundle]) -> dict[str, TopicBundle]:
    """The bundle of each env of an episode, its first robot's: the robots of one env share its pedestrians."""
    envs: dict[str, TopicBundle] = {}
    for key in sorted(bundles):
        envs.setdefault(bundle_env(key), bundles[key])
    return envs


def env_tracks(bundle: TopicBundle, episode_id: int | None = None) -> pl.DataFrame:
    """The tracks of one env's bundle, cut to its episode window as the metrics see the episode."""
    start_ns, end_ns = episode_window(bundle.episode_record)
    return track_table(native_topics(bundle, TRACK_TOPICS), start_ns, end_ns, episode_id)


def _episode_id(episode_dir: pathlib.Path) -> int | None:
    match = _EPISODE_DIR.match(episode_dir.name)
    return int(match.group(1)) if match else None


def tracks_path(episode_dir: pathlib.Path, env: str) -> pathlib.Path:
    """Where the kept tracks of `env` in an episode lie, beside the topics they are built from."""
    return TopicParquetStore.derived_dir(pathlib.Path(episode_dir) / TOPICS_DIR) / f"ped_tracks.v{TRACKS_VERSION}" / f"{env}.parquet"


def _keep(path: pathlib.Path, tracks: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".parquet.tmp")
    tracks.write_parquet(temp)
    temp.rename(path)


def write_tracks(episode_dir: pathlib.Path, bundles: Mapping[str, TopicBundle]) -> dict[str, pl.DataFrame]:
    """Build and keep the tracks of every env of an extracted episode, by env."""
    episode_dir = pathlib.Path(episode_dir)
    episode_id = _episode_id(episode_dir)
    built: dict[str, pl.DataFrame] = {}
    for env, bundle in episode_envs(bundles).items():
        built[env] = env_tracks(bundle, episode_id)
        _keep(tracks_path(episode_dir, env), built[env])
    return built


def tracks_for_episode(episode_dir: pathlib.Path, env: str | None = None) -> pl.DataFrame:
    """The kept tracks of `env` in an extracted episode, built and kept on first use; `env` may be left out only when the episode ran one."""
    episode_dir = pathlib.Path(episode_dir)
    topics_dir = episode_dir / TOPICS_DIR
    envs = episode_envs(TopicParquetStore.read(topics_dir) or {})
    if not envs:
        return empty_tracks()
    if env is None:
        if len(envs) > 1:
            raise ValueError(f"{episode_dir}: the episode ran {sorted(envs)}, name the env to read the tracks of")
        (env,) = envs
    if env not in envs:
        raise ValueError(f"{episode_dir}: no {env} in the episode, it ran {sorted(envs)}")
    path = tracks_path(episode_dir, env)
    if path.is_file():
        return pl.read_parquet(path)
    tracks = env_tracks(envs[env], _episode_id(episode_dir))
    if TopicParquetStore.is_current(topics_dir):
        _keep(path, tracks)
    return tracks


@dataclasses.dataclass(frozen=True)
class Pairing:
    """The pedestrians two runs share by key, and those only one of them has, each sorted."""

    keys: list[str]
    only_ref: list[str]
    only_run: list[str]


def pair(ref: pl.DataFrame, run: pl.DataFrame) -> Pairing:
    """Pair the pedestrians of a reference run and a run by `key`."""
    a = set(ref["key"].unique().to_list()) if len(ref) else set()
    b = set(run["key"].unique().to_list()) if len(run) else set()
    return Pairing(sorted(a & b), sorted(a - b), sorted(b - a))


__all__ = [
    "DEFAULT_AGENT_TYPE",
    "TOPICS_DIR",
    "TRACKS_VERSION",
    "TRACK_SCHEMA",
    "TRACK_TOPICS",
    "Pairing",
    "empty_tracks",
    "env_tracks",
    "episode_envs",
    "pair",
    "strip_env",
    "track_table",
    "tracks_for_episode",
    "tracks_path",
    "write_tracks",
]
