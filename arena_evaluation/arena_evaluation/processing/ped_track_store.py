"""Per-pedestrian tracks of an episode env in the world frame, keyed so the same pedestrian pairs across runs."""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing
from collections.abc import Mapping

import polars as pl

from arena_evaluation.processing.episode_topics import episode_start_stamp, episode_window
from arena_evaluation.processing.mcap_reader import bundle_env
from arena_evaluation.processing.parquet_store import TopicParquetStore

if typing.TYPE_CHECKING:
    from arena_evaluation.storage.schemas import TopicBundle

DEFAULT_AGENT_TYPE = "adult"
TOPICS_DIR = "topics"
#: Where tables built from an episode's extracted topics are kept, under its topics dir: re-extracting clears them
DERIVED_DIR = ".derived"
ENGINE_TABLE = "pedestrian_agent_states"
ARENA_TABLE = "pedestrian_arena_peds"
META_TABLE = "pedestrian_agent_meta"
LIFECYCLE_TABLE = "episode_lifecycle"
#: the extracted tables the tracks are built from: the crowd engine's agent states, else arena_peds, and the agent meta
TRACK_TOPICS = (ENGINE_TABLE, ARENA_TABLE, META_TABLE)
TRACKS_VERSION = 2
_PED_PREFIX = "Pedestrian_"
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


_FRAME_SCHEMA = {"time_ns": pl.Int64, "stamp_ns": pl.Int64, "agent_id": pl.Int64, "recorded": pl.String, "x": pl.Float64, "y": pl.Float64, "vx": pl.Float64, "vy": pl.Float64, "radius": pl.Float64}


def _frames(peds: pl.DataFrame | None) -> pl.DataFrame:
    """One row per agent per frame from a pedestrian table (`pedestrian_agent_states` or `pedestrian_arena_peds`, one
    row per pedestrian per frame): time_ns (the log time), stamp_ns, agent_id, recorded name, x, y, vx, vy, radius."""
    need = {"t_ns", "ped_id", "x", "y"}
    if peds is None or len(peds) == 0 or not need <= set(peds.columns):
        return pl.DataFrame(schema=_FRAME_SCHEMA)
    nan = pl.lit(float("nan"), dtype=pl.Float64)

    def column(name: str, dtype: pl.DataType, missing: pl.Expr) -> pl.Expr:
        return (pl.col(name).cast(dtype) if name in peds.columns else missing).alias(name)

    # an extraction without header stamps times its frames by the log time; a frame without velocities gets NaN ones
    # (derived from the positions downstream), one without a radius a NaN radius
    out = peds.select(
        pl.col("t_ns").cast(pl.Int64).alias("time_ns"),
        column("stamp_ns", pl.Int64(), pl.lit(None, dtype=pl.Int64)),
        # the engine's numeric agent id, `Pedestrian_<id>` in the tables
        pl.col("ped_id").cast(pl.String).str.strip_prefix(_PED_PREFIX).cast(pl.Int64, strict=False).alias("agent_id"),
        column("ped_name", pl.String(), pl.lit("")).fill_null("").alias("recorded"),
        pl.col("x").cast(pl.Float64),
        pl.col("y").cast(pl.Float64),
        column("vx", pl.Float64(), nan).fill_null(float("nan")).alias("vx"),
        column("vy", pl.Float64(), nan).fill_null(float("nan")).alias("vy"),
        column("radius_m", pl.Float64(), nan).alias("radius"),
    ).filter(pl.col("agent_id").is_not_null())
    # a radius of 0 is the producers' sentinel for unknown (the reader stores it as null)
    radius = pl.col("radius").fill_null(float("nan"))
    return out.with_columns(pl.when(radius.is_nan() | (radius <= 0)).then(float("nan")).otherwise(radius).alias("radius"))


def _meta(agent_meta: pl.DataFrame | None) -> pl.DataFrame:
    schema = {"time_ns": pl.Int64, "agent_id": pl.Int64, "meta_name": pl.String, "meta_type": pl.String, "source_name": pl.String, "source_seq": pl.Int64}
    if agent_meta is None or len(agent_meta) == 0 or "agent_id" not in agent_meta.columns:
        return pl.DataFrame(schema=schema)
    return agent_meta.select(
        pl.col("t_ns").cast(pl.Int64).alias("time_ns"),
        pl.col("agent_id").cast(pl.Int64),
        pl.col("name").cast(pl.String).alias("meta_name") if "name" in agent_meta.columns else pl.lit("").alias("meta_name"),
        pl.col("agent_type").cast(pl.String).alias("meta_type") if "agent_type" in agent_meta.columns else pl.lit("").alias("meta_type"),
        pl.col("source_name").cast(pl.String) if "source_name" in agent_meta.columns else pl.lit("").alias("source_name"),
        pl.col("source_seq").cast(pl.Int64) if "source_seq" in agent_meta.columns else pl.lit(0, dtype=pl.Int64).alias("source_seq"),
    ).sort("time_ns")


def track_table(topics: Mapping[str, pl.DataFrame | None], start_ns: int | None = None, end_ns: int | None = None, episode_id: int | None = None, start_stamp_ns: int | None = None) -> pl.DataFrame:
    """One env's pedestrian tracks (TRACK_SCHEMA) from its native topic frames, cut to [start_ns, end_ns]; the metrics README defines the columns.

    `topics` holds an env's extracted tables by name (TRACK_TOPICS); `start_ns`/`end_ns` are the recorder's log times of the episode. With `start_stamp_ns`, the sim time the episode
    started at, and frames that carry their header stamp, a frame's `time_ns` is its stamp and `t` its stamp minus the
    start: the crowd engine's tick time. The recorder logs a frame at its own clock when the frame arrives, up to ~0.4 s
    late at an episode's start and a few ticks late at random later, so two runs of the same crowd would otherwise
    disagree on when a frame happened. Without them the log time stands in, as before."""
    peds = topics.get(ENGINE_TABLE)
    if peds is None or len(peds) == 0:
        peds = topics.get(ARENA_TABLE)
    frames = _frames(peds)
    if start_ns is not None:
        frames = frames.filter(pl.col("time_ns") >= start_ns)
    if end_ns is not None:
        frames = frames.filter(pl.col("time_ns") <= end_ns)
    if len(frames) == 0:
        return empty_tracks()
    # as-of joins need both sides sorted by time; `by` matches each agent's own meta rows
    frames = frames.sort("time_ns")
    meta = _meta(topics.get(META_TABLE))
    if len(meta) > 0:
        before = frames.join_asof(meta, on="time_ns", by="agent_id", strategy="backward", check_sortedness=False)
        after = frames.join_asof(meta, on="time_ns", by="agent_id", strategy="forward", check_sortedness=False)
        later = after.select(pl.col(c).alias(f"{c}_after") for c in _META_COLUMNS)
        frames = pl.concat([before, later], how="horizontal").with_columns(pl.coalesce(pl.col(c), pl.col(f"{c}_after")).alias(c) for c in _META_COLUMNS).drop(f"{c}_after" for c in _META_COLUMNS)
    else:
        frames = frames.with_columns(pl.lit("").alias("meta_name"), pl.lit("").alias("meta_type"), pl.lit("").alias("source_name"), pl.lit(0, dtype=pl.Int64).alias("source_seq"))
    frames = frames.with_columns(pl.col("meta_name").fill_null(""), pl.col("meta_type").fill_null(""), pl.col("source_name").fill_null(""), pl.col("source_seq").fill_null(0))
    # time each frame by the engine's own clock where the episode's start and the frame's stamp are known
    stamped = start_stamp_ns is not None and frames["stamp_ns"].null_count() == 0 and bool((frames["stamp_ns"] > 0).all())
    if stamped:
        # a frame stamped before the start is a late copy of the previous episode's crowd
        frames = frames.with_columns(pl.col("stamp_ns").alias("time_ns")).filter(pl.col("time_ns") >= start_stamp_ns).unique(subset=["agent_id", "time_ns"], keep="first", maintain_order=True)
        if len(frames) == 0:
            return empty_tracks()
    # the engine table names agents by their id; a recorded name equal to the id is no name
    recorded = pl.when(pl.col("recorded") == pl.col("agent_id").cast(pl.String)).then(pl.lit("")).otherwise(pl.col("recorded"))
    name = pl.when(pl.col("meta_name") != "").then(pl.col("meta_name")).otherwise(recorded).str.replace(_ENV_PREFIX.pattern, "")
    frames = frames.with_columns(name.alias("name"))
    released = pl.col("source_name") + "#" + pl.col("source_seq").cast(pl.String)
    key = pl.when(pl.col("source_name") != "").then(released).when(pl.col("name") != "").then(pl.col("name")).otherwise(pl.lit("id") + pl.col("agent_id").cast(pl.String))
    t0 = start_stamp_ns if stamped else start_ns if start_ns is not None else int(frames["time_ns"].min())  # pyright: ignore[reportArgumentType]
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


def _collect(frame: pl.DataFrame | pl.LazyFrame | None) -> pl.DataFrame | None:
    if frame is None:
        return None
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def env_tracks(bundle: TopicBundle, episode_id: int | None = None) -> pl.DataFrame:
    """The tracks of one env's bundle, cut to its episode window as the metrics see the episode."""
    topics = {name: df for name in TRACK_TOPICS if (df := _collect(getattr(bundle, name, None))) is not None}
    record = _collect(bundle.episode_lifecycle)
    start_ns, end_ns = episode_window(record)
    return track_table(topics, start_ns, end_ns, episode_id, episode_start_stamp(record))


def _episode_id(episode_dir: pathlib.Path) -> int | None:
    match = _EPISODE_DIR.match(episode_dir.name)
    return int(match.group(1)) if match else None


def _env_table(topics_dir: pathlib.Path, name: str, env: str) -> pl.DataFrame | None:
    path = topics_dir / f"{name}.parquet"
    if not path.is_file():
        return None
    return pl.scan_parquet(path).filter(pl.col("env_id") == env).collect()


def topics_envs(topics_dir: pathlib.Path) -> list[str]:
    """The envs of an extracted episode, those its lifecycle or pedestrian tables hold rows of, sorted.

    Read from the env tables, not from the robot bundles: a run without robots (a crowd-only contestant) has
    pedestrians and no robot."""
    found: set[str] = set()
    for name in (LIFECYCLE_TABLE, *TRACK_TOPICS):
        path = pathlib.Path(topics_dir) / f"{name}.parquet"
        if path.is_file():
            found.update(str(v) for v in pl.scan_parquet(path).select("env_id").unique().collect()["env_id"].to_list() if v)
    return sorted(found)


def topics_tracks(topics_dir: pathlib.Path, env: str, episode_id: int | None = None) -> pl.DataFrame:
    """The tracks of `env` from an episode's extracted tables in `topics_dir`, cut to its episode window."""
    topics_dir = pathlib.Path(topics_dir)
    topics = {name: df for name in TRACK_TOPICS if (df := _env_table(topics_dir, name, env)) is not None}
    record = _env_table(topics_dir, LIFECYCLE_TABLE, env)
    start_ns, end_ns = episode_window(record)
    return track_table(topics, start_ns, end_ns, episode_id, episode_start_stamp(record))


def tracks_path(episode_dir: pathlib.Path, env: str) -> pathlib.Path:
    """Where the kept tracks of `env` in an episode lie, beside the topics they are built from."""
    return pathlib.Path(episode_dir) / TOPICS_DIR / DERIVED_DIR / f"ped_tracks.v{TRACKS_VERSION}" / f"{env}.parquet"


def _keep(path: pathlib.Path, tracks: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".parquet.tmp")
    tracks.write_parquet(temp)
    temp.rename(path)


def write_tracks(episode_dir: pathlib.Path, bundles: Mapping[str, TopicBundle] | None = None) -> dict[str, pl.DataFrame]:
    """Build and keep the tracks of every env of an extracted episode, by env (`bundles` is not needed: the tracks are
    built from the extracted tables, which also hold the envs without a robot)."""
    del bundles
    episode_dir = pathlib.Path(episode_dir)
    topics_dir = episode_dir / TOPICS_DIR
    episode_id = _episode_id(episode_dir)
    built: dict[str, pl.DataFrame] = {}
    for env in topics_envs(topics_dir):
        built[env] = topics_tracks(topics_dir, env, episode_id)
        _keep(tracks_path(episode_dir, env), built[env])
    return built


def tracks_for_episode(episode_dir: pathlib.Path, env: str | None = None) -> pl.DataFrame:
    """The kept tracks of `env` in an extracted episode, built and kept on first use; `env` may be left out only when the episode ran one."""
    episode_dir = pathlib.Path(episode_dir)
    topics_dir = episode_dir / TOPICS_DIR
    envs = topics_envs(topics_dir) if topics_dir.is_dir() else []
    if not envs:
        return empty_tracks()
    if env is None:
        if len(envs) > 1:
            raise ValueError(f"{episode_dir}: the episode ran {envs}, name the env to read the tracks of")
        (env,) = envs
    if env not in envs:
        raise ValueError(f"{episode_dir}: no {env} in the episode, it ran {envs}")
    path = tracks_path(episode_dir, env)
    if path.is_file():
        return pl.read_parquet(path)
    tracks = topics_tracks(topics_dir, env, _episode_id(episode_dir))
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
    "DERIVED_DIR",
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
    "topics_envs",
    "topics_tracks",
    "tracks_for_episode",
    "tracks_path",
    "write_tracks",
]
