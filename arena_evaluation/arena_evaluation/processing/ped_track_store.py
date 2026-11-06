"""Per-pedestrian tracks of an episode, keyed so the same pedestrian pairs across runs, and the pairing of two runs.

One row per pedestrian per frame, in the world frame:

    time_ns, t           recording time, and seconds from the episode start (else from the first frame)
    agent_id             the crowd engine's id in this run, not stable across runs
    key                  the pairing key: `<source_name>#<source_seq>` for a pedestrian a source released (HumanSim draws
                         such an agent from its source and release index, so two runs with one seed agree on it), else
                         its name without the `env_<n>/` prefix (env ids count up per spawn), else `id<agent_id>`
    name                 the name without the env prefix, "" when the engine names none
    source_name, source_seq
    agent_type           HumanSim's agent type, "adult" when empty
    x, y, vx, vy

The crowd engine's own `agent_states` (`peds_physics`) are the source when recorded: they carry every agent's id, are in
the world frame and come at the engine rate. `arena_peds` (`peds`) is the fallback, already moved into the world frame by
the reader. Names, types and sources come from `agent_meta`, the latest row at or before each frame, else the first later
one naming the agent.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import typing

import polars as pl

DEFAULT_AGENT_TYPE = "adult"
TRACKS_FILE = "ped_tracks.parquet"
_ENV_PREFIX = re.compile(r"^/?env_\d+/")

TRACK_SCHEMA = {
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
}


def empty_tracks() -> pl.DataFrame:
    return pl.DataFrame(schema=TRACK_SCHEMA)


def _frames(peds: pl.DataFrame) -> pl.DataFrame:
    """One row per agent per frame from a `peds_*` table: time_ns, agent_id, recorded name, x, y, vx, vy."""
    need = {"time_ns", "peds_ids", "peds_positions"}
    if peds is None or len(peds) == 0 or not need <= set(peds.columns):
        return pl.DataFrame(schema={"time_ns": pl.Int64, "agent_id": pl.Int64, "recorded": pl.String, "x": pl.Float64, "y": pl.Float64, "vx": pl.Float64, "vy": pl.Float64})
    cols = [pl.col("time_ns"), pl.col("peds_ids"), pl.col("peds_positions")]
    cols.append(pl.col("peds_twists") if "peds_twists" in peds.columns else pl.lit(None, dtype=pl.List(pl.Float64)).alias("peds_twists"))
    cols.append(pl.col("peds_names") if "peds_names" in peds.columns else pl.lit(None, dtype=pl.List(pl.String)).alias("peds_names"))
    df = peds.select(cols).filter(pl.col("peds_ids").list.len() > 0)
    n = pl.col("peds_ids").list.len()
    # positions and twists are flat [x, y, z, ...]; a row whose lists disagree with its ids is dropped, not misread
    df = df.filter(pl.col("peds_positions").list.len() == 3 * n)
    df = df.with_columns(pl.col("peds_positions").list.gather_every(3, 0).alias("x"), pl.col("peds_positions").list.gather_every(3, 1).alias("y"))
    # a frame without matching twists gets NaN velocities (derived from the positions downstream), one without matching
    # names gets the ids as names, which track_table reads as unnamed
    twists = pl.col("peds_twists").list.len() == 3 * n
    nan = pl.col("x").list.eval(pl.element() * float("nan"))
    df = df.with_columns(
        pl.when(twists).then(pl.col("peds_twists").list.gather_every(3, 0)).otherwise(nan).alias("vx"),
        pl.when(twists).then(pl.col("peds_twists").list.gather_every(3, 1)).otherwise(nan).alias("vy"),
        pl.when(pl.col("peds_names").list.len() == n).then(pl.col("peds_names")).otherwise(pl.col("peds_ids").list.eval(pl.element().cast(pl.String))).alias("recorded"),
    )
    out = df.select("time_ns", pl.col("peds_ids").alias("agent_id"), "recorded", "x", "y", "vx", "vy").explode("agent_id", "recorded", "x", "y", "vx", "vy")
    return out.with_columns(pl.col("agent_id").cast(pl.Int64), pl.col("recorded").fill_null("").cast(pl.String))


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


def strip_env(name: str) -> str:
    """A name without its `env_<n>/` prefix."""
    return _ENV_PREFIX.sub("", name, count=1)


def track_table(topics: typing.Mapping[str, pl.DataFrame | None], start_ns: int | None = None, end_ns: int | None = None) -> pl.DataFrame:
    """The episode's pedestrian tracks (see the module doc) from its native topic frames, cut to [start_ns, end_ns]."""
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
        filled = [pl.coalesce(pl.col(c), pl.col(f"{c}_after")).alias(c) for c in ("meta_name", "meta_type", "source_name", "source_seq")]
        frames = pl.concat([before, after.select(*(pl.col(c).alias(f"{c}_after") for c in ("meta_name", "meta_type", "source_name", "source_seq")))], how="horizontal").with_columns(filled).drop(f"{c}_after" for c in ("meta_name", "meta_type", "source_name", "source_seq"))
    else:
        frames = frames.with_columns(pl.lit("").alias("meta_name"), pl.lit("").alias("meta_type"), pl.lit("").alias("source_name"), pl.lit(0, dtype=pl.Int64).alias("source_seq"))
    frames = frames.with_columns(pl.col("meta_name").fill_null(""), pl.col("meta_type").fill_null(""), pl.col("source_name").fill_null(""), pl.col("source_seq").fill_null(0))
    # the engine table names agents by their id; a recorded name equal to the id is no name
    recorded = pl.when(pl.col("recorded") == pl.col("agent_id").cast(pl.String)).then(pl.lit("")).otherwise(pl.col("recorded"))
    name = pl.when(pl.col("meta_name") != "").then(pl.col("meta_name")).otherwise(recorded).str.replace(_ENV_PREFIX.pattern, "")
    frames = frames.with_columns(name.alias("name"))
    key = (
        pl.when(pl.col("source_name") != "")
        .then(pl.col("source_name") + "#" + pl.col("source_seq").cast(pl.String))
        .when(pl.col("name") != "")
        .then(pl.col("name"))
        .otherwise(pl.lit("id") + pl.col("agent_id").cast(pl.String))
    )
    t0 = start_ns if start_ns is not None else int(frames["time_ns"].min())
    out = frames.with_columns(
        key.alias("key"),
        ((pl.col("time_ns") - t0) / 1e9).alias("t"),
        pl.when(pl.col("meta_type") != "").then(pl.col("meta_type")).otherwise(pl.lit(DEFAULT_AGENT_TYPE)).alias("agent_type"),
    )
    return out.select([pl.col(c).cast(dtype) for c, dtype in TRACK_SCHEMA.items()]).sort("time_ns", "key")


def tracks_for_episode(episode_dir: pathlib.Path, start_ns: int | None = None, end_ns: int | None = None) -> pl.DataFrame:
    """The tracks of a processed episode, read from its extracted topics (`<episode_dir>/topics`), empty without them."""
    from .parquet_store import TopicParquetStore

    bundles = TopicParquetStore.read(pathlib.Path(episode_dir) / "topics")
    if not bundles:
        return empty_tracks()
    bundle = next(iter(bundles.values()))
    topics: dict[str, pl.DataFrame | None] = {}
    for name in ("peds_physics", "peds", "agent_meta"):
        df = getattr(bundle, name, None)
        topics[name] = df.collect() if isinstance(df, pl.LazyFrame) else df
    return track_table(topics, start_ns, end_ns)


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


__all__ = ["DEFAULT_AGENT_TYPE", "TRACKS_FILE", "TRACK_SCHEMA", "Pairing", "empty_tracks", "pair", "strip_env", "track_table", "tracks_for_episode"]
