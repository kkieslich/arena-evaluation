from __future__ import annotations

import json
import pathlib
import shutil

import polars as pl

from arena_evaluation.processing.mcap_reader import ENV_STATE_TABLES, KEPT_EMPTY_TABLES, bundle_env
from arena_evaluation.storage.exceptions import SchemaViolationError
from arena_evaluation.storage.schemas import RunMetadata, TopicBundle


def _without_empty_structs(value: object) -> object:
    """Empty dicts infer as zero-field structs, which Parquet cannot represent."""
    if isinstance(value, dict):
        if not value:
            return None
        return {k: _without_empty_structs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_empty_structs(v) for v in value]
    return value


def _writable_rows(rows: list[dict]) -> list[dict]:
    """Metric rows with empty dicts replaced by None, so an episode with no doors writes."""
    return [{k: _without_empty_structs(v) for k, v in row.items()} for row in rows]


class ParquetStore:
    """Reads and writes metric DataFrames to Parquet format with embedded metadata."""

    METADATA_KEY = "arena_evaluation_metadata"

    @staticmethod
    def write(df: pl.DataFrame, dest: pathlib.Path, metadata: RunMetadata | None = None) -> None:
        """Write DataFrame to Parquet, embedding optional RunMetadata."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        table = df.to_arrow()

        if metadata is not None:
            meta_dict = table.schema.metadata or {}
            meta_dict[ParquetStore.METADATA_KEY.encode()] = json.dumps(metadata.model_dump(exclude_none=True)).encode()
            table = table.replace_schema_metadata(meta_dict)

        import pyarrow.parquet as pq

        pq.write_table(table, dest)

    @staticmethod
    def write_rows(
        rows: list[dict],
        dest: pathlib.Path,
        *,
        schema_overrides: dict | None = None,
        metadata: RunMetadata | None = None,
    ) -> None:
        """Build a frame from metric rows and write it, dropping empty dicts on the way."""
        df = pl.DataFrame(_writable_rows(rows), schema_overrides=schema_overrides, infer_schema_length=None)
        ParquetStore.write(df, dest, metadata)

    @staticmethod
    def read(source: pathlib.Path) -> tuple[pl.DataFrame, dict | None]:
        """Read Parquet file and return (DataFrame, metadata_dict)."""
        if not source.exists():
            raise FileNotFoundError(f"Parquet file not found: {source}")

        import pyarrow.parquet as pq

        table = pq.read_table(source)

        metadata_dict = None
        if table.schema.metadata:
            meta_bytes = table.schema.metadata.get(ParquetStore.METADATA_KEY.encode())
            if meta_bytes:
                metadata_dict = json.loads(meta_bytes.decode())

        df = pl.from_arrow(table)
        assert isinstance(df, pl.DataFrame)
        return df, metadata_dict

    @staticmethod
    def combine(sources: list[pathlib.Path], dest: pathlib.Path) -> None:
        """Combine multiple metrics.parquet files into a single combined file."""
        if not sources:
            return

        dfs = []
        for src in sources:
            try:
                df, _ = ParquetStore.read(src)
                dfs.append(df)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Failed to read {src} for combining: {e}")

        if not dfs:
            return

        try:
            combined = pl.concat(dfs, how="diagonal_relaxed")
        except Exception as e:
            raise SchemaViolationError(f"Failed to combine parquet files due to schema mismatch: {e}") from e

        ParquetStore.write(combined, dest)


class TopicParquetStore:
    """Reads and writes TopicBundle DataFrames to per-topic Parquet files."""

    VERSION = 6
    VERSION_FILE = ".cache_version"
    # Beside the version of a cache extracted for some tables only (MCAPReader.read `tables`): their names, one per line.
    TABLES_FILE = ".cache_tables"
    # Tables built from the cache live here: every write clears them, and reading the topics skips them.
    DERIVED_DIR = ".derived"

    @staticmethod
    def derived_dir(source_dir: pathlib.Path) -> pathlib.Path:
        """Where tables built from the cache in source_dir are kept."""
        return source_dir / TopicParquetStore.DERIVED_DIR

    @staticmethod
    def cached_tables(source_dir: pathlib.Path) -> frozenset[str] | None:
        """The tables a partial cache was extracted for, None for a whole one."""
        try:
            text = (source_dir / TopicParquetStore.TABLES_FILE).read_text()
        except OSError:
            return None
        return frozenset(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def is_current(source_dir: pathlib.Path, tables: frozenset[str] | set[str] | None = None) -> bool:
        """True when source_dir holds an extraction cache written by this layout that covers `tables` (None: every table)."""
        try:
            stamp = (source_dir / TopicParquetStore.VERSION_FILE).read_text().strip()
        except OSError:
            return False
        if stamp != str(TopicParquetStore.VERSION):
            return False
        cached = TopicParquetStore.cached_tables(source_dir)
        return cached is None or (tables is not None and set(tables) <= cached)

    @staticmethod
    def write(bundles: dict[str, TopicBundle], dest_dir: pathlib.Path, overwrite: bool = False, tables: frozenset[str] | set[str] | None = None) -> None:
        """Write TopicBundle DataFrames to Parquet files using zstd compression; `tables` marks a cache extracted for those tables only."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / TopicParquetStore.TABLES_FILE).unlink(missing_ok=True)
        shutil.rmtree(TopicParquetStore.derived_dir(dest_dir), ignore_errors=True)
        import dataclasses

        # Shared by every robot, written once to the root dir
        global_fields = {"tf", "tf_static", "semantic_snapshot"}
        # Shared by the robots of one env: the first env's go to the root dir, another env's into each of its robot dirs, which read() lets override the root
        env_fields = {"peds", "episode_record", *ENV_STATE_TABLES}
        global_written = set()
        root_env: dict[str, str] = {}
        complete = True

        for robot_name, bundle in bundles.items():
            robot_dir = dest_dir / robot_name
            robot_dir.mkdir(parents=True, exist_ok=True)
            env = bundle_env(robot_name)

            for field in dataclasses.fields(bundle):
                df = getattr(bundle, field.name)
                if df is not None:
                    at_root = field.name in global_fields or (field.name in env_fields and root_env.setdefault(field.name, env) == env)
                    if at_root:
                        if field.name in global_written:
                            continue
                        final_path = dest_dir / f"{field.name}.parquet"
                        global_written.add(field.name)
                    else:
                        final_path = robot_dir / f"{field.name}.parquet"

                    if isinstance(df, pl.LazyFrame):
                        if not overwrite and final_path.exists():
                            complete = False
                            continue
                        df = df.collect()

                    if not df.is_empty() or field.name in KEPT_EMPTY_TABLES:
                        temp_path = final_path.with_suffix(".parquet.tmp")
                        df.write_parquet(temp_path, compression="zstd")
                        temp_path.rename(final_path)

        if complete:
            if tables is not None:
                (dest_dir / TopicParquetStore.TABLES_FILE).write_text("".join(f"{t}\n" for t in sorted(tables)))
            (dest_dir / TopicParquetStore.VERSION_FILE).write_text(f"{TopicParquetStore.VERSION}\n")

    @staticmethod
    def read(source_dir: pathlib.Path) -> dict[str, TopicBundle] | None:
        """
        Read Parquet files from source_dir to reconstruct a dict of TopicBundles.
        Returns None if no parquet files exist.
        """
        if not source_dir.exists() or not source_dir.is_dir():
            return None

        bundles = {}

        def load_parquet(path: pathlib.Path) -> pl.LazyFrame | None:
            if path.exists():
                try:
                    lf = pl.scan_parquet(path)
                    if "time_ns" in lf.collect_schema().names():
                        lf = lf.sort("time_ns")
                    return lf
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to read parquet {path}: {e}")
            return None

        # Load global topics from root
        global_bundle = TopicBundle()
        for p in source_dir.glob("*.parquet"):
            if p.stem == "peds_fallback":
                continue
            lf = load_parquet(p)
            if lf is not None:
                setattr(global_bundle, p.stem, lf)
        if global_bundle.peds is None:
            global_bundle.peds = load_parquet(source_dir / "peds_fallback.parquet")

        # Load robot specific topics
        robot_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name != TopicParquetStore.DERIVED_DIR]

        for robot_dir in robot_dirs:
            robot_name = robot_dir.name
            rb = TopicBundle()

            # Copy global references
            rb.peds = global_bundle.peds
            rb.episode_record = global_bundle.episode_record
            rb.tf = global_bundle.tf
            rb.tf_static = global_bundle.tf_static
            rb.semantic_snapshot = global_bundle.semantic_snapshot
            for name in ENV_STATE_TABLES:
                setattr(rb, name, getattr(global_bundle, name))

            for p in robot_dir.glob("*.parquet"):
                lf = load_parquet(p)
                if lf is not None:
                    setattr(rb, p.stem, lf)

            bundles[robot_name] = rb

        return bundles if bundles else None
