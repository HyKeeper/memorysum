"""SQLite scope routing, isolation, schema setup, and 30-day data lifecycle."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from memorysum.core.errors import PipelineMismatch, ScopeRetired, StorageError
from memorysum.core.identity import scope_id
from memorysum.core.metadata import parse_chunk_metadata
from memorysum.settings import PipelineConfig

logger = logging.getLogger(__name__)
_GIB = 1024**3
_SCHEMA_VERSION = 3


def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
    """Add the harness-metadata columns to an existing v2 scope and backfill them.

    Parsing the stored text is deterministic and needs no model calls, so an older
    scope is upgraded in place rather than rebuilt. The migration is idempotent:
    it re-checks for the columns and only bumps `user_version` after the backfill,
    so an interrupted run simply retries on the next open.
    """
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(messages)")}
    if "metadata_ids" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN metadata_ids TEXT")
    if "metadata_anchor" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN metadata_anchor TEXT")

    rows = connection.execute("SELECT message_id, raw_content FROM messages").fetchall()
    backfilled = 0
    for row in rows:
        metadata = parse_chunk_metadata(str(row["raw_content"]))
        if not metadata.identifiers and not metadata.anchor:
            continue
        connection.execute(
            "UPDATE messages SET metadata_ids = ?, metadata_anchor = ? WHERE message_id = ?",
            (
                json.dumps(list(metadata.identifiers), ensure_ascii=False)
                if metadata.identifiers
                else None,
                metadata.anchor,
                int(row["message_id"]),
            ),
        )
        backfilled += 1
    # `PRAGMA user_version` joins the pending transaction, while the DDL above is
    # implicitly committed: without an explicit commit the version bump is rolled
    # back on close and the scope would migrate again (and re-scan) on every open.
    connection.commit()
    connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
    connection.commit()
    logger.info(
        "Migrated scope schema v2 -> v%d: %d/%d messages carry harness metadata",
        _SCHEMA_VERSION,
        backfilled,
        len(rows),
    )


@dataclass(frozen=True)
class ScopeRef:
    scope_key: str
    db_path: Path
    expires_ms: int


class _TrackedConnection:
    """Release the scope connection lease exactly once on close."""

    def __init__(self, connection: sqlite3.Connection, on_close) -> None:
        self._connection = connection
        self._on_close = on_close
        self._closed = False

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            self._on_close()


@contextmanager
def transaction(
    connection: sqlite3.Connection, mode: str = "IMMEDIATE"
) -> Iterator[sqlite3.Connection]:
    connection.execute(f"BEGIN {mode}")
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


class DatabaseManager:
    """One SQLite database per opaque user_id plus a hash-only scope registry."""

    def __init__(
        self,
        data_root: Path,
        pipeline: PipelineConfig,
        embedding_space_id: str,
    ) -> None:
        self.root = data_root
        self.scopes_root = data_root / "scopes"
        self.registry_path = data_root / "registry.sqlite3"
        self.pipeline = pipeline
        self.embedding_space_id = embedding_space_id
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._connections: dict[str, int] = {}
        self._connection_condition = threading.Condition()
        self._schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self._prepare_root()
        self._initialize_registry()
        self._probe_sqlite_features()

    def _probe_sqlite_features(self) -> None:
        connection = sqlite3.connect(":memory:")
        try:
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
            connection.execute(
                """CREATE VIRTUAL TABLE vec_probe USING vec0(
                       embedding float[2] distance_metric=cosine,
                       published_epoch INTEGER
                   )"""
            )
            connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(tokens)")
            vector = sqlite3.Binary(bytes.fromhex("0000803f00000000"))
            connection.execute(
                "INSERT INTO vec_probe(rowid, embedding, published_epoch) VALUES (1, ?, 1)",
                (vector,),
            )
            connection.execute("INSERT INTO fts_probe(rowid, tokens) VALUES (1, 'memorysum probe')")
            nearest = connection.execute(
                """SELECT rowid FROM vec_probe
                   WHERE embedding MATCH ? AND k=1 AND published_epoch<=1
                   ORDER BY distance""",
                (vector,),
            ).fetchone()
            lexical = connection.execute(
                "SELECT rowid FROM fts_probe WHERE fts_probe MATCH 'probe'"
            ).fetchone()
            if nearest is None or int(nearest[0]) != 1 or lexical is None:
                raise RuntimeError("SQLite extension probe returned no rows")
            connection.commit()
            connection.execute("BEGIN")
            connection.execute(
                "INSERT INTO vec_probe(rowid, embedding, published_epoch) VALUES (2, ?, 1)",
                (vector,),
            )
            connection.rollback()
            if connection.execute("SELECT 1 FROM vec_probe WHERE rowid=2").fetchone():
                raise RuntimeError("sqlite-vec rollback probe failed")
        except (sqlite3.Error, RuntimeError) as exc:
            logger.error("SQLite vector/FTS probe failed (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    def _prepare_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.scopes_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            os.chmod(self.scopes_root, 0o700)
        except OSError as exc:
            logger.error("Cannot initialize the data directory (%s)", type(exc).__name__)
            raise StorageError() from None

    def _registry_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.registry_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(
            f"PRAGMA wal_autocheckpoint = {self.pipeline.capacity.wal_autocheckpoint_pages}"
        )
        return connection

    @contextmanager
    def _registry(self) -> Iterator[sqlite3.Connection]:
        connection = self._registry_connection()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_registry(self) -> None:
        try:
            with self._registry() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scopes (
                        scope_id TEXT PRIMARY KEY,
                        state TEXT NOT NULL CHECK (state IN ('provisioning', 'active', 'retired')),
                        pipeline_id TEXT NOT NULL,
                        first_seen_ms INTEGER NOT NULL,
                        expires_ms INTEGER NOT NULL,
                        retired_ms INTEGER
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS scopes_expiry ON scopes(state, expires_ms)"
                )
            os.chmod(self.registry_path, 0o600)
        except sqlite3.Error as exc:
            logger.error("Cannot initialize the scope registry (%s)", type(exc).__name__)
            raise StorageError() from None

    def scope_lock(self, scope_key: str) -> threading.RLock:
        with self._locks_guard:
            lock = self._locks.get(scope_key)
            if lock is None:
                lock = threading.RLock()
                self._locks[scope_key] = lock
            return lock

    def _lookup_scope_record(
        self,
        key: str,
        *,
        create: bool,
        now_ms: int,
        retention_ms: int,
    ) -> tuple[bool, bool, int, bool]:
        created = False
        expired = False
        missing = False
        expires_ms = now_ms + retention_ms
        try:
            # Serialize scope admission with connection leases and retirement. A
            # deleted SQLite path must never be reopened as a fresh empty database.
            with self._connection_condition:
                connection = self._registry_connection()
                try:
                    with transaction(connection):
                        row = connection.execute(
                            "SELECT * FROM scopes WHERE scope_id = ?", (key,)
                        ).fetchone()
                        if row is not None:
                            if row["state"] == "retired" or int(row["expires_ms"]) <= now_ms:
                                if row["state"] != "retired":
                                    connection.execute(
                                        "UPDATE scopes SET state='retired', retired_ms=? "
                                        "WHERE scope_id=?",
                                        (now_ms, key),
                                    )
                                expired = True
                                expires_ms = int(row["expires_ms"])
                            elif row["pipeline_id"] != self.pipeline.pipeline_id:
                                raise PipelineMismatch()
                            else:
                                expires_ms = int(row["expires_ms"])
                        elif create:
                            connection.execute(
                                """INSERT INTO scopes(
                                       scope_id, state, pipeline_id, first_seen_ms, expires_ms
                                   ) VALUES (?, 'provisioning', ?, ?, ?)""",
                                (
                                    key,
                                    self.pipeline.pipeline_id,
                                    now_ms,
                                    now_ms + retention_ms,
                                ),
                            )
                            created = True
                        else:
                            missing = True
                finally:
                    connection.close()
        except PipelineMismatch:
            raise
        except sqlite3.Error as exc:
            logger.error("Cannot resolve memory scope (%s)", type(exc).__name__)
            raise StorageError() from None
        return created, expired, expires_ms, missing

    def resolve_scope(self, user_id: str, *, create: bool) -> ScopeRef | None:
        """Return a live scope, provision it on first Add, or reject retired IDs."""
        key = scope_id(user_id)
        now = self._now_ms()
        retention_ms = self.pipeline.capacity.scope_retention_days * 24 * 60 * 60 * 1000
        created, expired, expires_ms, missing = self._lookup_scope_record(
            key, create=create, now_ms=now, retention_ms=retention_ms
        )
        if missing:
            return None

        reference = ScopeRef(
            scope_key=key,
            db_path=self.scopes_root / f"{key}.sqlite3",
            expires_ms=expires_ms,
        )
        if expired:
            with self.scope_lock(key):
                self._wait_for_scope_connections(key)
                self._delete_scope_files(reference)
            raise ScopeRetired()

        if created:
            self._assert_disk_space()

        with self.scope_lock(key):
            try:
                state = self._registry_state(key)
                if state == "provisioning":
                    if reference.db_path.exists():
                        self._verify_scope_file(reference, user_id)
                    else:
                        row = self._registry_row(key)
                        self._create_scope_file(reference, user_id, int(row["first_seen_ms"]))
                    self._set_scope_active(key)
                    state = self._registry_state(key)
                if state != "active":
                    raise ScopeRetired()
                if not reference.db_path.exists():
                    raise StorageError()
            except PipelineMismatch:
                raise
            except ScopeRetired:
                self._wait_for_scope_connections(key)
                self._delete_scope_files(reference)
                raise
            except StorageError:
                raise
            except sqlite3.Error as exc:
                logger.error("Cannot open the memory scope (%s)", type(exc).__name__)
                raise StorageError() from None
        return reference

    def connect(self, reference: ScopeRef, user_id: str) -> _TrackedConnection:
        self._acquire_scope_connection(reference)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(reference.db_path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                f"PRAGMA wal_autocheckpoint = {self.pipeline.capacity.wal_autocheckpoint_pages}"
            )
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
            self._verify_schema(connection)
            self._verify_manifest(connection, reference, user_id)
            tracked = _TrackedConnection(
                connection,
                lambda: self._release_scope_connection(reference.scope_key),
            )
            connection = None
            return tracked
        except (PipelineMismatch, ScopeRetired, StorageError):
            if connection is not None:
                connection.close()
            self._release_scope_connection(reference.scope_key)
            raise
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            if connection is not None:
                connection.close()
            self._release_scope_connection(reference.scope_key)
            logger.error("Cannot open scope database (%s)", type(exc).__name__)
            raise StorageError() from None

    def _acquire_scope_connection(self, reference: ScopeRef) -> None:
        try:
            with self._connection_condition:
                with self._registry() as connection:
                    row = connection.execute(
                        "SELECT state, pipeline_id, expires_ms FROM scopes WHERE scope_id=?",
                        (reference.scope_key,),
                    ).fetchone()
                if (
                    row is None
                    or row["state"] != "active"
                    or int(row["expires_ms"]) <= self._now_ms()
                ):
                    raise ScopeRetired()
                if row["pipeline_id"] != self.pipeline.pipeline_id:
                    raise PipelineMismatch()
                if not reference.db_path.exists():
                    raise StorageError()
                self._connections[reference.scope_key] = (
                    self._connections.get(reference.scope_key, 0) + 1
                )
        except (PipelineMismatch, ScopeRetired, StorageError):
            raise
        except sqlite3.Error as exc:
            logger.error("Cannot acquire a scope connection (%s)", type(exc).__name__)
            raise StorageError() from None

    def ensure_vector_tables(self, connection: sqlite3.Connection, dimension: int) -> None:
        if dimension <= 0:
            raise StorageError()
        row = connection.execute(
            "SELECT dimension FROM scope_manifest WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise StorageError()
        current = row["dimension"]
        if current is not None and int(current) != dimension:
            raise PipelineMismatch()
        # Create the virtual tables before updating dimension; some SQLite builds do
        # not permit virtual-table DDL inside an active transaction.
        connection.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_leaf USING vec0(
                   embedding float[{dimension}] distance_metric=cosine,
                   published_epoch INTEGER
               )"""
        )
        connection.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_context USING vec0(
                   embedding float[{dimension}] distance_metric=cosine,
                   published_epoch INTEGER
               )"""
        )
        if current is None:
            connection.execute(
                "UPDATE scope_manifest SET dimension=? WHERE singleton=1", (dimension,)
            )
            connection.commit()

    def assert_scope_active(self, user_id: str) -> None:
        key = scope_id(user_id)
        try:
            with self._registry() as connection:
                row = connection.execute(
                    "SELECT state, pipeline_id, expires_ms FROM scopes WHERE scope_id=?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.error("Cannot verify memory scope lifecycle (%s)", type(exc).__name__)
            raise StorageError() from None
        if row is None or row["state"] != "active" or int(row["expires_ms"]) <= self._now_ms():
            raise ScopeRetired()
        if row["pipeline_id"] != self.pipeline.pipeline_id:
            raise PipelineMismatch()

    def assert_disk_space(self) -> None:
        self._assert_disk_space()

    def retire_expired_scopes(self) -> list[str]:
        now = self._now_ms()
        try:
            with self._registry() as connection:
                rows = connection.execute(
                    "SELECT scope_id FROM scopes WHERE state='active' AND expires_ms <= ?",
                    (now,),
                ).fetchall()
        except sqlite3.Error as exc:
            logger.error("Cannot query expired scopes (%s)", type(exc).__name__)
            raise StorageError() from None

        retired: list[str] = []
        for row in rows:
            key = str(row["scope_id"])
            with self.scope_lock(key):
                try:
                    with self._connection_condition:
                        with self._registry() as connection:
                            with transaction(connection):
                                current = connection.execute(
                                    "SELECT state, expires_ms FROM scopes WHERE scope_id=?",
                                    (key,),
                                ).fetchone()
                                if (
                                    current is None
                                    or current["state"] != "active"
                                    or int(current["expires_ms"]) > self._now_ms()
                                ):
                                    continue
                                connection.execute(
                                    "UPDATE scopes SET state='retired', retired_ms=? "
                                    "WHERE scope_id=?",
                                    (self._now_ms(), key),
                                )
                    self._wait_for_scope_connections(key)
                    self._delete_scope_files(ScopeRef(key, self.scopes_root / f"{key}.sqlite3", 0))
                    retired.append(key)
                except sqlite3.Error as exc:
                    logger.error("Cannot retire expired scope (%s)", type(exc).__name__)
        return retired

    def _create_scope_file(self, reference: ScopeRef, user_id: str, first_seen_ms: int) -> None:
        reference.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(reference.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                f"PRAGMA wal_autocheckpoint = {self.pipeline.capacity.wal_autocheckpoint_pages}"
            )
            connection.executescript(self._schema)
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            expires_ms = first_seen_ms + (
                self.pipeline.capacity.scope_retention_days * 24 * 60 * 60 * 1000
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO scope_manifest(
                    singleton, user_id, pipeline_id, embedding_space_id,
                    dimension, state, active_epoch, first_seen_ms, expires_ms
                ) VALUES (1, ?, ?, ?, NULL, 'active', 0, ?, ?)
                """,
                (
                    user_id,
                    self.pipeline.pipeline_id,
                    self.embedding_space_id,
                    first_seen_ms,
                    expires_ms,
                ),
            )
            connection.commit()
            self._verify_manifest(connection, reference, user_id)
            os.chmod(reference.db_path, 0o600)
        except Exception:
            connection.close()
            self._delete_scope_files(reference)
            raise
        else:
            connection.close()

    def _verify_scope_file(self, reference: ScopeRef, user_id: str) -> None:
        connection = sqlite3.connect(reference.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                f"PRAGMA wal_autocheckpoint = {self.pipeline.capacity.wal_autocheckpoint_pages}"
            )
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
            self._verify_schema(connection)
            self._verify_manifest(connection, reference, user_id)
        finally:
            connection.close()

    @staticmethod
    def _verify_schema(connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == _SCHEMA_VERSION:
            return
        if version == 2:
            _migrate_v2_to_v3(connection)
            return
        raise StorageError()

    def _verify_manifest(
        self, connection: sqlite3.Connection, reference: ScopeRef, user_id: str
    ) -> None:
        row = connection.execute(
            "SELECT user_id, pipeline_id, embedding_space_id, state, expires_ms "
            "FROM scope_manifest WHERE singleton=1"
        ).fetchone()
        if row is None:
            raise StorageError()
        if (
            row["user_id"] != user_id
            or row["pipeline_id"] != self.pipeline.pipeline_id
            or row["embedding_space_id"] != self.embedding_space_id
            or int(row["expires_ms"]) != reference.expires_ms
        ):
            raise PipelineMismatch()
        if row["state"] != "active":
            raise ScopeRetired()

    def _registry_row(self, key: str) -> sqlite3.Row:
        with self._registry() as connection:
            row = connection.execute("SELECT * FROM scopes WHERE scope_id=?", (key,)).fetchone()
        if row is None:
            raise StorageError()
        return row

    def _registry_state(self, key: str) -> str | None:
        with self._registry() as connection:
            row = connection.execute("SELECT state FROM scopes WHERE scope_id=?", (key,)).fetchone()
        return str(row["state"]) if row else None

    def _set_scope_active(self, key: str) -> None:
        with self._registry() as connection:
            with transaction(connection):
                updated = connection.execute(
                    """UPDATE scopes SET state='active'
                       WHERE scope_id=? AND state='provisioning' AND expires_ms>?""",
                    (key, self._now_ms()),
                ).rowcount
                if updated != 1:
                    raise ScopeRetired()

    def _release_scope_connection(self, key: str) -> None:
        with self._connection_condition:
            remaining = self._connections.get(key, 0) - 1
            if remaining <= 0:
                self._connections.pop(key, None)
            else:
                self._connections[key] = remaining
            self._connection_condition.notify_all()

    def _wait_for_scope_connections(self, key: str) -> None:
        with self._connection_condition:
            while self._connections.get(key, 0) > 0:
                self._connection_condition.wait()

    def _delete_scope_files(self, reference: ScopeRef) -> None:
        for path in (
            reference.db_path,
            Path(str(reference.db_path) + "-wal"),
            Path(str(reference.db_path) + "-shm"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.error("Cannot remove retired scope data (%s)", type(exc).__name__)

    def _assert_disk_space(self) -> None:
        try:
            free_bytes = os.statvfs(self.root).f_bavail * os.statvfs(self.root).f_frsize
        except (AttributeError, OSError):
            return
        minimum = 40 * _GIB
        if free_bytes < minimum:
            from memorysum.core.errors import ServiceError

            raise ServiceError(
                "disk_space_low",
                "The memory service is temporarily not accepting Add requests.",
                status_code=503,
                retry_after=60,
            )

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
