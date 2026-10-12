from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from memorysum.core.errors import ScopeRetired
from memorysum.settings import load_pipeline_config
from memorysum.storage.database import DatabaseManager


def test_expired_scope_waits_for_open_connection_before_deleting_files(tmp_path):
    database = DatabaseManager(
        tmp_path / "memory",
        load_pipeline_config(),
        "embedding-space-test",
    )
    user_id = "expiring-user"
    reference = database.resolve_scope(user_id, create=True)
    assert reference is not None
    connection = database.connect(reference, user_id)

    with sqlite3.connect(database.registry_path) as registry:
        registry.execute(
            "UPDATE scopes SET expires_ms=? WHERE scope_id=?",
            (int(time.time() * 1000) - 1, reference.scope_key),
        )

    result: list[str] = []
    cleanup = threading.Thread(target=lambda: result.extend(database.retire_expired_scopes()))
    cleanup.start()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with sqlite3.connect(database.registry_path) as registry:
            row = registry.execute(
                "SELECT state FROM scopes WHERE scope_id=?", (reference.scope_key,)
            ).fetchone()
        if row and row[0] == "retired":
            break
        time.sleep(0.01)

    assert cleanup.is_alive()
    assert reference.db_path.exists()
    connection.close()
    cleanup.join(timeout=2)

    assert not cleanup.is_alive()
    assert result == [reference.scope_key]
    assert not reference.db_path.exists()
    with pytest.raises(ScopeRetired):
        database.resolve_scope(user_id, create=True)
