"""Synchronous, idempotent Add pipeline with staged source and atomic publication."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Sequence

from memorysum.api.schemas import AddRequest, AddResponse
from memorysum.core.chunking import WindowDraft, build_windows
from memorysum.core.errors import (
    AddInProgress,
    PipelineMismatch,
    ProviderError,
    RequestConflict,
    ScopeRetired,
    ServiceError,
    StorageError,
)
from memorysum.core.identity import source_id
from memorysum.core.rendering import message_token_estimate
from memorysum.core.text import proxy_tokens
from memorysum.core.vector import normalize_batch, vector_to_blob
from memorysum.providers.protocols import ModelProvider
from memorysum.settings import PipelineConfig
from memorysum.storage.database import DatabaseManager, ScopeRef, transaction

logger = logging.getLogger(__name__)
_BUILD_LEASE_MS = 180_000
_INPUT_LEASE_MS = 120_000
_RETRY_DELAY_MS = 30_000


class AddService:
    def __init__(
        self,
        database: DatabaseManager,
        provider: ModelProvider,
        pipeline: PipelineConfig,
    ) -> None:
        self._database = database
        self._provider = provider
        self._pipeline = pipeline

    def add(
        self,
        request: AddRequest,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> AddResponse:
        self._database.assert_disk_space()
        reference = self._database.resolve_scope(request.user_id, create=True)
        assert reference is not None
        owner = uuid.uuid4().hex
        digest = request.payload_digest()
        deadline = time.monotonic() + self._pipeline.capacity.add_deadline_seconds
        drafts = build_windows(
            [(message.role, message.content) for message in request.messages], self._pipeline
        )
        render_estimates = [
            message_token_estimate(
                source_id(request.user_id, request.request_id, ordinal),
                message.role,
                message.timestamp if "timestamp" in message.model_fields_set else None,
                message.content,
            )
            for ordinal, message in enumerate(request.messages)
        ]
        if diagnostics is not None:
            diagnostics.update(
                leaf_windows=sum(window.scale == "leaf" for window in drafts),
                context_windows=sum(window.scale == "context" for window in drafts),
                unique_document_inputs=len({window.embedding_text for window in drafts}),
            )
        if not drafts:
            raise ServiceError(
                "empty_memory", "No indexable source text was provided.", status_code=422
            )

        with self._database.scope_lock(reference.scope_key):
            self._database.assert_scope_active(request.user_id)
            staged = self._stage_request(
                reference, request, digest, drafts, render_estimates, owner
            )
            if isinstance(staged, AddResponse):
                return staged
            fence = staged
            try:
                self._ensure_embeddings(
                    reference,
                    request.user_id,
                    request.request_id,
                    owner,
                    fence,
                    deadline,
                    diagnostics,
                )
                if time.monotonic() >= deadline:
                    raise ServiceError(
                        "add_deadline_exceeded",
                        "Add processing exceeded its deadline.",
                        status_code=504,
                    )
                self._database.assert_scope_active(request.user_id)
                return self._publish(reference, request, owner, fence)
            except (ProviderError, ServiceError, PipelineMismatch, ScopeRetired) as exc:
                self._mark_request_retryable(
                    reference, request.user_id, request.request_id, owner, fence, exc
                )
                raise
            except Exception as exc:
                self._mark_request_retryable(
                    reference, request.user_id, request.request_id, owner, fence, exc
                )
                logger.error("Add failed before publication (%s)", type(exc).__name__)
                raise StorageError() from None

    def _stage_request(
        self,
        reference: ScopeRef,
        request: AddRequest,
        digest: bytes,
        drafts: list[WindowDraft],
        render_estimates: Sequence[int],
        owner: str,
    ) -> int | AddResponse:
        connection = self._database.connect(reference, request.user_id)
        now = self._now_ms()
        try:
            with transaction(connection):
                manifest = connection.execute(
                    "SELECT state FROM scope_manifest WHERE singleton=1"
                ).fetchone()
                if manifest is None or manifest["state"] != "active":
                    raise ScopeRetired()

                existing = connection.execute(
                    "SELECT * FROM requests WHERE request_id=?", (request.request_id,)
                ).fetchone()
                if existing is not None:
                    if bytes(existing["payload_hash"]) != digest:
                        raise RequestConflict()
                    if existing["state"] == "ready":
                        return self._response_from_json(existing["receipt_json"])
                    if (
                        existing["state"] == "building"
                        and existing["build_lease_until_ms"] is not None
                        and int(existing["build_lease_until_ms"]) > now
                    ):
                        raise AddInProgress()
                    fence = int(existing["build_fence"]) + 1
                    connection.execute(
                        """UPDATE requests SET state='building', build_fence=?, build_owner=?,
                               build_lease_until_ms=?, error_class=NULL
                           WHERE request_id=?""",
                        (fence, owner, now + _BUILD_LEASE_MS, request.request_id),
                    )
                    staged_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM messages WHERE request_id=?",
                            (request.request_id,),
                        ).fetchone()[0]
                    )
                    if staged_count == 0:
                        self._insert_staged_sources(connection, request, drafts, render_estimates)
                    elif staged_count != len(request.messages):
                        raise StorageError()
                    window_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM windows WHERE request_id=?",
                            (request.request_id,),
                        ).fetchone()[0]
                    )
                    if window_count == 0:
                        raise StorageError()
                    return fence

                connection.execute(
                    """INSERT INTO requests(
                           request_id, session_id, payload_hash, state, build_fence,
                           build_owner, build_lease_until_ms, received_ms
                       ) VALUES (?, ?, ?, 'building', 1, ?, ?, ?)""",
                    (
                        request.request_id,
                        request.session_id,
                        digest,
                        owner,
                        now + _BUILD_LEASE_MS,
                        now,
                    ),
                )
                self._insert_staged_sources(connection, request, drafts, render_estimates)
                return 1
        except (RequestConflict, AddInProgress, ScopeRetired, StorageError):
            raise
        except sqlite3.Error as exc:
            logger.error("Could not stage Add data (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    def _insert_staged_sources(
        self,
        connection,
        request: AddRequest,
        drafts: list[WindowDraft],
        render_estimates: Sequence[int],
    ) -> None:
        message_ids: dict[int, int] = {}
        for ordinal, message in enumerate(request.messages):
            public_id = source_id(request.user_id, request.request_id, ordinal)
            timestamp_ms = message.timestamp if "timestamp" in message.model_fields_set else None
            cursor = connection.execute(
                """INSERT INTO messages(
                       public_source_id, request_id, ordinal, role, timestamp_provided,
                       source_timestamp_ms, raw_content, reader_token_est
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    public_id,
                    request.request_id,
                    ordinal,
                    message.role,
                    int("timestamp" in message.model_fields_set),
                    timestamp_ms,
                    message.content,
                    render_estimates[ordinal],
                ),
            )
            message_ids[ordinal] = int(cursor.lastrowid)

        for draft in drafts:
            input_hash = hashlib.sha256(draft.embedding_text.encode("utf-8")).digest()
            connection.execute(
                """INSERT INTO embedding_inputs(
                       embedding_space_id, input_hash, input_text, state, sparse_present
                   ) VALUES (?, ?, ?, 'pending', 0)
                   ON CONFLICT(embedding_space_id, input_hash) DO NOTHING""",
                (
                    self._database.embedding_space_id,
                    input_hash,
                    draft.embedding_text,
                ),
            )
            input_row = connection.execute(
                """SELECT input_id, input_text FROM embedding_inputs
                   WHERE embedding_space_id=? AND input_hash=?""",
                (self._database.embedding_space_id, input_hash),
            ).fetchone()
            if input_row is None or input_row["input_text"] != draft.embedding_text:
                raise PipelineMismatch()
            connection.execute(
                """INSERT INTO windows(
                       request_id, ordinal, scale, message_id, input_id,
                       start_char, end_char, proxy_token_count, lexical_en, lexical_zh
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.request_id,
                    draft.ordinal,
                    draft.scale,
                    message_ids[draft.message_ordinal],
                    int(input_row["input_id"]),
                    draft.start_char,
                    draft.end_char,
                    draft.proxy_token_count,
                    draft.lexical_en,
                    draft.lexical_zh,
                ),
            )

    def _ensure_embeddings(
        self,
        reference: ScopeRef,
        user_id: str,
        request_id: str,
        owner: str,
        build_fence: int,
        deadline: float,
        diagnostics: dict[str, object] | None,
    ) -> None:
        while True:
            connection = self._database.connect(reference, user_id)
            try:
                pending = connection.execute(
                    """SELECT DISTINCT i.input_id, i.input_text, i.state, i.fence,
                              i.lease_owner, i.lease_until_ms, i.retry_at_ms
                       FROM embedding_inputs AS i
                       JOIN windows AS w ON w.input_id=i.input_id
                       WHERE w.request_id=? AND i.state!='ready'
                       ORDER BY i.input_id""",
                    (request_id,),
                ).fetchall()
            finally:
                connection.close()
            if not pending:
                return
            if time.monotonic() >= deadline:
                self._mark_request_retryable(
                    reference,
                    user_id,
                    request_id,
                    owner,
                    build_fence,
                    TimeoutError("add_deadline"),
                )
                raise ServiceError(
                    "add_deadline_exceeded",
                    "Add processing exceeded its deadline.",
                    status_code=504,
                )

            batch_rows = self._select_batch(pending)
            claims = self._claim_inputs(
                reference, user_id, request_id, owner, build_fence, batch_rows
            )
            if not claims:
                continue
            try:
                vectors = self._provider.embed([str(row["input_text"]) for row in claims])
                if diagnostics is not None:
                    diagnostics["embedding_batches"] = (
                        int(diagnostics.get("embedding_batches", 0)) + 1
                    )
                    diagnostics["embedding_inputs_embedded"] = int(
                        diagnostics.get("embedding_inputs_embedded", 0)
                    ) + len(claims)
                normalized, dimension = normalize_batch(vectors, len(claims))
                connection = self._database.connect(reference, user_id)
                try:
                    self._database.ensure_vector_tables(connection, dimension)
                    with transaction(connection):
                        self._heartbeat_request(connection, request_id, owner, build_fence)
                        for claim, vector in zip(claims, normalized, strict=True):
                            blob = vector_to_blob(vector)
                            updated = connection.execute(
                                """UPDATE embedding_inputs
                                   SET state='ready', dense_blob=?, sparse_present=0,
                                       lease_owner=NULL, lease_until_ms=NULL, retry_at_ms=NULL,
                                       last_error=NULL
                                   WHERE input_id=? AND fence=? AND lease_owner=? AND state='running'""",
                                (
                                    blob,
                                    int(claim["input_id"]),
                                    int(claim["fence"]),
                                    owner,
                                ),
                            ).rowcount
                            if updated != 1:
                                raise StorageError()
                finally:
                    connection.close()
            except ProviderError as exc:
                self._mark_inputs_retryable(
                    reference, user_id, request_id, owner, build_fence, claims, exc.code
                )
                raise
            except (PipelineMismatch, ScopeRetired, StorageError):
                raise
            except Exception as exc:
                self._mark_inputs_retryable(
                    reference,
                    user_id,
                    request_id,
                    owner,
                    build_fence,
                    claims,
                    type(exc).__name__,
                )
                if isinstance(exc, TimeoutError):
                    raise ServiceError(
                        "add_deadline_exceeded",
                        "Add processing exceeded its deadline.",
                        status_code=504,
                    ) from None
                logger.error("Embedding batch failed (%s)", type(exc).__name__)
                raise StorageError() from None

    def _select_batch(self, rows: Sequence[object]) -> list[object]:
        selected: list[object] = []
        total_tokens = 0
        for row in rows:
            state = str(row["state"])
            now = self._now_ms()
            if state == "retry_wait" and row["retry_at_ms"] is not None:
                retry_at = int(row["retry_at_ms"])
                if retry_at > now:
                    raise ServiceError(
                        "embedding_retry_wait",
                        "Add is waiting for a model retry window.",
                        status_code=503,
                        retry_after=max(1, (retry_at - now + 999) // 1000),
                    )
            if (
                state == "running"
                and row["lease_until_ms"] is not None
                and int(row["lease_until_ms"]) > now
            ):
                raise AddInProgress()
            cost = len(proxy_tokens(str(row["input_text"])))
            if cost > self._pipeline.embedding.input_proxy_tokens:
                raise ServiceError(
                    "embedding_input_too_large",
                    "A memory passage exceeds the configured embedding input limit.",
                    status_code=422,
                )
            if selected and (
                len(selected) >= self._pipeline.embedding.batch_items
                or total_tokens + cost > self._pipeline.embedding.batch_proxy_tokens
            ):
                break
            selected.append(row)
            total_tokens += cost
        if not selected:
            raise StorageError()
        return selected

    def _claim_inputs(
        self,
        reference: ScopeRef,
        user_id: str,
        request_id: str,
        owner: str,
        build_fence: int,
        rows: Sequence[object],
    ) -> list[dict[str, object]]:
        connection = self._database.connect(reference, user_id)
        now = self._now_ms()
        claims: list[dict[str, object]] = []
        try:
            with transaction(connection):
                self._heartbeat_request(connection, request_id, owner, build_fence)
                for row in rows:
                    input_id = int(row["input_id"])
                    current = connection.execute(
                        """SELECT input_text, state, fence, retry_at_ms, lease_until_ms
                           FROM embedding_inputs WHERE input_id=?""",
                        (input_id,),
                    ).fetchone()
                    if current is None:
                        raise StorageError()
                    if current["state"] == "ready":
                        continue
                    if (
                        current["state"] == "retry_wait"
                        and current["retry_at_ms"] is not None
                        and int(current["retry_at_ms"]) > now
                    ):
                        raise ServiceError(
                            "embedding_retry_wait",
                            "Add is waiting for a model retry window.",
                            status_code=503,
                            retry_after=max(1, (int(current["retry_at_ms"]) - now + 999) // 1000),
                        )
                    if (
                        current["state"] == "running"
                        and current["lease_until_ms"] is not None
                        and int(current["lease_until_ms"]) > now
                    ):
                        raise AddInProgress()
                    fence = int(current["fence"]) + 1
                    updated = connection.execute(
                        """UPDATE embedding_inputs
                           SET state='running', fence=?, lease_owner=?, lease_until_ms=?,
                               retry_at_ms=NULL, attempt_count=attempt_count+1
                           WHERE input_id=? AND fence=? AND state!='ready'""",
                        (fence, owner, now + _INPUT_LEASE_MS, input_id, int(current["fence"])),
                    ).rowcount
                    if updated != 1:
                        raise StorageError()
                    claims.append(
                        {
                            "input_id": input_id,
                            "input_text": str(current["input_text"]),
                            "fence": fence,
                        }
                    )
            return claims
        except (ServiceError, ScopeRetired, PipelineMismatch, StorageError):
            raise
        except sqlite3.Error as exc:
            logger.error("Could not claim embedding inputs (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    def _publish(
        self,
        reference: ScopeRef,
        request: AddRequest,
        owner: str,
        build_fence: int,
    ) -> AddResponse:
        connection = self._database.connect(reference, request.user_id)
        try:
            with transaction(connection):
                manifest = connection.execute(
                    "SELECT state, active_epoch, dimension, active_message_count, "
                    "full_render_tokens_est FROM scope_manifest WHERE singleton=1"
                ).fetchone()
                if manifest is None or manifest["state"] != "active":
                    raise ScopeRetired()
                current = connection.execute(
                    "SELECT state, build_owner, build_fence, payload_hash FROM requests WHERE request_id=?",
                    (request.request_id,),
                ).fetchone()
                if (
                    current is None
                    or current["build_owner"] != owner
                    or int(current["build_fence"]) != build_fence
                ):
                    raise StorageError()
                if bytes(current["payload_hash"]) != request.payload_digest():
                    raise RequestConflict()
                if current["state"] == "ready":
                    row = connection.execute(
                        "SELECT receipt_json FROM requests WHERE request_id=?",
                        (request.request_id,),
                    ).fetchone()
                    return self._response_from_json(row["receipt_json"])
                if current["state"] != "building":
                    raise StorageError()

                not_ready = connection.execute(
                    """SELECT 1 FROM embedding_inputs AS i
                       JOIN windows AS w ON w.input_id=i.input_id
                       WHERE w.request_id=? AND i.state!='ready' LIMIT 1""",
                    (request.request_id,),
                ).fetchone()
                if not_ready is not None or manifest["dimension"] is None:
                    raise StorageError()

                epoch = int(manifest["active_epoch"]) + 1
                message_stats = connection.execute(
                    """SELECT COUNT(*) AS message_count,
                              COALESCE(SUM(reader_token_est), 0) AS message_token_cost
                       FROM messages WHERE request_id=?""",
                    (request.request_id,),
                ).fetchone()
                message_count = int(message_stats["message_count"])
                message_token_cost = int(message_stats["message_token_cost"])
                windows = connection.execute(
                    """SELECT w.window_id, w.input_id, w.scale, w.lexical_en, w.lexical_zh,
                              i.dense_blob
                       FROM windows AS w JOIN embedding_inputs AS i ON i.input_id=w.input_id
                       WHERE w.request_id=? ORDER BY w.window_id""",
                    (request.request_id,),
                ).fetchall()
                if not windows:
                    raise StorageError()
                for window in windows:
                    if window["dense_blob"] is None:
                        raise StorageError()
                    connection.execute(
                        f"INSERT INTO vec_{window['scale']}(rowid, embedding, published_epoch) "
                        "VALUES (?, ?, ?)",
                        (int(window["window_id"]), window["dense_blob"], epoch),
                    )
                    connection.execute(
                        "INSERT INTO fts_en(rowid, tokens) VALUES (?, ?)",
                        (int(window["window_id"]), window["lexical_en"]),
                    )
                    connection.execute(
                        "INSERT INTO fts_zh(rowid, tokens) VALUES (?, ?)",
                        (int(window["window_id"]), window["lexical_zh"]),
                    )
                connection.execute(
                    "UPDATE windows SET published_epoch=? WHERE request_id=?",
                    (epoch, request.request_id),
                )
                connection.execute(
                    """INSERT INTO active_windows(window_id, input_id, scale, published_epoch)
                       SELECT window_id, input_id, scale, ? FROM windows WHERE request_id=?""",
                    (epoch, request.request_id),
                )
                connection.execute(
                    """INSERT INTO active_input_counts(input_id, n_windows)
                       SELECT input_id, COUNT(*) FROM windows WHERE request_id=? GROUP BY input_id
                       ON CONFLICT(input_id) DO UPDATE SET n_windows=n_windows+excluded.n_windows""",
                    (request.request_id,),
                )

                receipt = AddResponse(
                    success=True,
                    request_id=request.request_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                )
                receipt_json = json.dumps(
                    receipt.model_dump(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                updated_request = connection.execute(
                    """UPDATE requests SET state='ready', published_epoch=?, receipt_json=?,
                           build_owner=NULL, build_lease_until_ms=NULL, error_class=NULL
                       WHERE request_id=? AND build_owner=? AND build_fence=?""",
                    (epoch, receipt_json, request.request_id, owner, build_fence),
                ).rowcount
                if updated_request != 1:
                    raise StorageError()
                connection.execute(
                    """UPDATE scope_manifest SET active_epoch=?, active_message_count=?,
                           full_render_tokens_est=? WHERE singleton=1 AND state='active'""",
                    (
                        epoch,
                        int(manifest["active_message_count"]) + message_count,
                        int(manifest["full_render_tokens_est"]) + message_token_cost,
                    ),
                )
            return receipt
        except (RequestConflict, ScopeRetired, PipelineMismatch, StorageError):
            raise
        except sqlite3.Error as exc:
            logger.error("Atomic Add publication failed (%s)", type(exc).__name__)
            self._mark_request_retryable(
                reference, request.user_id, request.request_id, owner, build_fence, exc
            )
            raise StorageError() from None
        finally:
            connection.close()

    def _heartbeat_request(
        self,
        connection,
        request_id: str,
        owner: str,
        build_fence: int,
    ) -> None:
        updated = connection.execute(
            """UPDATE requests SET build_lease_until_ms=?
               WHERE request_id=? AND state='building' AND build_owner=? AND build_fence=?""",
            (self._now_ms() + _BUILD_LEASE_MS, request_id, owner, build_fence),
        ).rowcount
        if updated != 1:
            raise StorageError()

    def _mark_inputs_retryable(
        self,
        reference: ScopeRef,
        user_id: str,
        request_id: str,
        owner: str,
        build_fence: int,
        claims: Sequence[dict[str, object]],
        error_code: str,
    ) -> None:
        connection = self._database.connect(reference, user_id)
        now = self._now_ms()
        try:
            with transaction(connection):
                for claim in claims:
                    connection.execute(
                        """UPDATE embedding_inputs SET state='retry_wait', lease_owner=NULL,
                               lease_until_ms=NULL, retry_at_ms=?, last_error=?
                           WHERE input_id=? AND fence=? AND lease_owner=? AND state='running'""",
                        (
                            now + _RETRY_DELAY_MS,
                            error_code[:80],
                            int(claim["input_id"]),
                            int(claim["fence"]),
                            owner,
                        ),
                    )
                connection.execute(
                    """UPDATE requests SET state='retryable_failed', build_owner=NULL,
                           build_lease_until_ms=NULL, error_class=?
                       WHERE request_id=? AND build_owner=? AND build_fence=?""",
                    (error_code[:80], request_id, owner, build_fence),
                )
        except sqlite3.Error as exc:
            logger.error("Could not persist Add retry state (%s)", type(exc).__name__)
        finally:
            connection.close()

    def _mark_request_retryable(
        self,
        reference: ScopeRef,
        user_id: str,
        request_id: str,
        owner: str,
        build_fence: int,
        error: Exception,
    ) -> None:
        connection = None
        try:
            connection = self._database.connect(reference, user_id)
            with transaction(connection):
                connection.execute(
                    """UPDATE requests SET state='retryable_failed', build_owner=NULL,
                           build_lease_until_ms=NULL, error_class=?
                       WHERE request_id=? AND build_owner=? AND build_fence=? AND state='building'""",
                    (type(error).__name__[:80], request_id, owner, build_fence),
                )
        except Exception as exc:
            logger.error("Could not persist request retry state (%s)", type(exc).__name__)
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _response_from_json(value: str) -> AddResponse:
        payload = json.loads(value)
        return AddResponse.model_validate(payload)

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)
