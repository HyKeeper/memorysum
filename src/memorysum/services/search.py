"""Bounded hybrid retrieval, evidence-only reranking, and output budgeting."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass

from memorysum.api.schemas import SearchItem, SearchRequest, SearchResponse
from memorysum.core.cache import TTLCache
from memorysum.core.errors import (
    PipelineMismatch,
    ProviderError,
    ScopeRetired,
    ServiceError,
    StorageError,
)
from memorysum.core.identity import canonical_json, scope_id
from memorysum.core.query_modes import is_context_query, is_wide_query
from memorysum.core.ranking import FusedCandidate, reciprocal_rank_fusion
from memorysum.core.rendering import (
    FULL_BUNDLE_SEPARATOR,
    full_bundle_token_cost,
    render_full_bundle,
    render_window,
)
from memorysum.core.text import (
    chinese_terms,
    english_terms,
    query_options_text,
    reader_token_count,
    safe_fts_expression,
)
from memorysum.core.vector import normalize_batch, vector_to_blob
from memorysum.providers.protocols import ModelProvider
from memorysum.settings import PipelineConfig, RetrievalModeConfig
from memorysum.storage.database import DatabaseManager, ScopeRef, transaction

logger = logging.getLogger(__name__)
_RENDER_TOKEN_OVERHEAD = 8
_OPTION_LABEL_RE = re.compile(r"^\s*(?:[A-Za-z][.)]|[A-Za-z][:：]|\d+[.)])\s*")


@dataclass(frozen=True)
class Snapshot:
    epoch: int
    dimension: int | None
    active_message_count: int
    full_render_tokens_est: int


@dataclass(frozen=True)
class RerankEntry:
    candidate: FusedCandidate
    view_id: str
    content: str
    span: dict[str, object]
    pair_proxy_tokens: int


@dataclass
class CandidateGroup:
    candidate: FusedCandidate
    row: dict[str, object]
    contains_context_window: bool


class SearchService:
    def __init__(
        self,
        database: DatabaseManager,
        provider: ModelProvider,
        pipeline: PipelineConfig,
    ) -> None:
        self._database = database
        self._provider = provider
        self._pipeline = pipeline
        self._query_vectors: TTLCache[list[float]] = TTLCache(1024, 300)
        self._results: TTLCache[SearchResponse] = TTLCache(256, 60)
        self._full_bundles: TTLCache[tuple[str, str, int]] = TTLCache(64, 300)

    def clear_scope(self, scope_key: str) -> None:
        self._query_vectors.clear_scope(scope_key)
        self._results.clear_scope(scope_key)
        self._full_bundles.clear_scope(scope_key)

    def search(
        self,
        request: SearchRequest,
        *,
        diagnostics: dict[str, object] | None = None,
    ) -> SearchResponse:
        diagnostics = diagnostics if diagnostics is not None else {}
        try:
            return self._search(request, diagnostics)
        except ScopeRetired:
            self.clear_scope(scope_id(request.user_id))
            raise

    def _search(self, request: SearchRequest, diagnostics: dict[str, object]) -> SearchResponse:
        deadline = time.monotonic() + self._pipeline.capacity.search_deadline_seconds
        reference = self._database.resolve_scope(request.user_id, create=False)
        if reference is None:
            diagnostics["strategy"] = "empty_scope"
            return SearchResponse(data=[])
        self._database.assert_scope_active(request.user_id)
        snapshot = self._read_snapshot(reference, request.user_id)
        if request.top_k == 0 or snapshot.epoch == 0:
            diagnostics["strategy"] = "empty_top_k" if request.top_k == 0 else "empty_scope"
            self._database.assert_scope_active(request.user_id)
            return SearchResponse(data=[])

        scope_key = reference.scope_key
        options = request.options or []
        question_text = query_options_text(request.query, options)
        question_cost = reader_token_count(question_text)
        rerank_query_cost = reader_token_count(
            self._pipeline.reranker.instruction + "\n" + question_text
        )
        if rerank_query_cost > 6144:
            raise ServiceError(
                "query_options_too_large",
                "The query and options exceed the configured reranker input limit.",
                status_code=422,
            )

        result_cache_key = self._result_cache_key(scope_key, snapshot.epoch, request, options)
        cached = self._results.get(result_cache_key)
        if cached is not None:
            diagnostics["cache_hit"] = True
            diagnostics["strategy"] = "result_cache"
            self._database.assert_scope_active(request.user_id)
            return cached.model_copy(deep=True)

        wide_mode = is_wide_query(request.query)
        context_mode = is_context_query(request.query)
        diagnostics["wide_mode"] = wide_mode
        diagnostics["context_mode"] = context_mode
        full_bundle = self._try_full_bundle(
            reference,
            request.user_id,
            snapshot,
            question_cost,
            wide_mode or context_mode,
        )
        if full_bundle is not None:
            diagnostics["strategy"] = "full_bundle"
            diagnostics["full_bundle_tokens_est"] = full_bundle[2]
            diagnostics["returned_records"] = 1
            content, view_id, _ = full_bundle
            response = SearchResponse(data=[SearchItem(id=view_id, content=content)])
            self._database.assert_scope_active(request.user_id)
            self._results.set(result_cache_key, response)
            return response

        if snapshot.dimension is None:
            raise StorageError()
        query_vector = self._query_vector(reference, request.user_id, request.query)
        if time.monotonic() >= deadline:
            raise ServiceError(
                "search_deadline_exceeded",
                "Search processing exceeded its deadline.",
                status_code=504,
            )
        if len(query_vector) != snapshot.dimension:
            raise PipelineMismatch()

        channels, rows = self._retrieve_candidates(
            reference,
            request.user_id,
            snapshot,
            query_vector,
            request.query,
            options,
            wide_mode,
        )
        diagnostics["strategy"] = "hybrid_retrieval"
        diagnostics["channel_hits"] = {name: len(ids) for name, ids, _ in channels}
        mode = self._pipeline.retrieval.wide if wide_mode else self._pipeline.retrieval.ordinary
        fused = reciprocal_rank_fusion(
            channels,
            k=self._pipeline.retrieval.rrf_k,
            limit=mode.candidate_cap,
        )
        diagnostics["candidate_windows"] = len(fused)
        entries = self._prepare_rerank_entries(fused, rows, request, options, wide_mode)
        diagnostics["rerank_groups"] = len(entries)
        diagnostics["rerank_pair_tokens_est"] = sum(entry.pair_proxy_tokens for entry in entries)
        if not entries:
            diagnostics["returned_records"] = 0
            response = SearchResponse(data=[])
            self._database.assert_scope_active(request.user_id)
            self._results.set(result_cache_key, response)
            return response

        rerank_query = self._rerank_query(request.query, options)
        scores = self._rerank_entries(
            rerank_query,
            entries,
            deadline,
        )
        ranked = sorted(
            zip(entries, scores, strict=True),
            key=lambda pair: (
                -pair[1],
                -pair[0].candidate.score,
                pair[0].view_id,
            ),
        )
        response = self._select_output(ranked, request.top_k, mode, question_cost)
        diagnostics["returned_records"] = len(response.data)
        diagnostics["returned_chars"] = sum(len(item.content) for item in response.data)
        self._database.assert_scope_active(request.user_id)
        self._results.set(result_cache_key, response)
        return response

    def _read_snapshot(self, reference: ScopeRef, user_id: str) -> Snapshot:
        connection = self._database.connect(reference, user_id)
        try:
            with transaction(connection, "DEFERRED"):
                row = connection.execute(
                    """SELECT state, active_epoch, dimension, active_message_count,
                              full_render_tokens_est
                       FROM scope_manifest WHERE singleton=1"""
                ).fetchone()
                if row is None or row["state"] != "active":
                    raise ScopeRetired()
                return Snapshot(
                    epoch=int(row["active_epoch"]),
                    dimension=int(row["dimension"]) if row["dimension"] is not None else None,
                    active_message_count=int(row["active_message_count"]),
                    full_render_tokens_est=int(row["full_render_tokens_est"]),
                )
        except sqlite3.Error as exc:
            logger.error("Could not read Search snapshot (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    def _try_full_bundle(
        self,
        reference: ScopeRef,
        user_id: str,
        snapshot: Snapshot,
        question_cost: int,
        wide_or_context: bool,
    ) -> tuple[str, str, int] | None:
        estimate = snapshot.full_render_tokens_est + reader_token_count(
            FULL_BUNDLE_SEPARATOR
        ) * max(0, snapshot.active_message_count - 1)
        if estimate <= 0 or estimate > self._pipeline.coverage.conditional_full_tokens:
            return None

        cache_key = f"{reference.scope_key}:{snapshot.epoch}"
        cached = self._full_bundles.get(cache_key)
        if cached is None:
            messages = self._load_messages(reference, user_id, snapshot.epoch)
            if not messages:
                return None
            content, view_id = render_full_bundle(messages)
            full_cost = full_bundle_token_cost(messages)
            cached = (content, view_id, full_cost)
            self._full_bundles.set(cache_key, cached)
        content, view_id, full_cost = cached

        full_threshold = self._pipeline.coverage.always_full_tokens
        if full_cost > full_threshold:
            conditional_allowed = (
                wide_or_context and full_cost <= self._pipeline.coverage.conditional_full_tokens
            )
            if not conditional_allowed:
                return None
        safe_total = (
            full_cost + question_cost + self._pipeline.coverage.platform_instruction_reserve
        ) * self._pipeline.coverage.estimation_safety_factor
        if safe_total > self._pipeline.coverage.safe_input_tokens:
            return None
        return content, view_id, full_cost

    def _load_messages(self, reference: ScopeRef, user_id: str, epoch: int) -> list[sqlite3.Row]:
        connection = self._database.connect(reference, user_id)
        try:
            with transaction(connection, "DEFERRED"):
                rows = connection.execute(
                    """SELECT m.message_id, m.public_source_id, m.request_id, m.ordinal,
                              m.role, m.source_timestamp_ms, m.raw_content, m.reader_token_est
                       FROM messages AS m
                       JOIN requests AS r ON r.request_id=m.request_id
                       WHERE r.state='ready' AND r.published_epoch<=?
                       ORDER BY r.published_epoch, m.ordinal, m.message_id""",
                    (epoch,),
                ).fetchall()
            return rows
        except sqlite3.Error as exc:
            logger.error("Could not load the full source bundle (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    def _query_vector(self, reference: ScopeRef, user_id: str, query: str) -> list[float]:
        key = f"{reference.scope_key}:{hashlib.sha256(query.encode('utf-8')).hexdigest()}"
        cached = self._query_vectors.get(key)
        if cached is not None:
            return cached
        query_input = self._pipeline.embedding.query_instruction + "\n\n" + query
        vectors = self._provider.embed([query_input])
        normalized, _ = normalize_batch(vectors, 1)
        vector = normalized[0]
        self._query_vectors.set(key, vector)
        return vector

    def _retrieve_candidates(
        self,
        reference: ScopeRef,
        user_id: str,
        snapshot: Snapshot,
        query_vector: Sequence[float],
        query: str,
        options: Sequence[str],
        wide_mode: bool,
    ) -> tuple[list[tuple[str, list[int], float]], dict[int, sqlite3.Row]]:
        mode = self._pipeline.retrieval.wide if wide_mode else self._pipeline.retrieval.ordinary
        connection = self._database.connect(reference, user_id)
        try:
            with transaction(connection, "DEFERRED"):
                leaf = self._dense_hits(
                    connection,
                    "vec_leaf",
                    query_vector,
                    snapshot.epoch,
                    mode.leaf_dense,
                )
                context = self._dense_hits(
                    connection,
                    "vec_context",
                    query_vector,
                    snapshot.epoch,
                    mode.context_dense,
                )
                en = self._fts_hits(
                    connection,
                    "fts_en",
                    english_terms(query),
                    snapshot.epoch,
                    mode.english_fts,
                )
                zh = self._fts_hits(
                    connection,
                    "fts_zh",
                    chinese_terms(query),
                    snapshot.epoch,
                    mode.chinese_fts,
                )
                option_ids = self._option_hits(
                    connection,
                    options,
                    snapshot.epoch,
                    mode.options_lexical,
                )
                channels: list[tuple[str, list[int], float]] = [
                    ("leaf_dense", leaf, self._pipeline.retrieval.base_channel_weight),
                    ("context_dense", context, self._pipeline.retrieval.base_channel_weight),
                    ("english_fts", en, self._pipeline.retrieval.base_channel_weight),
                    ("chinese_fts", zh, self._pipeline.retrieval.base_channel_weight),
                    ("options_lexical", option_ids, self._pipeline.retrieval.base_channel_weight),
                ]
                all_ids = sorted({window_id for _, ids, _ in channels for window_id in ids})
                rows = self._load_window_rows(connection, all_ids, snapshot.epoch)
            return channels, rows
        except (PipelineMismatch, ScopeRetired, ServiceError):
            raise
        except sqlite3.Error as exc:
            logger.error("Local retrieval failed (%s)", type(exc).__name__)
            raise StorageError() from None
        finally:
            connection.close()

    @staticmethod
    def _dense_hits(
        connection: sqlite3.Connection,
        table: str,
        query_vector: Sequence[float],
        epoch: int,
        limit: int,
    ) -> list[int]:
        if limit <= 0:
            return []
        blob = vector_to_blob(query_vector)
        rows = connection.execute(
            f"""SELECT rowid, distance FROM {table}
                WHERE embedding MATCH ? AND k=? AND published_epoch<=?
                ORDER BY distance""",
            (blob, limit, epoch),
        ).fetchall()
        rows = sorted(rows, key=lambda row: (float(row["distance"]), int(row["rowid"])))
        return [int(row["rowid"]) for row in rows]

    @staticmethod
    def _fts_hits(
        connection: sqlite3.Connection,
        table: str,
        terms: Sequence[str],
        epoch: int,
        limit: int,
    ) -> list[int]:
        expression = safe_fts_expression(terms)
        if not expression or limit <= 0:
            return []
        rows = connection.execute(
            f"""SELECT f.rowid AS window_id, bm25({table}) AS rank_score
                FROM {table} AS f
                JOIN active_windows AS a ON a.window_id=f.rowid
                WHERE {table} MATCH ? AND a.published_epoch<=?
                ORDER BY rank_score ASC, f.rowid ASC LIMIT ?""",
            (expression, epoch, limit),
        ).fetchall()
        return [int(row["window_id"]) for row in rows]

    def _option_hits(
        self,
        connection: sqlite3.Connection,
        options: Sequence[str],
        epoch: int,
        limit: int,
    ) -> list[int]:
        if not options or limit <= 0:
            return []
        unique_options = sorted(
            set(options), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        )
        proposals: list[tuple[int, str, int]] = []
        for option in unique_options:
            option_hash = hashlib.sha256(option.encode("utf-8")).hexdigest()
            option_text = _OPTION_LABEL_RE.sub("", option, count=1)
            branch: dict[int, int] = {}
            en = self._fts_hits(connection, "fts_en", english_terms(option_text), epoch, 2)
            zh = self._fts_hits(connection, "fts_zh", chinese_terms(option_text), epoch, 2)
            for rank, window_id in enumerate(en, start=1):
                branch[window_id] = min(branch.get(window_id, 2**31 - 1), rank)
            for rank, window_id in enumerate(zh, start=1):
                branch[window_id] = min(branch.get(window_id, 2**31 - 1), rank)
            for window_id, rank in branch.items():
                proposals.append((rank, option_hash, window_id))
        proposals.sort()
        selected: list[int] = []
        seen: set[int] = set()
        for _, _, window_id in proposals:
            if window_id in seen:
                continue
            seen.add(window_id)
            selected.append(window_id)
            if len(selected) >= limit:
                break
        return selected

    @staticmethod
    def _load_window_rows(
        connection: sqlite3.Connection, window_ids: Sequence[int], epoch: int
    ) -> dict[int, sqlite3.Row]:
        if not window_ids:
            return {}
        placeholders = ",".join("?" for _ in window_ids)
        rows = connection.execute(
            f"""SELECT w.window_id, w.scale, w.message_id, w.start_char, w.end_char,
                       w.input_id, w.published_epoch, m.public_source_id, m.request_id,
                       m.ordinal AS message_ordinal, m.role, m.source_timestamp_ms, m.raw_content
                FROM windows AS w
                JOIN active_windows AS a ON a.window_id=w.window_id
                JOIN messages AS m ON m.message_id=w.message_id
                WHERE w.window_id IN ({placeholders}) AND a.published_epoch<=?
                ORDER BY w.window_id""",
            (*window_ids, epoch),
        ).fetchall()
        return {int(row["window_id"]): row for row in rows}

    def _group_overlapping_candidates(
        self,
        fused: Sequence[FusedCandidate],
        rows: Mapping[int, sqlite3.Row],
    ) -> list[CandidateGroup]:
        groups: list[CandidateGroup] = []
        for candidate in fused:
            source = rows.get(candidate.window_id)
            if source is None:
                continue
            row = dict(source)
            start = int(row["start_char"])
            end = int(row["end_char"])
            is_context = row["scale"] == "context"
            overlapping = [
                group
                for group in groups
                if int(group.row["message_id"]) == int(row["message_id"])
                and start <= int(group.row["end_char"]) + 1
                and end + 1 >= int(group.row["start_char"])
            ]

            if not overlapping:
                groups.append(CandidateGroup(candidate, row, is_context))
                continue

            # Merge the entire connected source view only while it fits the
            # frozen direct/context group budget. Never cut a source span.
            all_start = min(start, *(int(group.row["start_char"]) for group in overlapping))
            all_end = max(end, *(int(group.row["end_char"]) for group in overlapping))
            has_context = is_context or any(group.contains_context_window for group in overlapping)
            merged_row = {**row, "start_char": all_start, "end_char": all_end}
            merged_content, _, _ = render_window(merged_row)
            token_limit = (
                self._pipeline.groups.context_max_tokens
                if has_context
                else self._pipeline.groups.direct_max_tokens
            )
            if reader_token_count(merged_content) <= token_limit:
                merged_candidate = self._merge_cpu_candidates(
                    [candidate, *(group.candidate for group in overlapping)]
                )
                groups = [group for group in groups if group not in overlapping]
                groups.append(CandidateGroup(merged_candidate, merged_row, has_context))
                continue

            # If merging every touching group is too large, attach to the
            # strongest single compatible group; otherwise retain a separate
            # complete passage for reranking.
            attached = False
            for group in overlapping:
                pair_start = min(start, int(group.row["start_char"]))
                pair_end = max(end, int(group.row["end_char"]))
                pair_context = is_context or group.contains_context_window
                pair_row = {**row, "start_char": pair_start, "end_char": pair_end}
                pair_content, _, _ = render_window(pair_row)
                pair_limit = (
                    self._pipeline.groups.context_max_tokens
                    if pair_context
                    else self._pipeline.groups.direct_max_tokens
                )
                if reader_token_count(pair_content) > pair_limit:
                    continue
                group.row = pair_row
                group.candidate = self._merge_cpu_candidates([group.candidate, candidate])
                group.contains_context_window = pair_context
                attached = True
                break
            if not attached:
                groups.append(CandidateGroup(candidate, row, is_context))

        return sorted(
            groups,
            key=lambda group: (
                -round(group.candidate.score, 12),
                group.candidate.best_rank,
                group.candidate.window_id,
            ),
        )

    @staticmethod
    def _merge_cpu_candidates(candidates: Sequence[FusedCandidate]) -> FusedCandidate:
        return FusedCandidate(
            window_id=min(candidate.window_id for candidate in candidates),
            score=max(candidate.score for candidate in candidates),
            best_rank=min(candidate.best_rank for candidate in candidates),
            channels=set().union(*(candidate.channels for candidate in candidates)),
        )

    def _prepare_rerank_entries(
        self,
        fused: Sequence[FusedCandidate],
        rows: Mapping[int, sqlite3.Row],
        request: SearchRequest,
        options: Sequence[str],
        wide_mode: bool,
    ) -> list[RerankEntry]:
        max_groups = (
            self._pipeline.reranker.wide_candidate_groups
            if wide_mode
            else self._pipeline.reranker.ordinary_candidate_groups
        )
        pair_budget = (
            self._pipeline.reranker.wide_pair_proxy_budget
            if wide_mode
            else self._pipeline.reranker.ordinary_pair_proxy_budget
        )
        rerank_question = query_options_text(request.query, list(options) if options else None)
        candidates_by_view: dict[str, RerankEntry] = {}
        groups = self._group_overlapping_candidates(fused, rows)
        for group in groups:
            candidate = group.candidate
            content, view_id, span = render_window(group.row)
            pair_text = (
                f"<Instruct>: {self._pipeline.reranker.instruction}\n"
                f"<Query>: {rerank_question}\n<Document>: {content}"
            )
            pair_cost = reader_token_count(pair_text)
            existing = candidates_by_view.get(view_id)
            if existing is None or candidate.score > existing.candidate.score:
                candidates_by_view[view_id] = RerankEntry(
                    candidate=candidate,
                    view_id=view_id,
                    content=content,
                    span=span,
                    pair_proxy_tokens=pair_cost,
                )

        ordered = sorted(
            candidates_by_view.values(),
            key=lambda entry: (
                -round(entry.candidate.score, 12),
                entry.candidate.best_rank,
                entry.candidate.window_id,
            ),
        )
        selected: list[RerankEntry] = []
        spent = 0
        for entry in ordered:
            cost = entry.pair_proxy_tokens
            if cost > self._pipeline.reranker.single_pair_proxy_tokens:
                continue
            if spent + cost > pair_budget:
                continue
            selected.append(entry)
            spent += cost
            if len(selected) >= max_groups:
                break
        return selected

    def _rerank_entries(
        self,
        query: str,
        entries: Sequence[RerankEntry],
        deadline: float,
    ) -> list[float]:
        batch_size = self._pipeline.reranker.batch_groups
        batches = [
            list(range(start, min(start + batch_size, len(entries))))
            for start in range(0, len(entries), batch_size)
        ]
        result: list[float | None] = [None] * len(entries)
        executor = ThreadPoolExecutor(max_workers=self._pipeline.reranker.max_concurrency)
        futures = {
            executor.submit(
                self._provider.rerank,
                query,
                [entries[index].content for index in batch],
            ): batch
            for batch in batches
        }
        try:
            for future in as_completed(futures, timeout=max(0.0, deadline - time.monotonic())):
                batch = futures[future]
                scores = future.result()
                if len(scores) != len(batch):
                    raise ProviderError(
                        "model_provider_invalid_rerank",
                        "The reranker returned an incomplete result set.",
                        status_code=503,
                    )
                for score in scores:
                    if (
                        score.index < 0
                        or score.index >= len(batch)
                        or not math.isfinite(score.score)
                    ):
                        raise ProviderError(
                            "model_provider_invalid_rerank",
                            "The reranker returned an invalid result.",
                            status_code=503,
                        )
                    result[batch[score.index]] = float(score.score)
        except FuturesTimeout:
            raise ServiceError(
                "search_deadline_exceeded",
                "Search processing exceeded its deadline.",
                status_code=504,
            ) from None
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if any(score is None for score in result):
            raise ProviderError(
                "model_provider_invalid_rerank",
                "The reranker returned an incomplete result set.",
                status_code=503,
            )
        return [float(score) for score in result if score is not None]

    def _select_output(
        self,
        ranked: Sequence[tuple[RerankEntry, float]],
        top_k: int,
        mode: RetrievalModeConfig,
        question_cost: int,
    ) -> SearchResponse:
        memory_budget = max(
            0,
            min(
                mode.output_tokens,
                int(
                    self._pipeline.coverage.safe_input_tokens
                    / self._pipeline.coverage.estimation_safety_factor
                    - question_cost
                    - self._pipeline.coverage.platform_instruction_reserve
                ),
            ),
        )
        max_records = min(top_k, mode.output_records)
        selected: list[SearchItem] = []
        spent = 0
        covered: dict[str, list[tuple[int, int]]] = {}
        for entry, _score in ranked:
            if len(selected) >= max_records:
                break
            start = int(entry.span["start_char"])
            end = int(entry.span["end_char"])
            source_id = str(entry.span["source_id"])
            if any(start >= left and end <= right for left, right in covered.get(source_id, [])):
                continue
            cost = reader_token_count(entry.content) + _RENDER_TOKEN_OVERHEAD
            if spent + cost > memory_budget:
                continue
            selected.append(SearchItem(id=entry.view_id, content=entry.content))
            spent += cost
            covered.setdefault(source_id, []).append((start, end))
        return SearchResponse(data=selected)

    def _rerank_query(self, query: str, options: Sequence[str]) -> str:
        question = query_options_text(query, list(options) if options else None)
        return self._pipeline.reranker.instruction + "\n\nQuestion and options:\n" + question

    def _result_cache_key(
        self,
        scope_key: str,
        epoch: int,
        request: SearchRequest,
        options: Sequence[str],
    ) -> str:
        digest = hashlib.sha256(
            canonical_json(
                [self._pipeline.pipeline_id, epoch, request.query, list(options), request.top_k]
            )
        ).hexdigest()
        return f"{scope_key}:{digest}"
