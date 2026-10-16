"""Bounded hybrid retrieval, evidence-only reranking, and output budgeting."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import time
from collections import defaultdict
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
from memorysum.core.evidence import purify
from memorysum.core.identity import canonical_json, scope_id
from memorysum.core.metadata import (
    attach_anchor,
    cluster_session,
    has_visible_date,
    time_anchor_line,
)
from memorysum.core.query_modes import is_context_query, is_temporal_query, is_wide_query
from memorysum.core.ranking import FusedCandidate, reciprocal_rank_fusion
from memorysum.core.rendering import (
    _RESPONSE_SPAN_SEPARATOR,
    FULL_BUNDLE_SEPARATOR,
    full_bundle_token_cost,
    render_full_bundle,
    render_message,
    render_response_window,
    render_window,
)
from memorysum.core.text import (
    chinese_terms,
    clip_middle,
    collapse_whitespace,
    english_terms,
    query_options_text,
    reader_token_count,
    safe_fts_expression,
    utc_timestamp,
)
from memorysum.core.vector import normalize_batch, vector_to_blob
from memorysum.providers.protocols import ModelProvider
from memorysum.settings import PipelineConfig, RetrievalModeConfig
from memorysum.storage.database import DatabaseManager, ScopeRef, transaction

logger = logging.getLogger(__name__)
_RENDER_TOKEN_OVERHEAD = 8
_OPTION_LABEL_RE = re.compile(r"^\s*(?:[A-Za-z][.)]|[A-Za-z][:：]|\d+[.)])\s*")


def _resolved_created_at_ms(source_timestamp_ms: object, received_ms: object) -> int:
    timestamp_ms = source_timestamp_ms if source_timestamp_ms is not None else received_ms
    if timestamp_ms is None:
        raise StorageError()
    return int(timestamp_ms)


@dataclass(frozen=True)
class Snapshot:
    epoch: int
    dimension: int | None
    active_message_count: int
    full_render_tokens_est: int


@dataclass(frozen=True)
class FullBundle:
    messages: tuple[dict[str, object], ...]
    token_cost: int


@dataclass(frozen=True)
class RerankEntry:
    candidate: FusedCandidate
    view_id: str
    rerank_content: str
    response_content: str
    spans: list[dict[str, object]]
    created_at_ms: int
    pair_proxy_tokens: int
    #: Harness ids of the originating messages (order preserved, de-duplicated).
    source_ids: tuple[str, ...] = ()
    #: Time anchor to re-attach when the rendered slice does not contain it.
    anchor: str | None = None
    #: Session label from the platform's bracket header (`[..][session_1][D1:3][..]`),
    #: used when synthesising a visible time anchor for temporal questions.
    session: str | None = None


_SPEAKER_RE = re.compile(r"(?m)^([A-Z][a-z]{2,15}):\s")
_WORD_RE = re.compile(r"[a-z]{4,}")
_STOP = frozenset(
    """about after again against also because been before being below between both
could doing down during each few from further have having here into itself just like more most
only other over same should some such than that their them then there these they this those
through under until very were what when where which while will with would your yours""".split()
)
_WORD_CACHE: dict[str, re.Pattern[str]] = {}


def _word_re(name: str) -> re.Pattern[str]:
    cached = _WORD_CACHE.get(name)
    if cached is None:
        cached = re.compile(rf"\b{re.escape(name)}\b")
        _WORD_CACHE[name] = cached
    return cached


def _primary_message_id(entry: RerankEntry) -> str:
    """Source message of an entry's first span, used for chunk-level consensus.

    `entry.spans` carries the renderer's span identities (source_id / start / end),
    so the message key here is `source_id` - the same key `_select_output` uses to
    de-duplicate overlapping passages.
    """
    spans = entry.spans
    if not spans:
        return ""
    return str(spans[0].get("source_id", ""))


def entry_provenance(
    spans: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, ...], str | None]:
    """Collect the harness ids and time anchor carried by a window's source spans.

    Ids are the union across spans (a context window can cover several messages),
    order-preserving and de-duplicated. The anchor comes from the first span that
    has one; it is re-attached at render time when the slice lost it.
    """
    ids: list[str] = []
    seen: set[str] = set()
    anchor: str | None = None
    for span in spans:
        raw = span.get("metadata_ids")
        if raw:
            try:
                values = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                values = ()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    text = str(value)
                    if text and text not in seen:
                        seen.add(text)
                        ids.append(text)
        if anchor is None and span.get("metadata_anchor"):
            anchor = str(span["metadata_anchor"])
    return tuple(ids), anchor


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
        self._full_bundles: TTLCache[FullBundle] = TTLCache(64, 300)

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
        # Document-grounded questions arrive as a whole document glued to the question
        # (observed 13k-186k chars in one round). They must still be served, and each
        # stage gets the largest slice it can actually accept:
        #   * the retrieval channels are clipped to the embedding input budget -- only a
        #     truncated *copy* is embedded / matched, the query itself is never edited;
        #   * the reranker is handed a clip sized so that even the largest candidate
        #     group still fits a single pair, which keeps every candidate rerankable
        #     instead of dropping the stage (and with it the ordering signal) outright.
        # Only when even that clip cannot fit a pair does the pipeline fall back to the
        # fusion-only order -- the same path any other over-budget candidate takes.
        retrieval_query = request.query
        retrieval_query_cost = reader_token_count(request.query)
        if retrieval_query_cost > self._pipeline.embedding.input_proxy_tokens:
            retrieval_query = clip_middle(
                request.query, self._pipeline.embedding.input_proxy_tokens
            )
            diagnostics["retrieval_query_clipped"] = True
            diagnostics["retrieval_query_chars"] = len(retrieval_query)
            logger.warning(
                "Query over the embedding input budget (%d > %d proxy tokens); "
                "retrieval runs on a head+tail clip of it",
                retrieval_query_cost,
                self._pipeline.embedding.input_proxy_tokens,
            )
        rerank_question = question_text
        if rerank_query_cost > self._pipeline.reranker.max_query_proxy_tokens:
            rerank_question = clip_middle(question_text, self._rerank_clip_budget())
            diagnostics["rerank_query_clipped"] = True
            diagnostics["rerank_query_chars"] = len(rerank_question)
            logger.warning(
                "Query over the reranker input budget (%d > %d proxy tokens); "
                "reranking against a head+tail clip of it",
                rerank_query_cost,
                self._pipeline.reranker.max_query_proxy_tokens,
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
            mode = self._pipeline.retrieval.wide if wide_mode else self._pipeline.retrieval.ordinary
            entries = self._prepare_full_bundle_entries(
                full_bundle.messages,
                request,
                options,
                wide_mode,
                rerank_question=rerank_question,
            )
            if entries is not None:
                diagnostics["strategy"] = "full_bundle"
                diagnostics["full_bundle_tokens_est"] = full_bundle.token_cost
                diagnostics["reranked"] = self._pipeline.reranker.enabled
                diagnostics["rerank_groups"] = len(entries)
                diagnostics["rerank_pair_tokens_est"] = sum(
                    entry.pair_proxy_tokens for entry in entries
                )
                ranked = self._rank_entries(
                    entries, request.query, options, deadline, rerank_question=rerank_question
                )
                response = self._select_output(
                    ranked, request.top_k, mode, question_cost, request.query
                )
                diagnostics["returned_records"] = len(response.data)
                diagnostics["returned_chars"] = sum(len(item.content) for item in response.data)
                self._database.assert_scope_active(request.user_id)
                self._results.set(result_cache_key, response)
                return response

        if snapshot.dimension is None:
            raise StorageError()
        query_vector = self._query_vector(reference, request.user_id, retrieval_query)
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
            retrieval_query,
            options,
            wide_mode,
        )
        diagnostics["strategy"] = "hybrid_retrieval"
        diagnostics["channel_hits"] = {name: len(ids) for name, ids, _ in channels}
        mode = self._pipeline.retrieval.wide if wide_mode else self._pipeline.retrieval.ordinary
        rrf_k = self._pipeline.retrieval.rrf_k
        if getattr(self._pipeline.retrieval, "rrf_k_adaptive", False):
            # 按去重前的候选总量缩放 k：RRF 原论文的 k=60 是在 TREC 量级语料上调的，
            # 而这里的候选列表只有几十到几百条 —— 见 docs/round8-prereg.md 的扫描。
            unique = len(
                {
                    int(entry) if not isinstance(entry, (tuple, list)) else int(entry[0])
                    for _, entries, _ in channels
                    for entry in entries
                }
            )
            rrf_k = max(10, min(30, unique // 3))
            diagnostics["rrf_k"] = rrf_k
        fused = reciprocal_rank_fusion(
            channels,
            k=rrf_k,
            limit=mode.candidate_cap,
            similarity_weight=float(
                getattr(self._pipeline.retrieval, "similarity_weight", 0.0) or 0.0
            ),
            channel_consensus_bonus=float(
                getattr(self._pipeline.retrieval, "channel_consensus_bonus", 0.0) or 0.0
            ),
        )
        diagnostics["candidate_windows"] = len(fused)
        # The pair rules decide which candidates can be scored; a query over the reranker
        # budget arrives here already clipped to a size that keeps groups rerankable, so
        # an empty scored list now means "no pair fits at all".
        entries, leftover = self._prepare_rerank_entries(
            fused, rows, request, options, wide_mode, rerank_question=rerank_question
        )
        diagnostics["reranked"] = self._pipeline.reranker.enabled
        diagnostics["rerank_groups"] = len(entries)
        diagnostics["rerank_pair_tokens_est"] = sum(entry.pair_proxy_tokens for entry in entries)
        if leftover:
            diagnostics["unreranked_candidates"] = len(leftover)
        if not entries:
            if not fused:
                diagnostics["returned_records"] = 0
                response = SearchResponse(data=[])
                self._database.assert_scope_active(request.user_id)
                self._results.set(result_cache_key, response)
                return response
            # Nothing can be scored: every candidate pair is over the rerank budget.
            # Returning `data=[]` would tell the caller "no relevant memory" when the
            # evidence is right there, so fall back to the fusion order (the same ranking
            # the pipeline uses with `reranker.enabled=false`). The strategy string and
            # the warning keep the skipped rerank visible in the audit log.
            logger.warning(
                "No rerankable pair in scope %s (query %d proxy tokens > budget %d)",
                reference.scope_key[:12],
                rerank_query_cost,
                self._pipeline.reranker.single_pair_proxy_tokens,
            )
            diagnostics["rerank_skipped"] = "no_pair_within_budget"
            diagnostics["strategy"] = "hybrid_retrieval_unreranked"
            diagnostics["reranked"] = False
            ranked = [(entry, entry.candidate.score) for entry in leftover]
            response = self._select_output(
                ranked, request.top_k, mode, question_cost, request.query
            )
            diagnostics["returned_records"] = len(response.data)
            diagnostics["returned_chars"] = sum(len(item.content) for item in response.data)
            self._database.assert_scope_active(request.user_id)
            self._results.set(result_cache_key, response)
            return response

        ranked = self._rank_entries(
            entries,
            request.query,
            options,
            deadline,
            rerank_question=rerank_question,
            leftover=leftover,
        )
        response = self._select_output(ranked, request.top_k, mode, question_cost, request.query)
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
    ) -> FullBundle | None:
        if (
            self._pipeline.coverage.conditional_full_requires_wide_or_context_mode
            and not wide_or_context
        ):
            return None

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
            cached = FullBundle(
                messages=tuple(dict(message) for message in messages),
                token_cost=full_bundle_token_cost(messages),
            )
            self._full_bundles.set(cache_key, cached)
        full_cost = cached.token_cost

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
        return cached

    def _prepare_full_bundle_entries(
        self,
        messages: Sequence[Mapping[str, object]],
        request: SearchRequest,
        options: Sequence[str],
        wide_mode: bool,
        *,
        rerank_question: str | None = None,
    ) -> list[RerankEntry] | None:
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
        if not messages or len(messages) > max_groups:
            return None

        if rerank_question is None:
            rerank_question = query_options_text(request.query, list(options) if options else None)
        rendered_messages = render_full_bundle(messages)
        entries: list[RerankEntry] = []
        spent = 0
        for index, ((view_id, response_content), message) in enumerate(
            zip(rendered_messages, messages, strict=True)
        ):
            public_source_id = str(message["public_source_id"])
            source_timestamp_ms = message["source_timestamp_ms"]
            created_at_ms = _resolved_created_at_ms(
                source_timestamp_ms,
                message.get("request_received_ms"),
            )
            rerank_content = render_message(
                "msg_" + public_source_id[:16],
                str(message["role"]),
                source_timestamp_ms,  # type: ignore[arg-type]
                response_content,
            )
            pair_text = (
                f"<Instruct>: {self._pipeline.reranker.instruction}\n"
                f"<Query>: {rerank_question}\n<Document>: {rerank_content}"
            )
            pair_cost = reader_token_count(pair_text)
            if (
                pair_cost > self._pipeline.reranker.single_pair_proxy_tokens
                or spent + pair_cost > pair_budget
            ):
                return None
            span = {
                "source_id": public_source_id,
                "start_char": 0,
                "end_char": len(response_content),
                "metadata_ids": message["metadata_ids"],
                "metadata_anchor": message["metadata_anchor"],
            }
            source_ids, anchor = entry_provenance([span])
            entries.append(
                RerankEntry(
                    candidate=FusedCandidate(
                        window_id=index,
                        best_rank=index + 1,
                        channels={"full_bundle"},
                    ),
                    view_id=view_id,
                    rerank_content=rerank_content,
                    response_content=response_content,
                    spans=[span],
                    created_at_ms=created_at_ms,
                    pair_proxy_tokens=pair_cost,
                    source_ids=source_ids,
                    anchor=anchor,
                    session=cluster_session(str(message["raw_content"])),
                )
            )
            spent += pair_cost
        return entries

    def _load_messages(self, reference: ScopeRef, user_id: str, epoch: int) -> list[sqlite3.Row]:
        connection = self._database.connect(reference, user_id)
        try:
            with transaction(connection, "DEFERRED"):
                rows = connection.execute(
                    """SELECT m.message_id, m.public_source_id, m.request_id, m.ordinal,
                              m.role, m.source_timestamp_ms, m.raw_content, m.reader_token_est,
                              m.metadata_ids, m.metadata_anchor,
                              r.received_ms AS request_received_ms
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
                weights = self._pipeline.retrieval.channel_weights
                fallback = self._pipeline.retrieval.base_channel_weight

                def weight(name: str) -> float:
                    return weights.get(name, fallback)

                channels: list[tuple[str, list[object], float]] = [
                    ("leaf_dense", list(leaf), weight("leaf_dense")),
                    ("context_dense", list(context), weight("context_dense")),
                    ("english_fts", en, weight("english_fts")),
                    ("chinese_fts", zh, weight("chinese_fts")),
                    ("options_lexical", option_ids, weight("options_lexical")),
                ]
                # Dense channels yield (window_id, similarity); lexical ones yield
                # bare ids. Normalise here so both drive the same row fetch.
                all_ids = sorted(
                    {
                        int(entry[0]) if isinstance(entry, (tuple, list)) else int(entry)
                        for _, ids, _ in channels
                        for entry in ids
                    }
                )
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
    ) -> list[tuple[int, float]]:
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
        # Negated distance: closer vectors must score higher once the fusion step
        # min-max normalises the channel. Previously the distances were discarded
        # and only ranks survived, which made a 0.9-similarity top hit weigh the
        # same as a 0.3 one.
        return [(int(row["rowid"]), -float(row["distance"])) for row in rows]

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
    ) -> dict[int, dict[str, object]]:
        if not window_ids:
            return {}
        placeholders = ",".join("?" for _ in window_ids)
        ids = (*window_ids, epoch)
        windows = connection.execute(
            f"""SELECT w.window_id, w.scale, w.input_id, w.published_epoch,
                       w.forced_split
                FROM windows AS w
                JOIN active_windows AS a ON a.window_id=w.window_id
                WHERE w.window_id IN ({placeholders}) AND a.published_epoch<=?
                ORDER BY w.window_id""",
            ids,
        ).fetchall()
        span_rows = connection.execute(
            f"""SELECT s.window_id, s.ordinal AS span_ordinal, s.message_id,
                       s.start_char, s.end_char, m.public_source_id, m.request_id,
                       m.ordinal AS message_ordinal, m.role, m.source_timestamp_ms,
                       m.raw_content, r.received_ms AS request_received_ms,
                       m.metadata_ids, m.metadata_anchor
                FROM window_spans AS s
                JOIN messages AS m ON m.message_id=s.message_id
                JOIN requests AS r ON r.request_id=m.request_id
                WHERE s.window_id IN ({placeholders})
                ORDER BY s.window_id, s.ordinal""",
            window_ids,
        ).fetchall()
        spans_by_window: dict[int, list[dict[str, object]]] = {}
        span_by_message: dict[tuple[int, int], dict[str, object]] = {}
        for source in span_rows:
            window_id = int(source["window_id"])
            span = {
                "span_ordinal": int(source["span_ordinal"]),
                "message_id": int(source["message_id"]),
                "start_char": int(source["start_char"]),
                "end_char": int(source["end_char"]),
                "public_source_id": str(source["public_source_id"]),
                "request_id": str(source["request_id"]),
                "message_ordinal": int(source["message_ordinal"]),
                "role": str(source["role"]),
                "source_timestamp_ms": source["source_timestamp_ms"],
                "request_received_ms": int(source["request_received_ms"]),
                "raw_content": str(source["raw_content"]),
                "metadata_ids": source["metadata_ids"],
                "metadata_anchor": source["metadata_anchor"],
                "atoms": [],
            }
            spans_by_window.setdefault(window_id, []).append(span)
            span_by_message[(window_id, int(source["message_id"]))] = span

        atom_rows = connection.execute(
            f"""SELECT wa.window_id, a.message_id, a.start_char, a.end_char,
                       a.kind, a.in_code, a.in_quote
                FROM window_atoms AS wa
                JOIN atoms AS a ON a.atom_id=wa.atom_id
                WHERE wa.window_id IN ({placeholders})
                ORDER BY wa.window_id, a.message_id, a.start_char""",
            window_ids,
        ).fetchall()
        for atom in atom_rows:
            span = span_by_message.get((int(atom["window_id"]), int(atom["message_id"])))
            if span is not None:
                span["atoms"].append(
                    {
                        "start_char": int(atom["start_char"]),
                        "end_char": int(atom["end_char"]),
                        "kind": str(atom["kind"]),
                        "in_code": bool(atom["in_code"]),
                        "in_quote": bool(atom["in_quote"]),
                    }
                )

        result: dict[int, dict[str, object]] = {}
        for window in windows:
            window_id = int(window["window_id"])
            spans = spans_by_window.get(window_id, [])
            if not spans:
                continue
            primary = spans[0]
            result[window_id] = {
                "window_id": window_id,
                "scale": str(window["scale"]),
                "input_id": int(window["input_id"]),
                "published_epoch": int(window["published_epoch"]),
                "forced_split": bool(window["forced_split"]),
                "message_id": int(primary["message_id"]),
                "start_char": int(primary["start_char"]),
                "end_char": int(primary["end_char"]),
                "public_source_id": str(primary["public_source_id"]),
                "request_id": str(primary["request_id"]),
                "message_ordinal": int(primary["message_ordinal"]),
                "role": str(primary["role"]),
                "source_timestamp_ms": primary["source_timestamp_ms"],
                "raw_content": str(primary["raw_content"]),
                "source_spans": spans,
            }
        return result

    def _group_overlapping_candidates(
        self,
        fused: Sequence[FusedCandidate],
        rows: Mapping[int, dict[str, object]],
    ) -> list[CandidateGroup]:
        groups: list[CandidateGroup] = []
        for candidate in fused:
            source = rows.get(candidate.window_id)
            if source is None:
                continue
            row = dict(source)
            is_context = row["scale"] == "context"
            token_limit = (
                self._pipeline.groups.context_max_tokens
                if is_context
                else self._pipeline.groups.direct_max_tokens
            )
            source_spans = self._expand_spans_to_atoms(row["source_spans"], token_limit)
            row["source_spans"] = source_spans
            if not source_spans:
                continue
            primary = source_spans[0]
            row.update(
                message_id=int(primary["message_id"]),
                start_char=int(primary["start_char"]),
                end_char=int(primary["end_char"]),
                public_source_id=str(primary["public_source_id"]),
                role=str(primary["role"]),
                source_timestamp_ms=primary["source_timestamp_ms"],
                raw_content=str(primary["raw_content"]),
            )

            # Cross-message context windows are already bounded, ordered source
            # groups. Keep them intact rather than merging unrelated windows.
            if len(source_spans) > 1:
                groups.append(CandidateGroup(candidate, row, True))
                continue

            start = int(row["start_char"])
            end = int(row["end_char"])
            overlapping = [
                group
                for group in groups
                if group.contains_context_window is is_context
                and len(group.row["source_spans"]) == 1
                and int(group.row["message_id"]) == int(row["message_id"])
                and start <= int(group.row["end_char"]) + 1
                and end + 1 >= int(group.row["start_char"])
            ]
            if not overlapping:
                groups.append(CandidateGroup(candidate, row, is_context))
                continue

            all_start = min(start, *(int(group.row["start_char"]) for group in overlapping))
            all_end = max(end, *(int(group.row["end_char"]) for group in overlapping))
            merged_span = {
                **primary,
                "start_char": all_start,
                "end_char": all_end,
            }
            merged_row = self._row_with_spans(row, [merged_span])
            merged_content, _, _ = render_window(merged_row)
            if reader_token_count(merged_content) <= token_limit:
                merged_candidate = self._merge_cpu_candidates(
                    [candidate, *(group.candidate for group in overlapping)]
                )
                groups = [group for group in groups if group not in overlapping]
                groups.append(CandidateGroup(merged_candidate, merged_row, is_context))
                continue

            attached = False
            for group in overlapping:
                pair_start = min(start, int(group.row["start_char"]))
                pair_end = max(end, int(group.row["end_char"]))
                group_span = group.row["source_spans"][0]
                pair_span = {
                    **group_span,
                    "start_char": pair_start,
                    "end_char": pair_end,
                }
                pair_row = self._row_with_spans(group.row, [pair_span])
                pair_content, _, _ = render_window(pair_row)
                if reader_token_count(pair_content) > token_limit:
                    continue
                group.row = pair_row
                group.candidate = self._merge_cpu_candidates([group.candidate, candidate])
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

    def _expand_spans_to_atoms(
        self, source_spans: Sequence[object], token_limit: int
    ) -> list[dict[str, object]]:
        original = [dict(span) for span in source_spans]
        expanded: list[dict[str, object]] = []
        for span in original:
            atoms = span.get("atoms", [])
            start = int(span["start_char"])
            end = int(span["end_char"])
            if atoms:
                start = min(start, *(int(atom["start_char"]) for atom in atoms))
                end = max(end, *(int(atom["end_char"]) for atom in atoms))
            expanded.append({**span, "start_char": start, "end_char": end})

        if reader_token_count(render_window({"source_spans": expanded})[0]) <= token_limit:
            return expanded
        return original

    @staticmethod
    def _row_with_spans(
        row: Mapping[str, object], spans: list[dict[str, object]]
    ) -> dict[str, object]:
        primary = spans[0]
        return {
            **row,
            "source_spans": spans,
            "message_id": int(primary["message_id"]),
            "start_char": int(primary["start_char"]),
            "end_char": int(primary["end_char"]),
            "public_source_id": str(primary["public_source_id"]),
            "role": str(primary["role"]),
            "source_timestamp_ms": primary["source_timestamp_ms"],
            "raw_content": str(primary["raw_content"]),
        }

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
        rows: Mapping[int, dict[str, object]],
        request: SearchRequest,
        options: Sequence[str],
        wide_mode: bool,
        *,
        rerank_question: str | None = None,
    ) -> tuple[list[RerankEntry], list[RerankEntry]]:
        """Render candidates into (rerankable, over-budget) entries.

        The second list holds the candidates the pair budgets exclude: the reranker
        cannot score them, but they are still evidence, so the caller keeps them in
        fusion order behind the scored records instead of dropping them silently.
        """
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
        if rerank_question is None:
            rerank_question = query_options_text(request.query, list(options) if options else None)
        candidates_by_view: dict[str, RerankEntry] = {}
        groups = self._group_overlapping_candidates(fused, rows)
        for group in groups:
            candidate = group.candidate
            rerank_content, view_id, spans = render_window(group.row)
            response_content, _, _ = render_response_window(group.row)
            source_spans = group.row["source_spans"]
            primary_span = source_spans[0]  # type: ignore[index]
            created_at_ms = _resolved_created_at_ms(
                primary_span.get("source_timestamp_ms"),
                primary_span.get("request_received_ms"),
            )
            # The per-pair token accounting exists to enforce the rerank budgets. Even the
            # over-budget (leftover) entries need it -- that is the number that decides
            # they are over budget -- so it is always computed.
            pair_text = (
                f"<Instruct>: {self._pipeline.reranker.instruction}\n"
                f"<Query>: {rerank_question}\n<Document>: {rerank_content}"
            )
            pair_cost = reader_token_count(pair_text)
            existing = candidates_by_view.get(view_id)
            if existing is None or candidate.score > existing.candidate.score:
                candidates_by_view[view_id] = RerankEntry(
                    candidate=candidate,
                    view_id=view_id,
                    rerank_content=rerank_content,
                    response_content=response_content,
                    spans=spans,
                    created_at_ms=created_at_ms,
                    pair_proxy_tokens=pair_cost,
                    source_ids=entry_provenance(source_spans)[0],
                    anchor=entry_provenance(source_spans)[1],
                    session=cluster_session(str(group.row.get("raw_content") or "")),
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
        leftover: list[RerankEntry] = []
        spent = 0
        for entry in ordered:
            cost = entry.pair_proxy_tokens
            if (
                cost > self._pipeline.reranker.single_pair_proxy_tokens
                or spent + cost > pair_budget
            ):
                # Over the pair budget: unscored, but still delivered (see docstring).
                leftover.append(entry)
                continue
            selected.append(entry)
            spent += cost
            if len(selected) >= max_groups:
                break
        return selected, leftover

    def _rerank_clip_budget(self) -> int:
        """Largest query clip that still leaves room for the biggest candidate group.

        `reranker.single_pair_proxy_tokens` bounds a whole (query, document) pair; the
        document side is bounded by the group budgets (`groups.direct_max_tokens` /
        `groups.context_max_tokens`) plus the rendering prefix. Clipping the query to
        what is left keeps every candidate inside the pair budget, so an over-long
        query loses rerank *resolution* only -- never candidates.
        """
        render_allowance = 128
        return max(
            512,
            self._pipeline.reranker.single_pair_proxy_tokens
            - self._pipeline.groups.context_max_tokens
            - render_allowance,
        )

    def _rank_entries(
        self,
        entries: Sequence[RerankEntry],
        query: str,
        options: Sequence[str],
        deadline: float,
        *,
        rerank_question: str | None = None,
        leftover: Sequence[RerankEntry] = (),
    ) -> list[tuple[RerankEntry, float]]:
        """Order entries: ranked by the reranker when enabled, else by fusion.

        With `reranker.enabled = false` the pipeline stops after reciprocal-rank
        fusion and the returned score IS the fusion score. When enabled, the
        reranker's ranking is fused back with the fusion ranking (see
        `_fuse_rankings`) instead of replacing it outright, because measuring
        showed pure rerank ordering is unstable across query halves.
        """
        if not self._pipeline.reranker.enabled:
            return [(entry, entry.candidate.score) for entry in entries] + [
                (entry, entry.candidate.score) for entry in leftover
            ]
        question = (
            rerank_question
            if rerank_question is not None
            else query_options_text(query, list(options) if options else None)
        )
        scores = self._rerank_entries(self._rerank_query_text(question), entries, deadline)
        # Candidates the pair budgets excluded keep their fusion order behind the scored
        # ones: a query too large to rerank everything still delivers the full evidence
        # set (the response budget decides how far down the tail it actually reaches).
        return self._fuse_rankings(entries, scores, query) + [
            (entry, entry.candidate.score) for entry in leftover
        ]

    def _fuse_rankings(
        self,
        entries: Sequence[RerankEntry],
        scores: Sequence[float],
        query: str = "",
    ) -> list[tuple[RerankEntry, float]]:
        """Fuse rerank ranks with retrieval ranks; score = rerank + w * retrieval.

        Reciprocal-rank fusion keeps the multi-channel agreement signal that the
        cross-encoder alone discards, so the reranker reorders near-ties instead
        of being trusted absolutely. `reranker.fusion_weight` sets w.
        """
        k = self._pipeline.retrieval.rrf_k
        weight = self._pipeline.reranker.fusion_weight
        by_retrieval = sorted(
            range(len(entries)),
            key=lambda index: (
                -round(entries[index].candidate.score, 12),
                entries[index].candidate.best_rank,
                entries[index].candidate.window_id,
            ),
        )
        retrieval_rank = {index: rank + 1 for rank, index in enumerate(by_retrieval)}
        by_rerank = sorted(range(len(entries)), key=lambda index: (-scores[index], index))
        rerank_rank = {index: rank + 1 for rank, index in enumerate(by_rerank)}

        fused = [
            (
                entries[index],
                1.0 / (k + rerank_rank[index]) + weight / (k + retrieval_rank[index]),
            )
            for index in range(len(entries))
        ]
        coverage_alpha = float(
            getattr(self._pipeline.retrieval, "term_coverage_weight", 0.0) or 0.0
        )
        alpha = float(getattr(self._pipeline.retrieval, "consensus_weight", 0.0) or 0.0)
        speaker_alpha = float(getattr(self._pipeline.retrieval, "speaker_weight", 0.0) or 0.0)
        if (alpha > 0 or speaker_alpha > 0 or coverage_alpha > 0) and len(fused) > 1:
            high = max(score for _, score in fused) or 1.0
            low = min(score for _, score in fused)
            span = (high - low) or 1.0
            normalised = [(entry, (score - low) / span) for entry, score in fused]

            # Context-relationship vote: windows belonging to the same chunk as other
            # strong candidates are likelier to hold the same answer, which is what
            # multi-hop questions need (their two evidence turns usually sit in
            # different windows of one chunk).
            totals: dict[str, float] = defaultdict(float)
            counts: dict[str, int] = defaultdict(int)
            for entry, norm in normalised:
                key = _primary_message_id(entry)
                totals[key] += norm
                counts[key] += 1

            # Speakers named by the query: conversational questions usually target one
            # participant ("What did Melanie paint?"), so windows dominated by that
            # person's turns are better answers. Names are read off the candidate text
            # itself, so this needs no extra indexing or per-dataset configuration.
            targets: set[str] = set()
            if speaker_alpha > 0:
                names: set[str] = set()
                for entry, _ in normalised:
                    names.update(_SPEAKER_RE.findall(entry.response_content))
                targets = {name for name in names if _word_re(name).search(query)}

            # Content-word coverage of the question (soft lexical agreement).
            q_terms = set(_WORD_RE.findall(query.lower())) - _STOP
            boosted = []
            for entry, norm in normalised:
                score = norm
                if coverage_alpha and q_terms:
                    text = entry.response_content.lower()
                    hit = sum(1 for term in q_terms if term in text)
                    score += coverage_alpha * (hit / len(q_terms))
                if alpha > 0:
                    key = _primary_message_id(entry)
                    others = counts[key] - 1
                    if others > 0:
                        score += alpha * ((totals[key] - norm) / others)
                if targets:
                    hits = sum(
                        len(_word_re(name).findall(entry.response_content)) for name in targets
                    )
                    score += speaker_alpha * min(1.0, hits / 4.0)
                boosted.append((entry, score))
            fused = boosted
        return sorted(
            fused,
            key=lambda pair: (
                -round(pair[1], 12),
                pair[0].candidate.best_rank,
                pair[0].view_id,
            ),
        )

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
                [entries[index].rerank_content for index in batch],
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
        query: str = "",
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
        expand = max(0, self._pipeline.retrieval.neighbor_expand_chars)
        per_source_cap = int(getattr(self._pipeline.retrieval, "per_source_cap", 0) or 0)
        selected: list[SearchItem] = []
        spent = 0
        covered: dict[str, list[tuple[int, int]]] = {}
        seen_content: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)

        def bounds(span: Mapping[str, object]) -> tuple[int, int]:
            """Span interval widened by neighbor_expand_chars, clamped to the source text."""
            left = max(0, int(span["start_char"]) - expand)
            right = int(span["end_char"]) + expand
            raw = span.get("raw_content")
            if isinstance(raw, str) and raw:
                right = min(right, len(raw))
            return left, right

        for entry, score in ranked:
            if len(selected) >= max_records:
                break
            source_key = _primary_message_id(entry)
            if per_source_cap and source_counts[source_key] >= per_source_cap:
                # Budget spent on a chunk already represented; skip so evidence from
                # other sessions can still make it into the response.
                continue
            # Re-attach the chunk's time anchor when the rendered slice lost it
            # (only the first window of a chunk carries it). The anchor is plain
            # text such as "Session 1 (1:56 pm on 8 May, 2023)" — never a
            # "Time: " prefix, which the platform strips from the response text.
            content = attach_anchor(entry.response_content, entry.anchor)
            anchor_scope = self._pipeline.groups.time_anchor_scope
            if self._pipeline.groups.time_anchor_in_text and (
                anchor_scope == "all" or is_temporal_query(query)
            ):
                anchor_line = time_anchor_line(
                    timestamp_ms=entry.created_at_ms,
                    session=entry.session,
                    template=self._pipeline.groups.time_anchor_template,
                )
                if anchor_line and anchor_line not in content and not has_visible_date(content):
                    content = f"{anchor_line}\n{content}"
            if self._pipeline.groups.normalize_whitespace:
                # 默认关闭：原文保真要求正文是原文切片。实测只省 0~1.4% token
                # （XLM-R 口径 0%），仅用于平台侧 A/B。见 core/text.collapse_whitespace。
                content = collapse_whitespace(content)
            spans = entry.spans
            if expand:
                # Widen each passage to its neighbouring text so a hit carries the
                # context its own window cut off. Only the returned text changes;
                # ranking still uses the indexed window.
                widened = []
                for span in spans:
                    raw = span.get("raw_content")
                    if isinstance(raw, str) and raw:
                        left, right = bounds(span)
                        widened.append(raw[left:right])
                if widened:
                    content = attach_anchor(_RESPONSE_SPAN_SEPARATOR.join(widened), entry.anchor)
            head_records = int(getattr(self._pipeline.groups, "head_records", 0) or 0)
            tail_cut = int(getattr(self._pipeline.groups, "tail_cut_chars", 0) or 0)
            if (
                tail_cut
                and head_records
                and len(selected) >= head_records
                and len(content) > tail_cut
            ):
                # Planned shape: early records stay complete, later ones are trimmed.
                # Default off — the harness publishes no response-length policy, so
                # trimming costs content without a measured benefit. Revisit only if
                # the platform states a cap.
                content = content[:tail_cut].rstrip()
            content_key = " ".join(content.split()).casefold()
            if content_key in seen_content:
                continue
            already_covered = all(
                any(
                    bounds(span)[0] >= left and bounds(span)[1] <= right
                    for left, right in covered.get(str(span["source_id"]), [])
                )
                for span in spans
            )
            if already_covered:
                continue
            purify_config = self._pipeline.evidence_purify
            if purify_config.enabled and len(selected) < int(purify_config.max_records):
                # Verified evidence prefix; core/evidence.py fails open on any error so
                # the record is returned unchanged unless a high-confidence, verbatim
                # citation was found. Runs after de-dup and coverage so ordering and
                # budget accounting stay untouched.
                content = purify(content, query, purify_config)
            cost = reader_token_count(content) + _RENDER_TOKEN_OVERHEAD
            if spent + cost > memory_budget:
                continue
            selected.append(
                SearchItem(
                    id=entry.view_id,
                    content=content,
                    score=score,
                    created_at=utc_timestamp(entry.created_at_ms),
                    source_ids=list(entry.source_ids),
                )
            )
            spent += cost
            seen_content.add(content_key)
            source_counts[source_key] += 1
            for span in spans:
                covered.setdefault(str(span["source_id"]), []).append(bounds(span))
        return SearchResponse(data=selected)

    def _rerank_query(self, query: str, options: Sequence[str]) -> str:
        question = query_options_text(query, list(options) if options else None)
        return self._rerank_query_text(question)

    def _rerank_query_text(self, question: str) -> str:
        instruction = self._pipeline.reranker.instruction.strip()
        if not instruction:
            # Cross-encoders such as bge-reranker take a bare query; only models
            # that expect an instruction template (e.g. Qwen3-Reranker) need one.
            return question
        return instruction + "\n\nQuestion and options:\n" + question

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
