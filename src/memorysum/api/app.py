"""HTTP boundary and lifecycle wiring for the Add/Search service."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from memorysum import __version__
from memorysum.api.limits import RequestGate
from memorysum.api.middleware import RequestLoggingMiddleware, RequestSizeLimitMiddleware
from memorysum.api.schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from memorysum.core.errors import ScopeRetired, ServiceError
from memorysum.core.identity import scope_id
from memorysum.core.text import chinese_terms, reader_token_count
from memorysum.providers.factory import build_provider, embedding_endpoint
from memorysum.providers.protocols import ModelProvider
from memorysum.services.add import AddService
from memorysum.services.search import SearchService
from memorysum.settings import PipelineConfig, RuntimeSettings, load_pipeline_config
from memorysum.storage.database import DatabaseManager

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("uvicorn.error")


class _Services:
    def __init__(
        self,
        runtime: RuntimeSettings,
        pipeline: PipelineConfig,
        provider: ModelProvider,
        database: DatabaseManager,
        *,
        owns_provider: bool,
        start_cleanup: bool,
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.provider = provider
        self.database = database
        self.add = AddService(database, provider, pipeline)
        self.search = SearchService(database, provider, pipeline)
        self.add_gate = RequestGate(
            pipeline.capacity.add_concurrency, pipeline.capacity.interface_queue
        )
        self.search_gate = RequestGate(
            pipeline.capacity.search_concurrency, pipeline.capacity.interface_queue
        )
        self.owns_provider = owns_provider
        self.start_cleanup = start_cleanup
        self.stop_event = threading.Event()
        self.cleanup_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.start_cleanup:
            return
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="memorysum-ttl-cleanup",
            daemon=True,
        )
        self.cleanup_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.cleanup_thread is not None:
            self.cleanup_thread.join(timeout=2)
        if self.owns_provider:
            self.provider.close()

    def _cleanup_loop(self) -> None:
        interval = self.pipeline.capacity.cleanup_interval_seconds
        while not self.stop_event.wait(interval):
            try:
                retired = self.database.retire_expired_scopes()
                for key in retired:
                    self.search.clear_scope(key)
                if retired:
                    logger.info("Retired %d expired memory scopes", len(retired))
            except ServiceError as exc:
                logger.warning("Memory-scope cleanup did not complete (%s)", exc.code)
            except Exception as exc:
                logger.error("Memory-scope cleanup failed (%s)", type(exc).__name__)


def create_app(
    runtime: RuntimeSettings | None = None,
    pipeline: PipelineConfig | None = None,
    provider: ModelProvider | None = None,
    *,
    start_cleanup: bool = True,
) -> FastAPI:
    runtime = runtime or RuntimeSettings.from_environment()
    pipeline = pipeline or load_pipeline_config()
    # Fail before serving requests if the pinned local reader tokenizer/Jieba index
    # cannot initialize. These probes use synthetic strings only.
    reader_token_count("memorysum tokenizer startup probe")
    chinese_terms("记忆检索")
    owns_provider = provider is None
    provider = provider or build_provider(runtime, pipeline)
    embedding_space_id = hashlib.sha256(
        (pipeline.embedding_space_id + "\n" + embedding_endpoint(runtime, pipeline)).encode("utf-8")
    ).hexdigest()
    database = DatabaseManager(runtime.data_root, pipeline, embedding_space_id)
    services = _Services(
        runtime,
        pipeline,
        provider,
        database,
        owns_provider=owns_provider,
        start_cleanup=start_cleanup,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        services.start()
        try:
            yield
        finally:
            services.stop()

    app = FastAPI(
        title="MemorySum",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.services = services
    app.state.pipeline = pipeline
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_bytes=pipeline.capacity.max_body_bytes,
    )
    # Wrap the size limiter so rejected and normal requests both get one access record.
    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
        headers: dict[str, str] = {}
        if exc.retry_after is not None:
            headers["Retry-After"] = str(exc.retry_after)
        if exc.status_code == 401:
            headers["WWW-Authenticate"] = _auth_challenge(runtime.api_auth_scheme)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": {"code": exc.code, "message": exc.message}},
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Never echo Pydantic's `input` field: request payloads are evaluation data.
        errors = [
            {
                "loc": [part for part in error.get("loc", ()) if isinstance(part, (str, int))],
                "type": str(error.get("type", "value_error")),
                "msg": str(error.get("msg", "Invalid value")),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "invalid_request", "errors": errors}},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled request failure (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": {"code": "internal_error", "message": "Request failed."}},
        )

    @app.get("/health", response_model=None)
    def health() -> dict[str, str] | JSONResponse:
        needs_openrouter = "openrouter" in {
            pipeline.embedding.provider,
            pipeline.reranker.provider,
        }
        if services.owns_provider and needs_openrouter and runtime.openrouter_api_key is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable"},
            )
        return {"status": "ok"}

    @app.post("/add", response_model=AddResponse)
    @app.post("/v1/memories/add", response_model=AddResponse)
    def add_memory(request: Request, payload: AddRequest) -> AddResponse:
        trace_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        message_count = len(payload.messages)
        user_messages = sum(message.role == "user" for message in payload.messages)
        assistant_messages = message_count - user_messages
        timestamped_messages = sum(
            "timestamp" in message.model_fields_set for message in payload.messages
        )
        message_chars = sum(len(message.content) for message in payload.messages)
        diagnostics: dict[str, object] = {}
        try:
            _authenticate(request, runtime)
            if runtime.debug:
                _audit_log(
                    runtime,
                    "event=api_add_debug_payload trace_id=%s payload_json=%s",
                    trace_id,
                    json.dumps(
                        payload.canonical_payload(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            with services.add_gate.slot():
                response = services.add.add(payload, diagnostics=diagnostics)
        except ServiceError as exc:
            _audit_log(
                runtime,
                "event=api_add trace_id=%s pipeline=%s status=%d error=%s "
                "messages=%d user_messages=%d assistant_messages=%d timestamped=%d "
                "message_chars=%d leaf_windows=%d context_windows=%d unique_inputs=%d "
                "embedded_inputs=%d embedding_batches=%d latency_ms=%d",
                trace_id,
                pipeline.pipeline_id,
                exc.status_code,
                exc.code,
                message_count,
                user_messages,
                assistant_messages,
                timestamped_messages,
                message_chars,
                int(diagnostics.get("leaf_windows", 0)),
                int(diagnostics.get("context_windows", 0)),
                int(diagnostics.get("unique_document_inputs", 0)),
                int(diagnostics.get("embedding_inputs_embedded", 0)),
                int(diagnostics.get("embedding_batches", 0)),
                int((time.monotonic() - started) * 1000),
            )
            if isinstance(exc, ScopeRetired):
                services.search.clear_scope(scope_id(payload.user_id))
            raise
        _audit_log(
            runtime,
            "event=api_add trace_id=%s pipeline=%s status=200 messages=%d user_messages=%d "
            "assistant_messages=%d timestamped=%d message_chars=%d leaf_windows=%d "
            "context_windows=%d unique_inputs=%d embedded_inputs=%d embedding_batches=%d "
            "latency_ms=%d",
            trace_id,
            pipeline.pipeline_id,
            message_count,
            user_messages,
            assistant_messages,
            timestamped_messages,
            message_chars,
            int(diagnostics.get("leaf_windows", 0)),
            int(diagnostics.get("context_windows", 0)),
            int(diagnostics.get("unique_document_inputs", 0)),
            int(diagnostics.get("embedding_inputs_embedded", 0)),
            int(diagnostics.get("embedding_batches", 0)),
            int((time.monotonic() - started) * 1000),
        )
        return response

    @app.post("/search", response_model=SearchResponse)
    @app.post("/v1/memories/search", response_model=SearchResponse)
    def search_memory(request: Request, payload: SearchRequest) -> SearchResponse:
        trace_id = uuid.uuid4().hex[:12]
        started = time.monotonic()
        options = payload.options or []
        options_chars = sum(len(option) for option in options)
        diagnostics: dict[str, object] = {}
        try:
            _authenticate(request, runtime)
            if runtime.debug:
                _audit_log(
                    runtime,
                    "event=api_search_debug_payload trace_id=%s payload_json=%s",
                    trace_id,
                    json.dumps(
                        payload.model_dump(mode="json", exclude_unset=True),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            with services.search_gate.slot():
                response = services.search.search(payload, diagnostics=diagnostics)
        except ServiceError as exc:
            _audit_log(
                runtime,
                "event=api_search trace_id=%s pipeline=%s status=%d error=%s "
                "query_chars=%d options=%d options_chars=%d top_k=%d strategy=%s "
                "candidate_windows=%d rerank_groups=%d latency_ms=%d",
                trace_id,
                pipeline.pipeline_id,
                exc.status_code,
                exc.code,
                len(payload.query),
                len(options),
                options_chars,
                payload.top_k,
                str(diagnostics.get("strategy", "failed")),
                int(diagnostics.get("candidate_windows", 0)),
                int(diagnostics.get("rerank_groups", 0)),
                int((time.monotonic() - started) * 1000),
            )
            raise
        returned_chars = sum(len(item.content) for item in response.data)
        _audit_log(
            runtime,
            "event=api_search trace_id=%s pipeline=%s status=200 query_chars=%d "
            "options=%d options_chars=%d top_k=%d strategy=%s cache_hit=%s wide=%s "
            "context=%s full_tokens_est=%d channel_hits=%s candidate_windows=%d "
            "reranked=%s rerank_groups=%d rerank_pair_tokens_est=%d returned=%d "
            "returned_chars=%d latency_ms=%d",
            trace_id,
            pipeline.pipeline_id,
            len(payload.query),
            len(options),
            options_chars,
            payload.top_k,
            str(diagnostics.get("strategy", "unknown")),
            bool(diagnostics.get("cache_hit", False)),
            bool(diagnostics.get("wide_mode", False)),
            bool(diagnostics.get("context_mode", False)),
            int(diagnostics.get("full_bundle_tokens_est", 0)),
            str(diagnostics.get("channel_hits", {})),
            int(diagnostics.get("candidate_windows", 0)),
            bool(diagnostics.get("reranked", False)),
            int(diagnostics.get("rerank_groups", 0)),
            int(diagnostics.get("rerank_pair_tokens_est", 0)),
            len(response.data),
            returned_chars,
            int((time.monotonic() - started) * 1000),
        )
        return response

    return app


def _audit_log(runtime: RuntimeSettings, message: str, *args: object) -> None:
    if runtime.debug:
        audit_logger.info(message, *args)


def _authenticate(request: Request, runtime: RuntimeSettings) -> None:
    if not runtime.api_auth_tokens:
        raise ServiceError(
            "authentication_not_configured",
            "No Bearer tokens are configured.",
            status_code=503,
        )
    auth = request.headers.get("Authorization", "")
    scheme, separator, supplied = auth.partition(" ")
    valid_scheme = separator == " " and scheme.lower() == "bearer" and bool(supplied)
    supplied_bytes = supplied.encode("utf-8")
    token_matches = False
    for configured_token in runtime.api_auth_tokens:
        token_matches |= _constant_time_equal(
            supplied_bytes,
            configured_token.get_secret_value().encode("ascii"),
        )
    if not (valid_scheme and token_matches):
        raise ServiceError(
            "invalid_credentials",
            "The supplied memory-service credential is invalid.",
            status_code=401,
        )


def _constant_time_equal(left: bytes, right: bytes) -> bool:
    import hmac

    return hmac.compare_digest(left, right)


def _auth_challenge(_scheme: str) -> str:
    return "Bearer"
