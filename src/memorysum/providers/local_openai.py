"""Local OpenAI-compatible model providers (llama.cpp, vLLM, LM Studio, ...).

Only the local wire formats live here. The embedding parser is deliberately
shared with the OpenRouter adapter: both speak the OpenAI `/v1/embeddings`
dialect, so vector count/index/dimension validation stays identical.

`LocalRerankProvider` targets a local cross-encoder endpoint and accepts both
response shapes seen in the wild:

    {"results": [{"index": i, "relevance_score": s}]}   llama.cpp --reranking / OpenRouter
    [{"index": i, "score": s, "document": d}]           plain FastAPI implementations
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import threading
import time
from collections.abc import Sequence
from typing import Any

import httpx

from memorysum.core.errors import ProviderError
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import RerankScore
from memorysum.settings import PipelineConfig, RuntimeSettings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_CONCURRENCY = 8

#: llama.cpp rejects a prompt that does not fit in one physical batch, or in the slot
#: context, with HTTP 500. The prompt is simply too long for the model - retrying cannot
#: help, and waiting 1 s + 2 s for the same rejection burns the caller's deadline (measured
#: on 2026-10-14: every failed Search stalled for 3027 ms and then returned 503).
#: `reranker -ub` must therefore cover `reranker.single_pair_proxy_tokens` (see README).
_CAPACITY_ERROR_MARKERS = (
    "too large to process",
    "larger than the max context size",
    "exceeds the available context size",
)


def _error_message(response: httpx.Response, *, limit: int = 200) -> str:
    """Best-effort read of a provider's error text; never the request payload we sent."""
    decoded: object = None
    with contextlib.suppress(ValueError):
        decoded = response.json()
    message = ""
    if isinstance(decoded, dict):
        error = decoded.get("error", decoded)
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif isinstance(error, str):
            message = error
        if not message:
            message = str(decoded.get("message") or "")
    if not message:
        with contextlib.suppress(Exception):
            message = response.text
    return " ".join(message.split())[:limit]


def _post_json(
    client: httpx.Client,
    path: str,
    payload: dict[str, Any],
    *,
    attempts: int,
    headers: dict[str, str],
    what: str,
) -> dict[str, Any] | list[Any]:
    """POST with bounded retries; shared by the local embedding and rerank clients."""
    last_timeout = False

    for attempt in range(attempts):
        try:
            response = client.post(path, json=payload, headers=headers)
        except httpx.TimeoutException:
            last_timeout = True
            if attempt + 1 >= attempts:
                break
            time.sleep(1 if attempt == 0 else 2)
            continue
        except httpx.RequestError as exc:
            if attempt + 1 >= attempts:
                logger.warning("Local %s connection failure (%s)", what, type(exc).__name__)
                raise ProviderError(
                    "model_provider_unavailable",
                    f"The {what} server is temporarily unavailable.",
                    status_code=503,
                    retry_after=5,
                ) from None
            time.sleep(1 if attempt == 0 else 2)
            continue

        detail = _error_message(response) if response.is_error else ""
        if response.is_error and any(marker in detail for marker in _CAPACITY_ERROR_MARKERS):
            # Deterministic rejection: fail immediately instead of retrying a prompt that
            # no number of attempts can shrink. The message carries the server's own limit.
            logger.warning("Local %s server rejected an oversized input: %s", what, detail)
            raise ProviderError(
                "model_provider_input_too_large",
                f"The {what} server rejected an input that is too long for the model.",
                status_code=503,
            )
        if response.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
            time.sleep(1 if attempt == 0 else 2)
            continue
        if response.is_error:
            status = response.status_code
            logger.warning("Local %s server returned HTTP %d: %s", what, status, detail)
            if status == 429:
                raise ProviderError(
                    "model_provider_rate_limited",
                    f"The {what} server is rate limiting requests.",
                    status_code=429,
                    retry_after=5,
                )
            if status in {408, 504}:
                raise ProviderError(
                    "model_provider_timeout",
                    f"The {what} server request timed out.",
                    status_code=504,
                )
            raise ProviderError(
                "model_provider_error",
                f"The {what} server rejected or failed a request.",
                status_code=503,
                retry_after=5 if status >= 500 else None,
            )
        try:
            decoded = response.json()
        except (ValueError, json.JSONDecodeError):
            raise ProviderError(
                "model_provider_invalid_response",
                f"The {what} server returned an invalid response.",
                status_code=503,
            ) from None
        if not isinstance(decoded, (dict, list)):
            raise ProviderError(
                "model_provider_invalid_response",
                f"The {what} server returned an invalid response.",
                status_code=503,
            )
        return decoded

    if last_timeout:
        raise ProviderError(
            "model_provider_timeout",
            f"The {what} server request timed out.",
            status_code=504,
        )
    raise ProviderError(
        "model_provider_unavailable",
        f"The {what} server is temporarily unavailable.",
        status_code=503,
        retry_after=5,
    )


class LocalEmbeddingProvider:
    """Dense embeddings served by a local, OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        runtime: RuntimeSettings,
        pipeline: PipelineConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._runtime = runtime
        self._pipeline = pipeline
        self._client = httpx.Client(
            base_url=runtime.embedding_base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(
                pipeline.capacity.model_request_timeout_seconds,
                connect=5.0,
            ),
            limits=httpx.Limits(max_connections=_CONCURRENCY, max_keepalive_connections=4),
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._embedding_slots = threading.BoundedSemaphore(_CONCURRENCY)

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {
            "model": self._pipeline.embedding.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        with self._embedding_slots:
            body = self._post("embeddings", payload)
        return OpenRouterProvider._parse_embeddings(body, len(texts))

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        raise ProviderError(
            "model_provider_not_configured",
            "The local embedding provider does not implement reranking.",
            status_code=503,
        )

    def _request_headers(self) -> dict[str, str]:
        api_key = self._runtime.embedding_api_key
        if api_key is None:
            return {}
        return {"Authorization": f"Bearer {api_key.get_secret_value()}"}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = _post_json(
            self._client,
            path,
            payload,
            attempts=self._pipeline.capacity.model_request_attempts,
            headers=self._request_headers(),
            what="embedding",
        )
        if not isinstance(body, dict):
            raise ProviderError(
                "model_provider_invalid_response",
                "The embedding server returned an invalid response.",
                status_code=503,
            )
        return body


class LocalRerankProvider:
    """Cross-encoder reranking served by a local HTTP endpoint."""

    def __init__(
        self,
        runtime: RuntimeSettings,
        pipeline: PipelineConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._runtime = runtime
        self._pipeline = pipeline
        self._client = httpx.Client(
            base_url=runtime.rerank_base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(
                pipeline.capacity.model_request_timeout_seconds,
                connect=5.0,
            ),
            limits=httpx.Limits(
                max_connections=pipeline.reranker.max_concurrency * 2,
                max_keepalive_connections=2,
            ),
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._rerank_slots = threading.BoundedSemaphore(pipeline.reranker.max_concurrency)

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise ProviderError(
            "model_provider_not_configured",
            "The local rerank provider does not implement embeddings.",
            status_code=503,
        )

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        if not documents:
            return []
        payload: dict[str, Any] = {
            "model": self._pipeline.reranker.model,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        with self._rerank_slots:
            body = _post_json(
                self._client,
                "rerank",
                payload,
                attempts=self._pipeline.capacity.model_request_attempts,
                headers=self._request_headers(),
                what="rerank",
            )
        return self._parse_rerank(body, len(documents))

    def _request_headers(self) -> dict[str, str]:
        api_key = self._runtime.rerank_api_key
        if api_key is None:
            return {}
        return {"Authorization": f"Bearer {api_key.get_secret_value()}"}

    @staticmethod
    def _parse_rerank(body: object, expected: int) -> list[RerankScore]:
        if isinstance(body, list):
            entries: object = body
        elif isinstance(body, dict):
            entries = body.get("results")
        else:
            entries = None
        if not isinstance(entries, list) or len(entries) != expected:
            raise ProviderError(
                "model_provider_invalid_rerank",
                "The reranker returned an incomplete result set.",
                status_code=503,
            )
        parsed: list[RerankScore] = []
        seen: set[int] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProviderError(
                    "model_provider_invalid_rerank",
                    "The reranker returned an invalid result.",
                    status_code=503,
                )
            index = entry.get("index")
            score = entry.get("relevance_score", entry.get("score"))
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected
                or index in seen
                or not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
            ):
                raise ProviderError(
                    "model_provider_invalid_rerank",
                    "The reranker returned an invalid score or index.",
                    status_code=503,
                )
            seen.add(index)
            parsed.append(RerankScore(index=index, score=float(score)))
        if len(seen) != expected:
            raise ProviderError(
                "model_provider_invalid_rerank",
                "The reranker returned duplicate or missing indexes.",
                status_code=503,
            )
        return parsed
