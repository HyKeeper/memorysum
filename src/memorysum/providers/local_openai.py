"""Local OpenAI-compatible embeddings adapter (llama.cpp, vLLM, LM Studio, ...).

Only the embedding wire format lives here. The response parser is deliberately
shared with the OpenRouter adapter: both speak the OpenAI `/v1/embeddings`
dialect, so vector count/index/dimension validation must stay identical.

Reranking is not implemented here; `SplitProvider` pairs this client with the
remote reranker selected by the pipeline config.
"""

from __future__ import annotations

import json
import logging
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
_EMBEDDING_CONCURRENCY = 8


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
            limits=httpx.Limits(
                max_connections=_EMBEDDING_CONCURRENCY, max_keepalive_connections=4
            ),
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._embedding_slots = threading.BoundedSemaphore(_EMBEDDING_CONCURRENCY)

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
        attempts = self._pipeline.capacity.model_request_attempts
        headers = self._request_headers()
        last_timeout = False

        for attempt in range(attempts):
            try:
                response = self._client.post(path, json=payload, headers=headers)
            except httpx.TimeoutException:
                last_timeout = True
                if attempt + 1 >= attempts:
                    break
                time.sleep(1 if attempt == 0 else 2)
                continue
            except httpx.RequestError as exc:
                if attempt + 1 >= attempts:
                    logger.warning(
                        "Local embedding server connection failure (%s)", type(exc).__name__
                    )
                    raise ProviderError(
                        "model_provider_unavailable",
                        "The embedding server is temporarily unavailable.",
                        status_code=503,
                        retry_after=5,
                    ) from None
                time.sleep(1 if attempt == 0 else 2)
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
                time.sleep(1 if attempt == 0 else 2)
                continue
            if response.is_error:
                status = response.status_code
                logger.warning("Local embedding server returned HTTP %d", status)
                if status == 429:
                    raise ProviderError(
                        "model_provider_rate_limited",
                        "The embedding server is rate limiting requests.",
                        status_code=429,
                        retry_after=5,
                    )
                if status in {408, 504}:
                    raise ProviderError(
                        "model_provider_timeout",
                        "The embedding server request timed out.",
                        status_code=504,
                    )
                raise ProviderError(
                    "model_provider_error",
                    "The embedding server rejected or failed a request.",
                    status_code=503,
                    retry_after=5 if status >= 500 else None,
                )
            try:
                decoded = response.json()
            except (ValueError, json.JSONDecodeError):
                raise ProviderError(
                    "model_provider_invalid_response",
                    "The embedding server returned an invalid response.",
                    status_code=503,
                ) from None
            if not isinstance(decoded, dict):
                raise ProviderError(
                    "model_provider_invalid_response",
                    "The embedding server returned an invalid response.",
                    status_code=503,
                )
            return decoded

        if last_timeout:
            raise ProviderError(
                "model_provider_timeout",
                "The embedding server request timed out.",
                status_code=504,
            )
        raise ProviderError(
            "model_provider_unavailable",
            "The embedding server is temporarily unavailable.",
            status_code=503,
            retry_after=5,
        )
