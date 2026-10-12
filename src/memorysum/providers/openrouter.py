"""OpenRouter embeddings and rerank adapter.

This module is intentionally the only place that knows OpenRouter's wire format.
The API key is read from the environment and is never included in logs/errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Sequence
from typing import Any

import httpx

from memorysum.core.errors import ProviderError
from memorysum.providers.protocols import RerankScore
from memorysum.settings import PipelineConfig, RuntimeSettings

logger = logging.getLogger(__name__)
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 524, 525}


class OpenRouterProvider:
    """Synchronous OpenRouter client for dense embeddings and reranking."""

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
            base_url=runtime.openrouter_base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(
                pipeline.capacity.model_request_timeout_seconds,
                connect=5.0,
            ),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={"Content-Type": "application/json"},
            transport=transport,
        )
        self._embedding_slots = threading.BoundedSemaphore(8)
        self._rerank_slots = threading.BoundedSemaphore(pipeline.reranker.max_concurrency)

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = self._api_key()
        payload: dict[str, Any] = {
            "model": self._pipeline.embedding.model,
            "input": list(texts) if len(texts) > 1 else texts[0],
            "encoding_format": "float",
        }
        with self._embedding_slots:
            body = self._post("embeddings", payload, api_key)
        return self._parse_embeddings(body, len(texts))

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        if not documents:
            return []
        api_key = self._api_key()
        payload = {
            "model": self._pipeline.reranker.model,
            "query": query,
            "documents": [{"text": document} for document in documents],
            "top_n": len(documents),
        }
        with self._rerank_slots:
            body = self._post("rerank", payload, api_key)
        return self._parse_rerank(body, len(documents))

    def _api_key(self) -> str:
        value = self._runtime.openrouter_api_key
        if value is None:
            raise ProviderError(
                "model_provider_not_configured",
                "OPENROUTER_API_KEY is not configured.",
                status_code=503,
            )
        return value.get_secret_value()

    def _post(self, path: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        attempts = self._pipeline.capacity.model_request_attempts
        jitter = self._retry_jitter(payload)
        last_timeout = False

        for attempt in range(attempts):
            try:
                response = self._client.post(
                    path,
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            except httpx.TimeoutException:
                last_timeout = True
                if attempt + 1 >= attempts:
                    break
                time.sleep((1 if attempt == 0 else 2) + jitter)
                continue
            except httpx.RequestError as exc:
                if attempt + 1 >= attempts:
                    logger.warning("OpenRouter connection failure (%s)", type(exc).__name__)
                    raise ProviderError(
                        "model_provider_unavailable",
                        "The model provider is temporarily unavailable.",
                        status_code=503,
                        retry_after=5,
                    ) from None
                time.sleep((1 if attempt == 0 else 2) + jitter)
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt + 1 < attempts:
                wait = self._retry_delay(response, attempt, jitter)
                time.sleep(wait)
                continue
            if response.is_error:
                status = response.status_code
                logger.warning("OpenRouter returned HTTP %d for a model request", status)
                if status == 429:
                    raise ProviderError(
                        "model_provider_rate_limited",
                        "The model provider is rate limiting requests.",
                        status_code=429,
                        retry_after=self._retry_after(response),
                    )
                if status in {408, 504, 524}:
                    raise ProviderError(
                        "model_provider_timeout",
                        "The model provider request timed out.",
                        status_code=504,
                    )
                raise ProviderError(
                    "model_provider_error",
                    "The model provider rejected or failed a request.",
                    status_code=503,
                    retry_after=5 if status >= 500 else None,
                )
            try:
                decoded = response.json()
            except (ValueError, json.JSONDecodeError):
                raise ProviderError(
                    "model_provider_invalid_response",
                    "The model provider returned an invalid response.",
                    status_code=503,
                ) from None
            if not isinstance(decoded, dict):
                raise ProviderError(
                    "model_provider_invalid_response",
                    "The model provider returned an invalid response.",
                    status_code=503,
                )
            return decoded

        if last_timeout:
            raise ProviderError(
                "model_provider_timeout",
                "The model provider request timed out.",
                status_code=504,
            )
        raise ProviderError(
            "model_provider_unavailable",
            "The model provider is temporarily unavailable.",
            status_code=503,
            retry_after=5,
        )

    @staticmethod
    def _retry_jitter(payload: dict[str, Any]) -> float:
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).digest()
        return int.from_bytes(digest[:2], "big") / 65535 * 0.25

    @staticmethod
    def _retry_after(response: httpx.Response) -> int:
        value = response.headers.get("Retry-After", "")
        try:
            return min(60, max(1, int(float(value))))
        except (TypeError, ValueError):
            return 5

    @classmethod
    def _retry_delay(cls, response: httpx.Response, attempt: int, jitter: float) -> float:
        if response.status_code == 429:
            return float(cls._retry_after(response))
        return (1 if attempt == 0 else 2) + jitter

    @staticmethod
    def _parse_embeddings(body: dict[str, Any], expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ProviderError(
                "model_provider_invalid_embedding",
                "The embedding provider returned an unexpected number of vectors.",
                status_code=503,
            )
        indexed: dict[int, list[float]] = {}
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned an invalid vector.",
                    status_code=503,
                )
            index = item.get("index", fallback_index)
            vector = item.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(vector, list)
            ):
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned an invalid vector.",
                    status_code=503,
                )
            if index in indexed or index < 0 or index >= expected:
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned invalid vector indexes.",
                    status_code=503,
                )
            parsed: list[float] = []
            for value in vector:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ProviderError(
                        "model_provider_invalid_embedding",
                        "The embedding provider returned a non-numeric vector.",
                        status_code=503,
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ProviderError(
                        "model_provider_invalid_embedding",
                        "The embedding provider returned a non-finite vector.",
                        status_code=503,
                    )
                parsed.append(number)
            if not parsed:
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned an empty vector.",
                    status_code=503,
                )
            indexed[index] = parsed

        ordered = [indexed[index] for index in range(expected)]
        dimension = len(ordered[0])
        if any(len(vector) != dimension for vector in ordered):
            raise ProviderError(
                "model_provider_invalid_embedding",
                "The embedding provider returned inconsistent dimensions.",
                status_code=503,
            )
        return ordered

    @staticmethod
    def _parse_rerank(body: dict[str, Any], expected: int) -> list[RerankScore]:
        results = body.get("results")
        if not isinstance(results, list) or len(results) != expected:
            raise ProviderError(
                "model_provider_invalid_rerank",
                "The reranker returned an incomplete result set.",
                status_code=503,
            )
        parsed: list[RerankScore] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise ProviderError(
                    "model_provider_invalid_rerank",
                    "The reranker returned an invalid result.",
                    status_code=503,
                )
            index = result.get("index")
            score = result.get("relevance_score", result.get("score"))
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
