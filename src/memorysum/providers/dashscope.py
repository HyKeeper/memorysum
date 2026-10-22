"""Alibaba Cloud DashScope embedding provider (`text-embedding-v4`).

Wire format is the native DashScope synchronous embedding API, **not** the
OpenAI-compatible `/v1/embeddings` dialect:

    POST {base}/api/v1/services/embeddings/text-embedding/text-embedding
    Authorization: Bearer $DASHSCOPE_API_KEY
    {"model": "text-embedding-v4",
     "input": {"texts": ["..."]},
     "dimension": 2048}

    -> {"output": {"embeddings": [{"text_index": 0, "embedding": [...]}]},
        "usage": {"total_tokens": 23}, "request_id": "..."}

Documented limits (checked against the live endpoint on 2026-10-20):
at most 10 inputs per request, each input up to 8,192 tokens. The dimension must
be one of 64/128/256/512/768/1024/1536/2048/3072; omitting it silently yields the
1024-dimension default, which is why `embedding.dimension` is required and the
response is checked against it.

Measured on 2026-10-20: `https://dashscope.aliyuncs.com` answered a
`"dimension": 2048` request with **1024-d vectors** (the parameter was ignored),
while the dedicated MaaS endpoint
`https://llm-5qr9sf5t4nflguf4.cn-beijing.maas.aliyuncs.com` honoured it. This
provider therefore refuses a mismatched dimension instead of silently writing a
different embedding space into the scope database.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any

import httpx

from memorysum.core.errors import ProviderError
from memorysum.providers.http_post import post_json
from memorysum.providers.protocols import RerankScore
from memorysum.settings import PipelineConfig, RuntimeSettings

#: Dimensions accepted by text-embedding-v4 (probed against the live endpoint).
SUPPORTED_DIMENSIONS = frozenset({64, 128, 256, 512, 768, 1024, 1536, 2048, 3072})

#: Hard server limit: one request carries at most 10 inputs.
MAX_BATCH_ITEMS = 10

_PATH = "api/v1/services/embeddings/text-embedding/text-embedding"
_CONCURRENCY = 8


class DashScopeEmbeddingProvider:
    """Dense embeddings served by DashScope's native text-embedding API."""

    def __init__(
        self,
        runtime: RuntimeSettings,
        pipeline: PipelineConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._runtime = runtime
        self._pipeline = pipeline
        embedding = pipeline.embedding
        if not isinstance(embedding.dimension, int):
            # Auto-detection would accept the server's 1024 default, silently changing
            # the embedding space; DashScope callers must state the dimension.
            raise ValueError(
                "embedding.dimension must be an explicit integer for provider 'dashscope' "
                f"(got {embedding.dimension!r})"
            )
        if embedding.dimension not in SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"embedding.dimension={embedding.dimension} is not supported by "
                f"text-embedding-v4; choose one of {sorted(SUPPORTED_DIMENSIONS)}"
            )
        if embedding.batch_items > MAX_BATCH_ITEMS:
            raise ValueError(
                f"embedding.batch_items={embedding.batch_items} exceeds the DashScope "
                f"limit of {MAX_BATCH_ITEMS} inputs per request"
            )
        self._client = httpx.Client(
            base_url=runtime.dashscope_base_url.rstrip("/") + "/",
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
        if len(texts) > MAX_BATCH_ITEMS:
            raise ProviderError(
                "model_provider_invalid_embedding",
                f"The embedding batch has {len(texts)} inputs; DashScope accepts at most "
                f"{MAX_BATCH_ITEMS}.",
                status_code=503,
            )
        payload: dict[str, Any] = {
            "model": self._pipeline.embedding.model,
            "input": {"texts": list(texts)},
            "dimension": int(self._pipeline.embedding.dimension),
        }
        with self._embedding_slots:
            body = post_json(
                self._client,
                _PATH,
                payload,
                attempts=self._pipeline.capacity.model_request_attempts,
                headers=self._request_headers(),
                what="DashScope embedding",
            )
        if not isinstance(body, dict):
            raise ProviderError(
                "model_provider_invalid_embedding",
                "The embedding provider returned an invalid response.",
                status_code=503,
            )
        return self._parse_embeddings(body, len(texts))

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        raise ProviderError(
            "model_provider_not_configured",
            "The DashScope embedding provider does not implement reranking.",
            status_code=503,
        )

    def _request_headers(self) -> dict[str, str]:
        api_key = self._runtime.dashscope_api_key
        if api_key is None:
            raise ProviderError(
                "model_provider_not_configured",
                "DASHSCOPE_API_KEY (or MEMORYSUM_DASHSCOPE_API_KEY) is not configured.",
                status_code=503,
            )
        return {"Authorization": f"Bearer {api_key.get_secret_value()}"}

    def _parse_embeddings(self, body: dict[str, Any], expected: int) -> list[list[float]]:
        output = body.get("output")
        entries = output.get("embeddings") if isinstance(output, dict) else None
        if not isinstance(entries, list) or len(entries) != expected:
            raise ProviderError(
                "model_provider_invalid_embedding",
                "The embedding provider returned an unexpected number of vectors.",
                status_code=503,
            )

        indexed: dict[int, list[float]] = {}
        declared = int(self._pipeline.embedding.dimension)
        for fallback_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned an invalid vector.",
                    status_code=503,
                )
            index = entry.get("text_index", fallback_index)
            vector = entry.get("embedding")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= expected
                or index in indexed
                or not isinstance(vector, list)
            ):
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    "The embedding provider returned an invalid vector or index.",
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
            if len(parsed) != declared:
                # Never persist a different embedding space: the scope manifest and the
                # vec0 tables would be built for `declared`, so a silent fallback (observed:
                # the public endpoint returning 1024 for a 2048 request) corrupts the scope.
                raise ProviderError(
                    "model_provider_invalid_embedding",
                    f"The embedding provider returned {len(parsed)}-dimension vectors but "
                    f"the pipeline declares {declared}; check MEMORYSUM_DASHSCOPE_BASE_URL "
                    "and embedding.dimension.",
                    status_code=503,
                )
            indexed[index] = parsed

        return [indexed[index] for index in range(expected)]
