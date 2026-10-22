"""Build the ModelProvider implementation named by the pipeline configuration."""

from __future__ import annotations

import httpx

from memorysum.providers.dashscope import DashScopeEmbeddingProvider
from memorysum.providers.local_openai import LocalEmbeddingProvider, LocalRerankProvider
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import ModelProvider
from memorysum.providers.split import SplitProvider
from memorysum.settings import PipelineConfig, RuntimeSettings

#: Providers that are served by a local, OpenAI-compatible HTTP endpoint.
LOCAL_PROVIDER_NAMES = frozenset({"local-openai"})
#: Hosted DashScope embedding API (native wire format, no reranking).
DASHSCOPE_PROVIDER_NAMES = frozenset({"dashscope"})
#: Hosted OpenRouter API (OpenAI-compatible embeddings + rerank).
OPENROUTER_PROVIDER_NAMES = frozenset({"openrouter"})
_KNOWN_PROVIDER_NAMES = LOCAL_PROVIDER_NAMES | DASHSCOPE_PROVIDER_NAMES | OPENROUTER_PROVIDER_NAMES


def embedding_endpoint(runtime: RuntimeSettings, pipeline: PipelineConfig) -> str:
    """Return the endpoint that defines the document/query embedding space."""
    provider = pipeline.embedding.provider
    if provider in LOCAL_PROVIDER_NAMES:
        return runtime.embedding_base_url
    if provider in DASHSCOPE_PROVIDER_NAMES:
        return runtime.dashscope_base_url
    return runtime.openrouter_base_url


def _build_embedding_provider(
    name: str,
    runtime: RuntimeSettings,
    pipeline: PipelineConfig,
    *,
    transport: httpx.BaseTransport | None,
) -> ModelProvider:
    if name in LOCAL_PROVIDER_NAMES:
        return LocalEmbeddingProvider(runtime, pipeline, transport=transport)
    if name in DASHSCOPE_PROVIDER_NAMES:
        return DashScopeEmbeddingProvider(runtime, pipeline, transport=transport)
    return OpenRouterProvider(runtime, pipeline, transport=transport)


def _build_rerank_provider(
    name: str,
    runtime: RuntimeSettings,
    pipeline: PipelineConfig,
    *,
    transport: httpx.BaseTransport | None,
) -> ModelProvider:
    if name in DASHSCOPE_PROVIDER_NAMES:
        raise ValueError(
            "provider 'dashscope' implements embeddings only; "
            "configure reranker.provider as 'local-openai' or 'openrouter'"
        )
    if name in LOCAL_PROVIDER_NAMES:
        return LocalRerankProvider(runtime, pipeline, transport=transport)
    return OpenRouterProvider(runtime, pipeline, transport=transport)


def build_provider(
    runtime: RuntimeSettings,
    pipeline: PipelineConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ModelProvider:
    """Wire embedding and reranking to the implementations the config names."""
    names = {pipeline.embedding.provider, pipeline.reranker.provider}
    unsupported = sorted(names - _KNOWN_PROVIDER_NAMES)
    if unsupported:
        raise ValueError(f"Unsupported model provider in pipeline config: {unsupported}")

    # A single OpenRouter client can serve both stages; every other combination gets an
    # explicit split so the embedding endpoint and the rerank endpoint stay independent.
    if (
        pipeline.embedding.provider in OPENROUTER_PROVIDER_NAMES
        and pipeline.reranker.provider in OPENROUTER_PROVIDER_NAMES
    ):
        return OpenRouterProvider(runtime, pipeline, transport=transport)

    return SplitProvider(
        _build_embedding_provider(
            pipeline.embedding.provider, runtime, pipeline, transport=transport
        ),
        _build_rerank_provider(pipeline.reranker.provider, runtime, pipeline, transport=transport),
    )
