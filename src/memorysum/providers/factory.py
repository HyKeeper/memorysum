"""Build the ModelProvider implementation named by the pipeline configuration."""

from __future__ import annotations

import httpx

from memorysum.providers.local_openai import LocalEmbeddingProvider
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import ModelProvider
from memorysum.providers.split import SplitProvider
from memorysum.settings import PipelineConfig, RuntimeSettings

#: Providers that are served by a local, OpenAI-compatible HTTP endpoint.
LOCAL_PROVIDER_NAMES = frozenset({"local-openai"})


def embedding_endpoint(runtime: RuntimeSettings, pipeline: PipelineConfig) -> str:
    """Return the endpoint that defines the document/query embedding space."""
    if pipeline.embedding.provider in LOCAL_PROVIDER_NAMES:
        return runtime.embedding_base_url
    return runtime.openrouter_base_url


def build_provider(
    runtime: RuntimeSettings,
    pipeline: PipelineConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ModelProvider:
    """Wire embedding and reranking to the implementations the config names."""
    names = {pipeline.embedding.provider, pipeline.reranker.provider}
    unsupported = sorted(names - LOCAL_PROVIDER_NAMES - {"openrouter"})
    if unsupported:
        raise ValueError(f"Unsupported model provider in pipeline config: {unsupported}")

    embedding_is_local = pipeline.embedding.provider in LOCAL_PROVIDER_NAMES
    rerank_is_local = pipeline.reranker.provider in LOCAL_PROVIDER_NAMES

    if rerank_is_local:
        if pipeline.reranker.enabled:
            raise ValueError("A local reranker is not implemented; use a remote reranker provider")
        return LocalEmbeddingProvider(runtime, pipeline, transport=transport)

    if not embedding_is_local:
        return OpenRouterProvider(runtime, pipeline, transport=transport)

    return SplitProvider(
        LocalEmbeddingProvider(runtime, pipeline, transport=transport),
        OpenRouterProvider(runtime, pipeline, transport=transport),
    )
