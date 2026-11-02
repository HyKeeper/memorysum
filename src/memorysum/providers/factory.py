"""Build the ModelProvider implementation named by the pipeline configuration.

Every supported provider is a hosted HTTP API: the service never loads a model
into the process and never needs a local inference server.
"""

from __future__ import annotations

import httpx

from memorysum.providers.dashscope import DashScopeProvider
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import ModelProvider
from memorysum.providers.split import SplitProvider
from memorysum.settings import PipelineConfig, RuntimeSettings

#: Hosted DashScope API (native embedding + rerank wire formats).
DASHSCOPE_PROVIDER_NAMES = frozenset({"dashscope"})
#: Hosted OpenRouter API (OpenAI-compatible embeddings + rerank).
OPENROUTER_PROVIDER_NAMES = frozenset({"openrouter"})
_KNOWN_PROVIDER_NAMES = DASHSCOPE_PROVIDER_NAMES | OPENROUTER_PROVIDER_NAMES


def embedding_endpoint(runtime: RuntimeSettings, pipeline: PipelineConfig) -> str:
    """Return the endpoint that defines the document/query embedding space."""
    provider = pipeline.embedding.provider
    if provider in DASHSCOPE_PROVIDER_NAMES:
        return runtime.dashscope_base_url
    return runtime.openrouter_base_url


def _build_provider(
    name: str,
    runtime: RuntimeSettings,
    pipeline: PipelineConfig,
    *,
    transport: httpx.BaseTransport | None,
) -> ModelProvider:
    if name in DASHSCOPE_PROVIDER_NAMES:
        return DashScopeProvider(runtime, pipeline, transport=transport)
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

    # One hosted client serves both stages when they name the same provider; otherwise
    # each stage keeps its own client so the endpoints stay independent.
    if len(names) == 1:
        return _build_provider(pipeline.embedding.provider, runtime, pipeline, transport=transport)

    return SplitProvider(
        _build_provider(pipeline.embedding.provider, runtime, pipeline, transport=transport),
        _build_provider(pipeline.reranker.provider, runtime, pipeline, transport=transport),
    )
