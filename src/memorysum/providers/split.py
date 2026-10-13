"""Route embedding and reranking to different providers.

Champion configs may pin a local embedding endpoint while the reranker stays on
a hosted provider. `SplitProvider` keeps that split explicit so Add/Search keep
depending on a single `ModelProvider`.
"""

from __future__ import annotations

from collections.abc import Sequence

from memorysum.providers.protocols import ModelProvider, RerankScore


class SplitProvider:
    """Delegate embeddings and reranking to independent provider instances."""

    def __init__(
        self,
        embedding_provider: ModelProvider,
        rerank_provider: ModelProvider,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._rerank_provider = rerank_provider

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embedding_provider.embed(texts)

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]:
        return self._rerank_provider.rerank(query, documents)

    def close(self) -> None:
        try:
            self._embedding_provider.close()
        finally:
            self._rerank_provider.close()
