"""Small provider contracts; memory services do not depend on OpenRouter types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RerankScore:
    index: int
    score: float


class ModelProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[RerankScore]: ...

    def close(self) -> None: ...
