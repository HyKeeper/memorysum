"""Immutable pipeline configuration and environment-only runtime settings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkScaleConfig(FrozenConfig):
    min: int
    target: int
    max: int
    overlap: int


class ChunkingConfig(FrozenConfig):
    leaf: ChunkScaleConfig
    context: ChunkScaleConfig


class EmbeddingConfig(FrozenConfig):
    provider: str
    model: str
    dimension: str
    query_instruction: str
    document_instruction: str | None
    batch_items: int
    batch_proxy_tokens: int
    input_proxy_tokens: int


class RerankerConfig(FrozenConfig):
    #: Set false to stop after fusion (no rerank model call, score = RRF score).
    enabled: bool = True
    provider: str
    model: str
    instruction: str
    single_pair_proxy_tokens: int
    ordinary_pair_proxy_budget: int
    wide_pair_proxy_budget: int
    ordinary_candidate_groups: int
    wide_candidate_groups: int
    batch_groups: int
    max_concurrency: int


class RetrievalModeConfig(FrozenConfig):
    leaf_dense: int
    context_dense: int
    english_fts: int
    chinese_fts: int
    options_lexical: int
    candidate_cap: int
    output_records: int
    output_tokens: int


class RetrievalConfig(FrozenConfig):
    rrf_k: int
    base_channel_weight: float
    #: Per-channel fusion weights; unlisted channels fall back to base_channel_weight.
    channel_weights: dict[str, float]
    ordinary: RetrievalModeConfig
    wide: RetrievalModeConfig


class GroupConfig(FrozenConfig):
    max_spans: int
    direct_max_tokens: int
    context_max_tokens: int


class CoverageConfig(FrozenConfig):
    always_full_tokens: int
    conditional_full_tokens: int
    conditional_full_requires_wide_or_context_mode: bool
    safe_input_tokens: int
    estimation_safety_factor: float
    platform_instruction_reserve: int


class CapacityConfig(FrozenConfig):
    max_body_bytes: int
    max_add_messages: int
    max_search_options: int
    add_concurrency: int
    search_concurrency: int
    interface_queue: int
    add_deadline_seconds: int
    search_deadline_seconds: int
    model_request_timeout_seconds: int
    model_request_attempts: int
    scope_retention_days: int
    cleanup_interval_seconds: int
    wal_autocheckpoint_pages: int


class PipelineConfig(FrozenConfig):
    pipeline_id: str
    qualification_status: Literal["provisional", "ready"]
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    groups: GroupConfig
    coverage: CoverageConfig
    capacity: CapacityConfig

    @property
    def embedding_space_id(self) -> str:
        """Fingerprint the exact document/query embedding contract."""
        import hashlib

        identity = "\n".join(
            (
                self.embedding.provider,
                self.embedding.model,
                self.embedding.query_instruction,
                self.embedding.document_instruction or "",
                "dimension:auto",
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @property
    def leaf_chunks(self) -> ChunkScaleConfig:
        return self.chunking.leaf

    @property
    def context_chunks(self) -> ChunkScaleConfig:
        return self.chunking.context


@dataclass(frozen=True)
class RuntimeSettings:
    data_root: Path
    openrouter_base_url: str
    openrouter_api_key: SecretStr | None
    api_auth_scheme: Literal["bearer"]
    api_auth_tokens: tuple[SecretStr, ...]
    host: str
    port: int
    debug: bool = False
    embedding_base_url: str = "http://127.0.0.1:7999/v1"
    embedding_api_key: SecretStr | None = None

    @classmethod
    def from_environment(cls) -> RuntimeSettings:
        keys_path = Path(os.getenv("MEMORYSUM_API_KEYS_FILE", "keys.txt")).expanduser()
        if not keys_path.is_absolute():
            keys_path = Path.cwd() / keys_path
        try:
            raw_lines = keys_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"Cannot read Bearer token file ({type(exc).__name__}); configure MEMORYSUM_API_KEYS_FILE"
            ) from None

        tokens: list[SecretStr] = []
        seen_tokens: set[str] = set()
        for line_number, raw_line in enumerate(raw_lines, start=1):
            token = raw_line.strip()
            if not token or token.startswith("#"):
                continue
            if any(not 33 <= ord(character) <= 126 for character in token):
                raise ValueError(
                    f"Bearer token file must contain printable ASCII tokens (line {line_number})"
                )
            if token not in seen_tokens:
                seen_tokens.add(token)
                tokens.append(SecretStr(token))
        if not tokens:
            raise ValueError("Bearer token file contains no tokens")

        debug_value = os.getenv("MEMORYSUM_DEBUG", "false").strip().lower()
        if debug_value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise ValueError("MEMORYSUM_DEBUG must be a boolean value")
        debug = debug_value in {"1", "true", "yes", "on"}

        api_key = os.getenv("OPENROUTER_API_KEY")
        embedding_api_key = os.getenv("MEMORYSUM_EMBEDDING_API_KEY")
        return cls(
            data_root=Path(os.getenv("MEMORYSUM_DATA_ROOT", "./data")).expanduser().resolve(),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            openrouter_api_key=SecretStr(api_key) if api_key else None,
            api_auth_scheme="bearer",
            api_auth_tokens=tuple(tokens),
            host=os.getenv("MEMORYSUM_HOST", "0.0.0.0"),
            port=8001,
            debug=debug,
            embedding_base_url=os.getenv(
                "MEMORYSUM_EMBEDDING_BASE_URL", "http://127.0.0.1:7999/v1"
            ).rstrip("/"),
            embedding_api_key=SecretStr(embedding_api_key) if embedding_api_key else None,
        )


@lru_cache(maxsize=1)
def load_pipeline_config() -> PipelineConfig:
    """Read the packaged, versioned algorithm configuration (not env-tunable)."""
    config_path = files("memorysum").joinpath("config/champion-v1.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(payload)
