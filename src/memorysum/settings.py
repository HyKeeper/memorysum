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
    #: Weight of the retrieval ranking when fusing with the reranker ranking.
    fusion_weight: float = 2.0
    provider: str
    model: str
    instruction: str
    #: 单条 (query, document) 的代理 token 预算（cl100k 估算值）。必须是**保守值**：
    #: 真实口径是 llama.cpp 的 XLM-R 分词器，实测最坏 real/est = 1.494，所以预算×1.75
    #: 仍小于服务端硬上限（模型训练窗口 8192）才算安全。见 README「服务编排」。
    single_pair_proxy_tokens: int
    #: query(+options) 的代理 token 上限，超过直接 422；目前只作为荒谬输入的护栏。
    max_query_proxy_tokens: int
    ordinary_pair_proxy_budget: int
    wide_pair_proxy_budget: int
    ordinary_candidate_groups: int
    wide_candidate_groups: int
    batch_groups: int
    max_concurrency: int
    #: Feed the option texts to the cross-encoder alongside the question. On
    #: multiple-choice scopes the six rewritten option sentences dilute the
    #: question signal; `false` reranks on the question alone (options still drive
    #: the `options_lexical` channel and the response).
    include_options: bool = True


class PrfDenseConfig(FrozenConfig):
    """Second dense recall from the first-round document centroid.

    The first-round documents (vector + lexical leaders) are averaged into one
    centroid vector and used as a *second* query, the PRF recipe. Only its
    increments (windows the main channels did not return) enter the fusion, at a
    low weight, so a bad centroid can add noise but cannot displace the main
    ranking. See docs/todo-prf-query-expansion.md.
    """

    enabled: bool = True
    #: How many leaders from each of the two main routes feed the centroid.
    seed_docs: int = 16
    #: Windows kept after the second recall (before the increment filter).
    neighbors: int = 64
    #: RRF channel weight for the increments (main channels run at 1-2).
    weight: float = 0.6
    #: 0 = pure document centroid; 0.5 = half original query, half centroid.
    query_mix: float = 0.0
    #: Do not build a centroid from fewer documents than this.
    min_seeds: int = 4


class PrfLexicalConfig(FrozenConfig):
    """Second BM25 recall from terms mined out of the first-round documents."""

    enabled: bool = True
    #: How many BM25 leaders are mined for expansion terms.
    seed_docs: int = 8
    #: How many expansion terms enter the second query.
    max_terms: int = 8
    #: Windows kept after the second recall (before the increment filter).
    neighbors: int = 32
    weight: float = 0.3
    #: Drop short terms; they are mostly function words and tokenizer debris.
    min_term_length: int = 4
    #: Drop terms present in more than this fraction of seed documents.
    max_seed_df_ratio: float = 0.6


class PrfConfig(FrozenConfig):
    """Four-way recall: two main routes + two low-weight expansion routes.

    Expansion routes only contribute increments (never a main-route window) and
    are fused with a lower weight; the final ordering still comes from the
    original query (rerank + fusion). Disabled by default until an A/B shows it
    helps on the offline harness.
    """

    enabled: bool = False
    #: Short-list queries do not need a wider net; the article's own guard.
    min_top_k: int = 30
    dense: PrfDenseConfig = PrfDenseConfig()
    lexical: PrfLexicalConfig = PrfLexicalConfig()
    #: Also mine `lexical_zh` terms for the Chinese FTS index (jieba tokens).
    lexical_zh: bool = True


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
    #: Scale `rrf_k` to the candidate count instead of using a fixed value
    #: (``k = clamp(unique_candidates / 3, 10, 30)``). RRF's own paper tuned k=60 on
    #: TREC-scale corpora; here the candidate list runs 40-254 windows, and a fixed k
    #: under-discounts the head on small lists while flattening it on large ones.
    #: Measured: k=20 beats k=60 on both txt_all and ids_all at our typical size.
    rrf_k_adaptive: bool = False
    base_channel_weight: float
    #: Per-channel fusion weights; unlisted channels fall back to base_channel_weight.
    channel_weights: dict[str, float]
    #: Weight of a channel's own similarity scores relative to its rank term. RRF
    #: alone ignores confidence (a 0.9-similarity hit and a 0.3 one with the same
    #: rank contribute identically); >0 min-max normalises each channel and adds
    #: the score as a bounded tie-breaker. 0 = pure rank fusion.
    similarity_weight: float = 0.0
    #: Multiplicative reward for being found by several channels at once
    #: (``score *= 1 + bonus*(n-1)``). Plain RRF adds each channel's contribution,
    #: which is linear in the channel count; multiplying is what lets a three-channel
    #: hit clearly outrank a single-channel top-1 instead of merely tying with it.
    #: 0 keeps the historical additive behaviour.
    channel_consensus_bonus: float = 0.0
    #: Chunk-level agreement vote applied after ranking: an entry gains score when
    #: other strong candidates come from the same source message. Multi-hop answers
    #: often need two turns of one chunk, so this lifts that chunk's windows.
    consensus_weight: float = 0.0
    #: Boost windows dominated by a speaker the query names ("What did Melanie paint?").
    #: 0 disables it. Uses the candidate text's own "Name:" prefixes.
    speaker_weight: float = 0.0
    #: Cap how many returned records may come from one source message (one chunk).
    #: Multi-hop questions need evidence from several sessions, and an uncapped
    #: ranking can fill the whole budget with windows of a single chatty chunk.
    #: 0 disables the cap.
    per_source_cap: int = 0
    #: Reward windows that actually mention the query's content words (stopwords and
    #: short tokens dropped). Cheaper and softer than FTS: a window covering more of
    #: the question's terms is likelier to hold the answer, regardless of phrasing.
    term_coverage_weight: float = 0.0
    #: Widen every returned passage by N source characters on both sides. Neighbouring
    #: text often holds the second half of a multi-hop answer, which leaf windows cut
    #: away; 0 keeps the response byte-identical to the indexed window.
    neighbor_expand_chars: int = 0
    #: Prepend/append the adjacent *messages* (verbatim, up to this many characters
    #: each) when a returned window starts/ends at a message boundary. Scripts and
    #: chat turns are one message per line, so a hit like "He stops... He is
    #: embarrassed" loses the speaker without the previous turn. 0 disables it.
    neighbor_message_chars: int = 0
    #: Pseudo-relevance-feedback expansion (see PrfConfig). Off by default.
    prf: PrfConfig = PrfConfig()
    ordinary: RetrievalModeConfig
    wide: RetrievalModeConfig


class EvidencePurifyConfig(FrozenConfig):
    """Optional THX-01 evidence purification (see core/evidence.py). Off by default."""

    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8095/v1/decide"
    #: Only the first N records are considered.
    max_records: int = 3
    #: A citation must clear both thresholds to be used.
    min_probability: float = 0.85
    min_chars: int = 15
    #: The state sent to the decision service is truncated to this many characters.
    max_state_chars: int = 1000
    timeout_seconds: float = 15.0


class GroupConfig(FrozenConfig):
    max_spans: int
    direct_max_tokens: int
    context_max_tokens: int
    #: Keep the first N returned records at full length and trim the rest to
    #: `tail_cut_chars`. Whether the platform truncates the response by token budget
    #: or passes it all through is unknown, and the two regimes want opposite
    #: shapes (few long records vs many short ones). A long head plus a compact tail
    #: is near-optimal under both. 0 disables the split.
    head_records: int = 0
    tail_cut_chars: int = 0
    #: 是否把响应正文里的病态空白（\n×N、\t×N、连续空格）压缩。**默认关闭**：
    #: 原文保真约定要求正文是原文切片；实测该压缩只省 0~1.4% token（XLM-R 口径 0%），
    #: 仅在明确要做平台侧 A/B 时打开。见 core/text.collapse_whitespace。
    normalize_whitespace: bool = False
    #: 正文里没有"看得见"的绝对时间时，是否按平台保留的锚点形态补一行
    #: `Session 1 (17 Jan, 2024)`（默认关闭；`Time:` 前缀会被平台删掉，所以不用那种）。
    time_anchor_in_text: bool = False
    #: 锚点的适用范围：`temporal` 只给时间/顺序类问题（默认），`all` 给所有问题。
    #: 平台侧吃锚点的叶子（C1/C2/I1）全是时间/顺序类，而 41 条可观测 query 只覆盖其中
    #: 7 条 —— 实例集在观测之外，扩大覆盖面是唯一不依赖猜测的做法。非时间题多一行
    #: 真实元数据（不是编造内容），成本约每记录 10 proxy token。
    time_anchor_scope: Literal["temporal", "all"] = "temporal"
    time_anchor_template: str = "Session {session} ({date})"
    #: How to treat the platform's injected QA-feedback messages. Their boilerplate
    #: makes them dominate dense retrieval on streaming scopes (measured: 82% of the
    #: top-10 at steps 3-5), which pushes the actual source evidence out of the answer
    #: model's view. `demote_other` moves feedback about *other* questions behind all
    #: source evidence while leaving this question's own feedback in place (it is the
    #: online-learning signal). `off` keeps the historical behaviour.
    feedback_policy: Literal["off", "demote_other"] = "off"
    #: Prepend the source message's own `Document title: ...` line to a returned
    #: window that lost it (document corpora are split into several windows; the
    #: title only sits in the first one). The line is copied verbatim from the same
    #: source message -- no synthesis -- and is a no-op when the window already has
    #: it or the message is not a titled document.
    prepend_document_title: bool = False
    #: Return the first N records as their *whole source message* (verbatim, capped at
    #: `full_message_chars`) instead of the ranked window slice. A ranked slice can
    #: cut the fact or the table row that made the message relevant; the whole message
    #: is still original text (no synthesis). 0 disables it.
    full_message_records: int = 0
    full_message_chars: int = 3000


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
    evidence_purify: EvidencePurifyConfig = EvidencePurifyConfig()
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
    rerank_base_url: str = "http://127.0.0.1:7997"
    rerank_api_key: SecretStr | None = None

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
        rerank_api_key = os.getenv("MEMORYSUM_RERANK_API_KEY")
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
            rerank_base_url=os.getenv("MEMORYSUM_RERANK_BASE_URL", "http://127.0.0.1:7997").rstrip(
                "/"
            ),
            rerank_api_key=SecretStr(rerank_api_key) if rerank_api_key else None,
        )


@lru_cache(maxsize=1)
def load_pipeline_config() -> PipelineConfig:
    """Read the packaged, versioned algorithm configuration (not env-tunable)."""
    config_path = files("memorysum").joinpath("config/champion-v1.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(payload)
