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
    #: rerank 分数量化粒度（0.0 = 关闭）。>0 时把 cross-encoder 分数 round 到该粒度
    #: 再参与排序：llama.cpp GPU 批推理（连续 batching 混合并发请求）的 run-to-run
    #: 分数波动实测 ±0.02~0.06，粒度 0.1 可把边缘 pair 的排名抖动吸收成确定性 tie
    #: （tie-break 走 best_rank/window_id，本身确定）。实测（2026-10-15）：0.1 在两题
    #: 并发下把 Reach 24 题前 13 位锁死（含 few-shot 示例关键位 6/7），0.05 仍抖、
    #: ≥0.15 制造过多 tie 反而在别处抖。0.0 保持逐字使用模型返回分数。
    score_quantization: float = 0.0


class RetrievalModeConfig(FrozenConfig):
    leaf_dense: int
    context_dense: int
    english_fts: int
    chinese_fts: int
    options_lexical: int
    candidate_cap: int
    output_records: int
    output_tokens: int


class QueryExpansionConfig(FrozenConfig):
    """伪相关反馈（RM3 式）的查询扩展参数；机制见 `core/expansion.py`。

    **不引入任何生成式模型**：只有词频、文档频次与现成的 embedding。
    """

    enabled: bool = False
    #: 调用方要的结果少于这个数就不扩展：`top_k` 小说明只要少数结果，
    #: 此时扩展带来的漂移风险大于收益。
    min_top_k: int = 30
    #: 二次检索的初始召回条数（用户的 N）。
    recall: int = 100
    #: 扩展词个数 m。
    terms: int = 8
    #: 扩展权重 α（向量插值的系数）。
    weight: float = 0.4
    #: 门控阈值：只取 rerank 分数在该分位之上的文档当伪相关集。
    gate_percentile: int = 70
    #: 伪相关集小于这个数就整体回退到原始查询。
    min_related: int = 3
    #: 二次检索最多引入多少条新候选（防止新结果反客为主）。
    max_new: int = 40
    #: 丢掉出现在超过这个比例窗口里的词（区分度为零）。
    max_df_ratio: float = 0.35
    #: >0 时启用漂移哨兵：扩展词与原始查询向量的平均余弦低于它则回退。
    drift_guard: float = 0.0
    #: 新候选的插入策略：`score` = 与原有候选同池按分数排序（可能顶替原排名）；
    #: `append` = 新候选只追加在原排名之后（严格保证原排序不被改写，只在 top_k
    #: 与 token 预算允许时多交几条证据）。用户设计意图“扩出的新答案不应顶替
    #: 高相似度答案”对应 `append`；实测 `score` 会掉 top-1（见 sweep 台账）。
    merge_policy: str = "score"


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
    #: 时间/顺序类问题是否把返回里已有的会话锚点排成时间线，前置到响应最前面
    #: （qa_ab.py 的 C 方案，`Conversation timeline: Session N = <date>`）。
    #: **默认关闭**；只重排已有锚点、不新增事实，靶子是 C1/C2/I1 的顺序题。
    timeline_in_text: bool = False
    #: 锚点的适用范围：`temporal` 只给时间/顺序类问题（默认），`all` 给所有问题。
    #: 平台侧吃锚点的叶子（C1/C2/I1）全是时间/顺序类，而 41 条可观测 query 只覆盖其中
    #: 7 条 —— 实例集在观测之外，扩大覆盖面是唯一不依赖猜测的做法。非时间题多一行
    #: 真实元数据（不是编造内容），成本约每记录 10 proxy token。
    time_anchor_scope: Literal["temporal", "all"] = "temporal"
    time_anchor_template: str = "Session {session} ({date})"


class CoverageConfig(FrozenConfig):
    always_full_tokens: int
    conditional_full_tokens: int
    conditional_full_requires_wide_or_context_mode: bool
    #: 小于 `always_full_tokens` 的 scope 是否也对**普通查询**开放全量交付。
    #: 默认 False = 保持历史行为（只对 wide/context 查询开放）。设为 True 的实验假设：
    #: 小 scope 的全部内容都放得下（≤4096 token），而窗口切分会把“昨天 → 02 Jan”
    #: 这类跨窗上下文切碎——locomo 时间题恰好都是小 scope（29~700 条短消息）。
    #: 语料消息长（>窗口）时不开，因为整条消息会稀释 rerank 信号（实测 full_bundle
    #: 对长消息曾把 gold 从第 1 名砸到第 19 名）。
    always_full_for_all_queries: bool = False
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
    query_expansion: QueryExpansionConfig = QueryExpansionConfig()
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
