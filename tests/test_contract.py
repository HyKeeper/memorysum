from __future__ import annotations

import pytest
from pydantic import ValidationError

from memorysum.api.schemas import AddRequest, SearchRequest


def add_payload(**message_overrides):
    message = {"role": "user", "content": "I prefer tea."}
    message.update(message_overrides)
    return {
        "request_id": "request-1",
        "messages": [message],
        "user_id": "opaque User/1",
        "session_id": "session-1",
    }


def test_add_payload_hash_is_stable_and_preserves_timestamp_presence():
    missing_timestamp = AddRequest.model_validate(add_payload())
    same_payload = AddRequest.model_validate(add_payload())
    explicit_timestamp = AddRequest.model_validate(add_payload(timestamp=0))

    assert missing_timestamp.payload_digest() == same_payload.payload_digest()
    assert missing_timestamp.payload_digest() != explicit_timestamp.payload_digest()
    assert missing_timestamp.messages[0].content == "I prefer tea."


def test_add_echo_fields_are_not_normalized():
    payload = AddRequest.model_validate(
        {
            "request_id": " ID ",
            "messages": [{"role": "assistant", "content": "  exact content  "}],
            "user_id": " User/One ",
            "session_id": " Session/One ",
        }
    )
    assert payload.request_id == " ID "
    assert payload.user_id == " User/One "
    assert payload.session_id == " Session/One "
    assert payload.messages[0].content == "  exact content  "


@pytest.mark.parametrize(
    "overrides",
    [
        {"timestamp": None},
        {"timestamp": True},
        {"timestamp": 1.5},
        {"content": "   "},
        {"content": "bad\x00content"},
    ],
)
def test_add_rejects_invalid_or_unknown_fields(overrides):
    with pytest.raises(ValidationError):
        AddRequest.model_validate(add_payload(**overrides))


def test_unknown_top_level_fields_are_ignored_not_rejected():
    """未声明的入参字段要**忽略**，不能 422。

    2026-10-14 实测：平台会带 `session_id` / `filters` / `metadata` / `request_id` 来调 `/search`，
    而旧行为（`extra="forbid"`）把它们变成 422；平台文档写明 422 **不重试**且"当前阶段立即失败"，
    日志里全天累计 163 次。宽容处理不改变语义（我们只读声明的字段），但不会让平台阶段失败。
    """
    extra = {
        "session_id": "s1",
        "filters": {"a": 1},
        "metadata": {"b": 2},
        "request_id": "r1",
        "keyword_search": True,
        "rerank": False,
    }
    request = SearchRequest.model_validate({"query": "hello", "user_id": "u1", "top_k": 5, **extra})
    assert request.query == "hello" and request.top_k == 5
    add = AddRequest.model_validate({**add_payload(), **extra})
    assert add.messages[0].content == "I prefer tea."
    # 声明过的字段仍然严格：类型不对照样拒绝
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": "hello", "user_id": "u1", "top_k": "5"})


def test_search_strictly_validates_top_k_and_options():
    base = {"query": "What do I prefer?", "user_id": "u", "top_k": 0}
    assert SearchRequest.model_validate(base).top_k == 0
    for invalid in (-1, True, 1.0, "10"):
        with pytest.raises(ValidationError):
            SearchRequest.model_validate({**base, "top_k": invalid})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({**base, "options": [""]})
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({**base, "options": ["A"] * 33})


def test_wire_limits_track_pipeline_config():
    """API 长度上限必须来自 pipeline config，而不是散落的硬编码常量。"""
    from memorysum.api import schemas
    from memorysum.settings import load_pipeline_config

    capacity = load_pipeline_config().capacity
    assert schemas._CAPACITY is capacity

    def max_length(model, field_name):
        for meta in model.model_fields[field_name].metadata:
            if hasattr(meta, "max_length"):
                return meta.max_length
        return None

    assert max_length(AddRequest, "messages") == capacity.max_add_messages
    assert max_length(SearchRequest, "options") == capacity.max_search_options


def test_pipeline_config_has_no_unread_fields():
    """config 里不应再出现“声明了但代码从不读取”的装饰性字段。"""
    from memorysum.settings import load_pipeline_config

    dumped = load_pipeline_config().model_dump()
    forbidden = {
        "features",
        "language",
        "generation_enabled",
    }
    assert not (set(dumped) & forbidden), (
        f"decorative top-level field reintroduced: {set(dumped) & forbidden}"
    )
    assert "sparse_output" not in dumped["embedding"]
    assert "tokenizer" not in dumped["reranker"]
    assert "tokenizer_is_approximate" not in dumped["reranker"]
    assert "token_proxy" not in dumped["chunking"]
    assert "unused_quota_transfer" not in dumped["retrieval"]
    assert "sparse_channel" not in dumped["retrieval"]
    assert "reader_estimator" not in dumped["coverage"]


def test_time_anchor_scope_stays_declared_and_typed():
    """锚点覆盖面是平台叶子（C1/C2/I1）的开关，必须是显式配置且非法值报错。

    背景：这三片叶子在 41 条可观测 query 里只覆盖到 7 条，所以“覆盖面”本身是一个
    实验变量。若有人把这个键从 config 里删掉，行为会静默回到默认 `temporal` —— 等于
    无声地撤销实验。所以要同时守护“显式声明”与“非法值报错”（而不是静默降级）。
    """
    import json
    from importlib.resources import files

    from memorysum.settings import GroupConfig, load_pipeline_config

    payload = json.loads(
        files("memorysum").joinpath("config/champion-v1.json").read_text(encoding="utf-8")
    )
    assert "time_anchor_scope" in payload["groups"], "锚点覆盖面必须在 config 里显式声明"

    groups = load_pipeline_config().groups
    assert groups.time_anchor_scope in {"temporal", "all"}
    with pytest.raises(ValidationError):
        GroupConfig(**{**groups.model_dump(), "time_anchor_scope": "everything"})


def test_query_expansion_is_off_by_default_and_gated_by_top_k():
    """查询扩展（PRF）是叠加层：**默认关闭**，且 `top_k < min_top_k` 时不触发。

    它是在 74.90 那版之上加的，所以必须证明"关着的时候什么都没变"：
    默认值关、阈值挡住小 top_k、以及两个新配置的默认值都不改变历史行为。
    """
    from memorysum.core.expansion import blend_query_vector, expansion_weights
    from memorysum.settings import load_pipeline_config

    config = load_pipeline_config()
    assert config.query_expansion.enabled is False, "PRF 必须默认关闭"
    assert config.query_expansion.min_top_k >= 30, "小 top_k 查询要关掉扩展（用户要求）"
    assert config.retrieval.channel_consensus_bonus == 0.0, "共识奖励默认关"
    assert config.retrieval.rrf_k == 20 and config.retrieval.rrf_k_adaptive is True

    # 扩展词：排除已在 query 里的词，并按 TF/(1+DF) 排序
    df_table = {t: 5 for t in ("quantum", "plutonium", "alignment", "capacitor")}
    terms = expansion_weights(
        ["the quantum flux capacitor needs a plutonium core", "plutonium core alignment"],
        "what does the flux capacitor need",
        lambda term: df_table.get(term, 0),  # 表外词 DF=0（真实实现走 FTS 索引）
        limit=3,
        total_windows=1000,
    )
    picked = [term for term, _ in terms]
    assert "capacitor" not in picked and "flux" not in picked, "query 里已有的词不该重复加"
    assert "plutonium" in picked

    # 向量插值：alpha=0 时原样返回（关掉扩展的语义等价物）
    base = [1.0, 0.0, 0.0]
    assert blend_query_vector(base, [[0.0, 1.0, 0.0]], 0.0) == base
    blended = blend_query_vector(base, [[0.0, 1.0, 0.0]], 0.4)
    assert abs(sum(v * v for v in blended) - 1.0) < 1e-9, "插值后必须重新归一化"
    assert blended[1] > 0, "扩展词的方向应当被混入"


def test_prf_merge_policies_append_never_reorders_base():
    """`merge_policy="append"` 必须保证原排名逐位不变（用户设计意图的严格版）。

    实测（BEAM 60 题）：`score` 模式下新候选会顶掉原 top-1（txt1 49→46），
    而 `append` 把新条目全部放在原排名之后——原指标严格无损，只在 top_k /
    token 预算允许时多交几条证据。
    """
    from types import SimpleNamespace

    from memorysum.services.search import _merge_ranked

    def entry(view_id):
        return SimpleNamespace(view_id=view_id, candidate=SimpleNamespace(best_rank=1))

    base = [(entry("a"), 0.5), (entry("b"), 0.4), (entry("c"), 0.3)]
    extra = [(entry("d"), 0.9), (entry("a"), 0.95), (entry("e"), 0.8)]

    appended = _merge_ranked(base, extra, "append")
    assert [item.view_id for item, _score in appended] == ["a", "b", "c", "d", "e"]
    scores = {item.view_id: score for item, score in appended}
    assert scores["a"] == 0.5, "append 模式不得改写原有条目的分数"

    scored = _merge_ranked(base, extra, "score")
    assert [item.view_id for item, _score in scored] == ["d", "e", "a", "b", "c"], (
        "score 模式按分数降序；原条目保留 base 里的打分（先见先保留）"
    )
    assert {item.view_id: score for item, score in scored}["a"] == 0.5
