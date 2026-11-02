from __future__ import annotations

from pathlib import Path

import pytest

from memorysum.settings import RuntimeSettings


def test_runtime_loads_bearer_tokens_from_local_file(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(
        "# active callers\nfirst-secret-token\n\nsecond-secret-token\nfirst-secret-token\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", "keys.txt")
    monkeypatch.setenv("MEMORYSUM_DEBUG", "false")

    settings = RuntimeSettings.from_environment()

    assert settings.api_auth_scheme == "bearer"
    assert tuple(token.get_secret_value() for token in settings.api_auth_tokens) == (
        "first-secret-token",
        "second-secret-token",
    )


def test_runtime_requires_nonempty_bearer_key_file(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# no active keys\n\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", str(keys_file))
    with pytest.raises(ValueError, match="contains no tokens"):
        RuntimeSettings.from_environment()


def test_runtime_rejects_tokens_with_whitespace(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("bad token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", str(keys_file))
    with pytest.raises(ValueError, match="printable ASCII"):
        RuntimeSettings.from_environment()


def test_pipeline_channel_weights_reference_known_channels():
    from memorysum.settings import load_pipeline_config

    pipeline = load_pipeline_config()
    known = {"leaf_dense", "context_dense", "english_fts", "chinese_fts", "options_lexical"}
    weights = pipeline.retrieval.channel_weights
    assert set(weights) <= known, f"unknown channel in channel_weights: {set(weights) - known}"
    assert all(value > 0 for value in weights.values()), "channel weights must be positive"


def test_pipeline_falls_back_to_base_weight_for_unlisted_channels():
    from memorysum.settings import load_pipeline_config

    pipeline = load_pipeline_config()
    unlisted = {
        "leaf_dense",
        "context_dense",
        "english_fts",
        "chinese_fts",
        "options_lexical",
    } - set(pipeline.retrieval.channel_weights)
    for name in unlisted:
        assert (
            pipeline.retrieval.channel_weights.get(name, pipeline.retrieval.base_channel_weight) > 0
        )


def test_rerank_budgets_stay_under_the_hosted_per_item_limit():
    """The reranker is a hosted API with hard per-item limits.

    DashScope ``qwen3-rerank`` accepts at most 4,000 tokens per item (query or document)
    and rejects longer input with HTTP 400 instead of truncating. The pipeline compares
    those limits against tiktoken's cl100k estimate, which can undercount real tokens
    (measured worst ratio 1.494 over 1,335 real memory chunks plus adversarial samples,
    2026-10-14). Both budgets therefore have to keep ``proxy * 1.494`` under 4,000.
    """
    from memorysum.settings import load_pipeline_config

    worst_measured_ratio = 1.494
    per_item_limit = 4000

    reranker = load_pipeline_config().reranker
    assert reranker.enabled is True
    assert reranker.single_pair_proxy_tokens * worst_measured_ratio <= per_item_limit
    assert reranker.max_query_proxy_tokens * worst_measured_ratio <= per_item_limit
    # One request stays far below the documented 500-document cap.
    assert reranker.batch_groups <= 500


def test_rerank_query_cap_is_configured_and_finite():
    from memorysum.settings import load_pipeline_config

    reranker = load_pipeline_config().reranker
    # A long query still needs to fit the rerank budget together with a window's text,
    # and the response token budget shrinks with the question, so the cap stays finite.
    assert 0 < reranker.max_query_proxy_tokens <= reranker.ordinary_pair_proxy_budget


def test_whitespace_normalisation_is_off_by_default():
    """原文保真约定：响应正文默认是原文切片，空白压缩只在显式开启时生效。

    实测（2026-10-14）：\n×N/\t×N 压缩在 XLM-R 口径省 0.0% token、tiktoken 口径省
    1.4%，返回条数与命中率都不变 —— 是卫生性清理，不是分数杠杆，所以默认关闭。
    """
    from memorysum.settings import load_pipeline_config

    assert load_pipeline_config().groups.normalize_whitespace is False


def test_collapse_whitespace_is_idempotent_and_keeps_word_boundaries():
    from memorysum.core.text import collapse_whitespace

    text = "a\n\n\n\t\t\t\tb\nJoined   words  \ttab"
    once = collapse_whitespace(text)
    assert once == "a\nb\nJoined words tab"
    assert collapse_whitespace(once) == once
    # JSON 转义只在显式要求时处理（默认不改正文语义）
    assert collapse_whitespace('{\\"a\\": 1}') == '{\\"a\\": 1}'
    assert collapse_whitespace('{\\"a\\": 1}', unescape=True) == '{"a": 1}'


def test_packaged_pipeline_calls_hosted_apis_only():
    """The shipped config must not depend on a model running on the same machine."""
    from memorysum.settings import load_pipeline_config

    pipeline = load_pipeline_config()
    hosted = {"dashscope", "openrouter"}
    assert pipeline.embedding.provider in hosted
    assert pipeline.reranker.provider in hosted
    assert pipeline.evidence_purify.enabled is False


def test_runtime_settings_expose_no_local_model_endpoints():
    import dataclasses

    from memorysum.settings import RuntimeSettings

    fields = {field.name for field in dataclasses.fields(RuntimeSettings)}
    assert not {
        name for name in fields if "embedding_base_url" in name or "rerank_base_url" in name
    }


def test_evidence_purification_requires_an_explicit_endpoint():
    from memorysum.settings import EvidencePurifyConfig

    default = EvidencePurifyConfig()
    assert default.enabled is False
    assert default.endpoint == ""
    with pytest.raises(ValueError, match="endpoint is required"):
        EvidencePurifyConfig(enabled=True)
    enabled = EvidencePurifyConfig(enabled=True, endpoint="https://decision.example/v1/decide")
    assert enabled.endpoint.startswith("https://")
