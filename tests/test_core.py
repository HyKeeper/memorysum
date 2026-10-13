from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from memorysum.core.errors import ProviderError
from memorysum.core.ranking import reciprocal_rank_fusion
from memorysum.core.rendering import render_response_spans
from memorysum.core.vector import normalize_batch
from memorysum.providers.factory import build_provider, embedding_endpoint
from memorysum.providers.local_openai import LocalEmbeddingProvider
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import RerankScore
from memorysum.providers.split import SplitProvider
from memorysum.services.search import SearchService
from memorysum.settings import RuntimeSettings, load_pipeline_config


def test_response_renderer_preserves_tag_like_source_text_verbatim():
    source_text = "[Source: actually in source]\nRole: user\nTime: literal text"
    content, _view_id, spans = render_response_spans(
        [
            {
                "source_id": "source-1",
                "public_source_id": "source-1",
                "start_char": 0,
                "end_char": len(source_text),
                "raw_content": source_text,
            }
        ]
    )

    assert content == source_text
    assert spans == [{"source_id": "source-1", "start_char": 0, "end_char": len(source_text)}]


def test_rrf_merges_channels_and_uses_stable_ties():
    results = reciprocal_rank_fusion(
        [
            ("dense", [2, 1], 1.0),
            ("lexical", [1, 2], 1.0),
        ],
        k=60,
        limit=10,
    )
    assert [item.window_id for item in results] == [1, 2]
    assert results[0].channels == {"dense", "lexical"}
    assert math.isclose(results[0].score, 1 / 62 + 1 / 61)


def test_embedding_response_parser_checks_batch_order_and_shape():
    body = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    assert OpenRouterProvider._parse_embeddings(body, 2) == [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ProviderError):
        OpenRouterProvider._parse_embeddings({"data": [{"index": 0, "embedding": []}]}, 1)


def test_rerank_parser_preserves_document_indexes():
    parsed = OpenRouterProvider._parse_rerank(
        {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        },
        2,
    )
    assert parsed == [RerankScore(1, 0.9), RerankScore(0, 0.2)]
    with pytest.raises(ProviderError):
        OpenRouterProvider._parse_rerank({"results": [{"index": 0, "score": float("nan")}]}, 1)


def test_vectors_are_l2_normalized_and_zero_vectors_rejected():
    normalized, dimension = normalize_batch([[3.0, 4.0]], 1)
    assert dimension == 2
    assert normalized[0] == [0.6, 0.8]
    with pytest.raises(ProviderError):
        normalize_batch([[0.0, 0.0]], 1)


def test_openrouter_adapter_uses_expected_wire_contract_without_network():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]},
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            },
        )

    runtime = RuntimeSettings(
        data_root=Path("/tmp/memorysum-provider-test"),
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key=SecretStr("test-provider-key"),
        api_auth_scheme="bearer",
        api_auth_tokens=(SecretStr("test-memory-system-key"),),
        host="127.0.0.1",
        port=8001,
    )
    provider = OpenRouterProvider(
        runtime,
        load_pipeline_config(),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert provider.embed(["source passage"])[0] == [3.0, 4.0]
        assert provider.rerank("question", ["first", "second"]) == [
            RerankScore(1, 0.9),
            RerankScore(0, 0.1),
        ]
    finally:
        provider.close()

    embedding_request = json.loads(requests[0].content)
    rerank_request = json.loads(requests[1].content)
    assert requests[0].url.path == "/api/v1/embeddings"
    assert embedding_request["input"] == "source passage"
    assert embedding_request["encoding_format"] == "float"
    assert embedding_request["model"] == load_pipeline_config().embedding.model
    assert requests[1].url.path == "/api/v1/rerank"
    assert rerank_request["documents"] == [{"text": "first"}, {"text": "second"}]
    assert rerank_request["top_n"] == 2
    assert requests[0].headers["authorization"] == "Bearer test-provider-key"


def _local_runtime(**overrides) -> RuntimeSettings:
    values = {
        "data_root": Path("/tmp/memorysum-local-provider-test"),
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_api_key": SecretStr("test-provider-key"),
        "api_auth_scheme": "bearer",
        "api_auth_tokens": (SecretStr("test-memory-system-key"),),
        "host": "127.0.0.1",
        "port": 8001,
        "embedding_base_url": "http://127.0.0.1:7999/v1",
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def test_local_embedding_adapter_uses_openai_compatible_contract_without_network():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    provider = LocalEmbeddingProvider(
        _local_runtime(embedding_api_key=None),
        load_pipeline_config(),
        transport=httpx.MockTransport(handler),
    )
    try:
        assert provider.embed(["source passage"])[0] == [3.0, 4.0]
        with pytest.raises(ProviderError) as error:
            provider.rerank("question", ["first"])
        assert error.value.code == "model_provider_not_configured"
    finally:
        provider.close()

    request = requests[0]
    assert request.url.host == "127.0.0.1"
    assert request.url.port == 7999
    assert request.url.path == "/v1/embeddings"
    payload = json.loads(request.content)
    assert payload["model"] == load_pipeline_config().embedding.model
    assert payload["input"] == ["source passage"]
    assert payload["encoding_format"] == "float"
    assert "authorization" not in request.headers


def test_local_embedding_adapter_sends_optional_bearer_key():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    provider = LocalEmbeddingProvider(
        _local_runtime(embedding_api_key=SecretStr("local-embedding-secret")),
        load_pipeline_config(),
        transport=httpx.MockTransport(handler),
    )
    try:
        provider.embed(["passage"])
    finally:
        provider.close()

    assert requests[0].headers["authorization"] == "Bearer local-embedding-secret"


def test_provider_factory_splits_local_embeddings_from_remote_rerank():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [3.0, 4.0]}]})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})

    pipeline = load_pipeline_config()
    # Pin the remote reranker explicitly so this test does not depend on the
    # packaged config's current choice.
    pipeline = pipeline.model_copy(
        update={
            "reranker": pipeline.reranker.model_copy(
                update={"provider": "openrouter", "enabled": True}
            )
        }
    )
    provider = build_provider(
        _local_runtime(),
        pipeline,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert isinstance(provider, SplitProvider)
        assert provider.embed(["passage"]) == [[3.0, 4.0]]
        assert provider.rerank("question", ["first"]) == [RerankScore(0, 0.7)]
    finally:
        provider.close()

    assert calls[0] == "http://127.0.0.1:7999/v1/embeddings"
    assert calls[1] == "https://openrouter.ai/api/v1/rerank"


def test_provider_factory_serves_local_rerank_endpoint_with_plain_list_response():
    """Local embedding + local rerank must hit both local endpoints.

    Also covers the bare-array response shape used by plain FastAPI rerank
    services (``[{"index": i, "score": s}]``) rather than the OpenRouter shape.
    """
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "body": json.loads(request.content)})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
        return httpx.Response(200, json=[{"index": 0, "score": 0.9, "document": "first"}])

    pipeline = load_pipeline_config()
    pipeline = pipeline.model_copy(
        update={
            "reranker": pipeline.reranker.model_copy(
                update={"provider": "local-openai", "enabled": True}
            )
        }
    )
    provider = build_provider(
        _local_runtime(),
        pipeline,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert provider.embed(["passage"]) == [[1.0, 2.0]]
        assert provider.rerank("question", ["first"]) == [RerankScore(0, 0.9)]
    finally:
        provider.close()

    assert calls[0]["url"] == "http://127.0.0.1:7999/v1/embeddings"
    assert calls[1]["url"] == "http://127.0.0.1:7997/rerank"
    assert calls[1]["body"]["query"] == "question"
    assert calls[1]["body"]["documents"] == ["first"]


def test_embedding_endpoint_follows_the_configured_provider():
    pipeline = load_pipeline_config()
    runtime = _local_runtime()
    assert pipeline.embedding.provider == "local-openai"
    assert embedding_endpoint(runtime, pipeline) == "http://127.0.0.1:7999/v1"

    remote_pipeline = pipeline.model_copy(
        update={"embedding": pipeline.embedding.model_copy(update={"provider": "openrouter"})}
    )
    assert embedding_endpoint(runtime, remote_pipeline) == "https://openrouter.ai/api/v1"


def test_factory_rejects_unknown_provider_in_config():
    pipeline = load_pipeline_config()
    broken = pipeline.model_copy(
        update={"reranker": pipeline.reranker.model_copy(update={"provider": "mystery"})}
    )
    with pytest.raises(ValueError, match="Unsupported model provider"):
        build_provider(_local_runtime(), broken)


def test_rerank_fusion_blends_rerank_and_retrieval_rankings():
    """融合排序：既不忽略精排，也不丢检索侧的多通道共识。"""
    from memorysum.core.ranking import FusedCandidate
    from memorysum.services.search import RerankEntry

    def entry(window_id: int, rrf_score: float, best_rank: int) -> RerankEntry:
        return RerankEntry(
            candidate=FusedCandidate(
                window_id=window_id, score=rrf_score, best_rank=best_rank, channels={"leaf_dense"}
            ),
            view_id=f"view-{window_id}",
            rerank_content=f"doc {window_id}",
            response_content=f"doc {window_id}",
            spans=[],
            created_at_ms=0,
            pair_proxy_tokens=10,
        )

    # A: 检索侧最强（RRF rank 1），精排最弱（rank 3）
    # C: 精排最强（rank 1），检索侧最弱（rank 3）
    # B: 两边都居中
    entries = [
        entry(1, 0.030, 1),
        entry(2, 0.020, 2),
        entry(3, 0.010, 3),
    ]
    scores = [0.1, 0.5, 0.9]  # 精排把 C 排第一

    service = SearchService.__new__(SearchService)  # 只测纯函数，不建完整栈

    class _Pipeline:
        class reranker:
            fusion_weight = 2.0

        class retrieval:
            rrf_k = 60

    service._pipeline = _Pipeline()
    ordered = service._fuse_rankings(entries, scores)

    assert len(ordered) == 3
    # 融合分公式：1/(60+rerank_rank) + w/(60+retrieval_rank)，w=2.0
    # A: 1/63 + 2/61 = 0.04866   B: 1/62 + 2/62 = 0.04839   C: 1/61 + 2/63 = 0.04814
    assert [pair[0].candidate.window_id for pair in ordered] == [1, 2, 3]
    assert service._pipeline.reranker.fusion_weight == 2.0
