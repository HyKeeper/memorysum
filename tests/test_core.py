from __future__ import annotations

import json
import math
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from memorysum.core.errors import ProviderError
from memorysum.core.ranking import reciprocal_rank_fusion
from memorysum.core.vector import normalize_batch
from memorysum.providers.openrouter import OpenRouterProvider
from memorysum.providers.protocols import RerankScore
from memorysum.settings import RuntimeSettings, load_pipeline_config


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
