from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from memorysum.core.errors import ProviderError
from memorysum.core.text import compose_embedding_input
from memorysum.providers.dashscope import (
    MAX_BATCH_ITEMS,
    SUPPORTED_DIMENSIONS,
    DashScopeEmbeddingProvider,
)
from memorysum.providers.factory import build_provider
from memorysum.providers.split import SplitProvider
from memorysum.settings import RuntimeSettings, load_pipeline_config

_EMBEDDING_PATH = "/api/v1/services/embeddings/text-embedding/text-embedding"


def _runtime(**overrides) -> RuntimeSettings:
    values = {
        "data_root": Path("/tmp/memorysum-dashscope-test"),
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openrouter_api_key": None,
        "api_auth_scheme": "bearer",
        "api_auth_tokens": (SecretStr("test-memory-key"),),
        "host": "127.0.0.1",
        "port": 8001,
        "dashscope_base_url": "https://example.maas.aliyuncs.com",
        "dashscope_api_key": SecretStr("test-dashscope-key"),
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def _pipeline(*, dimension: int = 64, batch_items: int = MAX_BATCH_ITEMS):
    pipeline = load_pipeline_config()
    return pipeline.model_copy(
        update={
            "embedding": pipeline.embedding.model_copy(
                update={"dimension": dimension, "batch_items": batch_items}
            )
        }
    )


def _vector(index: int, dimension: int) -> list[float]:
    values = [0.0] * dimension
    values[index] = 1.0
    return values


def test_dashscope_uses_native_wire_contract_and_orders_text_indexes():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"text_index": 1, "embedding": _vector(1, 64)},
                        {"text_index": 0, "embedding": _vector(0, 64)},
                    ]
                },
                "usage": {"total_tokens": 4},
                "request_id": "req-1",
            },
        )

    provider = DashScopeEmbeddingProvider(
        _runtime(), _pipeline(), transport=httpx.MockTransport(handler)
    )
    try:
        vectors = provider.embed(["first passage", "second passage"])
    finally:
        provider.close()

    assert vectors == [_vector(0, 64), _vector(1, 64)]
    request = requests[0]
    assert request.url.host == "example.maas.aliyuncs.com"
    assert request.url.path == _EMBEDDING_PATH
    assert request.headers["authorization"] == "Bearer test-dashscope-key"
    payload = json.loads(request.content)
    assert payload == {
        "model": "text-embedding-v4",
        "input": {"texts": ["first passage", "second passage"]},
        "dimension": 64,
    }


def test_dashscope_rejects_a_batch_over_the_server_limit_without_calling_the_api():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"output": {"embeddings": []}})

    provider = DashScopeEmbeddingProvider(
        _runtime(), _pipeline(), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.embed([f"text {index}" for index in range(MAX_BATCH_ITEMS + 1)])
        assert error.value.code == "model_provider_invalid_embedding"
        assert calls == []
    finally:
        provider.close()


def test_dashscope_requires_an_explicit_supported_dimension():
    with pytest.raises(ValueError, match="explicit integer"):
        DashScopeEmbeddingProvider(
            _runtime(),
            _pipeline(dimension="auto-detect-on-first-successful-embedding"),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )
    with pytest.raises(ValueError, match="not supported"):
        DashScopeEmbeddingProvider(
            _runtime(),
            _pipeline(dimension=1000),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )
    assert 2048 in SUPPORTED_DIMENSIONS


def test_dashscope_rejects_a_batch_items_setting_above_the_server_limit():
    with pytest.raises(ValueError, match=f"limit of {MAX_BATCH_ITEMS}"):
        DashScopeEmbeddingProvider(
            _runtime(),
            _pipeline(batch_items=MAX_BATCH_ITEMS + 1),
            transport=httpx.MockTransport(lambda request: httpx.Response(500)),
        )


def test_dashscope_missing_api_key_is_a_non_2xx_provider_error():
    provider = DashScopeEmbeddingProvider(
        _runtime(dashscope_api_key=None),
        _pipeline(),
        transport=httpx.MockTransport(lambda request: httpx.Response(500)),
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.embed(["passage"])
        assert error.value.code == "model_provider_not_configured"
        assert error.value.status_code == 503
    finally:
        provider.close()


def test_dashscope_refuses_a_dimension_the_pipeline_did_not_declare():
    """Regression for the public DashScope endpoint returning 1024 for a 2048 request.

    Persisting those vectors would build vec0 tables and a scope manifest for 2048 while
    every stored vector is 1024-d. Fail instead of silently changing the embedding space.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": {"embeddings": [{"text_index": 0, "embedding": _vector(0, 8)}]}},
        )

    provider = DashScopeEmbeddingProvider(
        _runtime(), _pipeline(dimension=64), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(ProviderError) as error:
            provider.embed(["passage"])
        assert error.value.code == "model_provider_invalid_embedding"
        assert "64" in error.value.message and "8" in error.value.message
    finally:
        provider.close()


def test_factory_wires_dashscope_embedding_with_the_local_reranker():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == _EMBEDDING_PATH:
            return httpx.Response(
                200,
                json={"output": {"embeddings": [{"text_index": 0, "embedding": _vector(0, 64)}]}},
            )
        return httpx.Response(200, json=[{"index": 0, "score": 0.9}])

    provider = build_provider(_runtime(), _pipeline(), transport=httpx.MockTransport(handler))
    try:
        assert isinstance(provider, SplitProvider)
        assert provider.embed(["passage"]) == [_vector(0, 64)]
        assert provider.rerank("question", ["first"])[0].score == 0.9
    finally:
        provider.close()

    assert calls[0].url.host == "example.maas.aliyuncs.com"
    assert calls[0].url.path == _EMBEDDING_PATH
    assert calls[1].url.host == "127.0.0.1"
    assert calls[1].url.port == 7997
    assert calls[1].url.path == "/rerank"


def test_factory_rejects_dashscope_as_a_reranker():
    pipeline = load_pipeline_config()
    pipeline = pipeline.model_copy(
        update={"reranker": pipeline.reranker.model_copy(update={"provider": "dashscope"})}
    )
    with pytest.raises(ValueError, match="embeddings only"):
        build_provider(_runtime(), pipeline)


def test_packaged_config_pins_text_embedding_v4():
    pipeline = load_pipeline_config()
    assert pipeline.pipeline_id == "memorysum-hybrid-dashscope-v0.6.0"
    assert pipeline.embedding.provider == "dashscope"
    assert pipeline.embedding.model == "text-embedding-v4"
    assert pipeline.embedding.dimension == 2048
    assert pipeline.embedding.query_instruction == ""
    assert pipeline.embedding.batch_items <= MAX_BATCH_ITEMS
    assert pipeline.embedding.dimension in SUPPORTED_DIMENSIONS
    # The provider/model/dimension are part of the data contract: switching embeddings
    # must invalidate every stored scope instead of mixing vector spaces.
    local = pipeline.model_copy(
        update={"embedding": pipeline.embedding.model_copy(update={"provider": "local-openai"})}
    )
    other_model = pipeline.model_copy(
        update={"embedding": pipeline.embedding.model_copy(update={"model": "text-embedding-v3"})}
    )
    other_dimension = pipeline.model_copy(
        update={"embedding": pipeline.embedding.model_copy(update={"dimension": 1024})}
    )
    assert (
        len(
            {
                pipeline.embedding_space_id,
                local.embedding_space_id,
                other_model.embedding_space_id,
                other_dimension.embedding_space_id,
            }
        )
        == 4
    )


def test_database_fingerprint_includes_the_embedding_endpoint(tmp_path: Path):
    """`api/app.py` mixes the endpoint into the scope fingerprint, so a base-URL change
    must invalidate old scopes even when provider/model/dimension are unchanged."""
    from memorysum.api.app import create_app

    class NoopProvider:
        def embed(self, texts):
            return [[1.0] for _ in texts]

        def rerank(self, query, documents):
            return []

        def close(self):
            pass

    pipeline = load_pipeline_config()
    fingerprints = []
    for name, host in (("a", "https://a.example"), ("b", "https://b.example")):
        app = create_app(
            runtime=_runtime(data_root=tmp_path / name, dashscope_base_url=host),
            pipeline=pipeline,
            provider=NoopProvider(),
            start_cleanup=False,
        )
        fingerprints.append(app.state.services.database.embedding_space_id)

    assert fingerprints[0] != fingerprints[1]


def test_runtime_reads_dashscope_credentials_and_base_url(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("inbound-token\n", encoding="utf-8")
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", str(keys_file))
    monkeypatch.setenv("MEMORYSUM_DEBUG", "false")
    monkeypatch.setenv("MEMORYSUM_DASHSCOPE_BASE_URL", "https://dedicated.example/")
    monkeypatch.delenv("MEMORYSUM_DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-fallback")

    settings = RuntimeSettings.from_environment()
    assert settings.dashscope_base_url == "https://dedicated.example"
    assert settings.dashscope_api_key is not None
    assert settings.dashscope_api_key.get_secret_value() == "dashscope-fallback"

    monkeypatch.setenv("MEMORYSUM_DASHSCOPE_API_KEY", "namespaced-wins")
    settings = RuntimeSettings.from_environment()
    assert settings.dashscope_api_key is not None
    assert settings.dashscope_api_key.get_secret_value() == "namespaced-wins"


def test_empty_instruction_embeds_the_query_verbatim():
    assert compose_embedding_input("", "what is the policy?") == "what is the policy?"
    assert compose_embedding_input(None, "what is the policy?") == "what is the policy?"
    assert (
        compose_embedding_input("Instruct: retrieve\nQuery:", "what is the policy?")
        == "Instruct: retrieve\nQuery:\n\nwhat is the policy?"
    )
