from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from memorysum.api.app import create_app
from memorysum.providers.protocols import RerankScore
from memorysum.settings import RuntimeSettings, load_pipeline_config


class FakeProvider:
    def __init__(self) -> None:
        self.embedding_batches: list[list[str]] = []
        self.rerank_batches: list[tuple[str, list[str]]] = []

    def embed(self, texts):
        self.embedding_batches.append(list(texts))
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "tea" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "coffee" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def rerank(self, query, documents):
        self.rerank_batches.append((query, list(documents)))
        normalized_query = query.lower()
        return [
            RerankScore(
                index=index,
                score=1.0 if "tea" in normalized_query and "tea" in document.lower() else 0.1,
            )
            for index, document in enumerate(documents)
        ]

    def close(self) -> None:
        pass


def make_app(data_root: Path, provider: FakeProvider):
    runtime = RuntimeSettings(
        data_root=data_root,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key=None,
        api_auth_scheme="bearer",
        api_auth_tokens=(SecretStr("test-secret"), SecretStr("alternate-test-secret")),
        host="127.0.0.1",
        port=8001,
    )
    return create_app(
        runtime=runtime,
        pipeline=load_pipeline_config(),
        provider=provider,
        start_cleanup=False,
    )


def test_add_is_synchronous_idempotent_and_user_isolated(tmp_path):
    provider = FakeProvider()
    app = make_app(tmp_path / "data", provider)
    payload = {
        "request_id": "add-1",
        "user_id": "user/one",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "I prefer tea in the morning.",
            }
        ],
    }

    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        added = client.post("/add", json=payload)
        assert added.status_code == 200
        assert added.json() == {
            "success": True,
            "request_id": "add-1",
            "user_id": "user/one",
            "session_id": "session-1",
        }
        call_count = sum(len(batch) for batch in provider.embedding_batches)
        scope = app.state.services.database.resolve_scope("user/one", create=False)
        connection = app.state.services.database.connect(scope, "user/one")
        epoch_after_first_add = connection.execute(
            "SELECT active_epoch FROM scope_manifest"
        ).fetchone()[0]
        connection.close()

        repeated = client.post("/add", json=payload)
        assert repeated.status_code == 200
        assert sum(len(batch) for batch in provider.embedding_batches) == call_count
        connection = app.state.services.database.connect(scope, "user/one")
        epoch_after_retry = connection.execute(
            "SELECT active_epoch FROM scope_manifest"
        ).fetchone()[0]
        connection.close()
        assert epoch_after_retry == epoch_after_first_add == 1

        search = client.post(
            "/search", json={"query": "What do I prefer?", "user_id": "user/one", "top_k": 10}
        )
        assert search.status_code == 200
        assert len(search.json()["data"]) == 1
        assert "I prefer tea" in search.json()["data"][0]["content"]
        assert set(search.json()["data"][0]) == {"id", "content"}

        isolated = client.post(
            "/search", json={"query": "What do I prefer?", "user_id": "user/two", "top_k": 10}
        )
        assert isolated.status_code == 200
        assert isolated.json() == {"data": []}

        conflict = client.post(
            "/add",
            json={**payload, "messages": [{"role": "user", "content": "Different payload."}]},
        )
        assert conflict.status_code == 422
        assert conflict.json()["detail"]["code"] == "request_id_payload_conflict"


def test_long_scope_uses_retrieval_and_reranker(tmp_path):
    provider = FakeProvider()
    app = make_app(tmp_path / "data", provider)
    long_text = "We had tea after the hike. The walk home was quiet. " * 1100
    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        added = client.post(
            "/add",
            json={
                "request_id": "long-add",
                "user_id": "long-user",
                "session_id": "long-session",
                "messages": [{"role": "assistant", "content": long_text}],
            },
        )
        assert added.status_code == 200
        before = len(provider.embedding_batches)
        response = client.post(
            "/search",
            json={"query": "When did we have tea?", "user_id": "long-user", "top_k": 5},
        )
        assert response.status_code == 200
        assert response.json()["data"]
        assert provider.rerank_batches
        rerank_count = len(provider.rerank_batches)
        assert len(provider.embedding_batches) > before
        assert all("tea" in item["content"].lower() for item in response.json()["data"])

        cached = client.post(
            "/search",
            json={"query": "When did we have tea?", "user_id": "long-user", "top_k": 5},
        )
        assert cached.status_code == 200
        assert cached.json() == response.json()
        assert len(provider.rerank_batches) == rerank_count


def test_search_reranks_bounded_context_groups_across_messages(tmp_path):
    provider = FakeProvider()
    app = make_app(tmp_path / "data", provider)
    filler = (
        "The team reviewed design decisions, meeting outcomes, deployment schedules, owners, "
        "due dates, and open questions. " * 8
    )
    messages = []
    for index in range(60):
        fact = "The project codename is Amber Finch. " if index == 23 else ""
        messages.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"Record {index:03d}. {fact}{filler}",
            }
        )

    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        added = client.post(
            "/add",
            json={
                "request_id": "context-add",
                "user_id": "context-user",
                "session_id": "context-session",
                "messages": messages,
            },
        )
        assert added.status_code == 200
        searched = client.post(
            "/search",
            json={
                "query": "What is the project codename?",
                "user_id": "context-user",
                "top_k": 10,
            },
        )

    assert searched.status_code == 200
    assert provider.rerank_batches
    reranked_documents = [
        document for _query, documents in provider.rerank_batches for document in documents
    ]
    assert any("Adjacent source span" in document for document in reranked_documents)
    assert any("Amber Finch" in document for document in reranked_documents)

    app = make_app(tmp_path / "data", FakeProvider())
    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        unauthorized = client.post(
            "/search",
            headers={"Authorization": "Bearer wrong-secret"},
            json={"query": "anything", "user_id": "u", "top_k": 1},
        )
        assert unauthorized.status_code == 401
        alternate_key = client.post(
            "/search",
            headers={"Authorization": "Bearer alternate-test-secret"},
            json={"query": "anything", "user_id": "u", "top_k": 1},
        )
        assert alternate_key.status_code == 200
        assert client.get("/health").status_code == 200
        empty = client.post(
            "/search",
            headers={"Authorization": "Bearer test-secret"},
            json={"query": "anything", "user_id": "u", "top_k": 0},
        )
        assert empty.status_code == 200
        assert empty.json() == {"data": []}
