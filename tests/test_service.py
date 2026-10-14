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
        return [
            RerankScore(
                index=index,
                score=(
                    1.0
                    if "tea" in document.lower()
                    and any(term in query.lower() for term in ("tea", "prefer"))
                    else 0.0005
                ),
            )
            for index, document in enumerate(documents)
        ]

    def close(self) -> None:
        pass


class FruitProvider(FakeProvider):
    def rerank(self, query, documents):
        self.rerank_batches.append((query, list(documents)))
        return [
            RerankScore(
                index=index,
                score=0.0022 if "i love apple" in document.lower() else 0.0005,
            )
            for index, document in enumerate(documents)
        ]


def make_app(
    data_root: Path,
    provider: FakeProvider,
    *,
    rerank_enabled: bool = True,
    reranker_overrides: dict | None = None,
    groups_overrides: dict | None = None,
):
    runtime = RuntimeSettings(
        data_root=data_root,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key=None,
        api_auth_scheme="bearer",
        api_auth_tokens=(SecretStr("test-secret"), SecretStr("alternate-test-secret")),
        host="127.0.0.1",
        port=8001,
    )
    pipeline = load_pipeline_config()
    if pipeline.reranker.enabled != rerank_enabled or reranker_overrides:
        pipeline = pipeline.model_copy(
            update={
                "reranker": pipeline.reranker.model_copy(
                    update={
                        "enabled": rerank_enabled,
                        **(reranker_overrides or {}),
                    }
                )
            }
        )
    if groups_overrides:
        pipeline = pipeline.model_copy(
            update={"groups": pipeline.groups.model_copy(update=groups_overrides)}
        )
    return create_app(
        runtime=runtime,
        pipeline=pipeline,
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
        assert set(search.json()["data"][0]) == {
            "id",
            "content",
            "score",
            "created_at",
            "source_ids",
        }
        # With the reranker enabled the returned score is the fused rank score
        # (1/(k+rerank_rank) + w/(k+rrf_rank)), not the raw rerank logit.
        fused_score = search.json()["data"][0]["score"]
        assert 0.0 < fused_score < 0.2
        assert search.json()["data"][0]["created_at"] == "2024-01-01T00:00:00.000Z"

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


def test_small_scope_narrow_search_returns_ranked_clean_items_and_obeys_top_k(tmp_path):
    provider = FruitProvider()
    app = make_app(tmp_path / "data", provider)
    source_messages = [
        "raw memory text",
        "I love apple",
        "I love sing",
        "i love apple【Doing】",
        "i love apple【Doing】",
    ]

    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        for index, content in enumerate(source_messages):
            message = {"role": "user", "content": content}
            if index > 0:
                message["timestamp"] = 1704067200000 + index
            added = client.post(
                "/add",
                json={
                    "request_id": f"fruit-add-{index}",
                    "user_id": "fruit-user",
                    "session_id": f"fruit-session-{index}",
                    "messages": [message],
                },
            )
            assert added.status_code == 200

        narrow = client.post(
            "/search",
            json={
                "query": "Which fruit i love?",
                "options": ["A. First answer", "B. Second answer"],
                "user_id": "fruit-user",
                "top_k": 100,
            },
        )
        narrow_top_one = client.post(
            "/search",
            json={
                "query": "Which fruit i love?",
                "options": ["A. First answer", "B. Second answer"],
                "user_id": "fruit-user",
                "top_k": 1,
            },
        )
        broad = client.post(
            "/search",
            json={"query": "List all memories", "user_id": "fruit-user", "top_k": 100},
        )
        broad_limited = client.post(
            "/search",
            json={"query": "List all memories", "user_id": "fruit-user", "top_k": 2},
        )

    assert (
        narrow.status_code
        == narrow_top_one.status_code
        == broad.status_code
        == broad_limited.status_code
        == 200
    )
    narrow_items = narrow.json()["data"]
    narrow_contents = [item["content"] for item in narrow_items]
    assert "apple" in narrow_contents[0].lower()
    assert {"raw memory text", "I love sing", "i love apple【Doing】"}.issubset(
        set(narrow_contents)
    )
    assert len(narrow_contents) == len(
        {" ".join(content.split()).casefold() for content in narrow_contents}
    )
    assert len(narrow_top_one.json()["data"]) == 1
    assert "apple" in narrow_top_one.json()["data"][0]["content"].lower()
    assert all(
        set(item) == {"id", "content", "score", "created_at", "source_ids"} for item in narrow_items
    )
    assert all(isinstance(item["score"], (int, float)) for item in narrow_items)
    assert all(isinstance(item["created_at"], str) for item in narrow_items)
    raw_item = next(item for item in narrow_items if item["content"] == "raw memory text")
    assert raw_item["created_at"].endswith("Z")
    assert narrow_items[0]["score"] >= max(item["score"] for item in narrow_items[1:])
    assert all(
        label not in item["content"]
        for item in narrow_items
        for label in ("[Source:", "Role:", "Time:")
    )
    assert provider.rerank_batches
    rerank_queries = [query for query, _documents in provider.rerank_batches]
    rerank_documents = [
        document for _query, documents in provider.rerank_batches for document in documents
    ]
    assert any("A. First answer" in query for query in rerank_queries)
    assert any("[Source: msg_" in document and "Time:" in document for document in rerank_documents)

    broad_items = broad.json()["data"]
    assert set(item["content"] for item in broad_items) == set(source_messages[:-1])
    assert all(
        set(item) == {"id", "content", "score", "created_at", "source_ids"} for item in broad_items
    )
    assert all(
        label not in item["content"]
        for item in broad_items
        for label in ("[Source:", "Role:", "Time:")
    )
    limited_items = broad_limited.json()["data"]
    assert len(limited_items) == 2
    assert all(
        set(item) == {"id", "content", "score", "created_at", "source_ids"}
        for item in limited_items
    )


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


def test_rerank_disabled_skips_provider_and_scores_with_fusion(tmp_path):
    provider = FakeProvider()
    app = make_app(tmp_path / "data", provider, rerank_enabled=False)
    payload = {
        "request_id": "add-1",
        "user_id": "no-rerank-user",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "I prefer tea in the morning.",
            },
            {
                "role": "user",
                "timestamp": 1704067201000,
                "content": "I also drink coffee in the afternoon.",
            },
        ],
    }

    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        assert client.post("/add", json=payload).status_code == 200
        searched = client.post(
            "/search",
            json={"query": "What do I prefer?", "user_id": "no-rerank-user", "top_k": 10},
        )

    assert searched.status_code == 200
    # The rerank model must never be called when the layer is disabled.
    assert provider.rerank_batches == []
    items = searched.json()["data"]
    assert items
    assert "I prefer tea" in items[0]["content"]
    # Scores come from reciprocal-rank fusion, not from a rerank logit, so every
    # value stays far below the [0, 1] range the test provider hands out.
    assert all(0.0 < item["score"] < 0.5 for item in items)


def test_long_scope_returns_fusion_order_instead_of_empty_when_no_pair_fits_the_budget(tmp_path):
    """A pair the reranker cannot score must not turn into "this scope has no memory".

    The pair budget is deliberately conservative (see README「服务拓扑」), so a long query
    can leave every candidate over it. The pipeline then skips reranking but still returns
    the fusion-ranked evidence, and says so through the strategy string.
    """
    provider = FakeProvider()
    app = make_app(
        tmp_path / "data",
        provider,
        reranker_overrides={"single_pair_proxy_tokens": 1, "ordinary_pair_candidate_groups": 1}
        if False
        else {"single_pair_proxy_tokens": 1},
    )
    payload = {
        "request_id": "add-budget",
        "user_id": "budget-user",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "I prefer tea in the morning and coffee in the afternoon. " * 4,
            }
        ],
    }

    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        assert client.post("/add", json=payload).status_code == 200
        provider.rerank_batches.clear()
        searched = client.post(
            "/search",
            json={
                "query": "What do I prefer in the morning?",
                "user_id": "budget-user",
                "top_k": 5,
            },
        )
        assert searched.status_code == 200
        body = searched.json()
        assert body["data"], "evidence must still come back, not an empty array"
        assert provider.rerank_batches == [], "no pair fits, so the reranker must not be called"


def test_search_reports_the_unreranked_fallback_in_diagnostics(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="memorysum.services.search")
    provider = FakeProvider()
    app = make_app(tmp_path / "data", provider, reranker_overrides={"single_pair_proxy_tokens": 1})
    payload = {
        "request_id": "add-diag",
        "user_id": "diag-user",
        "session_id": "session-1",
        "messages": [
            {"role": "user", "timestamp": 1704067200000, "content": "I rate the tea highly."}
        ],
    }
    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        assert client.post("/add", json=payload).status_code == 200
        assert (
            client.post(
                "/search",
                json={"query": "What about tea?", "user_id": "diag-user", "top_k": 3},
            ).status_code
            == 200
        )
    assert any("No rerankable pair" in record.message for record in caplog.records)


def _add_two_line_message(app, user_id: str) -> None:
    payload = {
        "request_id": f"add-{user_id}",
        "user_id": user_id,
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": "Tea first.\n\n\t\t\t\tThen coffee.\nSecond line follows.",
            }
        ],
    }
    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        assert client.post("/add", json=payload).status_code == 200
        searched = client.post(
            "/search", json={"query": "Tea or coffee?", "user_id": user_id, "top_k": 3}
        )
        assert searched.status_code == 200
        return client, payload, searched.json()


def test_response_body_stays_verbatim_unless_whitespace_collapse_is_requested(tmp_path):
    """原文保真：默认必须逐字返回切片，病态空白也不能动。

    `groups.normalize_whitespace` 默认关闭。开启后只压空白（\n×N/\t×N/连续空格），
    实测只省 0~1.4% token（XLM-R 口径 0%），所以默认保持原文；这个测试同时守住
    "配置项必须接线"（开了要真的生效，而不是静默无效）。
    """
    body = _add_two_line_message(make_app(tmp_path / "data", FakeProvider()), "verbatim-user")[2]
    assert body["data"], "must return the stored slice"
    assert any("\n" in item["content"] for item in body["data"])
    assert any("\t\t\t\t" in item["content"] for item in body["data"])

    collapsed = _add_two_line_message(
        make_app(
            tmp_path / "data2", FakeProvider(), groups_overrides={"normalize_whitespace": True}
        ),
        "collapsed-user",
    )[2]
    assert collapsed["data"]
    assert all(
        "\t" not in item["content"] and "\n\n" not in item["content"] for item in collapsed["data"]
    )
    assert any("Tea first." in item["content"] for item in collapsed["data"])


def test_cluster_header_ids_reach_the_response_and_survive_the_v4_backfill(tmp_path):
    """End-to-end for the bracket-cluster header fix, including the v3->v4 backfill.

    The platform's smoke scopes use `[<suite>][session_N][D<d>:<t>][text]` headers, so
    before v4 every hit carried `source_ids: []`. This covers both halves: a fresh Add
    stores the lifted ids, and an older scope (ids NULLed, user_version=3) gets them
    backfilled on the next open.
    """
    import sqlite3

    from memorysum.core.identity import scope_id

    provider = FakeProvider()
    data_root = tmp_path / "data"
    app = make_app(data_root, provider)
    payload = {
        "request_id": "add-cluster",
        "user_id": "cluster-user",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1704067200000,
                "content": (
                    "[locomo_refined][session_1][D1:3][text] Caroline: I went to a LGBTQ "
                    "support group yesterday and it was so powerful. I also prefer tea."
                ),
            }
        ],
    }
    with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
        assert client.post("/add", json=payload).status_code == 200
        found = client.post(
            "/search",
            json={
                "query": "When did Caroline go to the support group?",
                "user_id": "cluster-user",
                "top_k": 5,
            },
        ).json()
        assert found["data"], "the passage must come back"
        assert found["data"][0]["source_ids"] == ["D1:3"]

    # 模拟 v3 时代的库：抹掉 ids、把版本退回 3，再开一次应触发 v3→v4 回填
    scope_file = data_root / "scopes" / f"{scope_id('cluster-user')}.sqlite3"
    connection = sqlite3.connect(scope_file)
    connection.execute("UPDATE messages SET metadata_ids = NULL")
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()

    with TestClient(
        make_app(data_root, FakeProvider()), headers={"Authorization": "Bearer test-secret"}
    ) as client:
        again = client.post(
            "/search",
            json={
                "query": "When did Caroline go to the support group?",
                "user_id": "cluster-user",
                "top_k": 5,
            },
        ).json()
    assert again["data"][0]["source_ids"] == ["D1:3"], "backfill must restore the id"
    connection = sqlite3.connect(scope_file)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    connection.close()


def test_time_anchor_is_added_only_for_temporal_questions_and_only_when_enabled(tmp_path):
    """时间类问题的正文必须能读到绝对时间，否则"yesterday"无法解析。

    平台侧证据：`locomo1`（问失业时间）答题模型答 "Yesterday."，被判 0.0 —— 正文里只有
    相对时间。绝对时间一直在 `created_at` 字段里，但只读正文的模型看不到。这里补一行
    平台保留形态的锚点，默认关闭、且只对时间/顺序类问题生效（其余题保持逐字原文）。
    """
    payload = {
        "request_id": "add-anchor",
        "user_id": "anchor-user",
        "session_id": "session-1",
        "messages": [
            {
                "role": "user",
                "timestamp": 1706227200000,  # 2024-01-26T00:00:00Z
                "content": "[locomo_refined][session_3][D3:7][text] Caroline: I went yesterday. I prefer tea.",
            }
        ],
    }

    def search(app, query):
        with TestClient(app, headers={"Authorization": "Bearer test-secret"}) as client:
            assert client.post("/add", json=payload).status_code == 200
            body = client.post(
                "/search", json={"query": query, "user_id": "anchor-user", "top_k": 3}
            ).json()
        return " ".join(body["data"][0]["content"].split())

    off = search(
        make_app(
            tmp_path / "a", FakeProvider(), groups_overrides={"time_anchor_in_text": False}
        ),
        "When did Caroline go?",
    )
    assert "26 Jan, 2024" not in off  # 默认关闭：逐字原文
    assert off.startswith("[locomo_refined]")

    on_app = make_app(
        tmp_path / "b", FakeProvider(), groups_overrides={"time_anchor_in_text": True}
    )
    on = search(on_app, "When did Caroline go?")
    assert on.startswith("Session 3 (26 Jan, 2024)"), on
    assert "D3:7" in on, "原文切片必须保留"

    non_temporal = search(
        make_app(tmp_path / "c", FakeProvider(), groups_overrides={"time_anchor_in_text": True}),
        "What does Caroline prefer to drink?",
    )
    assert "26 Jan, 2024" not in non_temporal, "非时间类问题不该被加锚点"
