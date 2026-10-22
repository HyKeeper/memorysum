from __future__ import annotations

import logging
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from memorysum.api.app import create_app
from memorysum.providers.protocols import RerankScore
from memorysum.settings import RuntimeSettings, load_pipeline_config


class FakeProvider:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def rerank(self, query, documents):
        return [RerankScore(index, 1.0) for index, _ in enumerate(documents)]

    def close(self):
        pass


def make_pipeline():
    """FakeProvider returns 3-d vectors; declare that dimension for the Add guard."""
    pipeline = load_pipeline_config()
    return pipeline.model_copy(
        update={"embedding": pipeline.embedding.model_copy(update={"dimension": 3})}
    )


def make_runtime(data_root: Path, *, debug: bool):
    return RuntimeSettings(
        data_root=data_root,
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key=SecretStr("provider-api-secret"),
        api_auth_scheme="bearer",
        api_auth_tokens=(SecretStr("memory-system-secret"),),
        host="127.0.0.1",
        port=8001,
        debug=debug,
    )


def test_debug_logs_include_payload_but_never_credentials_or_headers(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    runtime = make_runtime(tmp_path / "data", debug=True)
    app = create_app(
        runtime=runtime,
        pipeline=make_pipeline(),
        provider=FakeProvider(),
        start_cleanup=False,
    )
    message_text = "SYNTHETIC-MESSAGE-DEBUG-ONLY"
    query_text = "SYNTHETIC-QUERY-DEBUG-ONLY"
    option_text = "A. SYNTHETIC-OPTION-DEBUG-ONLY"
    user_id = "synthetic-user-debug"
    request_id = "synthetic-request-debug"
    headers = {"Authorization": "Bearer memory-system-secret"}

    with TestClient(app) as client:
        added = client.post(
            "/add",
            headers=headers,
            json={
                "request_id": request_id,
                "user_id": user_id,
                "session_id": "synthetic-session-debug",
                "messages": [{"role": "user", "content": message_text}],
            },
        )
        searched = client.post(
            "/search",
            headers=headers,
            json={
                "query": query_text,
                "options": [option_text],
                "user_id": user_id,
                "top_k": 5,
            },
        )
    assert added.status_code == searched.status_code == 200

    logs = caplog.text
    assert "event=api_add_debug_payload" in logs
    assert "event=api_search_debug_payload" in logs
    assert message_text in logs
    assert query_text in logs
    assert option_text in logs
    assert user_id in logs
    assert request_id in logs
    assert "memory-system-secret" not in logs
    assert "provider-api-secret" not in logs
    assert "Authorization" not in logs


def test_normal_mode_does_not_log_payload_or_audit_summaries(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    app = create_app(
        runtime=make_runtime(tmp_path / "data", debug=False),
        pipeline=make_pipeline(),
        provider=FakeProvider(),
        start_cleanup=False,
    )
    marker = "SYNTHETIC-NORMAL-MODE-MUST-NOT-BE-LOGGED"
    with TestClient(app) as client:
        added = client.post(
            "/add",
            headers={"Authorization": "Bearer memory-system-secret"},
            json={
                "request_id": "normal-request-id",
                "user_id": "normal-user-id",
                "session_id": "normal-session-id",
                "messages": [{"role": "user", "content": marker}],
            },
        )
    assert added.status_code == 200
    assert marker not in caplog.text
    assert "event=api_add" not in caplog.text
    assert "event=http_request timestamp=" in caplog.text
    assert "method=POST path=/add status=200 latency_ms=" in caplog.text


def test_debug_payload_logging_is_not_gated_by_pipeline_status(tmp_path):
    pipeline = make_pipeline().model_copy(update={"qualification_status": "ready"})
    app = create_app(
        runtime=make_runtime(tmp_path / "data", debug=True),
        pipeline=pipeline,
        provider=FakeProvider(),
        start_cleanup=False,
    )
    assert app.state.pipeline.qualification_status == "ready"
