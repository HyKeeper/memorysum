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
        {"extra": "not allowed"},
    ],
)
def test_add_rejects_invalid_or_unknown_fields(overrides):
    with pytest.raises(ValidationError):
        AddRequest.model_validate(add_payload(**overrides))


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
