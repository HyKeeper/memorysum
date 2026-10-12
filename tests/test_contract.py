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
