"""Competition response rendering from immutable source spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memorysum.core.identity import stable_view_id
from memorysum.core.text import reader_token_count, utc_timestamp

RENDERER_VERSION = "memorysum-source-render-v1"
FULL_BUNDLE_SEPARATOR = "\n\n---\n\n"


def render_message(source_id: str, role: str, timestamp_ms: int | None, content: str) -> str:
    return f"[Source: {source_id}]\nRole: {role}\nTime: {utc_timestamp(timestamp_ms)}\n{content}"


def message_token_estimate(
    public_source_id: str, role: str, timestamp_ms: int | None, content: str
) -> int:
    """Conservative additive estimate used to avoid loading a huge full scope."""
    rendered = render_message("msg_" + public_source_id[:16], role, timestamp_ms, content)
    return reader_token_count(rendered)


def full_bundle_token_cost(messages: Sequence[Mapping[str, object]]) -> int:
    if not messages:
        return 0
    message_cost = sum(int(message["reader_token_est"]) for message in messages)
    separator_cost = reader_token_count(FULL_BUNDLE_SEPARATOR) * (len(messages) - 1)
    return message_cost + separator_cost


def render_full_bundle(messages: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    blocks: list[str] = []
    spans: list[dict[str, object]] = []
    for message in messages:
        content = str(message["raw_content"])
        source = str(message["public_source_id"])
        blocks.append(
            render_message(
                "msg_" + source[:16],
                str(message["role"]),
                message["source_timestamp_ms"],  # type: ignore[arg-type]
                content,
            )
        )
        spans.append(
            {
                "source_id": source,
                "start_char": 0,
                "end_char": len(content),
            }
        )
    return FULL_BUNDLE_SEPARATOR.join(blocks), stable_view_id(RENDERER_VERSION, spans)


def render_window(window: Mapping[str, object]) -> tuple[str, str, dict[str, object]]:
    raw_content = str(window["raw_content"])
    start = int(window["start_char"])
    end = int(window["end_char"])
    passage = raw_content[start:end]
    if not passage or raw_content[start:end] != passage:
        raise ValueError("window span does not map to its immutable source message")
    source = str(window["public_source_id"])
    span = {
        "source_id": source,
        "start_char": start,
        "end_char": end,
    }
    content = render_message(
        "msg_" + source[:16],
        str(window["role"]),
        window["source_timestamp_ms"],  # type: ignore[arg-type]
        passage,
    )
    return content, stable_view_id(RENDERER_VERSION, [span]), span
