"""Competition response rendering from immutable source spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memorysum.core.identity import stable_view_id
from memorysum.core.text import reader_token_count, utc_timestamp

RENDERER_VERSION = "memorysum-source-render-v3"
FULL_BUNDLE_SEPARATOR = "\n\n---\n\n"
CONTEXT_SPAN_SEPARATOR = "\n\n--- Adjacent source span ---\n\n"
_RESPONSE_SPAN_SEPARATOR = "\n\n"


def render_message(source_id: str, role: str, timestamp_ms: int | None, content: str) -> str:
    """Render metadata-rich text for the internal reranker prompt."""
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


def render_full_bundle(messages: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str], ...]:
    """Return one clean API item per original message, with stable evidence IDs."""
    items: list[tuple[str, str]] = []
    for message in messages:
        content = str(message["raw_content"])
        identity = {
            "source_id": str(message["public_source_id"]),
            "start_char": 0,
            "end_char": len(content),
        }
        view_id = stable_view_id(RENDERER_VERSION, [identity])
        items.append((view_id, content))
    return tuple(items)


def _collect_source_passages(
    spans: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[Mapping[str, object], str, str]], list[dict[str, object]]]:
    if not spans:
        raise ValueError("An evidence view must include at least one source span")

    passages: list[tuple[Mapping[str, object], str, str]] = []
    identities: list[dict[str, object]] = []
    for span in spans:
        raw_content = str(span["raw_content"])
        start = int(span["start_char"])
        end = int(span["end_char"])
        passage = raw_content[start:end]
        if start < 0 or end > len(raw_content) or start >= end or not passage:
            raise ValueError("window span does not map to its immutable source message")
        source = str(span["public_source_id"])
        passages.append((span, source, passage))
        identities.append({"source_id": source, "start_char": start, "end_char": end})
    return passages, identities


def render_source_spans(
    spans: Sequence[Mapping[str, object]],
) -> tuple[str, str, list[dict[str, object]]]:
    """Render metadata-rich source spans for internal retrieval and reranking."""
    passages, identities = _collect_source_passages(spans)
    blocks = [
        render_message(
            "msg_" + source[:16],
            str(span["role"]),
            span["source_timestamp_ms"],  # type: ignore[arg-type]
            passage,
        )
        for span, source, passage in passages
    ]
    return (
        CONTEXT_SPAN_SEPARATOR.join(blocks),
        stable_view_id(RENDERER_VERSION, identities),
        identities,
    )


def render_response_spans(
    spans: Sequence[Mapping[str, object]],
) -> tuple[str, str, list[dict[str, object]]]:
    """Render exact source passages for the public API without synthetic labels."""
    passages, identities = _collect_source_passages(spans)
    content = _RESPONSE_SPAN_SEPARATOR.join(passage for _, _, passage in passages)
    return content, stable_view_id(RENDERER_VERSION, identities), identities


def render_window(
    window: Mapping[str, object],
) -> tuple[str, str, list[dict[str, object]]]:
    source_spans = window.get("source_spans")
    if isinstance(source_spans, Sequence) and not isinstance(source_spans, (str, bytes)):
        if source_spans:
            return render_source_spans(source_spans)  # type: ignore[arg-type]
    return render_source_spans([window])


def render_response_window(
    window: Mapping[str, object],
) -> tuple[str, str, list[dict[str, object]]]:
    source_spans = window.get("source_spans")
    if isinstance(source_spans, Sequence) and not isinstance(source_spans, (str, bytes)):
        if source_spans:
            return render_response_spans(source_spans)  # type: ignore[arg-type]
    return render_response_spans([window])
