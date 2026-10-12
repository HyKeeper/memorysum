"""Source-preserving leaf windows and bounded same-request context windows."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from memorysum.core.atoms import AtomDraft, atomize_messages
from memorysum.core.text import chinese_terms, proxy_tokens
from memorysum.settings import ChunkScaleConfig, PipelineConfig

logger = logging.getLogger(__name__)
_BOUNDARY_TOKENS = {".", "!", "?", ";", ":", "。", "！", "？", "；", "："}


@dataclass(frozen=True)
class SpanDraft:
    message_ordinal: int
    start_char: int
    end_char: int
    atom_ordinals: tuple[int, ...]


@dataclass(frozen=True)
class WindowDraft:
    message_ordinal: int
    scale: str
    ordinal: int
    start_char: int
    end_char: int
    raw_text: str
    embedding_text: str
    lexical_en: str
    lexical_zh: str
    proxy_token_count: int
    source_spans: tuple[SpanDraft, ...]
    atom_refs: tuple[tuple[int, int], ...]
    forced_split: bool


@dataclass(frozen=True)
class IngestPlan:
    atoms: tuple[AtomDraft, ...]
    windows: tuple[WindowDraft, ...]


@dataclass(frozen=True)
class _TokenRef:
    message_ordinal: int
    token: re.Match[str]


def build_ingest_plan(messages: list[tuple[str, str]], pipeline: PipelineConfig) -> IngestPlan:
    """Build exact source atoms, per-message leaf windows, and bounded context windows."""
    atoms = atomize_messages(messages)
    atoms_by_message: dict[int, list[AtomDraft]] = {}
    for atom in atoms:
        atoms_by_message.setdefault(atom.message_ordinal, []).append(atom)

    windows: list[WindowDraft] = []
    for message_ordinal, (role, content) in enumerate(messages):
        windows.extend(
            _chunk_message_scale(
                content,
                role,
                message_ordinal,
                "leaf",
                pipeline.leaf_chunks,
                pipeline.embedding.input_proxy_tokens,
                atoms_by_message.get(message_ordinal, []),
            )
        )
    windows.extend(_chunk_context_scale(messages, atoms_by_message, pipeline))

    logger.debug(
        "Built %d atoms and %d windows across %d messages", len(atoms), len(windows), len(messages)
    )
    return IngestPlan(atoms=tuple(atoms), windows=tuple(windows))


def build_windows(messages: list[tuple[str, str]], pipeline: PipelineConfig) -> list[WindowDraft]:
    """Compatibility helper for tests and small local callers."""
    return list(build_ingest_plan(messages, pipeline).windows)


def _chunk_message_scale(
    content: str,
    role: str,
    message_ordinal: int,
    scale: str,
    spec: ChunkScaleConfig,
    max_input_proxy_tokens: int,
    atoms: Sequence[AtomDraft],
) -> list[WindowDraft]:
    tokens = proxy_tokens(content)
    if not tokens:
        return []

    drafts: list[WindowDraft] = []
    start = 0
    ordinal = 0
    while start < len(tokens):
        end = _choose_end(tokens, content, start, spec)
        start_char = tokens[start].start()
        end_char = tokens[end - 1].end()
        source_span = _make_span(message_ordinal, start_char, end_char, atoms)
        draft = _make_window(
            [source_span],
            {message_ordinal: (role, content)},
            scale,
            ordinal,
            max_input_proxy_tokens,
            {message_ordinal: list(atoms)},
        )
        drafts.append(draft)
        if end >= len(tokens):
            break
        next_start = end - min(spec.overlap, max(0, end - start - 1))
        start = max(start + 1, next_start)
        ordinal += 1
    return drafts


def _choose_end(tokens: list[re.Match[str]], text: str, start: int, spec: ChunkScaleConfig) -> int:
    remaining = len(tokens) - start
    if remaining <= spec.max:
        return len(tokens)

    target_end = min(start + spec.target, len(tokens))
    min_end = min(start + spec.min, len(tokens))
    max_end = min(start + spec.max, len(tokens))
    sentence_boundaries: list[int] = []
    whitespace_boundaries: list[int] = []
    for end in range(min_end, max_end + 1):
        if end >= len(tokens):
            continue
        previous = tokens[end - 1]
        next_token = tokens[end]
        gap = text[previous.end() : next_token.start()]
        if previous.group(0) in _BOUNDARY_TOKENS or "\n" in gap:
            sentence_boundaries.append(end)
        elif gap:
            whitespace_boundaries.append(end)

    before_target = [end for end in sentence_boundaries if end <= target_end]
    if before_target:
        return before_target[-1]
    before_target = [end for end in whitespace_boundaries if end <= target_end]
    if before_target:
        return before_target[-1]
    after_target = [end for end in sentence_boundaries if end > target_end]
    if after_target:
        return after_target[0]
    return target_end


def _chunk_context_scale(
    messages: list[tuple[str, str]],
    atoms_by_message: dict[int, list[AtomDraft]],
    pipeline: PipelineConfig,
) -> list[WindowDraft]:
    """Window over ordered messages from this Add only, capped at four source spans."""
    tokens = [
        _TokenRef(message_ordinal, token)
        for message_ordinal, (_role, content) in enumerate(messages)
        for token in proxy_tokens(content)
    ]
    if not tokens:
        return []

    roles_and_texts = {ordinal: item for ordinal, item in enumerate(messages)}
    spec = pipeline.context_chunks
    max_spans = pipeline.groups.max_spans
    drafts: list[WindowDraft] = []
    start = 0
    ordinal = 0
    while start < len(tokens):
        proposed_end = _choose_flat_end(tokens, start, spec)
        end, span_limited = _cap_span_count(tokens, start, proposed_end, max_spans)
        if end <= start:
            end = min(start + 1, len(tokens))

        spans = _spans_for_tokens(tokens, start, end, atoms_by_message)
        draft = _make_window(
            spans,
            roles_and_texts,
            "context",
            ordinal,
            pipeline.embedding.input_proxy_tokens,
            atoms_by_message,
        )
        drafts.append(draft)

        if end >= len(tokens):
            break
        if span_limited:
            last_message = tokens[end - 1].message_ordinal
            next_start = next(
                index
                for index in range(start + 1, end)
                if tokens[index].message_ordinal == last_message
            )
        else:
            overlap = min(spec.overlap, max(0, end - start - 1))
            next_start = end - overlap
        start = max(start + 1, next_start)
        ordinal += 1
    return drafts


def _choose_flat_end(tokens: Sequence[_TokenRef], start: int, spec: ChunkScaleConfig) -> int:
    if len(tokens) - start <= spec.max:
        return len(tokens)

    target_end = min(start + spec.target, len(tokens))
    min_end = min(start + spec.min, len(tokens))
    max_end = min(start + spec.max, len(tokens))
    message_boundaries: list[int] = []
    sentence_boundaries: list[int] = []
    whitespace_boundaries: list[int] = []

    for end in range(min_end, max_end + 1):
        if end >= len(tokens):
            continue
        previous = tokens[end - 1]
        following = tokens[end]
        if previous.message_ordinal != following.message_ordinal:
            message_boundaries.append(end)
            continue
        if previous.token.group(0) in _BOUNDARY_TOKENS:
            sentence_boundaries.append(end)
            continue
        content_gap = previous.token.string[previous.token.end() : following.token.start()]
        if content_gap:
            whitespace_boundaries.append(end)

    for boundaries in (message_boundaries, sentence_boundaries, whitespace_boundaries):
        before_target = [end for end in boundaries if end <= target_end]
        if before_target:
            return before_target[-1]
        after_target = [end for end in boundaries if end > target_end]
        if after_target:
            return after_target[0]
    return target_end


def _cap_span_count(
    tokens: Sequence[_TokenRef], start: int, proposed_end: int, max_spans: int
) -> tuple[int, bool]:
    seen_messages: set[int] = set()
    for index in range(start, proposed_end):
        message_ordinal = tokens[index].message_ordinal
        seen_messages.add(message_ordinal)
        if len(seen_messages) > max_spans:
            return index, True
    return proposed_end, False


def _spans_for_tokens(
    tokens: Sequence[_TokenRef],
    start: int,
    end: int,
    atoms_by_message: dict[int, list[AtomDraft]],
) -> list[SpanDraft]:
    grouped: list[tuple[int, int, int]] = []
    for token_ref in tokens[start:end]:
        message_ordinal = token_ref.message_ordinal
        token_start = token_ref.token.start()
        token_end = token_ref.token.end()
        if grouped and grouped[-1][0] == message_ordinal:
            previous = grouped[-1]
            grouped[-1] = (message_ordinal, previous[1], token_end)
        else:
            grouped.append((message_ordinal, token_start, token_end))

    return [
        _make_span(message_ordinal, span_start, span_end, atoms_by_message.get(message_ordinal, []))
        for message_ordinal, span_start, span_end in grouped
    ]


def _make_span(
    message_ordinal: int,
    start_char: int,
    end_char: int,
    atoms: Sequence[AtomDraft],
) -> SpanDraft:
    atom_ordinals = tuple(
        atom.ordinal for atom in atoms if atom.start_char < end_char and atom.end_char > start_char
    )
    return SpanDraft(message_ordinal, start_char, end_char, atom_ordinals)


def _make_window(
    spans: Sequence[SpanDraft],
    roles_and_texts: dict[int, tuple[str, str]],
    scale: str,
    ordinal: int,
    max_input_proxy_tokens: int,
    atoms_by_message: dict[int, list[AtomDraft]],
) -> WindowDraft:
    if not spans:
        raise ValueError("A window must include at least one source span")
    raw_parts: list[str] = []
    input_parts: list[str] = []
    for span in spans:
        role, source_text = roles_and_texts[span.message_ordinal]
        raw = source_text[span.start_char : span.end_char]
        if not raw:
            raise ValueError("A window span must not be empty")
        raw_parts.append(raw)
        input_parts.append(f"Source role: {role}\nPassage:\n{raw}")
    raw_text = "\n\n".join(raw_parts)
    embedding_text = "\n\n---\n\n".join(input_parts)
    proxy_count = len(proxy_tokens(embedding_text))
    if proxy_count > max_input_proxy_tokens:
        raise ValueError("A generated window exceeds the embedding input proxy-token limit")

    atom_refs = tuple(
        (span.message_ordinal, atom_ordinal)
        for span in spans
        for atom_ordinal in span.atom_ordinals
    )
    forced_split = any(
        atom.start_char < span.end_char
        and atom.end_char > span.start_char
        and (span.start_char > atom.start_char or span.end_char < atom.end_char)
        for span in spans
        for atom in atoms_by_message.get(span.message_ordinal, [])
    )
    primary = spans[0]
    return WindowDraft(
        message_ordinal=primary.message_ordinal,
        scale=scale,
        ordinal=ordinal,
        start_char=primary.start_char,
        end_char=primary.end_char,
        raw_text=raw_text,
        embedding_text=embedding_text,
        lexical_en=raw_text,
        lexical_zh=" ".join(chinese_terms(raw_text)),
        proxy_token_count=proxy_count,
        source_spans=tuple(spans),
        atom_refs=atom_refs,
        forced_split=forced_split,
    )
