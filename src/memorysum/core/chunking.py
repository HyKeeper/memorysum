"""Source-preserving dual-scale window construction."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from memorysum.core.text import chinese_terms, proxy_tokens
from memorysum.settings import ChunkScaleConfig, PipelineConfig

logger = logging.getLogger(__name__)

_BOUNDARY_TOKENS = {".", "!", "?", ";", ":", "。", "！", "？", "；", "：", "\n"}


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


def _choose_end(tokens: list[re.Match[str]], text: str, start: int, spec: ChunkScaleConfig) -> int:
    remaining = len(tokens) - start
    if remaining <= spec.max:
        return len(tokens)

    target_end = min(start + spec.target, len(tokens))
    min_end = min(start + spec.min, len(tokens))
    max_end = min(start + spec.max, len(tokens))
    candidates = range(min_end, max_end + 1)

    sentence_boundaries: list[int] = []
    whitespace_boundaries: list[int] = []
    for end in candidates:
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


def _chunk_scale(
    content: str,
    role: str,
    message_ordinal: int,
    scale: str,
    spec: ChunkScaleConfig,
    max_input_proxy_tokens: int,
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
        raw_text = content[start_char:end_char]
        embedding_text = f"Source role: {role}\nPassage:\n{raw_text}"
        proxy_count = len(proxy_tokens(embedding_text))
        if proxy_count > max_input_proxy_tokens:
            raise ValueError("A generated window exceeds the embedding input proxy-token limit")
        drafts.append(
            WindowDraft(
                message_ordinal=message_ordinal,
                scale=scale,
                ordinal=ordinal,
                start_char=start_char,
                end_char=end_char,
                raw_text=raw_text,
                embedding_text=embedding_text,
                lexical_en=raw_text,
                lexical_zh=" ".join(chinese_terms(raw_text)),
                proxy_token_count=proxy_count,
            )
        )
        if end >= len(tokens):
            break
        next_start = end - min(spec.overlap, max(0, end - start - 1))
        start = max(start + 1, next_start)
        ordinal += 1
    return drafts


def build_windows(messages: list[tuple[str, str]], pipeline: PipelineConfig) -> list[WindowDraft]:
    """Create leaf/context windows while retaining exact Python string offsets.

    Context windows currently remain within one source message. Cross-message
    context assembly is deliberately a later pipeline stage; no source order is
    inferred across Add requests.
    """
    result: list[WindowDraft] = []
    for message_ordinal, (role, content) in enumerate(messages):
        result.extend(
            _chunk_scale(
                content,
                role,
                message_ordinal,
                "leaf",
                pipeline.leaf_chunks,
                pipeline.embedding.input_proxy_tokens,
            )
        )
        result.extend(
            _chunk_scale(
                content,
                role,
                message_ordinal,
                "context",
                pipeline.context_chunks,
                pipeline.embedding.input_proxy_tokens,
            )
        )
    logger.debug("Built %d source windows across %d messages", len(result), len(messages))
    return result
