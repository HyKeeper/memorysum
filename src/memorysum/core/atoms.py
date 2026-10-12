"""Deterministic source atoms with exact Python string offsets."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AtomDraft:
    message_ordinal: int
    ordinal: int
    start_char: int
    end_char: int
    raw_hash: bytes
    kind: str
    in_code: bool
    in_quote: bool


_FENCE_RE = re.compile(r"^\s{0,3}(?:```+|~~~+)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_QUOTE_RE = re.compile(r"^\s{0,3}>")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CONTROL_RE = re.compile(
    r'^(?:Correction:\s*"[^"\n]{8,512}"\s*->\s*"[^"\n]{1,512}"[.!]?'
    r"|更正：“[^”\n]{8,512}”→“[^”\n]{1,512}”[。]?)$"
)
_SENTENCE_END = set(".!?。！？；;")
_CLOSING_MARKS = set("\"'”’）)]】")


def atomize_messages(messages: Sequence[tuple[str, str]]) -> list[AtomDraft]:
    """Split each source message without rewriting or trimming stored content."""
    result: list[AtomDraft] = []
    for message_ordinal, (_role, content) in enumerate(messages):
        result.extend(_atomize_message(message_ordinal, content))
    return result


def _atomize_message(message_ordinal: int, content: str) -> list[AtomDraft]:
    lines: list[tuple[int, int, str]] = []
    offset = 0
    for raw_line in content.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        lines.append((offset, offset + len(body), body))
        offset += len(raw_line)
    if not lines and content:
        lines.append((0, len(content), content))

    atoms: list[AtomDraft] = []
    in_fence = False
    for line_index, (line_start, line_end, _body) in enumerate(lines):
        stripped_start, stripped_end = _trim_bounds(content, line_start, line_end)
        if stripped_start >= stripped_end:
            continue
        line = content[stripped_start:stripped_end]
        is_fence = bool(_FENCE_RE.match(line))
        if in_fence or is_fence or line.startswith("    ") or line.startswith("\t"):
            _append_atom(
                atoms, message_ordinal, content, stripped_start, stripped_end, "code", True, False
            )
            if is_fence:
                in_fence = not in_fence
            continue
        if _CONTROL_RE.fullmatch(line):
            _append_atom(
                atoms,
                message_ordinal,
                content,
                stripped_start,
                stripped_end,
                "control_statement",
                False,
                False,
            )
            continue
        if _QUOTE_RE.match(line):
            _append_atom(
                atoms, message_ordinal, content, stripped_start, stripped_end, "quote", False, True
            )
            continue
        if _HEADING_RE.match(line):
            _append_atom(
                atoms,
                message_ordinal,
                content,
                stripped_start,
                stripped_end,
                "heading",
                False,
                False,
            )
            continue
        if _LIST_RE.match(line):
            _append_atom(
                atoms,
                message_ordinal,
                content,
                stripped_start,
                stripped_end,
                "list_item",
                False,
                False,
            )
            continue
        if "|" in line:
            next_body = _next_nonempty_line(lines, line_index + 1, content)
            if next_body is not None and _TABLE_SEPARATOR_RE.fullmatch(next_body):
                kind = "table_header"
            elif _TABLE_SEPARATOR_RE.fullmatch(line):
                kind = "table_separator"
            else:
                kind = "table_row"
            _append_atom(
                atoms, message_ordinal, content, stripped_start, stripped_end, kind, False, False
            )
            continue
        _append_sentences(atoms, message_ordinal, content, stripped_start, stripped_end)
    return atoms


def _append_sentences(
    output: list[AtomDraft], message_ordinal: int, content: str, start: int, end: int
) -> None:
    sentence_start = start
    cursor = start
    while cursor < end:
        if content[cursor] not in _SENTENCE_END:
            cursor += 1
            continue
        boundary = cursor + 1
        while boundary < end and content[boundary] in _CLOSING_MARKS:
            boundary += 1
        punctuation = content[cursor]
        if punctuation in ".!?;" and boundary < end and not content[boundary].isspace():
            cursor += 1
            continue
        _append_atom(
            output, message_ordinal, content, sentence_start, boundary, "sentence", False, False
        )
        sentence_start = boundary
        while sentence_start < end and content[sentence_start].isspace():
            sentence_start += 1
        cursor = sentence_start
    _append_atom(output, message_ordinal, content, sentence_start, end, "sentence", False, False)


def _append_atom(
    output: list[AtomDraft],
    message_ordinal: int,
    content: str,
    start: int,
    end: int,
    kind: str,
    in_code: bool,
    in_quote: bool,
) -> None:
    start, end = _trim_bounds(content, start, end)
    if start >= end:
        return
    raw = content[start:end]
    output.append(
        AtomDraft(
            message_ordinal=message_ordinal,
            ordinal=len(output),
            start_char=start,
            end_char=end,
            raw_hash=hashlib.sha256(raw.encode("utf-8", errors="strict")).digest(),
            kind=kind,
            in_code=in_code,
            in_quote=in_quote,
        )
    )


def _trim_bounds(content: str, start: int, end: int) -> tuple[int, int]:
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _next_nonempty_line(
    lines: Sequence[tuple[int, int, str]], start_index: int, content: str
) -> str | None:
    for start, end, _body in lines[start_index:]:
        start, end = _trim_bounds(content, start, end)
        if start < end:
            return content[start:end]
    return None
