"""Lift the harness's per-chunk metadata header into structured identifiers.

The evaluation harness wraps every ingested chunk with a single header line, e.g.

    [LOCOMO_META chunk_id=conv-26_chunk_0 source_ids=D1:1,D1:2,...,D1:18]
    Session 1 (1:56 pm on 8 May, 2023)
    Caroline: Hey Mel! Good to see you! ...

(BEAM and PersonaMem use the same shape with their own tag:

    [BEAM_META chunk_id=beam128k_conv1_session1_chunk0 source_ids=1:msg0,1:msg1])

The platform maps a retrieved item back to the gold turn by reading identifiers
from the item: `source_ids` first, then `sourceIds`/`chunk_ids`/`ids`, and only
as a last resort by parsing the header out of the returned text. MemorySum keeps
the header verbatim inside the stored message, so we can lift the ids once at
Add time (deterministic, no extra request) and return them as a structured
field — that is the difference between "the platform can locate the source" and
"the platform silently gives up".

The line right after the header is the time anchor (`Session 1 (1:56 pm on
8 May, 2023)`); temporal questions need it, and it is only present in the first
window of a chunk, so callers may re-attach it when rendering a later window.

This module never rewrites stored content: parsing is read-only, and re-attaching
the anchor happens at render time only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: A metadata header is a whole line: `[<TAG>_META key=value ...]`.
_META_LINE_RE = re.compile(
    r"^[ \t]*\[(?P<tag>[A-Z][A-Z0-9_]*_META)\b(?P<body>[^\]]*)\][ \t]*$", re.MULTILINE
)
_SOURCE_IDS_RE = re.compile(r"\bsource_ids[ \t]*=[ \t]*(?P<values>[^\s\]]+)", re.IGNORECASE)
_CHUNK_ID_RE = re.compile(r"\bchunk_id[ \t]*=[ \t]*(?P<value>[^\s\]]+)", re.IGNORECASE)

#: Cap the number of ids kept per message so a pathological header cannot bloat
#: the response payload.
_MAX_IDS = 512

#: The platform's own message headers are a leading bracket cluster rather than a
#: `[<TAG>_META ...]` line, e.g.
#:     [locomo_refined][session_1][D1:3][text] Caroline: I went to a LGBTQ support group…
#:     [memtrapbench][session_1][D1:1][text] user: I want to configure a high-fidelity CFD…
#:     [angry][session_1][D1:34][narration] NARRATION: The whole feeling is one of movement…
#: The id the platform can match against gold lives in the `D<session>:<turn>` token —
#: the same id space as the `[LOCOMO_META ... source_ids=D19:1,D19:2]` form we already
#: parse. 2026-10-14 measurement: 2695/2695 messages in the smoke scopes carried *no*
#: ids, so every hit came back with `source_ids: []` even though the harness id was
#: right there in the text.
_BRACKET_RE = re.compile(r"\[([^\[\]]{1,64})\]")
#: `D1:3`, `D19:1`, `1:msg108`, or a comma-separated list of the same.
_ID_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*:\d+")


#: 流式任务载荷（ALF/DB）里唯一的标识就是 `dataset` + `task_index`：
#:     Streaming intra-transfer train trajectory: {"dataset": "ALF", "task_index": 86, …}
#: 平台的问法也是字面量（"ALF task index 86"），所以照它的字段提两个形态：
#: 裸索引 `86` 与复合 `ALF#86`（不发明别的格式；多给的 id 在交集匹配下无害）。
_TRAJECTORY_RE = re.compile(
    r'"dataset"\s*:\s*"(?P<dataset>[A-Za-z0-9_]+)".{0,200}?"task_index"\s*:\s*(?P<index>\d+)',
    re.DOTALL,
)


def _trajectory_identifiers(content: str) -> tuple[str, ...]:
    head = content[:600]
    if "train trajectory" not in head:
        return ()
    match = _TRAJECTORY_RE.search(head)
    if match is None:
        return ()
    return (f"{match.group('dataset')}#{match.group('index')}", match.group("index"))


def _cluster_identifiers(content: str) -> tuple[str, ...]:
    """Ids from a leading bracket cluster; empty when the cluster is not there.

    Only the *contiguous* run of brackets at the start counts: text further in can
    contain unrelated brackets (MCQ option lists, arithmetic like `[5, 5, 5, 8]`).
    """
    text = content.lstrip()
    ids: list[str] = []
    position = 0
    while position < len(text) and text[position] == "[":
        match = _BRACKET_RE.match(text, position)
        if match is None:
            break
        for token in match.group(1).split(","):
            token = token.strip()
            if _ID_TOKEN_RE.fullmatch(token):
                ids.append(token)
        position = match.end()
        while position < len(text) and text[position] in " \t":
            position += 1
        if len(ids) >= _MAX_IDS:
            break
    return tuple(dict.fromkeys(ids))[:_MAX_IDS]


@dataclass(frozen=True)
class ChunkMetadata:
    """Structured view of one stored message's harness header."""

    source_ids: tuple[str, ...] = ()
    chunk_id: str | None = None
    anchor: str | None = None

    @property
    def identifiers(self) -> tuple[str, ...]:
        """Ids to report for this message, most specific first."""
        return self.source_ids or ((self.chunk_id,) if self.chunk_id else ())


def parse_chunk_metadata(content: str) -> ChunkMetadata:
    """Parse the header of a stored message; returns empty metadata if absent.

    Only the first header line is considered (a chunk carries exactly one).
    """
    match = _META_LINE_RE.search(content)
    if match is None:
        # A leading bracket cluster is the platform's other header shape; it carries
        # ids but no anchor (there is no date line to lift). Streaming-task payloads
        # identify themselves by (dataset, task_index) instead of a header.
        return ChunkMetadata(
            source_ids=_cluster_identifiers(content) or _trajectory_identifiers(content)
        )

    body = match.group("body")
    source_ids: tuple[str, ...] = ()
    ids_match = _SOURCE_IDS_RE.search(body)
    if ids_match is not None:
        values = [value.strip() for value in ids_match.group("values").split(",")]
        source_ids = tuple(value for value in values if value)[:_MAX_IDS]

    chunk_match = _CHUNK_ID_RE.search(body)
    chunk_id = chunk_match.group("value").strip() if chunk_match is not None else None

    # The anchor is the first non-empty line after the header.
    anchor = None
    for line in content[match.end() :].splitlines():
        stripped = line.strip()
        if stripped:
            anchor = stripped
            break

    return ChunkMetadata(source_ids=source_ids, chunk_id=chunk_id, anchor=anchor)


def attach_anchor(content: str, anchor: str | None) -> str:
    """Prefix `anchor` when the rendered slice does not already contain it.

    Returns `content` unchanged when there is no anchor or it is already present,
    so repeated rendering stays idempotent.
    """
    if not anchor:
        return content
    if anchor in content:
        return content
    return f"{anchor}\n{content}"
