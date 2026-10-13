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
        return ChunkMetadata()

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
