"""Stable identities and canonical Add payload hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically without normalizing strings or arrays."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8", errors="strict")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_hash(payload: Mapping[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(payload)).digest()


def source_id(user_id: str, request_id: str, ordinal: int) -> str:
    """Stable, opaque identity for one source message."""
    return sha256_hex(canonical_json([user_id, request_id, ordinal]))


def scope_id(user_id: str) -> str:
    """Filesystem-safe scope key; the raw user ID is never used in a path."""
    return sha256_hex(user_id.encode("utf-8", errors="strict"))


def stable_view_id(renderer_version: str, spans: Sequence[Mapping[str, Any]]) -> str:
    """Stable ID derived only from renderer version and exact source spans."""
    return "ev_" + sha256_hex(canonical_json([renderer_version, list(spans)]))
