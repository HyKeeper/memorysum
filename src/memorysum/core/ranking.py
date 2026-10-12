"""Deterministic reciprocal-rank fusion for bounded retrieval channels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class FusedCandidate:
    window_id: int
    score: float = 0.0
    best_rank: int = 2**31 - 1
    channels: set[str] = field(default_factory=set)


def reciprocal_rank_fusion(
    channels: Iterable[tuple[str, Iterable[int], float]],
    *,
    k: int = 60,
    limit: int,
) -> list[FusedCandidate]:
    """Fuse ranked window IDs; duplicate results add channel evidence only once."""
    candidates: dict[int, FusedCandidate] = {}
    for channel_name, window_ids, weight in channels:
        seen: set[int] = set()
        for rank, raw_window_id in enumerate(window_ids, start=1):
            window_id = int(raw_window_id)
            if window_id in seen:
                continue
            seen.add(window_id)
            candidate = candidates.setdefault(window_id, FusedCandidate(window_id))
            candidate.score += weight / (k + rank)
            candidate.best_rank = min(candidate.best_rank, rank)
            candidate.channels.add(channel_name)

    return sorted(
        candidates.values(),
        key=lambda item: (-round(item.score, 12), item.best_rank, item.window_id),
    )[:limit]
