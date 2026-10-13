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
    channels: Iterable[tuple[str, Iterable[object], float]],
    *,
    k: int = 60,
    limit: int,
    similarity_weight: float = 0.0,
) -> list[FusedCandidate]:
    """Fuse ranked window IDs; duplicate results add channel evidence only once.

    A channel may yield bare window IDs or ``(window_id, similarity)`` pairs. Ranks
    alone discard how *confident* a channel was: its best hit always contributes
    ``weight/(k+1)`` whether the cosine similarity was 0.9 or 0.3. Feeding the
    similarities in adds a bounded tie-breaker scaled to the channel's own range
    (min-max), so channels with different scoring scales stay comparable and the
    rank term keeps dominating. ``similarity_weight`` 0 reproduces the pure-rank
    behaviour exactly.
    """
    candidates: dict[int, FusedCandidate] = {}
    for channel_name, entries, weight in channels:
        ids: list[int] = []
        sims: list[float | None] = []
        for entry in entries:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                ids.append(int(entry[0]))  # type: ignore[arg-type]
                sims.append(None if entry[1] is None else float(entry[1]))  # type: ignore[arg-type]
            else:
                ids.append(int(entry))  # type: ignore[arg-type]
                sims.append(None)
        present = [s for s in sims if s is not None]
        low = min(present) if present else 0.0
        high = max(present) if present else 1.0
        span = (high - low) or 1.0
        seen: set[int] = set()
        for rank, (window_id, sim) in enumerate(zip(ids, sims, strict=False), start=1):
            if window_id in seen:
                continue
            seen.add(window_id)
            candidate = candidates.setdefault(window_id, FusedCandidate(window_id))
            candidate.score += weight / (k + rank)
            if similarity_weight and sim is not None:
                # Not multiplied by the channel weight: the rank term already carries
                # it, and scaling here twice made "0.3" already 36x the rank term.
                candidate.score += similarity_weight * ((sim - low) / span)
            candidate.best_rank = min(candidate.best_rank, rank)
            candidate.channels.add(channel_name)

    return sorted(
        candidates.values(),
        key=lambda item: (-round(item.score, 12), item.best_rank, item.window_id),
    )[:limit]
