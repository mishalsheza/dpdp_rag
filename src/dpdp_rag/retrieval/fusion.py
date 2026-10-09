"""Reciprocal rank fusion of several ranked result lists."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(
    rankings: Sequence[Sequence[str]], k: int, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of w / (k + rank), rank from 1.

    `weights` (default all 1) scale each list's contribution. Ties keep first-seen order,
    so the order of `rankings` is a stable tie-breaker.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
