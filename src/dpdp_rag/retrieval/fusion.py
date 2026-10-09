"""Reciprocal rank fusion of several ranked result lists."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank), rank from 1.

    Ties keep first-seen order, so the order of `rankings` is a stable tie-breaker.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
