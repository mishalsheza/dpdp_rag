"""Retrieval metrics over a ranked list of chunk ids and a set of gold chunk ids."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of gold chunks that appear in the top k."""
    if not gold:
        raise ValueError("recall is undefined without gold chunks")
    return len(set(ranked[:k]) & set(gold)) / len(set(gold))


def hit_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if at least one gold chunk is in the top k, else 0.0."""
    return 1.0 if set(ranked[:k]) & set(gold) else 0.0


def reciprocal_rank(ranked: Sequence[str], gold: Sequence[str], depth: int) -> float:
    """1 / rank of the first gold chunk within the top `depth`, else 0.0."""
    gold_set = set(gold)
    for rank, cid in enumerate(ranked[:depth], start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def first_gold_rank(ranked: Sequence[str], gold: Sequence[str]) -> int | None:
    gold_set = set(gold)
    return next((i for i, cid in enumerate(ranked, start=1) if cid in gold_set), None)


def item_metrics(
    ranked: Sequence[str], gold: Sequence[str], ks: Sequence[int], depth: int
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"hit_rate@{k}"] = hit_at_k(ranked, gold, k)
    out["mrr"] = reciprocal_rank(ranked, gold, depth)
    return out
