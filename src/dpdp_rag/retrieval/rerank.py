"""Optional cross-encoder reranker, selected by the `reranker` section of the config."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(documents))]


def make_reranker(cfg: dict[str, Any]) -> Reranker | None:
    if not cfg.get("enabled"):
        return None
    if cfg["provider"] == "fastembed":
        return FastEmbedReranker(cfg["model"])
    raise ValueError(f"Unknown reranker provider {cfg['provider']!r}")
