"""Embedding models, selected by the `embedding` section of the config."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any, Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed (models download on first use)."""

    def __init__(self, model: str, batch_size: int) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self._batch = batch_size
        self.dim = len(next(iter(self._model.query_embed("dimension probe"))))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._model.passage_embed(list(texts), batch_size=self._batch)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _make_base(cfg: dict[str, Any]) -> Embedder:
    provider = cfg["provider"]
    if provider == "fastembed":
        return FastEmbedEmbedder(cfg["model"], int(cfg["batch_size"]))
    if provider == "hash":
        if not cfg.get("dim"):
            raise ValueError("embedding.dim is required for the hash provider")
        return HashEmbedder(int(cfg["dim"]))
    raise ValueError(f"Unknown embedding provider {provider!r}")


def make_embedder(cfg: dict[str, Any]) -> Embedder:
    """The configured embedder, behind the on-disk cache when embedding.cache.enabled."""
    base = _make_base(cfg)
    cache = cfg.get("cache") or {}
    if not cache.get("enabled"):
        return base
    from dpdp_rag.config import resolve
    from dpdp_rag.kvcache import SqliteKV
    from dpdp_rag.retrieval.embedding_cache import CachingEmbedder

    model_id = f"{cfg['provider']}:{cfg.get('model') or ''}"
    return CachingEmbedder(base, SqliteKV(resolve(cache["path"]), "embeddings"), model_id)
