"""Disk cache for embeddings, keyed by model and exact text (documents and queries apart)."""

from __future__ import annotations

from collections.abc import Sequence

from dpdp_rag.kvcache import SqliteKV, digest
from dpdp_rag.retrieval.embeddings import Embedder


class CachingEmbedder:
    def __init__(self, inner: Embedder, kv: SqliteKV, model_id: str) -> None:
        self.inner = inner
        self.kv = kv
        self.model_id = model_id
        self.dim = inner.dim
        self.hits = 0
        self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        return digest(self.model_id, self.dim, kind, text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key("doc", t) for t in texts]
        found = self.kv.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            fresh = self.inner.embed_documents([texts[i] for i in todo])
            new = {keys[i]: vec for i, vec in zip(todo, fresh, strict=True)}
            self.kv.put_many(new)
            found.update(new)
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key("query", text)
        vec = self.kv.get(key)
        if vec is None:
            vec = self.inner.embed_query(text)
            self.kv.put(key, vec)
            self.misses += 1
        else:
            self.hits += 1
        return vec
