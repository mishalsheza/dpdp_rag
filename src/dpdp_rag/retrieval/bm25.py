"""Sparse lexical retrieval with rank_bm25."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from rank_bm25 import BM25Okapi

from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval.filters import Filters
from dpdp_rag.retrieval.store import render

_TOKEN = re.compile(r"[a-z0-9]+")


class Tokenizer:
    def __init__(self, stopwords: Iterable[str]) -> None:
        self.stopwords = frozenset(stopwords)

    def __call__(self, text: str) -> list[str]:
        return [t for t in _TOKEN.findall(text.lower()) if t not in self.stopwords]


class BM25Index:
    def __init__(self, chunks: list[Chunk], cfg: dict[str, Any], template: str) -> None:
        self.chunks = chunks
        self.tokenize = Tokenizer(cfg.get("stopwords", []))
        corpus = [self.tokenize(render(template, c)) for c in chunks]
        self._bm25 = BM25Okapi(corpus, k1=float(cfg["k1"]), b=float(cfg["b"]))

    def search(self, query: str, limit: int, filters: Filters) -> list[tuple[Chunk, float]]:
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            (
                (c, float(s))
                for c, s in zip(self.chunks, scores, strict=True)
                if s > 0 and filters.matches(c)
            ),
            key=lambda cs: cs[1],
            reverse=True,
        )
        return ranked[:limit]
