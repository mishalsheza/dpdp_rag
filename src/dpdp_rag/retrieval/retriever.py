"""Dense, BM25 and hybrid (RRF) retrieval with optional reranking and cross-reference expansion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from dpdp_rag.config import load_system_config, resolve
from dpdp_rag.generation.llm import LLMError
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval.bm25 import BM25Index
from dpdp_rag.retrieval.boost import MetadataBooster
from dpdp_rag.retrieval.embeddings import Embedder, make_embedder
from dpdp_rag.retrieval.filters import Filters
from dpdp_rag.retrieval.fusion import rrf
from dpdp_rag.retrieval.qdrant_index import QdrantIndex, make_client
from dpdp_rag.retrieval.rerank import Reranker, make_reranker
from dpdp_rag.retrieval.rewrite import STRATEGIES, QueryRewriter, make_rewriter
from dpdp_rag.retrieval.store import ChunkStore, render

MODES = ("dense", "bm25", "hybrid")

log = logging.getLogger(__name__)


class RetrievalUnavailable(RuntimeError):
    """The Qdrant collection is missing or out of date."""


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    # Per-stage scores, e.g. {"dense": 0.81, "bm25": 12.4, "rrf": 0.032, "rerank": 5.1}.
    scores: dict[str, float] = field(default_factory=dict)
    # Set on chunks added by cross-reference expansion.
    expanded_from: str | None = None
    via: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "score": self.score,
            "scores": self.scores,
            "expanded_from": self.expanded_from,
            "via": self.via,
            "chunk": self.chunk.model_dump(mode="json"),
        }


class Retriever:
    def __init__(
        self,
        config: dict[str, Any],
        store: ChunkStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.config = config
        self.store = store or ChunkStore.from_file(resolve(config["data"]["chunks_file"]))
        self.filters = Filters.from_config(config.get("filters", {}))
        self._embedder = embedder
        self._reranker = reranker if reranker is not None else make_reranker(config["reranker"])
        self._index: QdrantIndex | None = None
        self._bm25: BM25Index | None = None
        self._rewriter = rewriter
        self.rewrite_fallbacks = 0  # failed rewrites answered with the original query
        self._booster: MetadataBooster | None = None

    # Components are built on first use so that, e.g., BM25-only runs need no Qdrant.
    @property
    def index(self) -> QdrantIndex:
        if self._index is None:
            embedder = self._embedder or make_embedder(self.config["embedding"])
            qcfg = self.config["qdrant"]
            index = QdrantIndex(
                make_client(qcfg),
                qcfg,
                embedder,
                self.config["embedding"]["document_template"],
                self.config["embedding"].get("index_max_chars"),
            )
            if not index.is_current(self.store.chunks):
                if not qcfg.get("auto_index"):
                    raise RetrievalUnavailable(
                        f"Qdrant collection {qcfg['collection']!r} is missing or out of date; "
                        "run `uv run dpdp-index` (or set qdrant.auto_index: true)"
                    )
                index.build(self.store.chunks)
            self._index = index
        return self._index

    @property
    def booster(self) -> MetadataBooster:
        if self._booster is None:
            self._booster = MetadataBooster(self.store, self.config["boost"])
        return self._booster

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(
                self.store.chunks,
                self.config["bm25"],
                self.config["embedding"]["document_template"],
                self.config["embedding"].get("index_max_chars"),
            )
        return self._bm25

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        *,
        filters: Filters | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        cross_refs: bool | None = None,
        rewrite: str | None = None,
        boost: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k chunks for `query`, best first, followed by any cross-referenced chunks.

        `rewrite` is a query_rewrite strategy (see retrieval/rewrite.py). With more than
        one query, each is ranked on its own and the rankings are fused with RRF.
        `boost` overrides `boost.weights` (retrieval/boost.py), which re-rank the fused
        candidates using the original query and chunk metadata.
        Arguments left as None take their value from the config.
        """
        rcfg = self.config["retrieval"]
        k = int(k or rcfg["k"])
        mode = mode or rcfg["mode"]
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        filters = filters if filters is not None else self.filters
        limit = max(int(rcfg["candidates"]), k)

        queries = self._queries(query, rewrite)
        if len(queries) == 1:
            ranked, per_stage = self._rank(queries[0][0], mode, filters, limit)
        else:
            per_query = [self._rank(q, mode, filters, limit) for q, _ in queries]
            ranked = rrf(
                [[cid for cid, _ in r] for r, _ in per_query],
                int(rcfg["rrf_k"]),
                [w for _, w in queries],
            )
            per_stage = per_query[0][1]  # stage scores of the original query, for debugging
            per_stage["multi_query"] = dict(ranked)

        if "boost" in self.config:
            boosted = self.booster.apply(query, ranked, filters, boost)
            if boosted is not ranked:
                per_stage["boost"] = dict(boosted)
                ranked = boosted

        results = [
            RetrievedChunk(
                chunk=self.store.by_id[cid],
                score=score,
                scores={stage: s[cid] for stage, s in per_stage.items() if cid in s},
            )
            for cid, score in ranked
        ]
        use_rerank = self._reranker is not None if rerank is None else rerank
        if use_rerank:
            results = self._rerank(query, results)
        results = results[:k]

        expand = self.config["cross_refs"]["enabled"] if cross_refs is None else cross_refs
        if expand:
            results += self._expand(results, filters)
        return results

    def _queries(self, query: str, strategy: str | None) -> list[tuple[str, float]]:
        """(query, RRF weight): the original, plus rewrites when a strategy is active."""
        qcfg = self.config.get("query_rewrite") or {}
        strategy = strategy or qcfg.get("strategy", "off")
        if strategy not in STRATEGIES:
            raise ValueError(f"rewrite must be one of {STRATEGIES}, got {strategy!r}")
        if strategy == "off":
            return [(query, 1.0)]
        if self._rewriter is None:
            self._rewriter = make_rewriter({**qcfg, "strategy": strategy})
        assert self._rewriter is not None
        try:
            plan = self._rewriter.plan(query)
            return plan.queries(query, strategy, float(qcfg.get("sub_issue_weight", 1.0)))
        except LLMError as exc:
            if not qcfg.get("fallback_on_error", True):
                raise RetrievalUnavailable(f"query rewrite failed: {exc}") from exc
            self.rewrite_fallbacks += 1
            log.warning("query rewrite failed, using the original query: %s", exc)
            return [(query, 1.0)]

    def _rank(
        self, query: str, mode: str, filters: Filters, limit: int
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, float]]]:
        """One query's ranking (dense, BM25 or their RRF) and its per-stage scores."""
        rcfg = self.config["retrieval"]
        per_stage: dict[str, dict[str, float]] = {}
        rankings: list[list[str]] = []
        if mode in ("dense", "hybrid"):
            dense = self.index.search(query, limit, filters)
            per_stage["dense"] = dict(dense)
            rankings.append([cid for cid, _ in dense])
        if mode in ("bm25", "hybrid"):
            sparse = [(c.chunk_id, s) for c, s in self.bm25.search(query, limit, filters)]
            per_stage["bm25"] = dict(sparse)
            rankings.append([cid for cid, _ in sparse])

        if mode == "hybrid":
            ranked = rrf(rankings, int(rcfg["rrf_k"]))
            per_stage["rrf"] = dict(ranked)
        else:
            ranked = list(per_stage[mode].items())
        return ranked, per_stage

    def _rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._reranker is None:
            self._reranker = make_reranker({**self.config["reranker"], "enabled": True})
        assert self._reranker is not None
        cfg = self.config["reranker"]
        top, rest = results[: int(cfg["top_n"])], results[int(cfg["top_n"]) :]
        texts = [render(cfg["document_template"], r.chunk) for r in top]
        for r, s in zip(top, self._reranker.score(query, texts), strict=True):
            r.scores["rerank"] = s
            r.score = s
        top.sort(key=lambda r: r.score, reverse=True)
        return top + rest

    def _expand(self, results: list[RetrievedChunk], filters: Filters) -> list[RetrievedChunk]:
        """Add the Act chunks named in each result's refers_to (e.g. rule 12 -> section 9)."""
        cfg = self.config["cross_refs"]
        keep = filters.date_only() if cfg.get("apply_in_force_filter", True) else Filters()
        seen = {r.chunk.chunk_id for r in results}
        added: list[RetrievedChunk] = []
        for parent in results:
            labels = parent.chunk.refers_to_provisions or parent.chunk.refers_to
            # Most specific references first: "section 9(1)" before "section 9".
            labels = sorted(labels, key=lambda lbl: -lbl.count("("))
            budget = int(cfg["max_per_chunk"])
            for label in labels:
                for chunk in self.store.provision(label):
                    if budget == 0:
                        break
                    if chunk.chunk_id in seen or not keep.matches(chunk):
                        continue
                    seen.add(chunk.chunk_id)
                    budget -= 1
                    added.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=parent.score * float(cfg["score_factor"]),
                            expanded_from=parent.chunk.chunk_id,
                            via=label,
                        )
                    )
        return added


@lru_cache(maxsize=4)
def get_retriever(config_name: str = "default.yaml") -> Retriever:
    return Retriever(load_system_config(config_name))


def retrieve(query: str, k: int | None = None, **kwargs: Any) -> list[RetrievedChunk]:
    """Retrieve chunks for `query` using configs/default.yaml. See Retriever.retrieve."""
    return get_retriever().retrieve(query, k, **kwargs)
