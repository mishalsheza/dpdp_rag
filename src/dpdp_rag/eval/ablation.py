"""Retrieval ablation over the golden set: `uv run dpdp-ablation`.

Compares retrieval setups (dense, BM25, hybrid, hybrid + reranker) and structure-aware
vs fixed-size chunks by recall@k, hit rate and MRR. Retrieval only: no LLM calls.

Fixed-size chunks don't share ids with the gold labels, so a fixed chunk counts as
retrieving a gold chunk when their character spans overlap by at least `min_overlap` of
the shorter one (see configs/ablation.yaml).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from dpdp_rag.config import load_config, load_system_config, merge, resolve
from dpdp_rag.eval.golden import GoldenItem, load_golden
from dpdp_rag.eval.retrieval_metrics import hit_at_k, reciprocal_rank
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval.retriever import Retriever
from dpdp_rag.retrieval.store import ChunkStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def overlap(self, other: Span) -> int:
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


def document_spans(chunks: list[Chunk]) -> tuple[dict[str, str], dict[str, Span]]:
    """Each document's text (chunks joined in order) and every chunk's span within it."""
    texts: dict[str, list[str]] = defaultdict(list)
    lengths: dict[str, int] = defaultdict(int)
    spans: dict[str, Span] = {}
    for c in chunks:
        start = lengths[c.doc_id]
        texts[c.doc_id].append(c.text)
        spans[c.chunk_id] = Span(c.doc_id, start, start + len(c.text))
        lengths[c.doc_id] = start + len(c.text) + 1  # "\n" separator
    return {d: "\n".join(parts) for d, parts in texts.items()}, spans


def fixed_chunks(
    chunks: list[Chunk], size: int, overlap: int
) -> tuple[list[Chunk], dict[str, Span]]:
    """Fixed-size windows over each document's text, with their spans."""
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    docs, _ = document_spans(chunks)
    meta = {c.doc_id: c for c in chunks}
    out: list[Chunk] = []
    spans: dict[str, Span] = {}
    for doc_id, text in docs.items():
        first = meta[doc_id]
        for i, start in enumerate(range(0, max(len(text) - overlap, 1), size - overlap)):
            piece = text[start : start + size]
            cid = f"fixed:{doc_id}:{i}"
            out.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_type=first.doc_type,
                    gsr_no=first.gsr_no,
                    text=piece,
                )
            )
            spans[cid] = Span(doc_id, start, start + len(piece))
    return out, spans


def covers(fixed: Span, gold: Span, min_overlap: float) -> bool:
    shared = fixed.overlap(gold)
    shorter = min(fixed.end - fixed.start, gold.end - gold.start)
    return shorter > 0 and shared >= min_overlap * shorter


def score_item(
    ranked: list[str],
    gold: list[str],
    ks: list[int],
    depth: int,
    relevant: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    """Metrics for one item. `relevant` maps a retrieved id to the gold ids it covers
    (fixed chunks); for structure chunks a retrieved id covers itself."""

    def covered(cid: str) -> set[str]:
        return relevant.get(cid, set()) if relevant is not None else ({cid} & set(gold))

    out: dict[str, float] = {}
    for k in ks:
        found = set().union(*(covered(c) for c in ranked[:k])) if ranked else set()
        out[f"recall@{k}"] = len(found & set(gold)) / len(set(gold))
    marks = ["g" if covered(c) else "-" for c in ranked]
    out[f"hit_rate@{ks[-1]}"] = hit_at_k(marks, ["g"], ks[-1])
    out["mrr"] = reciprocal_rank(marks, ["g"], depth)
    return out


def run_ablation(cfg: dict[str, Any], system: dict[str, Any]) -> dict[str, Any]:
    base = merge(
        system,
        {"qdrant": {"location": ":memory:", "auto_index": True}, "cross_refs": {"enabled": False}},
    )
    structure = ChunkStore.from_file(resolve(system["data"]["chunks_file"]))
    items = [
        i
        for i in load_golden(
            resolve(cfg["golden_file"]),
            set(structure.by_id),
            load_config("eval.yaml")["categories"],
        )
        if i.gold_chunk_ids
    ]
    ks, depth = [int(k) for k in cfg["ks"]], int(cfg["mrr_depth"])
    fc = cfg["fixed_chunks"]
    fixed, fixed_spans = fixed_chunks(
        structure.chunks, int(fc["size_chars"]), int(fc["overlap_chars"])
    )
    _, gold_spans = document_spans(structure.chunks)
    fixed_store = ChunkStore(fixed)

    retrievers: dict[str, Retriever] = {}
    results: list[dict[str, Any]] = []
    for v in cfg["variants"]:
        key = v["chunks"]
        if key not in retrievers:
            store = structure if key == "structure" else fixed_store
            coll = f"ablation_{key}"
            retrievers[key] = Retriever(merge(base, {"qdrant": {"collection": coll}}), store=store)
        r = retrievers[key]
        per_item = []
        for item in items:
            hits = r.retrieve(
                item.question,
                max(depth, *ks),
                mode=v["mode"],
                rerank=bool(v["rerank"]),
                cross_refs=False,
            )
            ranked = [h.chunk.chunk_id for h in hits]
            relevant = None
            if key == "fixed":
                relevant = {
                    cid: {
                        g
                        for g in item.gold_chunk_ids
                        if covers(fixed_spans[cid], gold_spans[g], fc["min_overlap"])
                    }
                    for cid in ranked
                }
            per_item.append(
                {
                    "id": item.id,
                    "category": item.category,
                    **score_item(ranked, item.gold_chunk_ids, ks, depth, relevant),
                }
            )
        metrics = {
            m: sum(p[m] for p in per_item) / len(per_item)
            for m in per_item[0]
            if m not in ("id", "category")
        }
        results.append(
            {"variant": v["name"], **v, "n": len(per_item), "metrics": metrics, "items": per_item}
        )
        log.info("%s: %s", v["name"], {k: round(x, 3) for k, x in metrics.items()})
    return {
        "n_items": len(items),
        "n_structure_chunks": len(structure.chunks),
        "n_fixed_chunks": len(fixed),
        "embedding_model": system["embedding"]["model"],
        "reranker_model": system["reranker"]["model"],
        "settings": cfg,
        "variants": results,
    }


def markdown(res: dict[str, Any]) -> str:
    ks = res["settings"]["ks"]
    cols = [f"recall@{k}" for k in ks] + ["mrr", f"hit_rate@{ks[-1]}"]
    lines = ["| Variant | Chunks | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for v in res["variants"]:
        lines.append(
            f"| {v['variant']} | {v['chunks']} | "
            + " | ".join(f"{v['metrics'][c]:.2f}" for c in cols)
            + " |"
        )
    return "\n".join(lines) + (
        f"\n\nN = {res['n_items']} answerable golden questions · {res['n_structure_chunks']} "
        f"structure-aware chunks vs {res['n_fixed_chunks']} fixed "
        f"{res['settings']['fixed_chunks']['size_chars']}-char chunks · embeddings "
        f"`{res['embedding_model']}` · reranker `{res['reranker_model']}`\n"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dpdp-ablation", description=__doc__)
    p.add_argument("--config", default="ablation.yaml")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    res = run_ablation(cfg, load_system_config())
    out = resolve(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    table = markdown(res)
    out.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()


__all__ = ["GoldenItem", "covers", "fixed_chunks", "main", "markdown", "run_ablation", "score_item"]
