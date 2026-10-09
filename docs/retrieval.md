# Retrieval

Every option below is set in `configs/default.yaml`. Call-time arguments and CLI flags
override it for one query only.

## Setup

```bash
docker compose up -d          # local Qdrant on :6333 (volume: qdrant_storage)
uv run dpdp-index             # embed chunks.jsonl and upsert into Qdrant
uv run dpdp-index --if-stale  # rebuild only if missing / chunk count or vector size changed
```

To use Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`; `.env.example` lists
both. When `QDRANT_URL` is set it takes priority over `qdrant.location` and
`qdrant.local_url`. Setting `qdrant.location: ":memory:"` with `auto_index: true` runs
Qdrant in-process with no server, which is what the tests do.

## Querying

```python
from dpdp_rag.retrieval import retrieve, Filters
from datetime import date

hits = retrieve("Is parental consent needed for a child's data?", k=5)
hits = retrieve(
    "tracking of children",
    5,
    filters=Filters(doc_type=["rules"], in_force_on=date.today()),
    mode="hybrid",
    rerank=True,
    cross_refs=True,
)
for h in hits:
    print(h.score, h.chunk.chunk_id, h.scores, h.expanded_from, h.via)
```

```bash
python -m dpdp_rag.retrieval "query" [-k 5] [--mode dense|bm25|hybrid] \
    [--doc-type rules] [--rule 12] [--section 9] [--in-force [YYYY-MM-DD]] \
    [--rerank/--no-rerank] [--cross-refs/--no-cross-refs] [--json] [--config PATH]
```

`retrieve` returns `RetrievedChunk` objects. Each holds:

- `chunk`: the full chunk with all its metadata
- `score`: the final score
- `scores`: the score from each stage (`dense`, `bm25`, `rrf`, `rerank`)
- `expanded_from` and `via`: set only on chunks added by cross-reference expansion

## How a query runs

1. **Dense search.** Qdrant cosine search over embeddings of
   `embedding.document_template` (default `"{title}\n{text}"`). The default model is
   `BAAI/bge-small-en-v1.5` via fastembed, which runs locally on ONNX. Point ids are
   UUID5s derived from `chunk_id`, so re-indexing overwrites existing points instead of
   adding duplicates. The payload holds the whole chunk plus `in_force_int` (yyyymmdd),
   which is used for the date filter.
2. **BM25.** `rank_bm25` (BM25Okapi) runs in-process over the same rendered text, using
   lowercase alphanumeric tokens and the configured stopwords.
3. **Hybrid.** The two rankings are combined with reciprocal rank fusion:
   `sum(1 / (rrf_k + rank))`. Each retriever contributes `candidates` results.
4. **Reranker (optional).** A fastembed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`)
   rescores the top `reranker.top_n` results. Then the top `k` are kept.
5. **Cross-reference expansion (optional).** For each result, the Act chunks listed in its
   `refers_to_provisions` are added, most specific reference first, up to `max_per_chunk`
   per result. For example, rule 12(1) brings in s. 9(1) and s. 9(3). Each added chunk
   scores `parent_score * score_factor` and is placed after the top `k`.

## Filters

| Filter | Matches |
|---|---|
| `doc_type` | `act`, `rules`, `notification`, `corrigendum` (one or a list) |
| `rule` | the chunk's `rule` field, e.g. `"12"` (Schedule rows have no rule) |
| `section` | the chunk's `section` field, i.e. Act chunks, e.g. `"9"` |
| `in_force_only` / `--in-force` | `in_force_date <= today`; `filters.today` overrides the date |

Filters run inside Qdrant for dense search and in Python for BM25. Expanded chunks are
checked against the date filter only (`cross_refs.apply_in_force_filter`). Applying the
`doc_type`, `rule` or `section` filters to them would remove the Act sections the
expansion exists to bring in.
