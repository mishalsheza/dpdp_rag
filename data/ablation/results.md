| Variant | Chunks | recall@1 | recall@3 | recall@5 | mrr | hit_rate@5 |
|---|---|---|---|---|---|---|
| dense | structure | 0.46 | 0.62 | 0.69 | 0.93 | 1.00 |
| bm25 | structure | 0.21 | 0.50 | 0.50 | 0.52 | 0.71 |
| hybrid | structure | 0.39 | 0.50 | 0.68 | 0.76 | 0.86 |
| hybrid + reranker | structure | 0.32 | 0.36 | 0.50 | 0.63 | 0.71 |
| dense (fixed chunks) | fixed | 0.29 | 0.39 | 0.50 | 0.61 | 0.86 |
| hybrid (fixed chunks) | fixed | 0.36 | 0.54 | 0.68 | 0.65 | 0.86 |

N = 7 answerable golden questions · 364 structure-aware chunks vs 152 fixed 1000-char chunks · embeddings `BAAI/bge-small-en-v1.5` · reranker `Xenova/ms-marco-MiniLM-L-6-v2`
