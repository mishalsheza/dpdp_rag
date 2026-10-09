# Decision log

A short record of the major design choices: what was chosen, what else was considered,
the evidence, and the cost trade-off. Numbers marked **measured** come from runs in this
repo. Numbers marked `‹TBD›` are placeholders to fill from real runs:

- `uv run dpdp-eval run` for answer quality
- `uv run dpdp-ablation` for retrieval
- `docs/LOADTEST.md` for latency

**Caveat for every measured number:** the golden set has 8 questions, 7 of them with gold
chunks. At this size, a difference of one question moves a score by about 0.14. Treat the
numbers as direction, not proof, until the set is larger.

Retrieval numbers come from `data/ablation/results.json`, measured on 2026-10-09 with the
local CPU models, retrieval only, cross-reference expansion off.

---

## D1. Embedding model: `BAAI/bge-small-en-v1.5` via fastembed (local, CPU)

- **Context.** The corpus is 364 chunks of English legal text. Questions are short.
  Embeddings are needed at index time and for each query.
- **Alternatives.**
  - `BAAI/bge-base-en-v1.5` (768 dimensions) and `bge-large-en-v1.5` (1024 dimensions),
    both local
  - hosted embedding APIs (per-token cost, network latency, and a key in CI)
- **Evidence.**
  - Dense retrieval with bge-small: recall@5 **0.69**, MRR **0.93**, hit@5 **1.00**
    (measured, N = 7).
  - Indexing the full corpus in memory, including model load, took about 40 s on an
    Apple M3 (measured, cold start).
  - Query embedding fits inside an API-internal p50 of 11 ms for the whole retrieval path
    (measured, stub-LLM load test).
  - bge-base and bge-large: recall@5 `‹TBD›`, MRR `‹TBD›`, index time `‹TBD›`.
- **Cost trade-off.** $0 per query and no API key. The model is about 130 MB, baked into
  the Docker image and cached in CI. Larger models cost more CPU and memory, and their
  vectors aren't compatible, so switching means re-indexing.
- **Revisit when** the golden set has more than 50 questions. Re-run the ablation with
  `embedding.model` changed.

## D2. Chunking: structure-aware (section / sub-section / rule / Schedule row), not fixed windows

- **Context.**
  - Answers must cite pinpoints such as "Rule 8(3)" or "Third Schedule, row 1".
  - Commencement dates and corrigendum corrections attach to specific provisions.
  - Table rows and illustrations must stay whole.
- **Alternatives.** Fixed-size windows (1,000 characters with 200 overlap was tested),
  recursive splitting, and page-level chunks.
- **Evidence** (measured, N = 7). Fixed chunks are scored by character overlap with the
  gold provisions; see `src/dpdp_rag/eval/ablation.py`.

  | | recall@5 | MRR |
  |---|---|---|
  | Dense, structure-aware chunks | **0.69** | **0.93** |
  | Dense, fixed chunks | 0.50 | 0.61 |
  | Hybrid, structure-aware chunks | 0.68 | 0.76 |
  | Hybrid, fixed chunks | 0.68 | 0.65 |

  Answer quality with fixed chunks: `‹TBD›`. Fixed chunks can't carry a single pinpoint,
  in-force date or correction, so the citation and date features would degrade by design.
- **Cost trade-off.** The structure-aware ingestion is the most complex code in the repo
  (about 1,800 lines in `src/dpdp_rag/ingest/`, plus tests), and it is specific to these documents. New document
  types need parser work. Fixed windows would be generic and nearly free to build.
- **Revisit when** documents with different layouts are added, such as future amendment
  notifications.

## D3. Hybrid (dense + BM25, reciprocal rank fusion) vs dense only

- **Context.** Legal questions mix paraphrase ("can a company register as …") with exact
  terms ("section 9", "G.S.R. 892(E)", "Third Schedule").
- **Alternatives.** Dense only; BM25 only; weighted score fusion instead of reciprocal
  rank fusion.
- **Evidence** (measured, N = 7, structure-aware chunks).

  | | recall@5 | MRR |
  |---|---|---|
  | Dense | **0.69** | **0.93** |
  | BM25 | 0.50 | 0.52 |
  | Hybrid (RRF, k = 60) | 0.68 | 0.76 |

  On this tiny set, **dense alone beat hybrid on MRR**: BM25 pulls lexically similar but
  wrong provisions above the right ones. End-to-end answer scores, dense vs hybrid:
  `‹TBD›`.
- **Decision for now.** Hybrid stays the default because exact-citation queries ("rule
  12", "section 9(3)") are under-represented in 7 questions. **This should be re-decided**
  once the golden set has enough citation-style and paraphrase questions. If dense still
  wins, switch with `retrieval.mode: dense`.
- **Cost trade-off.** BM25 runs in memory at negligible CPU cost. The cost is ranking
  quality when its lexical matches are wrong, as seen above.

## D4. Reranker: off by default

- **Context.** A cross-encoder can reorder the fused top 20 before the top 5 are chosen.
- **Alternatives.**
  - `Xenova/ms-marco-MiniLM-L-6-v2` (tested)
  - `BAAI/bge-reranker-base`
  - an LLM reranker (Claude Haiku, per-query cost)
- **Evidence** (measured, N = 7). Hybrid with the MiniLM reranker: recall@5 **0.50**
  (vs 0.68 without), MRR **0.63** (vs 0.76). The general web-search reranker made legal
  ranking worse. bge-reranker-base: `‹TBD›`. Added latency per query: `‹TBD› ms`.
- **Cost trade-off.** About 90 MB more model and a CPU cross-encoder pass over 20
  candidates per query. An LLM reranker would add about `‹TBD›` USD per query.
- **Decision.** Keep it off (`reranker.enabled: false`). Use `--rerank` only to
  experiment.

## D5. Answer model: Claude Haiku 5.5 rather than a larger model

> **Superseded by D7** (Groq free tier). Kept for the record.

- **Context.** Answers are short, grounded in the supplied provisions (`generation.k` = 8
  plus cross-referenced Act sections), with
  structured JSON output. Quality depends mostly on retrieval and the prompt rules
  (citations, dates, refusals).
- **Alternatives.** `claude-sonnet-5-5` ($2 / $10 per million tokens) and
  `claude-opus-5-5` ($4 / $20). Haiku 5.5 is $0.10 / $0.50, all published first-party
  prices.
- **Evidence.**

  | | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
  |---|---|---|---|
  | Faithfulness / relevance / refusal correctness | `‹TBD›` | `‹TBD›` | `‹TBD›` |
  | p50 latency | `‹TBD›` | `‹TBD›` | `‹TBD›` |
  | Cost per request (estimate) | ≈ $0.0014 | ≈ $0.027 | ≈ $0.054 |

  The cost estimates assume about 6K input and 1.5K output tokens per request; replace
  them with measured `cost_usd`.
- **Cost trade-off.** Roughly 20× (Sonnet) to 40× (Opus) more per request. Switch only if
  the eval shows a quality gap Haiku can't close with prompt changes. Changing the model
  changes `config_hash`, so the CI gate compares like with like.
- **Related: the judge model.** The judge is `claude-haiku-4-5` at temperature 0, because
  Haiku 5.5 rejects non-default temperatures. It costs $1 / $5 per million tokens.
  Judge-vs-human agreement on a sample: `‹TBD›`. See docs/EVAL.md.

## D6. Vector store: Qdrant rather than pgvector

- **Context.** A small corpus (364 vectors). The search needs payload filters (doc_type,
  rule, section, in-force date) and must run in tests and CI without a server.
- **Alternatives.**
  - pgvector, a good choice if a Postgres database already exists
  - FAISS or numpy in-process, simplest but with no filtering or persistence service
  - Qdrant Cloud, which the code already supports via `QDRANT_URL`
- **Evidence.**
  - Qdrant's in-process `:memory:` mode runs the full retrieval path in tests and CI
    (measured: 268 tests in about 8 s).
  - Payload-indexed filters cover every metadata filter used.
  - pgvector latency and recall for the same queries: `‹TBD›`. At 364 vectors, both
    should be well under the measured 11 ms API-internal p50.
- **Cost trade-off.** Running Qdrant adds a service to operate, which docker-compose
  handles locally; Qdrant Cloud has a free tier. pgvector would reuse existing Postgres
  operations, but would need a database container in CI.
- **Revisit when** the deployment platform already provides Postgres, or the corpus grows
  past what one Qdrant node serves comfortably (`‹TBD›` vectors).

## D7. LLM provider: Groq free tier (`openai/gpt-oss-120b`) instead of Anthropic

- **Context.** The project must run at no cost and without Anthropic/Claude models. This
  replaces D5's answer model and the Claude Haiku 4.5 judge.
- **Choice.** `openai/gpt-oss-120b` on Groq for both answering (`reasoning_effort:
  medium`) and judging (temperature 0, `reasoning_effort: low`). It supports Groq's strict
  `json_schema` structured outputs, so replies always match `prompts/*schema.json`.
- **Evidence.** A warm `/ask` took about 1.4 s and about 2.8K tokens, and cited the right
  provision (Rule 7(2) for the 72-hour breach report). Eval scores: `‹TBD›`.
- **Trade-offs.**
  - Free-tier rate limits (tokens per minute and per day) cap throughput and eval size.
  - The judge is the same model as the answerer (self-preference risk; docs/EVAL.md).
  - `cost_usd` is always $0, so `budget_usd` no longer guards anything; tests use a paid
    `TEST_PRICING` to keep the cost code covered.
  - The `anthropic` provider and SDK remain in the code but no config uses them.
- **Revisit when** free-tier limits block normal use or CI, or eval quality falls short.

## D8. The eval fails loudly when the judge can't run

- **Context.** Eval run `20261009T142550Z_a138cc2f` reported faithfulness, relevance
  and refusal correctness as `None` for every category. It ran before the Groq
  `finish_reason` mapping existed, so every judge reply was rejected (`stop_reason
  'stop'`). The run still wrote results and exited 0, and the dashboard showed blanks
  that looked like "not applicable".
- **Choice.**
  - A one-call judge preflight before the run.
  - A `judge_status` (ok / degraded / unavailable) in `results.json`.
  - Non-zero exit codes (3 unavailable, 4 degraded).
  - A banner in the summary and on the dashboard.
  - A dedicated pass/fail `injection` judge metric (`prompts/judge_injection.md`) for
    `prompt_injection` items. Before this, injection items were scored only on retrieval
    and the generic refusal rubric.
- **Alternatives.**
  - Abort on the first judge error: one transient 429 would throw away a 20-minute run.
  - Keep `None` and only add a log line: this is what failed before.
- **Trade-offs.**
  - The preflight costs one judge call per run.
  - The extra injection metric adds one judge call per injection item.
  - Judge `max_tokens` dropped from 4000 to 2000 to fit the free tier's TPM accounting.
    A verdict that truncates surfaces as a judge failure, not a silent `None`.

