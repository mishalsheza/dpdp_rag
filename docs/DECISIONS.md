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
- **Re-measured 2026-10-10 on N = 27** (`dpdp-ablation`, retrieval only):

  | | recall@5 | MRR | hit@5 |
  |---|---|---|---|
  | Dense | 0.65 | 0.77 | 0.85 |
  | BM25 | 0.57 | 0.66 | 0.81 |
  | Hybrid (RRF) | 0.65 | 0.74 | 0.85 |
  | Hybrid + reranker | 0.62 | 0.77 | 0.85 |

  Dense and hybrid are tied overall. They differ by category: dense is better on
  lookup and corrigendum, hybrid is better on cross_reference and temporal recall. Each
  difference is one or two items. Hybrid stays the base because query rewriting (D9)
  gains more on top of it. The reranker helps scenario (0.31 / 0.55) but drops table
  recall and puts the injection item's gold chunk out of the top 5, so it stays off.

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

## D9. Query rewriting for plain-language (scenario) questions: `fuse`

- **Context.** Scenario questions ("we run a school that tracks students' behaviour; are
  we compliant?") had recall@5 0.05 and MRR 0.11. The words "DPDP Act" pulled in generic
  chunks (notification preambles, s3, s4(1)), and nothing in the question matched the
  Act's terms ("Data Fiduciary", "child", "behavioural monitoring", "verifiable
  consent").
- **Choice.** Before retrieval, one LLM call (`prompts/query_rewrite.md`,
  `openai/gpt-oss-20b`, temperature 0, disk-cached) returns a statutory restatement and
  sub-issues. With `query_rewrite.strategy: fuse`, the original question and the
  restatement are each ranked by hybrid retrieval, and the two rankings are fused with
  RRF. Sub-issues are generated but unused under `fuse`.
- **Evidence** (measured 2026-10-10, `dpdp-ablation`, N = 27 answerable items; recall@5 /
  MRR):

  | Strategy | overall | lookup (8) | xref (4) | table (5) | temporal (3) | corrigendum (2) | scenario (4) | injection (1) |
  |---|---|---|---|---|---|---|---|---|
  | off (hybrid) | 0.65 / 0.74 | 0.79 / 0.79 | 0.81 / 1.00 | 0.73 / 1.00 | 0.72 / 0.53 | 0.50 / 0.75 | 0.05 / 0.11 | 1.00 / 1.00 |
  | replace | 0.68 / 0.76 | 0.62 / 0.77 | 0.81 / 0.88 | 0.83 / 1.00 | 0.56 / 0.42 | 0.75 / 0.75 | 0.43 / 0.55 | 1.00 / 1.00 |
  | **fuse** | **0.71 / 0.78** | 0.79 / 0.78 | 0.81 / 1.00 | 0.73 / 1.00 | 0.72 / 0.50 | 0.75 / 0.75 | **0.31 / 0.47** | 1.00 / 1.00 |
  | decompose (weighted) | 0.63 / 0.76 | 0.67 / 0.75 | 0.71 / 1.00 | 0.63 / 0.83 | 0.72 / 0.53 | 0.50 / 0.75 | 0.39 / 0.58 | 1.00 / 1.00 |
  | fuse + decompose (weighted) | 0.67 / 0.76 | 0.67 / 0.76 | 0.71 / 1.00 | 0.73 / 0.85 | 0.72 / 0.50 | 0.75 / 0.75 | 0.39 / 0.53 | 1.00 / 1.00 |

  Rewrites are LLM outputs: temperature 0 reduces variation but doesn't remove it. An
  earlier measurement the same day gave the same `fuse` row and slightly different
  `replace` and decompose rows, with the same conclusions. The disk cache makes reruns
  identical.
- **Why not decompose.** It finds more scenario gold chunks, but it regresses lookup,
  cross_reference and table. The 20b model also emits sub-issues for some narrow
  questions (for example the territorial-scope lookup), and those sub-queries pull in
  off-target chunks. Giving all sub-issues a combined RRF weight of 1
  (`sub_issue_weight`) reduced the damage but did not remove it. `replace` drops the
  user's own words, which costs lookup and temporal.
- **Noise.** Scenario has n = 4, and scenario-002 has 7 gold chunks, so its recall@5
  can't exceed 0.71. The 0.05 → 0.31 gain is about one more gold chunk found per
  question. The MRR gain is larger: the first gold chunk moves into the top 3 on 3 of 4
  items. Everything else is within one item: temporal MRR 0.53 → 0.50 is one rank,
  corrigendum 0.50 → 0.75 is one chunk. The prompt was revised once after the first
  version produced no sub-issues. The glossary in it is generic DPDP vocabulary, not
  tied to the golden questions, but with 4 scenario items some overfitting can't be
  ruled out. Re-check this on new scenario questions.
- **Cost trade-off.**
  - One extra LLM call per uncached question: latency (sub-second on Groq, not yet
    load-tested) and quota on a separate model, so it doesn't eat the answerer's daily
    tokens.
  - If the call fails, retrieval falls back to the original query and logs a warning.
    The eval records each fallback as an item error, and the ablation fails instead.
  - Load tests and unit tests run with the rewrite off.
- **Revisit when** there are 10+ scenario questions, or when a classifier can reliably
  tell situations from narrow questions (then decompose only situations).

## D10. Metadata boosts for citations, definitions, commencement and corrections

- **Context.** After D9, the misses in lookup, temporal, corrigendum and table had
  specific causes:
  - **lookup:** no lookup question cites a section number. The misses were a
    definition ("what counts as 'personal data'" → s2(t), crowded out by preambles and
    s1) and territorial scope (s3).
  - **temporal:** for "when do the Rules come into force", G.S.R. 843(E)'s Act
    commencement items outranked rule 1.
  - **corrigendum:** the corrected Schedule row was not retrieved next to its
    corrigendum entry.
  - **table:** tables are **not** split. Each Schedule row is one complete chunk
    (largest 1,657 characters). The misses were the companion provisions: the parent
    rule (r4(1), r8(1)) or the Act Schedule penalty row for a section (s15 →
    schedule:5).
- **Choice.** `retrieval/boost.py`. Each signal builds a ranked list from the original
  query and chunk metadata, and that list is fused into the candidates with weighted
  RRF (`boost.weights`):
  - **citations:** "Section 8(5)", "sub-sections (1) and (3) of section 9", "rule
    1(3)", "item 11 of Part B of the First Schedule" → those exact chunks. The most
    specific match wins, and a citation resolving to more than 5 chunks (a whole
    Schedule) is skipped.
  - **definitions:** a quoted defined term, or a "what counts as / meaning of" question,
    → the clause defining it. The 29 defined terms are parsed from the documents'
    `“x” means` clauses.
  - **commencement:** "come into force / commence / effective from" → commencement
    provisions, narrowed to the document named (Rules → rule 1; Act → s1 and G.S.R.
    843(E)). A plain "in force" doesn't trigger it, because "is rule 7 in force?" is
    about rule 7.
  - **corrections:** on a question about corrections, each retrieved corrigendum chunk
    pulls in the chunk it corrected, and the reverse.
  - **links:** Schedule row ↔ parent rule and Act section. Off; see the evidence below.
- **New chunk metadata** (re-ingested; only these fields changed, chunk ids unchanged):
  - `corrects` on corrigendum chunks: the chunk ids each patch was applied to. Before
    this, it was known at ingest but discarded.
  - `see_rules` on Schedule rows, from the "[See rule N]" line.

  `in_force_date`, `section` / `rule` / `clause` / `item` and `corrected_by` already
  existed. `text` already holds the corrected wording, so the index already prefers it.
- **Evidence** (measured 2026-10-10, `dpdp-ablation`, each on top of hybrid + `fuse`;
  recall@5 / MRR):

  | Variant | overall | lookup (8) | xref (4) | table (5) | temporal (3) | corrigendum (2) | scenario (4) |
  |---|---|---|---|---|---|---|---|
  | fuse (D9) | 0.71 / 0.78 | 0.79 / 0.78 | 0.81 / 1.00 | 0.73 / 1.00 | 0.72 / 0.50 | 0.75 / 0.75 | 0.31 / 0.47 |
  | + citations | 0.75 / 0.80 | 0.79 / 0.78 | 0.94 / 1.00 | 0.73 / 1.00 | 0.72 / 0.50 | 1.00 / 1.00 | 0.31 / 0.47 |
  | + definitions | 0.75 / 0.82 | 0.92 / 0.91 | 0.81 / 1.00 | 0.73 / 1.00 | 0.72 / 0.50 | 0.75 / 0.75 | 0.31 / 0.47 |
  | + commencement | 0.70 / 0.81 | 0.79 / 0.78 | 0.81 / 1.00 | 0.73 / 1.00 | 0.83 / 0.75 | 0.50 / 0.75 | 0.31 / 0.47 |
  | + corrections | 0.73 / 0.80 | 0.79 / 0.78 | 0.81 / 1.00 | 0.73 / 1.00 | 0.72 / 0.50 | 1.00 / 1.00 | 0.31 / 0.47 |
  | + links (0.25) | 0.72 / 0.68 | 0.79 / 0.72 | 0.73 / 0.83 | 0.80 / 0.70 | 0.72 / 0.53 | 0.75 / 0.75 | 0.36 / 0.42 |
  | **+ all but links** | **0.80 / 0.87** | 0.92 / 0.91 | 0.94 / 1.00 | 0.73 / 1.00 | 0.83 / 0.75 | 1.00 / 1.00 | 0.31 / 0.47 |
  | + all | 0.76 / 0.72 | 0.88 / 0.75 | 0.62 / 0.78 | 0.80 / 0.69 | 0.83 / 0.78 | 1.00 / 1.00 | 0.36 / 0.42 |

  Two fixes came from the first pass:
  - Citations initially boosted all 7 chunks of "the Third Schedule" and cost table a
    gold chunk; that led to the 5-chunk cap.
  - Corrections initially re-added the candidates themselves, so it had no effect; it
    now adds only the partners.

  Ablation variants are explicit: a variant with no `boost:` map runs with no boosts,
  not with the defaults in `default.yaml`.
- **Why links stays off.** It raises table recall (0.73 → 0.80) but costs MRR almost
  everywhere. A parent rule or penalty row is linked to many questions where it's not the
  answer. With cross-reference expansion, the answer model already gets the Act
  sections a rule refers to (context_recall).
- **Noise.** Every per-category gain here is one or two items:
  - lookup-003 for definitions
  - xref-001 and xref-002 for citations
  - temporal-002 for commencement
  - corrigendum-001 and corrigendum-002 for corrections and citations

  The mechanisms are deterministic and keyed to explicit wording, so they won't help
  questions without that wording; the signals are tested on hand-made chunks in
  `tests/test_boost.py`. Lookup with explicit section citations has no golden items yet
  ("Section 8(5)" only appears in an xref question), so the lookup gain comes from
  definitions, not citations.
- **Cost trade-off.** Pure Python over the candidate list, with no model calls and no
  measurable latency. The patterns live in `configs/default.yaml`. The risk is false
  triggers from wording, which the per-signal weights let you turn off.

