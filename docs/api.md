# API

```bash
docker compose up -d && uv run dpdp-index   # retrieval backend (see docs/retrieval.md)
uv run dpdp-api                              # or: python -m dpdp_rag.api  (host/port: api.*)
```

Set `GROQ_API_KEY` in `.env` (Groq free tier). Langfuse tracing
turns on when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set; `LANGFUSE_HOST` is
optional.

## Endpoints

`POST /ask` takes `{"question": str, "as_of_date": "YYYY-MM-DD" | null}`. When
`as_of_date` is omitted, today's date is used.

```json
{
  "answer": "…",
  "citations": [{"chunk_id": "dpdp_rules_2025:r12(1)", "pinpoint": "Rule 12(1)",
                 "doc_type": "rules", "gsr_no": "G.S.R. 846(E)", "page": 29,
                 "in_force_date": "2027-05-13", "in_force": false, "corrected_by": null}],
  "refused": false,
  "refusal_reason": null,
  "as_of_date": "2026-01-01",
  "latency_ms": 812.4,
  "tokens": {"input": 2310, "output": 402, "cache_read": 1450, "cache_write": 0, "total": 4162},
  "cost_usd": 0.0,
  "config_hash": "b556…",
  "model": "openai/gpt-oss-120b",
  "cached": false
}
```

`refusal_reason` takes these values:

| Value | Meaning |
|---|---|
| `insufficient_context` | Nothing relevant was retrieved, the model said the documents don't answer the question, or the model cited nothing it was given |
| `legal_advice` | Advice or compliance questions. The answer still explains the provisions, with citations |
| `model_refusal` | The model returned `stop_reason: "refusal"` |

Error responses:

| Status | When |
|---|---|
| 422 | Empty or over-long question, or a bad date |
| 502 | The LLM failed or its output didn't match the schema |
| 503 | Qdrant is unreachable or the collection hasn't been indexed |

`GET /healthz` returns `{"status": "ok"}`.

`GET /version` returns `{"git_sha", "config_hash", "model"}`. `git_sha` comes from
`$GIT_SHA` if set, otherwise from `git rev-parse HEAD`, with `-dirty` appended when there
are local changes.

## How an answer is produced

1. **Retrieve.** `generation.k` chunks are retrieved, plus their cross-referenced Act
   sections. Retrieval never filters by date, because provisions that aren't in force yet
   still have to be explained.
2. **Build the prompt.** The system prompt is fixed text (`prompts/answer_system.md`) and
   is prompt-cached. The user turn (`prompts/answer_user.md`) holds the as-of date, the
   `<documents>` and the `<question>`. Each document carries attributes computed in code
   rather than left to the model:
   - `chunk_id` and `pinpoint` (e.g. `Rule 8(3)`, `Third Schedule, row 1`, `Section 9(1)`)
   - `in_force_date`
   - `status`, e.g. "NOT YET IN FORCE on 2026-01-01 (comes into force on 2027-05-13)"
   - `corrected_by` with a note, on chunks changed by G.S.R. 892(E). These chunks contain
     the corrected text only.
3. **Untrusted input.** Question and document text are HTML-escaped (`&`, `<`, `>`), so
   they can't close the surrounding tags. The system prompt tells the model that anything
   inside them is data, never instructions.
4. **Call the model.** `openai/gpt-oss-120b` on Groq is called through
   `client.chat.completions.create` with a strict `json_schema` response format built from
   `prompts/answer_schema.json`, at `reasoning_effort: medium`. On the free tier the
   reported `cost_usd` is always 0.
5. **Validate.** The model's JSON is checked against the `LLMAnswer` model. Then:
   - Citations to chunks that weren't supplied are dropped, and duplicates are merged.
   - Pinpoint labels and in-force flags come from chunk metadata, not from the model.
   - A non-refused answer with no valid citation is replaced by the insufficient-context
     refusal text in `prompts/refusal_insufficient_context.md`.
6. **Cost.** `cost_usd` uses `llm.pricing`: $0.10 / $0.50 per million input / output
   tokens, and $0.50 / $2.50 for prompts over 100K tokens. Cache writes cost 1.25× the
   input rate and cache reads 0.1×.

## config_hash, cache, tracing, metrics

- **`config_hash`** is the sha256 of the canonical JSON of `configs/default.yaml` plus the
  bytes of every file in `generation.prompt_files`. Changing any setting or prompt changes
  it. It is attached to every response, trace, cache key and metrics row.
- **Response cache.** When `api.response_cache.enabled` is set (it is meant for dev
  only), answers are stored as one JSON file per
  `sha256(config_hash, normalised question, as_of_date)` under `data/cache/responses`.
  A cache hit returns `cached: true` with zero tokens and zero cost.
- **Langfuse.** Each request creates one trace named `ask`. The config_hash appears in
  the trace's metadata, in its `version` and in a `config:<hash12>` tag. The trace
  contains a `retrieve` step (retriever type) and a `generate` step (generation type,
  with model, usage and cost). Cached requests are traced as well. Traces are flushed on
  shutdown.
- **Metrics.** Each request writes one row to `data/metrics.db`, table `requests`, with
  columns `timestamp, config_hash, latency_ms, tokens_in, tokens_out, cost_usd, cached,
  refused, source` (`source` is `api` or `eval`). Summaries are available through
  `MetricsStore`:

  ```python
  from dpdp_rag.metrics import MetricsStore, Where

  m = MetricsStore(Path("data/metrics.db"))
  m.p50_latency(Where(config_hash=h)), m.p99_latency(), m.mean_cost(Where(source="eval"))
  m.request_counts()  # {"total", "cached", "refused", "api", "eval"}
  m.summary(Where(include_cached=False))
  ```
