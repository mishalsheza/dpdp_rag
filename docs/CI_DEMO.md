# CI and the eval gate, plus how to demo a blocked PR

## What runs

The workflow is `.github/workflows/ci.yml`. It has two jobs.

**`Lint + tests`** runs on every PR and every push to `main`:

- `uv sync --frozen`
- `ruff check`
- `ruff format --check`
- `pytest`

The tests are offline: the LLM is mocked and Qdrant runs in memory. They need no secrets.

**`Eval gate`** runs after `Lint + tests`, on PRs and on pushes to `main`:

1. `dpdp-eval validate`. Fails if a golden question references a chunk id that is no
   longer in `chunks.jsonl`.
2. `dpdp-eval --override configs/ci.yaml run`. Runs the full golden set through the real
   system, with Groq answering, and judges the answers. `configs/ci.yaml`:
   - uses in-memory Qdrant, so no Qdrant service is needed
   - turns on the embedding cache and the LLM-call cache
   - caps uncached spend at `budget_usd` ($0.50)
   - pins the as-of date for questions without one
3. `dpdp-eval-gate`. Compares the run with `eval/baseline.json` using
   `configs/gates.yaml`, writes the PR comment and decides pass or fail.
4. **PR comment.** A single comment is posted and then updated on each push, found by
   the `<!-- dpdp-eval-gate -->` marker. It has a before/after table for the gated metrics
   (recall@5, MRR, faithfulness, relevance), plus collapsible sections for the ungated
   metrics and each category.
5. **Job result.** The job fails when a gate fails, so a branch-protection rule can block
   the merge. `eval-out/`, holding `results.json`, `summary.md` and `comment.md`, is
   uploaded as the `eval-results` artifact.

### When a gate fails

A gated metric fails when `baseline - current > max_drop + tolerance`:

| Metric | max_drop | tolerance | Why this tolerance |
|---|---|---|---|
| recall@5, MRR | 0.05 | 0.01 | Retrieval is deterministic for the same chunks, model and index, so the tolerance only absorbs float rounding |
| faithfulness, relevance (mean of 1–5 scores) | 0.25 | 0.15 | With ~30 items, one judge score moving one point shifts the mean by about 0.03. The tolerance absorbs a one-point wobble |

The gate also fails if any item errored (`max_item_errors: 0`) or if a gated metric is
missing from the run.

### Keeping runs cheap and stable

- **LLM-call cache** (`data/cache/llm_calls.sqlite`): every answer and judge call is
  keyed by model, parameters, system prompt, user message and schema.
  - A PR that changes nothing the model sees replays every call for $0, with identical
    scores and no judge noise at all.
  - A PR that changes retrieval or prompts pays only for the calls whose inputs changed.
- **Embedding cache** (`data/cache/embeddings.sqlite`): chunk and query vectors are
  keyed by model and text. The fastembed model itself is cached in `.cache/fastembed`.
- **How the caches move between runs:** both files are restored with `actions/cache`
  using a per-run key and a prefix restore-key, so every run starts from the newest cache
  and saves an updated one. Runs on `main` warm the cache that PRs restore from.
- **Judge at temperature 0:** `openai/gpt-oss-120b` on Groq, the same model as the
  answerer (see docs/EVAL.md).
- **Spend cap:** `budget_usd` aborts the run, failing the job with a "❌ Eval run failed"
  comment, before a call that would go over the cap.
- **Cost:** $0 on Groq's free tier, so `budget_usd` never trips. The binding limit is
  the free tier's rate limits; the LLM-call cache keeps reruns from spending quota. The
  comment shows the cache hit counts.

### Forks and missing secrets

Without the `GROQ_API_KEY` secret (e.g. a PR from a fork), the eval doesn't run and
the comment says "⏭️ Eval gate skipped". Lint and tests still run.

## One-time setup

1. **Secrets.** In the repo: Settings → Secrets and variables → Actions. Add
   `GROQ_API_KEY`. Optionally add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and
   `LANGFUSE_HOST` to trace CI runs.
2. **Branch protection.** Settings → Branches → rule for `main`. Require the status
   checks **Lint + tests** and **Eval gate**.
3. **Record the first baseline.** `eval/baseline.json` starts as an `uninitialized`
   placeholder. Until a baseline exists, the gate reports metrics but can't fail on
   regressions, and the comment says "no baseline yet".
   - Merge the workflow, let the `main` run finish, and download its `eval-results`
     artifact. Using CI's own run means the baseline comes from the same environment.
   - Or run it locally with the same settings:
     `uv run dpdp-eval --override configs/ci.yaml run --out /tmp/base`. This needs
     `GROQ_API_KEY`; Docker isn't needed.
   - Then record it and commit the change in its own PR:

     ```bash
     uv run dpdp-eval-baseline --results /tmp/base/results.json --reason "initial baseline"
     git checkout -b eval/baseline && git add eval/baseline.json && git commit -m "Record eval baseline"
     ```

### Updating the baseline later (deliberately)

`dpdp-eval-baseline` is the only intended way to change `eval/baseline.json`.

- It requires `--reason`, which is stored in the file with the author, date, run id,
  config_hash, judge_hash and golden-set hash.
- It refuses runs that have errored items.
- It refuses runs that would fail the gates against the current baseline, unless you pass
  `--allow-regression`. That makes accepting a regression an explicit, reviewable choice.
- `--dry-run` shows the comparison without writing anything.

Update the baseline when you deliberately change something that moves the numbers, such
as growing the golden set, changing the judge rubric, or improving retrieval. When the
golden set, judge config or as-of date differ from the baseline's, the PR comment warns
that the before/after numbers aren't strictly comparable.

## Demo: a PR that the gate blocks (for the README screenshot)

The chunker splits by legal structure (section, rule, Schedule row), not by a fixed size.
So the closest thing to a "bad chunk size" is to make the index see only a tiny prefix of
every chunk. `embedding.index_max_chars` truncates the text that both dense retrieval and
BM25 index. Set it very small and retrieval can no longer see most of each provision.

You need the one-time setup above, including a recorded baseline. Then:

```bash
git checkout -b demo/bad-chunk-size
# configs/default.yaml -> embedding.index_max_chars: 60
sed -i.bak 's/^  index_max_chars: null/  index_max_chars: 60/' configs/default.yaml && rm configs/default.yaml.bak
git commit -am "Demo: index only the first 60 characters of each chunk"
git push -u origin demo/bad-chunk-size
gh pr create --title "Demo: bad chunk size" --body "Deliberately degrades retrieval to show the eval gate."
```

What to expect:

- **Lint + tests** passes; the change is valid code and config.
- **Eval gate** re-indexes in memory. The documents retrieved for each question change,
  so those answer and judge calls are cache misses and are billed (cents).
- **recall@5 and MRR** should fall well past their 0.06 limit (0.05 + 0.01). This is a
  prediction; the demo hasn't been run yet. Faithfulness and relevance usually fall too,
  as answers lose their supporting provisions.
- **The comment** reads **❌ Eval gate failed**, with the failing rows marked ❌. The job
  goes red and branch protection blocks the merge.
- **Screenshots:** take one of the PR comment and one of the checks box showing
  "Eval gate — failing". Put them in the README, e.g. `docs/img/blocked-pr.png`.
- **Afterwards:** close the PR without merging. **Don't** update the baseline from this
  run.

Illustrative layout of the comment (the numbers are placeholders, not measured results):

```
## ❌ Eval gate failed
| Metric | Baseline | This PR | Δ | Allowed drop (+noise) | |
|---|---|---|---|---|---|
| **recall@5** | 0.xxx | 0.yyy | -0.zzz | 0.05 (+0.01) | ❌ |
| **mrr** | … | … | … | 0.05 (+0.01) | ❌ |
| **faithfulness** | … | … | … | 0.25 (+0.15) | ✅ or ❌ |
| **relevance** | … | … | … | 0.25 (+0.15) | ✅ or ❌ |
| items with errors | – | 0 | – | max 0 | ✅ |
```

Other ways to trigger the gate:

| Change | Main effect |
|---|---|
| `generation.k: 1` | Context recall drops; relevance falls on multi-provision questions |
| `retrieval.mode: bm25` | Recall and MRR drop on paraphrased questions such as `temporal` and `scenario` |
| Deleting the citation and date rules from `prompts/answer_system.md` | Faithfulness and relevance drop. Retrieval metrics stay the same, which shows that the answer gates are independent of the retrieval gates |

To preview the comment locally without GitHub, given any `results.json`:

```bash
uv run dpdp-eval-gate --results data/eval_runs/<run_id>/results.json --comment-out /tmp/comment.md
```

When using a persistent local Qdrant (`docker compose`) instead of CI's in-memory one,
run `uv run dpdp-index` after changing `index_max_chars`. The collection is only rebuilt
automatically when the chunk count or vector size changes.
