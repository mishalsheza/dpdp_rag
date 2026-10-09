# Load test

`loadtest/locustfile.py` sends questions from `data/golden.jsonl` to `POST /ask`, with a
`GET /healthz` call 1 time in 11. Each simulated user waits 0.5–1.5 s between requests.

## Cost-free run: serving path only (stub LLM)

`configs/loadtest.yaml` replaces Claude with a stub that cites the top document and labels
its answer `[stub answer: no language model was called]`. It also:

- keeps Qdrant in memory
- turns off the response cache, so every request does real retrieval
- writes metrics to `data/cache/loadtest_metrics.db`

The test therefore measures the API, query embedding, Qdrant, BM25, fusion,
cross-reference expansion and prompt building, at no cost.

```bash
# terminal 1: the API with the load-test overrides
DPDP_CONFIG_OVERRIDES=loadtest.yaml uv run dpdp-api
# warm up once: the first request embeds the 364 chunks into in-memory Qdrant (~40 s)
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Within how many hours must a breach be reported?"}' > /dev/null
# terminal 2: 10 users for 60 s
uv run --group load locust -f loadtest/locustfile.py --headless \
    -u 10 -r 5 -t 60s --host http://127.0.0.1:8000
```

### Sample output

Recorded 2026-10-09 on an Apple M3 MacBook Air (8 cores), one uvicorn worker,
`BAAI/bge-small-en-v1.5` on CPU, commit `fd962e2` plus local changes.

10 users, 60 s:

```
Type     Name      # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
--------|---------|-------|-------------|-------|-------|-------|-------|--------|-----------
POST     /ask         524     0(0.00%) |     19       9     156     16 |    8.79        0.00
GET      /healthz      52     0(0.00%) |      3       0      22      2 |    0.87        0.00
--------|---------|-------|-------------|-------|-------|-------|-------|--------|-----------
         Aggregated   576     0(0.00%) |     18       0     156     15 |    9.66        0.00

Response time percentiles (approximated)
Type     Name          50%    66%    75%    80%    90%    95%    98%    99%  99.9%   100% # reqs
POST     /ask           16     18     20     21     28     42     81     96    160    160    524
GET      /healthz        2      2      3      3      4     11     19     23     23     23     52
```

50 users, 60 s (`-u 50 -r 10`):

```
Type     Name      # reqs      # fails |    Avg     Min     Max    Med |   req/s  failures/s
POST     /ask        2595     0(0.00%) |     19       8     439     14 |   43.36        0.00
GET      /healthz     245     0(0.00%) |      3       0     101      1 |    4.09        0.00
         Aggregated  2840     0(0.00%) |     18       0     439     14 |   47.46        0.00

Response time percentiles (approximated)
POST     /ask           14     16     18     19     25     35     73    110    430    440   2595
```

### Reading these numbers

- **The server wasn't saturated.** At about 1 s between requests per user, 50 users offer
  roughly 50 requests per second. The API served 43 `/ask` per second with p50 14 ms and
  no failures, so 43 req/s is a lower bound on serving throughput on this machine, not a
  maximum. To find the ceiling, raise `-u` until p99 climbs or failures appear.
- **The API's own log agrees.** Its `metrics.db` recorded 3,121 requests with p50 11 ms
  and p99 79 ms. These are API-internal latencies, without client and HTTP overhead. The
  dashboard (`uv run streamlit run src/dpdp_rag/ui/app.py` with the same
  `DPDP_CONFIG_OVERRIDES`) shows them.
- **These are not end-user latencies.** With Claude Haiku, each uncached `/ask` adds the
  model's time to generate the answer, likely around a second or more. That hasn't been
  measured yet. Throughput then depends mainly on the Anthropic rate limits for the
  account, not on this server.
- **Cold start:** the first request after start-up embeds the whole corpus when Qdrant
  is in memory (about 40 s here). With docker-compose, the indexer does this once ahead
  of time.

## Real-model run (costs money)

To measure end-to-end latency, drop the stub with `DPDP_CONFIG_OVERRIDES=` (empty), set
`ANTHROPIC_API_KEY`, start Qdrant, run `uv run dpdp-index`, and run locust with fewer
users and a short duration. The cost is roughly requests × the per-request cost reported
by `/ask`. The response cache is on by default, so repeated questions will be served from
it; set `api.response_cache.enabled: false` to measure model calls.
