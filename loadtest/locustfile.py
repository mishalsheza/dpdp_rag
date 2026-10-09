"""Load test for the API.

    uv run --group load locust -f loadtest/locustfile.py --headless \
        -u 10 -r 5 -t 60s --host http://127.0.0.1:8000

Questions come from data/golden.jsonl, so requests look like real traffic. See
docs/LOADTEST.md for how to start the API for a cost-free run and for sample output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

GOLDEN = Path(__file__).resolve().parents[1] / "data" / "golden.jsonl"
QUESTIONS = [
    json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()
]


class AskUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def ask(self) -> None:
        item = random.choice(QUESTIONS)
        payload = {"question": item["question"]}
        if item.get("as_of_date"):
            payload["as_of_date"] = item["as_of_date"]
        with self.client.post("/ask", json=payload, name="/ask", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}: {r.text[:200]}")
            elif "answer" not in r.json():
                r.failure("response has no answer")

    @task(1)
    def healthz(self) -> None:
        self.client.get("/healthz", name="/healthz")
