"""POST /ask, /healthz and /version with a mocked LLM and the tiny fixture corpus."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpdp_rag.api import create_app
from dpdp_rag.api.versioning import config_hash
from dpdp_rag.generation.cost import Usage, cost_usd
from dpdp_rag.metrics import MetricsStore, Where
from fakes import FakeLLM, RecordingTracer

R12_QUESTION = "Which Data Fiduciaries are exempt from section 9 under the Fourth Schedule?"


def cite(*ids: str, refused: bool = False, reason: str = "none") -> dict[str, Any]:
    return {
        "answer": "The text says ...",
        "refused": refused,
        "refusal_reason": reason,
        "citations": [{"chunk_id": i, "pinpoint": "x"} for i in ids],
    }


@pytest.fixture
def make(api_config):
    def _make(
        llm: FakeLLM | None = None, **overrides: Any
    ) -> tuple[TestClient, FakeLLM, RecordingTracer]:
        llm = llm or FakeLLM()
        tracer = RecordingTracer()
        app = create_app(api_config, llm=llm, tracer=tracer, **overrides)
        return TestClient(app), llm, tracer

    return _make


def test_ask_response_shape(make, api_config) -> None:
    llm = FakeLLM(
        cite("dpdp_rules_2025:r12(1)", "dpdp_act_2023:s9(1)"),
        usage=Usage(input_tokens=1200, output_tokens=300, cache_read_tokens=800),
    )
    client, _, _ = make(llm)
    r = client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "answer",
        "citations",
        "refused",
        "refusal_reason",
        "as_of_date",
        "latency_ms",
        "tokens",
        "cost_usd",
        "config_hash",
        "model",
        "cached",
    }
    assert body["answer"] == "The text says ..."
    assert body["refused"] is False and body["refusal_reason"] is None
    assert body["latency_ms"] > 0
    assert body["tokens"] == {
        "input": 1200,
        "output": 300,
        "cache_read": 800,
        "cache_write": 0,
        "total": 2300,
    }
    assert body["cost_usd"] == pytest.approx(cost_usd(llm.usage, api_config["llm"]["pricing"]))
    assert body["config_hash"] == config_hash(api_config)
    assert body["model"] == "fake-haiku" and body["cached"] is False


def test_citations_get_pinpoints_and_in_force_status(make) -> None:
    client, _, _ = make(FakeLLM(cite("dpdp_rules_2025:r12(1)", "dpdp_act_2023:s9(1)")))
    body = client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"}).json()
    c12, c9 = body["citations"]
    assert c12["pinpoint"] == "Rule 12(1)"  # computed from metadata, not taken from the model
    assert c9["pinpoint"] == "Section 9(1)"
    assert c12["in_force_date"] == "2027-05-13" and c12["in_force"] is False
    assert c12["gsr_no"] == "G.S.R. 846(E)" and c12["page"] == 29


def test_unknown_citations_dropped_and_duplicates_merged(make) -> None:
    client, _, _ = make(
        FakeLLM(cite("dpdp_rules_2025:r12(1)", "made-up:id", "dpdp_rules_2025:r12(1)"))
    )
    body = client.post("/ask", json={"question": R12_QUESTION}).json()
    assert [c["chunk_id"] for c in body["citations"]] == ["dpdp_rules_2025:r12(1)"]


def test_uncited_answer_becomes_refusal(make, api_config) -> None:
    client, _, _ = make(FakeLLM(cite("made-up:id")))
    body = client.post("/ask", json={"question": R12_QUESTION}).json()
    assert body["refused"] is True and body["refusal_reason"] == "insufficient_context"
    assert body["citations"] == []
    assert body["answer"].startswith("I can't answer that")
    assert body["cost_usd"] > 0  # the model was still called and billed


def test_model_refuses_insufficient_context(make) -> None:
    llm = FakeLLM(
        cite(refused=True, reason="insufficient_context")
        | {"answer": "The documents cover consent but not cookies."}
    )
    client, _, _ = make(llm)
    body = client.post("/ask", json={"question": "What about cookie banners?"}).json()
    assert body["refused"] is True and body["refusal_reason"] == "insufficient_context"
    assert body["answer"] == "The documents cover consent but not cookies."


def test_legal_advice_refusal_still_explains_the_text(make, api_config) -> None:
    llm = FakeLLM(
        cite("dpdp_act_2023:s9(1)", refused=True, reason="legal_advice")
        | {"answer": "I can't assess compliance. Section 9(1) requires ..."}
    )
    client, _, _ = make(llm)
    body = client.post(
        "/ask", json={"question": "Is my edtech company compliant with s. 9?"}
    ).json()
    assert body["refused"] is True and body["refusal_reason"] == "legal_advice"
    assert body["citations"][0]["pinpoint"] == "Section 9(1)"
    assert "Section 9(1) requires" in body["answer"]
    store = MetricsStore(Path(api_config["metrics"]["db_path"]))
    assert store.request_counts()["refused"] == 1


def test_no_retrieval_results_refuses_without_calling_llm(api_config) -> None:
    from dpdp_rag.config import merge

    cfg = merge(api_config, {"retrieval": {"mode": "bm25"}})
    llm = FakeLLM()
    client = TestClient(create_app(cfg, llm=llm, tracer=RecordingTracer()))
    body = client.post("/ask", json={"question": "the of and"}).json()  # stopwords only
    assert body["refused"] is True and body["refusal_reason"] == "insufficient_context"
    assert llm.calls == [] and body["cost_usd"] == 0.0


def test_model_safety_refusal(make) -> None:
    client, _, _ = make(FakeLLM(stop_reason="refusal", raw_text=""))
    body = client.post("/ask", json={"question": R12_QUESTION}).json()
    assert body["refused"] is True and body["refusal_reason"] == "model_refusal"


@pytest.mark.parametrize("llm", [FakeLLM(raw_text="not json"), FakeLLM(stop_reason="max_tokens")])
def test_bad_model_output_is_502(make, llm) -> None:
    client, _, _ = make(llm)
    r = client.post("/ask", json={"question": R12_QUESTION})
    assert r.status_code == 502


def test_validation(make) -> None:
    client, _, _ = make()
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
    assert (
        client.post("/ask", json={"question": "q", "as_of_date": "13/05/2027"}).status_code == 422
    )


def test_as_of_date_defaults_to_today(make) -> None:
    client, llm, _ = make()
    body = client.post("/ask", json={"question": R12_QUESTION}).json()
    assert body["as_of_date"] == date.today().isoformat()
    assert f"As-of date: {date.today().isoformat()}" in llm.calls[0]["user"]


def test_response_cache(make, api_config) -> None:
    client, llm, _ = make()
    first = client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"}).json()
    second = client.post(
        "/ask", json={"question": "  " + R12_QUESTION + " ", "as_of_date": "2026-01-01"}
    ).json()
    assert len(llm.calls) == 1
    assert first["cached"] is False and second["cached"] is True
    assert second["answer"] == first["answer"] and second["citations"] == first["citations"]
    assert second["cost_usd"] == 0.0 and second["tokens"]["total"] == 0
    # A different as-of date is a different answer.
    client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2028-01-01"})
    assert len(llm.calls) == 2
    store = MetricsStore(Path(api_config["metrics"]["db_path"]))
    assert store.request_counts() == {"total": 3, "cached": 1, "refused": 0, "api": 3, "eval": 0}


def test_cache_disabled(api_config) -> None:
    from dpdp_rag.config import merge

    cfg = merge(api_config, {"api": {"response_cache": {"enabled": False}}})
    llm = FakeLLM()
    client = TestClient(create_app(cfg, llm=llm, tracer=RecordingTracer()))
    for _ in range(2):
        client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"})
    assert len(llm.calls) == 2


def test_metrics_row_per_request(make, api_config) -> None:
    client, _, _ = make(
        FakeLLM(usage=Usage(input_tokens=1000, output_tokens=200, cache_write_tokens=500))
    )
    body = client.post("/ask", json={"question": R12_QUESTION}).json()
    store = MetricsStore(Path(api_config["metrics"]["db_path"]))
    import sqlite3

    rows = (
        sqlite3.connect(store.db_path)
        .execute(
            "SELECT config_hash, tokens_in, tokens_out, cost_usd, cached, refused, source,"
            " latency_ms FROM requests"
        )
        .fetchall()
    )
    assert len(rows) == 1
    chash, tin, tout, cost, cached, refused, source, latency = rows[0]
    assert (chash, tin, tout, cached, refused, source) == (
        body["config_hash"],
        1500,
        200,
        0,
        0,
        "api",
    )
    assert cost == pytest.approx(body["cost_usd"]) and latency == pytest.approx(body["latency_ms"])
    assert store.request_counts(Where(source="eval"))["total"] == 0


def test_trace_carries_config_hash_and_steps(make, api_config) -> None:
    client, _, tracer = make()
    body = client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"}).json()
    (trace,) = tracer.traces
    assert trace["config_hash"] == body["config_hash"] == config_hash(api_config)
    assert trace["input"] == {"question": R12_QUESTION, "as_of_date": "2026-01-01"}
    retrieve, generate = trace["steps"]
    assert retrieve["name"] == "retrieve" and retrieve["output"]
    assert generate["as_type"] == "generation" and generate["model"] == "fake-haiku"
    assert generate["usage"]["output"] == 200
    assert generate["cost_usd"] == pytest.approx(body["cost_usd"])
    assert trace["output"]["answer"] == body["answer"]
    # Cached requests are traced too, without LLM steps.
    client.post("/ask", json={"question": R12_QUESTION, "as_of_date": "2026-01-01"})
    assert tracer.traces[1]["steps"] == [] and tracer.traces[1]["end_metadata"]["cached"] is True


def test_healthz_and_version(make, api_config, monkeypatch) -> None:
    monkeypatch.setenv("GIT_SHA", "abc123")
    client, _, _ = make()
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/version").json() == {
        "git_sha": "abc123",
        "config_hash": config_hash(api_config),
        "model": api_config["llm"]["model"],
    }


def test_tracer_flushed_on_shutdown(api_config) -> None:
    tracer = RecordingTracer()
    with TestClient(create_app(api_config, llm=FakeLLM(), tracer=tracer)):
        pass
    assert tracer.flushed == 1


def test_retrieval_backend_down_is_503(api_config) -> None:
    from dpdp_rag.config import merge

    cfg = merge(api_config, {"qdrant": {"auto_index": False}})  # empty in-memory Qdrant
    llm = FakeLLM()
    client = TestClient(create_app(cfg, llm=llm, tracer=RecordingTracer()))
    r = client.post("/ask", json={"question": R12_QUESTION})
    assert r.status_code == 503 and "dpdp-index" in r.json()["detail"]
    assert llm.calls == []
