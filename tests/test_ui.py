"""Chat UI helpers, dashboard data and charts, Streamlit pages (AppTest), stub LLM,
config overrides, the ablation, and the Docker / load-test files."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from conftest import GOLDEN_TINY, TINY_CHUNKS
from dpdp_rag.config import REPO_ROOT, load_config, load_system_config, merge
from dpdp_rag.eval.ablation import Span as ASpan
from dpdp_rag.eval.ablation import covers, fixed_chunks, run_ablation, score_item
from dpdp_rag.generation.llm import StubLLM, make_answer_llm
from dpdp_rag.metrics import MetricsStore, RequestMetric
from dpdp_rag.retrieval.store import ChunkStore
from dpdp_rag.ui import dashboard_data as dd
from dpdp_rag.ui import render
from dpdp_rag.ui.charts import PALETTE, TEXT, line_chart

CIT = {
    "chunk_id": "dpdp_rules_2025:r13(5)",
    "pinpoint": "Rule 13(5)",
    "gsr_no": "G.S.R. 846(E)",
    "page": 29,
    "in_force_date": "2027-05-13",
    "in_force": False,
    "corrected_by": "G.S.R. 892(E)",
    "title": "Additional obligations",
    "text": "(5) ...",
}


# -- chat helpers --------------------------------------------------------------------------


def test_badges_not_yet_in_force_and_corrected() -> None:
    badges = render.citation_badges(CIT, date(2026, 1, 1))
    assert [b.label for b in badges] == [
        "In force from 2027-05-13 · not yet on 2026-01-01",
        "Corrected by G.S.R. 892(E)",
    ]
    assert [b.color for b in badges] == ["orange", "violet"]
    md = render.badge_markdown(badges)
    assert md.startswith(":orange-badge[:material/schedule: In force from 2027-05-13")
    assert ":violet-badge[:material/edit_note: Corrected by G.S.R. 892(E)]" in md


def test_badges_in_force_without_correction() -> None:
    cit = CIT | {"in_force": True, "corrected_by": None}
    (badge,) = render.citation_badges(cit, date(2027, 6, 1))
    assert badge.label == "In force from 2027-05-13" and badge.color == "green"


def test_source_line_and_footer() -> None:
    assert render.source_line(CIT) == "G.S.R. 846(E) · p. 29 · `dpdp_rules_2025:r13(5)`"
    resp = {
        "latency_ms": 812.4,
        "cost_usd": 0.00045,
        "tokens": {"total": 4100},
        "model": "claude-haiku-5-5",
        "config_hash": "abcdef123456",
        "cached": True,
    }
    assert render.footer(resp) == (
        "812 ms · $0.00045 · 4100 tokens · model `claude-haiku-5-5` · config `abcdef12` · cached"
    )


def test_ask_and_errors(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        calls.append({"url": url, "json": json})
        if "fail" in json["question"]:
            return httpx.Response(503, json={"detail": "Cannot reach Qdrant"})
        return httpx.Response(200, json={"answer": "ok"})

    monkeypatch.setattr(render.httpx, "post", fake_post)
    assert render.ask("http://api:8000/", "q", date(2026, 1, 1), 5) == {"answer": "ok"}
    assert calls[0] == {
        "url": "http://api:8000/ask",
        "json": {"question": "q", "as_of_date": "2026-01-01"},
    }
    with pytest.raises(RuntimeError, match="API error 503: Cannot reach Qdrant"):
        render.ask("http://api:8000", "fail", date(2026, 1, 1), 5)


def test_api_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("DPDP_API_URL", "http://api:8000")
    assert render.api_url({"api_url": "http://127.0.0.1:8000"}) == "http://api:8000"


# -- dashboard data ------------------------------------------------------------------------


def _metrics_db(path: Path) -> MetricsStore:
    store = MetricsStore(path)
    t0 = datetime(2026, 10, 1, 9, tzinfo=UTC)
    for i in range(10):
        store.record(
            RequestMetric(
                "h1",
                100.0 + i,
                1000,
                100,
                0.001,
                cached=i == 0,
                refused=i >= 8,
                source="api",
                timestamp=t0 + timedelta(hours=i),
            )
        )
    store.record(
        RequestMetric(
            "h2", 900.0, 2000, 200, 0.004, False, False, "eval", timestamp=t0 + timedelta(days=1)
        )
    )
    return store


def _eval_runs(root: Path) -> None:
    for i, (r5, faith) in enumerate([(0.6, 4.0), (0.7, 4.4)]):
        run = {
            "run_id": f"run{i}",
            "started_at": f"2026-10-0{i + 1}T10:00:00+00:00",
            "n": 8,
            "config_hash": f"hash{i}" * 4,
            "overall": {
                "retrieval": {"recall@5": r5, "mrr": 0.5},
                "answers": {
                    "faithfulness_mean": faith,
                    "relevance_mean": 4.0,
                    "refusal_correctness": 1.0,
                },
            },
            "by_category": {
                "lookup": {
                    "n": 1,
                    "retrieval": {"recall@5": 1.0, "mrr": 1.0},
                    "answers": {"faithfulness_mean": 5.0},
                }
            },
        }
        (root / f"run{i}").mkdir(parents=True)
        (root / f"run{i}" / "results.json").write_text(json.dumps(run))
    (root / "broken").mkdir()
    (root / "broken" / "results.json").write_text("{not json")


def test_requests_summary_and_buckets(tmp_path) -> None:
    _metrics_db(tmp_path / "m.db")
    df = dd.load_requests(tmp_path / "m.db")
    assert len(df) == 11
    api = dd.filter_requests(df, source="api")
    s = dd.summary(api)
    assert s["requests"] == 10 and s["p50_latency_ms"] == 104.0 and s["p99_latency_ms"] == 109.0
    assert s["refusal_rate"] == pytest.approx(0.2)
    assert dd.summary(dd.filter_requests(df, include_cached=False))["requests"] == 10
    assert dd.filter_requests(df, config_hashes=["h2"])["source"].tolist() == ["eval"]
    daily = dd.over_time(df, "day")
    assert daily["requests"].tolist() == [10, 1]
    assert dd.over_time(df, "hour")["requests"].sum() == 11
    assert dd.load_requests(tmp_path / "missing.db").empty


def test_eval_history_and_categories(tmp_path) -> None:
    _eval_runs(tmp_path / "runs")
    runs = dd.load_eval_runs(tmp_path / "runs")
    assert [r["run_id"] for r in runs] == ["run0", "run1"]  # broken file skipped
    hist = dd.eval_history(runs)
    r5 = hist[hist["metric"] == "recall@5"]["value"].tolist()
    assert r5 == [0.6, 0.7]
    assert set(hist["metric"]) == set(dd.EVAL_SCORES)
    table = dd.category_table(runs[-1])
    assert table.loc[0, "category"] == "lookup" and table.loc[0, "recall@5"] == 1.0


# -- charts --------------------------------------------------------------------------------


def _frame() -> Any:
    import pandas as pd

    t = pd.date_range("2026-10-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "t": list(t) * 2,
            "v": [1, 2, 3, 4, 10, 20, 30, 40],
            "s": ["p50"] * 4 + ["p99"] * 4,
            "requests": [5] * 8,
        }
    )


def test_chart_colours_follow_series_and_mode() -> None:
    spec = line_chart(
        _frame(), x="t", y="v", series="s", series_order=["p50", "p99"], y_title="ms", mode="dark"
    ).to_dict()
    text = json.dumps(spec)
    assert all(c in text for c in PALETTE["dark"][:2])
    assert PALETTE["light"][0] not in text
    labels = [lyr for lyr in spec["layer"] if lyr.get("mark", {}).get("type") == "text"]
    assert labels and labels[0]["encoding"]["color"] == {"value": TEXT["dark"]}  # text ink


def test_chart_single_series_and_log_scale() -> None:
    df = _frame()
    spec = line_chart(
        df[df["s"] == "p50"],
        x="t",
        y="v",
        series=None,
        series_order=["cost"],
        y_title="USD",
        log_scale=True,
    ).to_dict()
    text = json.dumps(spec)
    assert '"type": "log"' in text
    assert not [lyr for lyr in spec["layer"] if lyr.get("mark", {}).get("type") == "text"]
    assert '"legend"' not in text or '"legend": null' in text


def test_chart_skips_colliding_direct_labels() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-10-01", "2026-10-02"] * 2),
            "v": [1.0, 100.0, 1.0, 99.5],
            "s": ["a", "a", "b", "b"],
        }
    )
    spec = line_chart(df, x="t", y="v", series="s", series_order=["a", "b"], y_title="x").to_dict()
    assert not [lyr for lyr in spec["layer"] if lyr.get("mark", {}).get("type") == "text"]


# -- Streamlit pages -----------------------------------------------------------------------


@pytest.fixture
def ui_overrides(tmp_path, monkeypatch) -> Path:
    _metrics_db(tmp_path / "m.db")
    _eval_runs(tmp_path / "runs")
    over = tmp_path / "ui_test.yaml"
    over.write_text(
        yaml.safe_dump(
            {
                "metrics": {"db_path": str(tmp_path / "m.db")},
                "ui": {"eval_runs_dir": str(tmp_path / "runs"), "api_url": "http://127.0.0.1:9"},
            }
        )
    )
    monkeypatch.setenv("DPDP_CONFIG_OVERRIDES", str(over))
    return tmp_path


def test_dashboard_page_renders(ui_overrides) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(
        str(REPO_ROOT / "src/dpdp_rag/ui/dashboard_page.py"), default_timeout=60
    ).run()
    assert not at.exception
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Requests"] == "11" and tiles["Refusal rate"] == "18%"
    assert [h.value for h in at.subheader][:3] == ["Latency", "Cost per request", "Refusal rate"]
    assert any("run1" in h.value for h in at.subheader)


def test_chat_page_renders_answer_with_badges(ui_overrides, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    resp = {
        "answer": "Rule 13(5) defines the committee.",
        "citations": [CIT],
        "refused": False,
        "refusal_reason": None,
        "latency_ms": 10,
        "cost_usd": 0.0001,
        "tokens": {"total": 10},
        "model": "stub",
        "config_hash": "abc12345",
        "cached": False,
    }
    monkeypatch.setattr(render, "ask", lambda *a, **k: resp)
    monkeypatch.setattr(render, "version", lambda *a, **k: None)
    at = AppTest.from_file(str(REPO_ROOT / "src/dpdp_rag/ui/chat_page.py"), default_timeout=30)
    at.run()
    assert not at.exception and at.title[0].value == "DPDP Act & Rules assistant"
    at.sidebar.date_input[0].set_value(date(2026, 1, 1)).run()
    at.chat_input[0].set_value("What is the committee?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Rule 13(5) defines the committee." in text
    assert "Corrected by G.S.R. 892(E)" in text and "not yet on 2026-01-01" in text
    assert at.expander[0].label.startswith("Rule 13(5)")


def test_chat_page_shows_api_errors(ui_overrides, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    def boom(*a: Any, **k: Any) -> dict[str, Any]:
        raise RuntimeError("API error 503: Cannot reach Qdrant")

    monkeypatch.setattr(render, "ask", boom)
    monkeypatch.setattr(render, "version", lambda *a, **k: None)
    at = AppTest.from_file(str(REPO_ROOT / "src/dpdp_rag/ui/chat_page.py"), default_timeout=30)
    at.run()
    at.chat_input[0].set_value("q").run()
    assert at.error[0].value == "API error 503: Cannot reach Qdrant"


# -- stub LLM and config overrides ---------------------------------------------------------


def test_stub_llm_is_labelled_and_cites_first_document() -> None:
    llm = make_answer_llm({"provider": "stub"})
    assert isinstance(llm, StubLLM) and llm.model == "stub"
    out = json.loads(llm.generate("s", '<document chunk_id="x:1">t</document>', {}).text)
    assert out["answer"].startswith("[stub answer: no language model was called]")
    assert out["citations"][0]["chunk_id"] == "x:1"
    with pytest.raises(ValueError):
        make_answer_llm({"provider": "nope"})


def test_system_config_overrides_from_env(tmp_path, monkeypatch) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("api: {port: 9001}\nllm: {provider: stub}\n")
    b.write_text("api: {port: 9002}\n")
    monkeypatch.setenv("DPDP_CONFIG_OVERRIDES", f"{a}, {b}")
    cfg = load_system_config()
    assert cfg["api"]["port"] == 9002 and cfg["llm"]["provider"] == "stub"
    assert cfg["api"]["host"] == load_config("default.yaml")["api"]["host"]
    monkeypatch.setenv("DPDP_CONFIG_OVERRIDES", "loadtest.yaml")
    lt = load_system_config()
    assert lt["llm"]["provider"] == "stub" and lt["qdrant"]["location"] == ":memory:"
    assert lt["api"]["response_cache"]["enabled"] is False


def test_version_reports_stub_model(api_config) -> None:
    from fastapi.testclient import TestClient

    from dpdp_rag.api import create_app

    cfg = merge(api_config, {"llm": {"provider": "stub"}})
    assert TestClient(create_app(cfg)).get("/version").json()["model"] == "stub"


# -- ablation ------------------------------------------------------------------------------


def test_fixed_chunks_spans_and_coverage() -> None:
    store = ChunkStore.from_file(TINY_CHUNKS)
    fixed, spans = fixed_chunks(store.chunks, 300, 50)
    assert all(len(c.text) <= 300 for c in fixed)
    by_doc: dict[str, list[ASpan]] = {}
    for c in fixed:
        by_doc.setdefault(c.doc_id, []).append(spans[c.chunk_id])
    for doc_spans in by_doc.values():  # consecutive windows overlap by 50 chars
        for a, b in zip(doc_spans, doc_spans[1:], strict=False):
            assert b.start == a.start + 250
    gold, window = ASpan("d", 100, 200), ASpan("d", 140, 400)
    assert covers(window, gold, 0.5) and not covers(window, gold, 0.7)
    assert not covers(ASpan("other", 100, 200), gold, 0.1)
    with pytest.raises(ValueError):
        fixed_chunks(store.chunks, 100, 100)


def test_score_item_structure_and_fixed() -> None:
    m = score_item(["x", "g1", "g2"], ["g1", "g2"], [1, 3], 10)
    assert m == {"recall@1": 0.0, "recall@3": 1.0, "hit_rate@3": 1.0, "mrr": 0.5}
    fixed = score_item(
        ["f1", "f2"], ["g1", "g2"], [1, 3], 10, relevant={"f1": {"g1", "g2"}, "f2": set()}
    )
    assert fixed["recall@1"] == 1.0 and fixed["mrr"] == 1.0


def test_run_ablation_on_fixture(retrieval_config) -> None:
    cfg = load_config("ablation.yaml") | {
        "golden_file": str(GOLDEN_TINY),
        "variants": [
            {"name": "dense", "chunks": "structure", "mode": "dense", "rerank": False},
            {"name": "bm25 fixed", "chunks": "fixed", "mode": "bm25", "rerank": False},
        ],
        "fixed_chunks": {"size_chars": 400, "overlap_chars": 50, "min_overlap": 0.5},
    }
    res = run_ablation(cfg, retrieval_config)
    assert res["n_items"] == 4 and res["n_structure_chunks"] == 15
    assert [v["variant"] for v in res["variants"]] == ["dense", "bm25 fixed"]
    for v in res["variants"]:
        assert 0 <= v["metrics"]["recall@5"] <= 1 and v["n"] == 4


# -- Docker and load test ------------------------------------------------------------------


def test_compose_defines_the_stack() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    services = compose["services"]
    assert set(services) == {"qdrant", "indexer", "api", "ui"}
    assert services["api"]["depends_on"]["indexer"]["condition"] == "service_completed_successfully"
    assert services["ui"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert services["api"]["environment"]["DPDP_CONFIG_OVERRIDES"] == "docker.yaml"
    assert services["ui"]["environment"]["DPDP_API_URL"] == "http://api:8000"
    docker_cfg = load_config("docker.yaml")
    assert docker_cfg["api"]["host"] == "0.0.0.0"
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    for needed in (
        "COPY src",
        "COPY configs",
        "COPY prompts",
        "COPY data/processed",
        "uv sync --frozen --no-dev --group ui",
    ):
        assert needed in dockerfile


def test_locustfile_targets_the_api() -> None:
    """Checked statically: importing locust under pytest triggers gevent monkey-patching."""
    import ast

    source = (REPO_ROOT / "loadtest/locustfile.py").read_text()
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    user = classes["AskUser"]
    assert [b.id for b in user.bases if isinstance(b, ast.Name)] == ["HttpUser"]
    tasks = [
        f.name
        for f in user.body
        if isinstance(f, ast.FunctionDef)
        and any("task" in ast.unparse(d) for d in f.decorator_list)
    ]
    assert tasks == ["ask", "healthz"]
    assert '"/ask"' in source and "golden.jsonl" in source
    assert (REPO_ROOT / "data/golden.jsonl").exists()
