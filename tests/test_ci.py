"""CI pieces: LLM/embedding caches, spend cap, regression gates, baseline updates,
config overrides, and consistency between the workflow and the configs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from dpdp_rag.config import REPO_ROOT, load_config, merge
from dpdp_rag.eval.baseline import main as baseline_main
from dpdp_rag.eval.cli import load_configs
from dpdp_rag.eval.gates import COMMENT_MARKER, compare, lookup, render_comment, snapshot
from dpdp_rag.eval.gates import main as gate_main
from dpdp_rag.eval.runner import EvalRunner
from dpdp_rag.generation.cost import Usage
from dpdp_rag.generation.llm_cache import Budget, BudgetExceeded, CachingLLM
from dpdp_rag.kvcache import SqliteKV, digest
from dpdp_rag.metrics import MetricsStore
from dpdp_rag.retrieval import Retriever
from dpdp_rag.retrieval.embedding_cache import CachingEmbedder
from dpdp_rag.retrieval.embeddings import HashEmbedder, make_embedder
from dpdp_rag.retrieval.store import render
from fakes import FakeLLM, RecordingTracer
from test_eval import FakeJudgeLLM

GATES = load_config("gates.yaml")
PRICING = {
    "input": 1.0,
    "output": 5.0,
    "long_context_threshold": 10**9,
    "long_input": 1.0,
    "long_output": 5.0,
    "cache_write_multiplier": 1.25,
    "cache_read_multiplier": 0.1,
}


# -- caches --------------------------------------------------------------------------------


def test_kv_roundtrip(tmp_path) -> None:
    kv = SqliteKV(tmp_path / "c.sqlite", "t")
    kv.put_many({"a": [1, 2], "b": {"x": 1}})
    assert kv.get("a") == [1, 2] and kv.get("missing") is None
    assert kv.get_many(["a", "b", "c"]) == {"a": [1, 2], "b": {"x": 1}}
    assert len(SqliteKV(tmp_path / "c.sqlite", "t")) == 2  # persisted
    assert digest("a", {"y": 1, "x": 2}) == digest("a", {"x": 2, "y": 1})


def test_llm_cache_replays_identical_calls_only(tmp_path) -> None:
    inner = FakeLLM(usage=Usage(input_tokens=1000, output_tokens=100))
    llm = CachingLLM(
        inner,
        SqliteKV(tmp_path / "l.sqlite", "llm_calls"),
        {"model": "m", "temperature": 0},
        PRICING,
    )
    first = llm.generate("sys", 'chunk_id="a"', {"type": "object"})
    again = llm.generate("sys", 'chunk_id="a"', {"type": "object"})
    assert len(inner.calls) == 1 and again.text == first.text and again.usage == first.usage
    llm.generate("sys", 'chunk_id="b"', {"type": "object"})  # different prompt -> miss
    llm.generate("sys2", 'chunk_id="a"', {"type": "object"})  # different system -> miss
    assert len(inner.calls) == 3
    assert llm.stats.hits == 1 and llm.stats.misses == 3
    assert llm.stats.fresh_cost_usd == pytest.approx(3 * (1000 * 1.0 + 100 * 5.0) / 1e6)
    # A new process with other parameters (e.g. temperature) does not reuse the entry.
    other = CachingLLM(
        FakeLLM(),
        SqliteKV(tmp_path / "l.sqlite", "llm_calls"),
        {"model": "m", "temperature": 1},
        PRICING,
    )
    other.generate("sys", 'chunk_id="a"', {"type": "object"})
    assert other.stats.misses == 1


def test_llm_cache_does_not_store_truncated_output(tmp_path) -> None:
    inner = FakeLLM(stop_reason="max_tokens")
    llm = CachingLLM(inner, SqliteKV(tmp_path / "l.sqlite", "llm_calls"), {}, PRICING)
    llm.generate("s", "u", {})
    llm.generate("s", "u", {})
    assert len(inner.calls) == 2


def test_budget_stops_fresh_calls_but_not_cache_hits(tmp_path) -> None:
    budget = Budget(max_usd=0.002)
    inner = FakeLLM(usage=Usage(input_tokens=1000, output_tokens=200))  # $0.002 per call
    llm = CachingLLM(inner, SqliteKV(tmp_path / "l.sqlite", "llm_calls"), {}, PRICING, budget)
    llm.generate("s", "u1", {})
    assert budget.spent_usd == pytest.approx(0.002)
    llm.generate("s", "u1", {})  # cached: still allowed
    with pytest.raises(BudgetExceeded, match="budget"):
        llm.generate("s", "u2", {})
    assert len(inner.calls) == 1


def test_embedding_cache(tmp_path) -> None:
    class Counting(HashEmbedder):
        calls = 0

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            Counting.calls += len(texts)
            return super().embed_documents(texts)

    kv = SqliteKV(tmp_path / "e.sqlite", "embeddings")
    emb = CachingEmbedder(Counting(32), kv, "hash:")
    first = emb.embed_documents(["a b", "c d"])
    assert emb.embed_documents(["c d", "a b", "e f"])[:2] == [first[1], first[0]]
    assert Counting.calls == 3 and emb.hits == 2 and emb.misses == 3
    q = emb.embed_query("a b")  # queries are cached separately from documents
    assert emb.embed_query("a b") == q and emb.misses == 4
    cfg = {
        "provider": "hash",
        "dim": 32,
        "cache": {"enabled": True, "path": str(tmp_path / "x.sqlite")},
    }
    assert isinstance(make_embedder(cfg), CachingEmbedder)
    assert isinstance(make_embedder(cfg | {"cache": {"enabled": False}}), HashEmbedder)


def test_index_max_chars_truncates_indexed_text(retrieval_config) -> None:
    r = Retriever(retrieval_config)
    chunk = r.store.by_id["dpdp_rules_2025:r8(3)"]
    assert len(render("{title}\n{text}", chunk, 40)) == 40
    full = Retriever(merge(retrieval_config, {"retrieval": {"mode": "bm25"}}))
    short = Retriever(
        merge(
            retrieval_config, {"retrieval": {"mode": "bm25"}, "embedding": {"index_max_chars": 40}}
        )
    )
    q = "associated traffic data and other logs of the processing"  # body-only words
    assert full.retrieve(q, 1)[0].chunk.chunk_id == "dpdp_rules_2025:r8(3)"
    assert all(h.chunk.chunk_id != "dpdp_rules_2025:r8(3)" for h in short.retrieve(q, 3))


# -- gates ---------------------------------------------------------------------------------


def results(
    recall5: float = 0.8,
    mrr: float = 0.7,
    faith: float | None = 4.5,
    rel: float | None = 4.2,
    errors: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    overall = {
        "errors": errors,
        "retrieval": {
            "recall@1": 0.4,
            "recall@3": 0.6,
            "recall@5": recall5,
            "mrr": mrr,
            "hit_rate@5": 0.9,
            "context_recall": 0.95,
        },
        "answers": {
            "faithfulness_mean": faith,
            "relevance_mean": rel,
            "refusal_correctness": 1.0,
            "faithfulness_pass_rate": 1.0,
            "relevance_pass_rate": 0.9,
        },
    }
    return {
        "run_id": "r1",
        "n": 8,
        "config_hash": "c" * 64,
        "judge_hash": "j",
        "golden_hash": "g",
        "judge_model": "claude-haiku-4-5",
        "settings": {"default_as_of_date": "2026-10-09"},
        "overall": overall,
        "by_category": {"lookup": overall},
        "cost": {"fresh_usd": 0.01, "budget_usd": 0.5},
        "llm_cache": {"answer": {"hits": 6, "misses": 2}},
    } | extra


def baseline_of(res: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "run_id": "base",
        "reason": "initial",
        "judge_hash": "j",
        "golden_hash": "g",
        "default_as_of_date": "2026-10-09",
        **snapshot(res, GATES),
    } | extra


def statuses(report) -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {c.name: c.status for c in report.checks}


def test_lookup_paths() -> None:
    assert lookup(results(), "overall.retrieval.recall@5") == 0.8
    assert lookup(results(), "overall.nope.x") is None
    assert lookup(results(faith=None), "overall.answers.faithfulness_mean") is None


def test_identical_run_passes() -> None:
    r = results()
    report = compare(r, baseline_of(r), GATES)
    assert report.passed and set(statuses(report).values()) == {"pass"}


@pytest.mark.parametrize(
    ("current", "failed"),
    [
        (results(recall5=0.75), None),  # drop 0.05 == max_drop: allowed
        (results(recall5=0.741), None),  # within max_drop + tolerance (0.06)
        (results(recall5=0.73), "recall@5"),  # 0.07 > 0.06
        (results(mrr=0.5), "mrr"),
        (results(faith=4.1), None),  # 0.4 == 0.25 + 0.15 noise tolerance
        (results(faith=4.0), "faithfulness"),
        (results(rel=3.5), "relevance"),
        (results(rel=None), "relevance"),  # metric disappeared
    ],
)
def test_threshold_and_tolerance(current, failed) -> None:
    report = compare(current, baseline_of(results()), GATES)
    bad = [n for n, s in statuses(report).items() if s in ("fail", "missing")]
    assert bad == ([failed] if failed else [])
    assert report.passed is (failed is None)


def test_improvement_and_errors_gate() -> None:
    report = compare(results(recall5=0.9), baseline_of(results()), GATES)
    assert statuses(report)["recall@5"] == "improved" and report.passed
    errored = compare(results(errors=1), baseline_of(results()), GATES)
    assert not errored.passed and errored.errors_failed


def test_no_baseline_never_fails_but_says_so() -> None:
    placeholder = json.loads((REPO_ROOT / "eval/baseline.json").read_text())
    assert placeholder["metrics"] == {}
    report = compare(results(recall5=0.0), placeholder, GATES)
    assert report.passed and not report.has_baseline
    assert set(statuses(report).values()) == {"new"}
    comment = render_comment(report, results(), placeholder)
    assert "no baseline yet" in comment and "dpdp-eval-baseline" in comment


def test_comparability_warnings() -> None:
    base = baseline_of(results(), golden_hash="old", judge_hash="old-judge")
    report = compare(results(), base, GATES)
    assert any("golden set" in w for w in report.warnings)
    assert any("judge" in w for w in report.warnings)
    assert report.passed  # warnings do not fail the gate


def test_comment_table() -> None:
    base = baseline_of(results())
    current = results(recall5=0.5)
    comment = render_comment(compare(current, base, GATES), current, base, "https://ci/run/1")
    assert comment.startswith(COMMENT_MARKER)
    assert "## ❌ Eval gate failed" in comment
    assert "| **recall@5** | 0.800 | 0.500 | -0.300 | 0.05 (+0.01) | ❌ |" in comment
    assert "| **mrr** | 0.700 | 0.700 | 0.000 | 0.05 (+0.01) | ✅ |" in comment
    assert "By category" in comment and "| lookup |" in comment
    assert "spent $0.0100 of $0.50 budget" in comment and "[details](https://ci/run/1)" in comment
    assert "baseline `base` (initial)" in comment


def test_gate_cli_exit_codes(tmp_path, capsys) -> None:
    base_path, res_path, out = tmp_path / "b.json", tmp_path / "r.json", tmp_path / "c.md"
    base_path.write_text(json.dumps(baseline_of(results())))
    res_path.write_text(json.dumps(results()))
    assert (
        gate_main(
            ["--results", str(res_path), "--baseline", str(base_path), "--comment-out", str(out)]
        )
        == 0
    )
    assert out.read_text().startswith(COMMENT_MARKER)
    res_path.write_text(json.dumps(results(mrr=0.2)))
    assert gate_main(["--results", str(res_path), "--baseline", str(base_path)]) == 1


# -- baseline updates ----------------------------------------------------------------------


def test_baseline_update_is_deliberate(tmp_path, capsys) -> None:
    target, res_path = tmp_path / "baseline.json", tmp_path / "results.json"
    res_path.write_text(json.dumps(results()))
    args = ["--results", str(res_path), "--baseline", str(target)]
    with pytest.raises(SystemExit):  # --reason is required
        baseline_main(args)
    assert baseline_main(args + ["--reason", "first", "--dry-run"]) == 0
    assert not target.exists()
    assert baseline_main(args + ["--reason", "first baseline"]) == 0
    written = json.loads(target.read_text())
    assert written["reason"] == "first baseline" and written["status"] == "recorded"
    assert written["metrics"]["recall@5"] == 0.8 and written["updated_by"]
    assert written["golden_hash"] == "g" and written["default_as_of_date"] == "2026-10-09"

    res_path.write_text(json.dumps(results(recall5=0.3)))  # a regression
    assert baseline_main(args + ["--reason", "worse"]) == 1
    assert json.loads(target.read_text())["metrics"]["recall@5"] == 0.8  # unchanged
    assert baseline_main(args + ["--reason", "accepted trade-off", "--allow-regression"]) == 0
    assert json.loads(target.read_text())["metrics"]["recall@5"] == 0.3

    res_path.write_text(json.dumps(results(errors=2)))
    assert baseline_main(args + ["--reason", "x"]) == 1
    assert "errored" in capsys.readouterr().err


# -- overrides and the runner under CI settings --------------------------------------------


def test_ci_override_merges_both_configs(tmp_path) -> None:
    cfg, system = load_configs("eval.yaml", ["ci.yaml"])
    assert cfg["llm_cache"]["enabled"] is True and cfg["budget_usd"] == 0.5
    assert str(cfg["default_as_of_date"]) == "2026-10-09"
    assert system["qdrant"]["location"] == ":memory:" and system["qdrant"]["auto_index"]
    assert system["embedding"]["cache"]["enabled"] is True
    assert system["llm"]["model"] == "claude-haiku-5-5"  # untouched keys survive
    bad = tmp_path / "bad.yaml"
    bad.write_text("judge: {}\n")
    with pytest.raises(SystemExit, match="unknown sections"):
        load_configs("eval.yaml", [str(bad)])


def _runner(
    eval_config: dict[str, Any],
    api_config: dict[str, Any],
    tmp_path: Path,
    answer: FakeLLM,
    judge: FakeJudgeLLM,
) -> EvalRunner:
    return EvalRunner(
        eval_config,
        api_config,
        retriever=Retriever(api_config),
        answer_llm=answer,
        judge_llm=judge,
        tracer=RecordingTracer(),
        metrics=MetricsStore(tmp_path / "m.db"),
    )


def test_second_ci_run_is_served_from_cache(eval_config, api_config, tmp_path) -> None:
    cfg = merge(
        eval_config,
        {
            "llm_cache": {"enabled": True, "path": str(tmp_path / "llm.sqlite")},
            "default_as_of_date": "2026-10-09",
            "budget_usd": 1.0,
        },
    )
    answer, judge = FakeLLM(), FakeJudgeLLM()
    first = _runner(cfg, api_config, tmp_path, answer, judge).run().results
    calls = (len(answer.calls), len(judge.calls))
    second = _runner(cfg, api_config, tmp_path, answer, judge).run().results
    assert (len(answer.calls), len(judge.calls)) == calls  # nothing re-sent
    assert first["cost"]["fresh_usd"] > 0 and second["cost"]["fresh_usd"] == 0
    assert second["llm_cache"]["answer"]["misses"] == 0
    assert second["llm_cache"]["judge"]["hits"] == calls[1]
    assert second["overall"] == first["overall"]  # identical scores: no judge noise
    assert second["cost"]["judge_usd"] == first["cost"]["judge_usd"]  # nominal cost kept
    assert first["settings"]["default_as_of_date"] == "2026-10-09"
    assert len(first["golden_hash"]) == 64


def test_budget_aborts_the_run(eval_config, api_config, tmp_path) -> None:
    cfg = merge(eval_config, {"budget_usd": 0.0})
    with pytest.raises(BudgetExceeded):
        _runner(cfg, api_config, tmp_path, FakeLLM(), FakeJudgeLLM()).run()


def test_snapshot_paths_exist_in_real_results(eval_config, api_config, tmp_path) -> None:
    res = (
        _runner(eval_config, api_config, tmp_path, FakeLLM(), FakeJudgeLLM())
        .run(today=date(2026, 10, 9))
        .results
    )
    snap = snapshot(res, GATES)
    assert all(v is not None for v in snap["metrics"].values()), snap["metrics"]
    assert set(snap["by_category"]) == set(eval_config["categories"])


# -- workflow ------------------------------------------------------------------------------

WORKFLOW = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())


def _steps(job: str) -> list[dict[str, Any]]:
    return WORKFLOW["jobs"][job]["steps"]


def test_workflow_runs_lint_and_tests_on_prs() -> None:
    triggers = WORKFLOW[True]  # YAML 1.1 parses the `on:` key as boolean True
    assert "pull_request" in triggers
    runs = " ".join(s.get("run", "") for s in _steps("lint-test"))
    assert "ruff check" in runs and "ruff format --check" in runs and "pytest" in runs


def test_workflow_eval_job_matches_configs() -> None:
    steps = _steps("eval")
    runs = " ".join(s.get("run", "") for s in steps)
    assert "dpdp-eval --override configs/ci.yaml run" in runs
    assert "dpdp-eval-gate --results eval-out/results.json" in runs
    assert WORKFLOW["jobs"]["eval"]["needs"] == "lint-test"
    assert WORKFLOW["jobs"]["eval"]["permissions"]["pull-requests"] == "write"
    # The cached paths are the ones the configs write to.
    cached = next(s for s in steps if s.get("name") == "Restore embedding and LLM-call caches")
    paths = cached["with"]["path"].split()
    eval_cfg, system = load_configs("eval.yaml", ["ci.yaml"])
    assert eval_cfg["llm_cache"]["path"] in paths
    assert system["embedding"]["cache"]["path"] in paths
    comment = next(s for s in steps if s.get("name") == "Post before/after comment")
    assert COMMENT_MARKER in comment["with"]["script"]
    skipped = next(s for s in steps if s.get("name") == "Explain a skipped eval")
    assert COMMENT_MARKER in skipped["run"]


def test_cache_files_are_gitignored() -> None:
    ignored = (REPO_ROOT / ".gitignore").read_text()
    assert "data/cache/" in ignored and ".cache/" in ignored
