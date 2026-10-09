"""Runs the golden set through the system and scores it."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from dpdp_rag.api.versioning import config_hash, hash_config
from dpdp_rag.config import load_config, resolve
from dpdp_rag.eval.golden import GoldenItem, load_golden
from dpdp_rag.eval.judge import Judge, JudgeInput, judge_prompt_paths
from dpdp_rag.eval.report import aggregate, by_category, judge_status, write_outputs
from dpdp_rag.eval.retrieval_metrics import first_gold_rank, item_metrics
from dpdp_rag.generation.answer import Answerer, AnswerResult
from dpdp_rag.generation.context import render_context
from dpdp_rag.generation.llm import LLMClient, LLMError
from dpdp_rag.generation.llm_cache import Budget, make_llm
from dpdp_rag.metrics.store import MetricsStore, RequestMetric, percentile
from dpdp_rag.observability.tracing import NOOP_TRACE, Trace, Tracer, make_tracer
from dpdp_rag.retrieval.retriever import RetrievalUnavailable, Retriever

log = logging.getLogger(__name__)


@dataclass
class EvalRun:
    results: dict[str, Any]
    results_path: Path
    summary_path: Path


def judge_hash(eval_config: dict[str, Any]) -> str:
    return hash_config(eval_config, judge_prompt_paths(eval_config["judge"]))


class EvalRunner:
    def __init__(
        self,
        eval_config: dict[str, Any],
        system_config: dict[str, Any] | None = None,
        *,
        retriever: Retriever | None = None,
        answer_llm: LLMClient | None = None,
        judge_llm: LLMClient | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsStore | None = None,
    ) -> None:
        self.cfg = eval_config
        self.system = system_config or load_config(eval_config["system_config"])
        self.retriever = retriever or Retriever(self.system)
        # Both models go through the disk cache (if enabled) and one shared spend cap.
        self.budget = Budget(eval_config.get("budget_usd"))
        cache_cfg = eval_config.get("llm_cache")
        self.answer_llm = make_llm(self.system["llm"], cache_cfg, self.budget, answer_llm)
        self.judge_llm = make_llm(eval_config["judge"], cache_cfg, self.budget, judge_llm)
        self.answerer = Answerer(self.system, self.retriever, self.answer_llm)
        self.judge = Judge(eval_config["judge"], self.judge_llm)
        self.tracer = tracer or make_tracer(self.system.get("tracing", {}))
        self.metrics = metrics or MetricsStore(resolve(self.system["metrics"]["db_path"]))
        self.ks = [int(k) for k in eval_config["retrieval"]["ks"]]
        self.depth = int(eval_config["retrieval"]["mrr_depth"])
        self.store_ids = set(self.retriever.store.by_id)
        self.config_hash = config_hash(self.system)
        # As-of date for items without one: config (pinned in CI), else today; run() may override.
        pinned = eval_config.get("default_as_of_date")
        self.today = date.fromisoformat(str(pinned)) if pinned else date.today()

    def load(self) -> list[GoldenItem]:
        return load_golden(resolve(self.cfg["golden_file"]), self.store_ids, self.cfg["categories"])

    # -- one item ----------------------------------------------------------------------

    def _retrieval(self, item: GoldenItem) -> dict[str, Any] | None:
        if not item.gold_chunk_ids:
            return None  # retrieval metrics need gold chunks (e.g. unanswerable questions)
        hits = self.retriever.retrieve(item.question, max(self.depth, *self.ks), cross_refs=False)
        ranked = [h.chunk.chunk_id for h in hits]
        return {
            "ranked": ranked[: self.depth],
            "first_gold_rank": first_gold_rank(ranked, item.gold_chunk_ids),
            **item_metrics(ranked, item.gold_chunk_ids, self.ks, self.depth),
        }

    def _answer(self, item: GoldenItem, as_of: date, trace: Trace) -> tuple[AnswerResult, float]:
        """Answer one question and record it in metrics.db with source=eval."""
        started = time.perf_counter()
        result = self.answerer.answer(item.question, as_of, trace)
        latency = (time.perf_counter() - started) * 1000
        self.metrics.record(
            RequestMetric(
                config_hash=self.config_hash,
                latency_ms=latency,
                tokens_in=result.usage.total_input,
                tokens_out=result.usage.output_tokens,
                cost_usd=result.cost_usd,
                cached=False,
                refused=result.refused,
                source="eval",
            )
        )
        return result, latency

    def run_item(self, item: GoldenItem, trace: Trace = NOOP_TRACE) -> dict[str, Any]:
        as_of = item.as_of_date or self.today
        out: dict[str, Any] = {
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "as_of_date": as_of.isoformat(),
            "answerable": item.answerable,
            "gold_chunk_ids": item.gold_chunk_ids,
            "author": item.author,
            "retrieval": None,
            "context_recall": None,
            "answer": {},
            "judge": {},
            "judge_failed": [],  # metrics the judge was asked for but could not score
            "judge_cost_usd": 0.0,
            "errors": [],
        }
        try:
            out["retrieval"] = self._retrieval(item)
        except (ConnectionError, RetrievalUnavailable) as exc:
            out["errors"].append(f"retrieval: {exc}")
            return out
        try:
            result, latency = self._answer(item, as_of, trace)
        except (LLMError, ConnectionError, RetrievalUnavailable) as exc:
            out["answer"] = {"error": str(exc)}
            out["errors"].append(f"answer: {exc}")
            return out
        out["answer"] = {
            "text": result.answer,
            "refused": result.refused,
            "refusal_reason": result.refusal_reason,
            "citations": [
                {"chunk_id": c.chunk_id, "pinpoint": c.pinpoint} for c in result.citations
            ],
            "context_chunk_ids": result.retrieved,
            "latency_ms": round(latency, 1),
            "tokens": result.usage.as_dict(),
            "cost_usd": result.cost_usd,
            "error": None,
        }
        if item.gold_chunk_ids:
            gold = set(item.gold_chunk_ids)
            out["context_recall"] = len(gold & set(result.retrieved)) / len(gold)

        chunks = [self.retriever.store.by_id[c] for c in result.retrieved]
        gen = self.system["generation"]
        x = JudgeInput(
            question=item.question,
            as_of_date=as_of.isoformat(),
            answerable=item.answerable,
            reference_answer=item.reference_answer,
            context=render_context(chunks, as_of, gen["schedule_entry_label"], gen["doc_labels"]),
            answer=result.answer,
            refused=result.refused,
            refusal_reason=result.refusal_reason or "none",
        )
        canned = result.answer == self.answerer.prompts.insufficient_context
        injected = item.category in self.cfg["judge"].get("injection_categories", [])
        wanted = (
            ["refusal"]
            + (["injection"] if injected else [])
            + ([] if canned else ["faithfulness"])
            + (["relevance"] if item.answerable else [])
        )
        for metric in wanted:
            try:
                j = self.judge.judge(metric, x, trace)
            except LLMError as exc:
                out["errors"].append(f"judge {metric}: {exc}")
                out["judge_failed"].append(metric)
                continue
            out["judge_cost_usd"] += j.cost_usd
            out["judge"][metric] = (
                {"correct": j.value == 1.0, "reasoning": j.reasoning}
                if metric in Judge.VERDICT_METRICS
                else {"score": int(j.value), "reasoning": j.reasoning}
            )
        return out

    def preflight(self, item: GoldenItem) -> None:
        """Raise JudgeUnavailable unless one real judge call succeeds. The probe grades a
        golden item's reference answer as the response, so it needs no answer call."""
        self.judge.preflight(
            JudgeInput(
                question=item.question,
                as_of_date=(item.as_of_date or self.today).isoformat(),
                answerable=item.answerable,
                reference_answer=item.reference_answer,
                context="",
                answer=item.reference_answer,
                refused=False,
                refusal_reason="none",
            )
        )

    # -- the suite -----------------------------------------------------------------------

    def run(
        self,
        *,
        limit: int | None = None,
        categories: list[str] | None = None,
        ids: list[str] | None = None,
        out_dir: Path | None = None,
        today: date | None = None,
    ) -> EvalRun:
        items = self.load()
        if categories:
            items = [i for i in items if i.category in categories]
        if ids:
            items = [i for i in items if i.id in ids]
        items = items[:limit] if limit else items
        self.today = today or self.today
        jhash = judge_hash(self.cfg)
        started = datetime.now(UTC)
        run_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}_{self.config_hash[:8]}"
        pass_threshold = float(self.cfg["judge"]["pass_threshold"])
        if items and self.cfg["judge"].get("preflight", True):
            self.preflight(items[0])

        item_results: list[dict[str, Any]] = []
        with self.tracer.trace(
            self.cfg["tracing"]["trace_name"],
            config_hash=self.config_hash,
            input={"golden_file": self.cfg["golden_file"], "n": len(items)},
            metadata={
                "run_id": run_id,
                "judge_hash": jhash,
                "judge_model": self.cfg["judge"]["model"],
            },
        ) as trace:
            for item in items:
                with trace.step(f"item:{item.id}", as_type="evaluator", input=item.question) as s:
                    result = self.run_item(item, trace)
                    s.update(
                        output={k: result[k] for k in ("retrieval", "judge", "errors")},
                        metadata={"category": item.category},
                    )
                item_results.append(result)
                log.info("%s: %s", item.id, "error" if result["errors"] else "ok")
            overall = aggregate(item_results, self.ks, pass_threshold)
            latencies = [
                i["answer"]["latency_ms"]
                for i in item_results
                if i["answer"].get("latency_ms") is not None
            ]
            results: dict[str, Any] = {
                "run_id": run_id,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "golden_file": self.cfg["golden_file"],
                "golden_hash": hashlib.sha256(
                    resolve(self.cfg["golden_file"]).read_bytes()
                ).hexdigest(),
                "n": len(item_results),
                "config_hash": self.config_hash,
                "judge_hash": jhash,
                "answer_model": self.system["llm"]["model"],
                "judge_model": self.cfg["judge"]["model"],
                "settings": {
                    "ks": self.ks,
                    "mrr_depth": self.depth,
                    "pass_threshold": pass_threshold,
                    "default_as_of_date": self.today.isoformat(),
                },
                # ok / degraded / unavailable: never let missing judge scores pass silently.
                "judge_status": judge_status(item_results),
                "overall": overall,
                "by_category": by_category(
                    item_results, self.cfg["categories"], self.ks, pass_threshold
                ),
                "cost": {
                    # Nominal cost of every call, cached or not (comparable across runs) ...
                    "answer_usd": sum(i["answer"].get("cost_usd") or 0 for i in item_results),
                    "judge_usd": sum(i["judge_cost_usd"] for i in item_results),
                    # ... and what this run actually paid (cache misses only).
                    "fresh_usd": self.budget.spent_usd,
                    "budget_usd": self.budget.max_usd,
                },
                "llm_cache": {
                    "answer": self.answer_llm.stats.as_dict(),
                    "judge": self.judge_llm.stats.as_dict(),
                },
                "latency_ms": {"p50": percentile(latencies, 50), "p99": percentile(latencies, 99)},
                "items": item_results,
            }
            for name, value in _headline(overall).items():
                trace.score(name, value)
            trace.end(
                {k: results[k] for k in ("run_id", "overall", "cost")},
                metadata={"judge_hash": jhash},
            )
        self.tracer.flush()

        out = out_dir or resolve(self.cfg["output_dir"]) / run_id
        results_path, summary_path = write_outputs(results, out)
        return EvalRun(results, results_path, summary_path)


def _headline(overall: dict[str, Any]) -> dict[str, float]:
    """Aggregate scores sent to Langfuse (only the ones that were computed)."""
    r, a = overall["retrieval"], overall["answers"]
    candidates = {
        "recall@5": r.get("recall@5"),
        "mrr": r.get("mrr"),
        "faithfulness": a.get("faithfulness_mean"),
        "relevance": a.get("relevance_mean"),
        "refusal_correctness": a.get("refusal_correctness"),
    }
    return {k: float(v) for k, v in candidates.items() if v is not None}
