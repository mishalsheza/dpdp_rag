"""Aggregation of per-item results and the results.json / summary.md outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(
    items: Sequence[dict[str, Any]], ks: Sequence[int], pass_threshold: float
) -> dict[str, Any]:
    """Means over the items where each metric applies, with the n behind each mean."""
    retrieval_keys = [f"recall@{k}" for k in ks] + [f"hit_rate@{k}" for k in ks] + ["mrr"]
    with_retrieval = [i["retrieval"] for i in items if i.get("retrieval")]
    out: dict[str, Any] = {"n": len(items), "errors": sum(1 for i in items if i["errors"])}
    out["retrieval"] = {key: _mean([r[key] for r in with_retrieval]) for key in retrieval_keys}
    out["retrieval"]["n"] = len(with_retrieval)
    ctx = [i["context_recall"] for i in items if i.get("context_recall") is not None]
    out["retrieval"]["context_recall"] = _mean(ctx)

    answers: dict[str, Any] = {}
    for metric in ("faithfulness", "relevance"):
        scores = [i["judge"][metric]["score"] for i in items if i["judge"].get(metric)]
        answers[f"{metric}_mean"] = _mean(scores)
        answers[f"{metric}_pass_rate"] = _mean(
            [1.0 if s >= pass_threshold else 0.0 for s in scores]
        )
        answers[f"{metric}_n"] = len(scores)
    for metric, key in (("refusal", "refusal_correctness"), ("injection", "injection_resisted")):
        verdicts = [
            1.0 if i["judge"][metric]["correct"] else 0.0 for i in items if i["judge"].get(metric)
        ]
        answers[key] = _mean(verdicts)
        answers[f"{metric}_n"] = len(verdicts)
    # Judge calls that failed: a None metric with judge_failed > 0 means "judge broke",
    # not "does not apply".
    answers["judge_failed"] = sum(len(i.get("judge_failed", [])) for i in items)
    answers["refusal_rate"] = _mean(
        [
            1.0 if i["answer"].get("refused") else 0.0
            for i in items
            if i["answer"].get("error") is None
        ]
    )
    out["answers"] = answers
    return out


def by_category(
    items: Sequence[dict[str, Any]],
    categories: Sequence[str],
    ks: Sequence[int],
    pass_threshold: float,
) -> dict[str, Any]:
    return {
        cat: aggregate([i for i in items if i["category"] == cat], ks, pass_threshold)
        for cat in categories
    }


def judge_status(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """ok: every wanted judge call scored. degraded: some calls failed or some items never
    reached the judge (answer or retrieval error). unavailable: nothing was scored."""
    scored = sum(len(i["judge"]) for i in items)
    failed = sum(len(i.get("judge_failed", [])) for i in items)
    unjudged = sum(1 for i in items if not i["answer"] or i["answer"].get("error"))
    if items and scored == 0:
        status = "unavailable"
    elif failed or unjudged:
        status = "degraded"
    else:
        status = "ok"
    return {"status": status, "scored": scored, "failed": failed, "unjudged_items": unjudged}


def judge_banner(results: dict[str, Any]) -> str | None:
    """A one-line warning when judge metrics are missing or partial (None when healthy).
    Runs from before judge_status existed are inferred from their errors."""
    js = results.get("judge_status")
    if js is None:
        a = results.get("overall", {}).get("answers", {})
        if not results.get("n") or a.get("refusal_n") or a.get("faithfulness_n"):
            return None
        js = {"status": "unavailable", "failed": None}
    if js["status"] == "ok":
        return None
    if js["status"] == "unavailable":
        return (
            "JUDGE UNAVAILABLE: no answer was scored, so faithfulness, relevance, refusal "
            "correctness and injection resistance are missing for this run. See item errors."
        )
    return (
        f"JUDGE DEGRADED: {js['failed']} judge call(s) failed and {js['unjudged_items']} "
        "item(s) never reached the judge; answer-quality means cover the rest only."
    )


def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "–"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def summary_markdown(results: dict[str, Any]) -> str:
    ks = results["settings"]["ks"]
    lines = [
        f"# Eval run {results['run_id']}",
        "",
        f"- Golden set: `{results['golden_file']}` ({results['n']} questions)",
        f"- System config_hash: `{results['config_hash']}`",
        f"- Judge hash: `{results['judge_hash']}`",
        f"- Answer model: `{results['answer_model']}` · Judge model: `{results['judge_model']}`",
        f"- Cost: answers ${results['cost']['answer_usd']:.4f}, judge "
        f"${results['cost']['judge_usd']:.4f}",
        f"- Answer latency: p50 {_fmt(results['latency_ms']['p50'])} ms, "
        f"p99 {_fmt(results['latency_ms']['p99'])} ms",
        f"- Items with errors: {results['overall']['errors']}",
        "",
    ]
    if banner := judge_banner(results):
        lines += [f"> **{banner}**", ""]
    lines += ["## By category", ""]
    head = (
        ["Category", "n"]
        + [f"R@{k}" for k in ks]
        + ["MRR", f"Hit@{ks[-1]}", "Ctx recall", "Faithful", "Relevant", "Refusal ok"]
        + ["Injection resisted"]
    )
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = list(results["by_category"].items()) + [("**overall**", results["overall"])]
    for cat, agg in rows:
        r, a = agg["retrieval"], agg["answers"]
        cells = (
            [cat, str(agg["n"])]
            + [_fmt(r[f"recall@{k}"]) for k in ks]
            + [
                _fmt(r["mrr"]),
                _fmt(r[f"hit_rate@{ks[-1]}"]),
                _fmt(r["context_recall"]),
                _fmt(a["faithfulness_mean"]),
                _fmt(a["relevance_mean"]),
                _fmt(a["refusal_correctness"], pct=True),
                _fmt(a.get("injection_resisted"), pct=True),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Faithfulness and relevance are mean judge scores on a 1-5 scale; refusal ok is the "
        "share judged correct; injection resisted is the share of prompt-injection items where "
        "the judge found the injected instruction was not followed. `–` means the metric does "
        "not apply to any item in that row, or the judge failed (see the banner and errors).",
        "",
        "## Items",
        "",
        "| id | category | first gold rank | refused | faithful | relevant | refusal ok "
        "| injection resisted | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in results["items"]:
        j = item["judge"]
        rank = item["retrieval"]["first_gold_rank"] if item.get("retrieval") else "–"
        lines.append(
            "| "
            + " | ".join(
                [
                    item["id"],
                    item["category"],
                    str(rank if rank is not None else "none"),
                    str(item["answer"].get("refused", "–")),
                    str(j["faithfulness"]["score"]) if j.get("faithfulness") else "–",
                    str(j["relevance"]["score"]) if j.get("relevance") else "–",
                    ("yes" if j["refusal"]["correct"] else "no") if j.get("refusal") else "–",
                    ("yes" if j["injection"]["correct"] else "no") if j.get("injection") else "–",
                    "; ".join(item["errors"]) or "",
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"N = {results['n']}. Small samples: treat differences of a single item as "
        "noise. Judge scores come from an LLM and share its biases (see docs/EVAL.md).",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(results: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path, summary_path = out_dir / "results.json", out_dir / "summary.md"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    summary_path.write_text(summary_markdown(results), encoding="utf-8")
    return results_path, summary_path
