"""Regression gates: compare an eval run with eval/baseline.json and render a PR comment.

`uv run dpdp-eval-gate --results <run>/results.json [--comment-out comment.md]`
exits 1 when a gated metric dropped by more than max_drop + tolerance (configs/gates.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dpdp_rag.config import load_config, resolve

COMMENT_MARKER = "<!-- dpdp-eval-gate -->"


def lookup(data: dict[str, Any], path: str) -> float | None:
    """Value at a dotted path such as "overall.retrieval.recall@5" (None if absent)."""
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, int | float) else None


def snapshot(results: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """The metrics a baseline stores: gated + info, overall and per category."""
    paths = {name: g["path"] for name, g in gates["gated"].items()} | dict(gates["info"])
    metrics = {name: lookup(results, path) for name, path in paths.items()}
    by_category = {
        cat: {name: lookup(agg, path.removeprefix("overall.")) for name, path in paths.items()}
        for cat, agg in results.get("by_category", {}).items()
    }
    return {"metrics": metrics, "by_category": by_category}


@dataclass
class Check:
    name: str
    baseline: float | None
    current: float | None
    max_drop: float
    tolerance: float

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    @property
    def status(self) -> str:
        if self.baseline is None:
            return "new"
        if self.current is None:
            return "missing"
        drop = self.baseline - self.current
        if drop > self.max_drop + self.tolerance + 1e-12:
            return "fail"
        return "improved" if drop < -1e-12 else "pass"

    @property
    def failed(self) -> bool:
        return self.status in ("fail", "missing")


@dataclass
class GateReport:
    checks: list[Check]
    info: list[tuple[str, float | None, float | None]]
    categories: dict[str, list[tuple[str, float | None, float | None]]]
    has_baseline: bool
    errors: int
    max_errors: int
    warnings: list[str] = field(default_factory=list)

    @property
    def errors_failed(self) -> bool:
        return self.errors > self.max_errors

    @property
    def passed(self) -> bool:
        return not self.errors_failed and not any(c.failed for c in self.checks)


def compare(
    results: dict[str, Any], baseline: dict[str, Any] | None, gates: dict[str, Any]
) -> GateReport:
    current = snapshot(results, gates)
    base_metrics = (baseline or {}).get("metrics") or {}
    has_baseline = bool(base_metrics)
    checks = [
        Check(
            name,
            base_metrics.get(name),
            current["metrics"][name],
            float(g["max_drop"]),
            float(g["tolerance"]),
        )
        for name, g in gates["gated"].items()
    ]
    info = [(name, base_metrics.get(name), current["metrics"][name]) for name in gates["info"]]
    base_cats = (baseline or {}).get("by_category") or {}
    categories = {
        cat: [
            (m, (base_cats.get(cat) or {}).get(m), vals.get(m)) for m in gates["category_metrics"]
        ]
        for cat, vals in current["by_category"].items()
    }
    report = GateReport(
        checks,
        info,
        categories,
        has_baseline,
        errors=int(results.get("overall", {}).get("errors", 0)),
        max_errors=int(gates["max_item_errors"]),
    )
    if not has_baseline:
        report.warnings.append(
            "No baseline recorded yet, so regressions cannot be detected. "
            "Record one deliberately with `uv run dpdp-eval-baseline`."
        )
    else:
        for key, label in (
            ("golden_hash", "golden set"),
            ("judge_hash", "judge config/rubrics"),
            ("default_as_of_date", "default as-of date"),
        ):
            if baseline and baseline.get(key) and baseline.get(key) != _run_value(results, key):
                report.warnings.append(
                    f"The {label} differs from the baseline's; "
                    "before/after numbers are not strictly comparable."
                )
    return report


def _run_value(results: dict[str, Any], key: str) -> Any:
    if key == "default_as_of_date":
        return results.get("settings", {}).get("default_as_of_date")
    return results.get(key)


def _num(v: float | None) -> str:
    return "–" if v is None else f"{v:.3f}"


def _delta(base: float | None, cur: float | None) -> str:
    if base is None or cur is None:
        return "–"
    d = cur - base
    return "0.000" if abs(d) < 5e-4 else f"{d:+.3f}"


_ICON = {"pass": "✅", "improved": "✅ ⬆", "fail": "❌", "missing": "❌ missing", "new": "🆕"}


def render_comment(
    report: GateReport,
    results: dict[str, Any],
    baseline: dict[str, Any] | None,
    run_url: str | None = None,
) -> str:
    if not report.has_baseline:
        title = "⚠️ Eval gate: no baseline yet (not enforced)"
    else:
        title = "✅ Eval gate passed" if report.passed else "❌ Eval gate failed"
    lines = [COMMENT_MARKER, f"## {title}", ""]
    lines += [
        "| Metric | Baseline | This PR | Δ | Allowed drop (+noise) | |",
        "|---|---|---|---|---|---|",
    ]
    for c in report.checks:
        lines.append(
            f"| **{c.name}** | {_num(c.baseline)} | {_num(c.current)} | "
            f"{_delta(c.baseline, c.current)} | {c.max_drop:g} (+{c.tolerance:g}) | "
            f"{_ICON[c.status]} |"
        )
    err_icon = "❌" if report.errors_failed else "✅"
    lines.append(
        f"| items with errors | – | {report.errors} | – | max {report.max_errors} | {err_icon} |"
    )
    lines += [
        "",
        "<details><summary>Other metrics (not gated)</summary>",
        "",
        "| Metric | Baseline | This PR | Δ |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} | {_num(b)} | {_num(c)} | {_delta(b, c)} |" for n, b, c in report.info]
    lines += ["", "</details>", ""]
    if report.categories:
        names = [m for m, _, _ in next(iter(report.categories.values()))]
        lines += [
            "<details><summary>By category (baseline → this PR)</summary>",
            "",
            "| Category | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for cat, vals in report.categories.items():
            cells = [f"{_num(b)} → {_num(c)}" for _, b, c in vals]
            lines.append(f"| {cat} | " + " | ".join(cells) + " |")
        lines += ["", "</details>", ""]
    for w in report.warnings:
        lines.append(f"> ⚠️ {w}")
    cost, cache = results.get("cost", {}), results.get("llm_cache", {})
    hit = " · ".join(
        f"{k} cache {v['hits']}/{v['hits'] + v['misses']} hits" for k, v in cache.items()
    )
    budget = cost.get("budget_usd")
    lines += [
        "",
        f"<sub>Run `{results.get('run_id')}` · n={results.get('n')} · config_hash "
        f"`{str(results.get('config_hash'))[:12]}` · judge `{results.get('judge_model')}` · "
        f"spent ${cost.get('fresh_usd', 0):.4f}"
        + (f" of ${budget:.2f} budget" if budget is not None else "")
        + (f" · {hit}" if hit else "")
        + (
            f" · baseline `{baseline.get('run_id')}` ({baseline.get('reason')})"
            if report.has_baseline and baseline
            else ""
        )
        + (f" · [details]({run_url})" if run_url else "")
        + "</sub>",
    ]
    return "\n".join(lines) + "\n"


def load_baseline(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dpdp-eval-gate", description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--gates", default="gates.yaml", help="gates config name or path")
    p.add_argument("--baseline", type=Path, default=None, help="default: gates baseline_file")
    p.add_argument("--comment-out", type=Path, default=None, help="write the PR comment here")
    p.add_argument("--run-url", default=None, help="link to the CI run, shown in the comment")
    args = p.parse_args(argv)
    gates = load_config(args.gates)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    baseline = load_baseline(args.baseline or resolve(gates["baseline_file"]))
    report = compare(results, baseline, gates)
    comment = render_comment(report, results, baseline, args.run_url)
    if args.comment_out:
        args.comment_out.parent.mkdir(parents=True, exist_ok=True)
        args.comment_out.write_text(comment, encoding="utf-8")
    print(comment)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
