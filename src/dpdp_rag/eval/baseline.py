"""Deliberately record an eval run as the new baseline: `uv run dpdp-eval-baseline`.

    uv run dpdp-eval-baseline --results data/eval_runs/<run_id>/results.json \\
        --reason "switch to hybrid retrieval"

The baseline is what CI compares every PR against, so updating it is an explicit,
reviewed act: a reason is required; runs with errored items are refused; and a run that
would fail the current gates is refused unless --allow-regression is given.
"""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dpdp_rag.config import REPO_ROOT, load_config, resolve
from dpdp_rag.eval.gates import compare, load_baseline, render_comment, snapshot


def _who() -> str:
    try:
        name = subprocess.run(
            ["git", "config", "user.name"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or getpass.getuser()


def build_baseline(
    results: dict[str, Any], gates: dict[str, Any], reason: str, author: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "recorded",
        "updated_at": datetime.now(UTC).isoformat(),
        "updated_by": author,
        "reason": reason,
        "run_id": results.get("run_id"),
        "config_hash": results.get("config_hash"),
        "judge_hash": results.get("judge_hash"),
        "golden_hash": results.get("golden_hash"),
        "golden_file": results.get("golden_file"),
        "n": results.get("n"),
        "default_as_of_date": results.get("settings", {}).get("default_as_of_date"),
        "answer_model": results.get("answer_model"),
        "judge_model": results.get("judge_model"),
        **snapshot(results, gates),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="dpdp-eval-baseline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results", type=Path, required=True, help="results.json of a trusted run")
    p.add_argument("--reason", required=True, help="why the baseline changes (kept in the file)")
    p.add_argument("--gates", default="gates.yaml", help="gates config name or path")
    p.add_argument("--baseline", type=Path, default=None, help="default: gates baseline_file")
    p.add_argument(
        "--allow-regression",
        action="store_true",
        help="accept a run that fails the gates against the current baseline",
    )
    p.add_argument("--allow-errors", action="store_true", help="accept a run with errored items")
    p.add_argument("--dry-run", action="store_true", help="show the comparison, write nothing")
    args = p.parse_args(argv)

    if not args.reason.strip():
        print("--reason must not be empty", file=sys.stderr)
        return 1
    gates = load_config(args.gates)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    target = args.baseline or resolve(gates["baseline_file"])
    current = load_baseline(target)

    errors = int(results.get("overall", {}).get("errors", 0))
    if errors and not args.allow_errors:
        print(
            f"Refusing: {errors} item(s) errored in this run (use --allow-errors).", file=sys.stderr
        )
        return 1
    report = compare(results, current, gates)
    print(render_comment(report, results, current))
    if report.has_baseline and not report.passed and not args.allow_regression:
        print(
            "Refusing: this run fails the gates against the current baseline. Re-run with "
            "--allow-regression if the drop is intended.",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        print("Dry run: baseline not written.")
        return 0
    new = build_baseline(results, gates, args.reason.strip(), _who())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target}. Commit it in its own PR so the change is reviewed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
