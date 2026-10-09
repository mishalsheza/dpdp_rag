"""Evaluation CLI: `uv run dpdp-eval run|validate` (or `python -m dpdp_rag.eval`)."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from dpdp_rag.config import REPO_ROOT, load_config, resolve
from dpdp_rag.eval.golden import GoldenSetError, load_golden
from dpdp_rag.retrieval.store import load_chunks


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dpdp-eval", description=__doc__)
    p.add_argument("--config", default="eval.yaml", help="eval config name or path")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="check data/golden.jsonl against chunks.jsonl")
    run = sub.add_parser("run", help="run the suite (calls the answer model and the judge)")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--category", action="append", default=None)
    run.add_argument("--id", action="append", default=None)
    run.add_argument("--out", type=Path, default=None, help="output directory")
    run.add_argument(
        "--today",
        type=date.fromisoformat,
        default=None,
        help="as-of date for questions without one (default: today)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    system = load_config(cfg["system_config"])
    chunk_ids = {c.chunk_id for c in load_chunks(resolve(system["data"]["chunks_file"]))}
    try:
        items = load_golden(resolve(cfg["golden_file"]), chunk_ids, cfg["categories"])
    except GoldenSetError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.command == "validate":
        counts = {c: sum(1 for i in items if i.category == c) for c in cfg["categories"]}
        print(f"OK: {len(items)} questions. " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return 0

    from dpdp_rag.eval.runner import EvalRunner  # imports the SDKs; not needed to validate

    run = EvalRunner(cfg, system).run(
        limit=args.limit, categories=args.category, ids=args.id, out_dir=args.out, today=args.today
    )
    print(run.summary_path.read_text(encoding="utf-8"))
    print(f"Wrote {run.results_path} and {run.summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
