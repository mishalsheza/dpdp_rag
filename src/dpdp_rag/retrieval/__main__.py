"""Query the index: `python -m dpdp_rag.retrieval "query" [options]`."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date

from dpdp_rag.config import load_system_config
from dpdp_rag.retrieval.filters import Filters
from dpdp_rag.retrieval.retriever import MODES, RetrievedChunk, Retriever


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m dpdp_rag.retrieval", description=__doc__)
    p.add_argument("query")
    p.add_argument("-k", type=int, default=None, help="number of results (config: retrieval.k)")
    p.add_argument("--config", default="default.yaml", help="config name or path")
    p.add_argument("--mode", choices=MODES, default=None)
    p.add_argument("--doc-type", action="append", default=None)
    p.add_argument("--rule", action="append", default=None)
    p.add_argument("--section", action="append", default=None)
    p.add_argument(
        "--in-force",
        nargs="?",
        const="today",
        default=None,
        metavar="DATE",
        help="only chunks in force on DATE (default: today)",
    )
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cross-refs", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--json", action="store_true", help="print results as JSON")
    return p


def _filters(args: argparse.Namespace, base: Filters) -> Filters:
    out = base
    if args.doc_type:
        out = replace(out, doc_type=args.doc_type)
    if args.rule:
        out = replace(out, rule=args.rule)
    if args.section:
        out = replace(out, section=args.section)
    if args.in_force:
        on = date.today() if args.in_force == "today" else date.fromisoformat(args.in_force)
        out = replace(out, in_force_on=on)
    return out


def _format(r: RetrievedChunk) -> str:
    c = r.chunk
    where = c.chunk_id + (f"  [via {r.via} from {r.expanded_from}]" if r.expanded_from else "")
    stages = " ".join(f"{k}={v:.4g}" for k, v in r.scores.items())
    text = c.text if len(c.text) <= 300 else c.text[:297] + "..."
    return (
        f"{r.score:.4f}  {where}\n    in force {c.in_force_date}  {stages}\n"
        f"    {c.title or ''}\n    {text}"
    )


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    retriever = Retriever(load_system_config(args.config))
    results = retriever.retrieve(
        args.query,
        args.k,
        filters=_filters(args, retriever.filters),
        mode=args.mode,
        rerank=args.rerank,
        cross_refs=args.cross_refs,
    )
    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    else:
        print("\n\n".join(_format(r) for r in results) or "No results.")


if __name__ == "__main__":
    main()
