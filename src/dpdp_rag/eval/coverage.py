"""List chunks that no golden question covers, to guide labelling: `uv run dpdp-eval-coverage`."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from dpdp_rag.config import load_config, resolve
from dpdp_rag.eval.golden import GoldenItem, load_golden
from dpdp_rag.generation.pinpoint import pinpoint
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval.store import load_chunks


def _group(chunk: Chunk) -> str:
    if chunk.schedule:
        return f"{chunk.doc_id} · {chunk.schedule}"
    if chunk.section:
        return f"{chunk.doc_id} · Chapter {chunk.chapter or '-'}"
    if chunk.rule:
        return f"{chunk.doc_id} · Rules"
    return chunk.doc_id


def uncovered(chunks: list[Chunk], items: list[GoldenItem]) -> list[Chunk]:
    covered = {cid for item in items for cid in item.gold_chunk_ids}
    return [c for c in chunks if c.chunk_id not in covered]


def report(chunks: list[Chunk], items: list[GoldenItem], labels: dict[str, str]) -> dict[str, Any]:
    missing = uncovered(chunks, items)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for c in missing:
        groups[_group(c)].append(
            {"chunk_id": c.chunk_id, "pinpoint": pinpoint(c, labels), "title": c.title or ""}
        )
    return {
        "total_chunks": len(chunks),
        "covered": len(chunks) - len(missing),
        "uncovered": len(missing),
        "groups": dict(sorted(groups.items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval.yaml", help="eval config name or path")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    system = load_config(cfg["system_config"])
    chunks = load_chunks(resolve(system["data"]["chunks_file"]))
    items = load_golden(
        resolve(cfg["golden_file"]), {c.chunk_id for c in chunks}, cfg["categories"]
    )
    rep = report(chunks, items, system["generation"]["schedule_entry_label"])
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return
    print(
        f"{rep['covered']}/{rep['total_chunks']} chunks covered by {len(items)} questions; "
        f"{rep['uncovered']} uncovered.\n"
    )
    for group, rows in rep["groups"].items():
        print(f"{group} ({len(rows)} uncovered)")
        for row in rows:
            print(f"  {row['pinpoint']:<40} {row['chunk_id']}")
        print()


if __name__ == "__main__":
    main()
