"""Embed chunks.jsonl and upsert it into Qdrant: `uv run dpdp-index`."""

from __future__ import annotations

import argparse
import logging

from dpdp_rag.config import load_config, resolve
from dpdp_rag.retrieval.embeddings import make_embedder
from dpdp_rag.retrieval.qdrant_index import QdrantIndex, make_client
from dpdp_rag.retrieval.store import load_chunks

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml", help="config name or path")
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="only rebuild if the collection is missing or out of date",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    chunks = load_chunks(resolve(config["data"]["chunks_file"]))
    qcfg = config["qdrant"]
    index = QdrantIndex(
        make_client(qcfg),
        qcfg,
        make_embedder(config["embedding"]),
        config["embedding"]["document_template"],
    )
    current = index.is_current(chunks)  # also checks that Qdrant is reachable
    if args.if_stale and current:
        log.info("Collection %s is up to date", qcfg["collection"])
        return
    index.build(chunks)


if __name__ == "__main__":
    main()
