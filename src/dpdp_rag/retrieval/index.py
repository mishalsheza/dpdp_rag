"""Embed chunks.jsonl and upsert it into Qdrant: `uv run dpdp-index`."""

from __future__ import annotations

import argparse
import logging
import time

from dpdp_rag.config import load_system_config, resolve
from dpdp_rag.retrieval.embeddings import make_embedder
from dpdp_rag.retrieval.qdrant_index import QdrantIndex, make_client
from dpdp_rag.retrieval.store import load_chunks

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default.yaml", help="config name or path")
    parser.add_argument(
        "--wait", type=float, default=0, help="seconds to keep retrying while Qdrant starts up"
    )
    parser.add_argument(
        "--if-stale",
        action="store_true",
        help="only rebuild if the collection is missing or out of date",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_system_config(args.config)
    chunks = load_chunks(resolve(config["data"]["chunks_file"]))
    qcfg = config["qdrant"]
    index = QdrantIndex(
        make_client(qcfg),
        qcfg,
        make_embedder(config["embedding"]),
        config["embedding"]["document_template"],
        config["embedding"].get("index_max_chars"),
    )
    deadline = time.monotonic() + args.wait
    while True:
        try:
            current = index.is_current(chunks)  # also checks that Qdrant is reachable
            break
        except ConnectionError:
            if time.monotonic() >= deadline:
                raise
            log.info("Waiting for Qdrant ...")
            time.sleep(2)
    if args.if_stale and current:
        log.info("Collection %s is up to date", qcfg["collection"])
        return
    index.build(chunks)


if __name__ == "__main__":
    main()
