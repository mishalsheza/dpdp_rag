from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dpdp_rag.config import load_config, merge, resolve
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.ingest.pipeline import IngestResult, build


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="session")
def ingest(config: dict[str, Any]) -> IngestResult:
    raw = resolve(config["paths"]["raw_dir"])
    if not (raw / config["sources"]["rules"]["file"]).exists():
        pytest.skip("source PDFs not present in data/raw")
    return build(config)


@pytest.fixture(scope="session")
def by_id(ingest: IngestResult) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in ingest.chunks}


FIXTURES = Path(__file__).parent / "fixtures"
TINY_CHUNKS = FIXTURES / "chunks_tiny.jsonl"


@pytest.fixture
def retrieval_config() -> dict[str, Any]:
    """configs/default.yaml pointed at the tiny fixture, with offline components only."""
    return merge(
        load_config("default.yaml"),
        {
            "data": {"chunks_file": str(TINY_CHUNKS)},
            "qdrant": {"location": ":memory:", "auto_index": True, "url_env": "DPDP_TEST_NO_URL"},
            "embedding": {"provider": "hash", "dim": 256},
            "reranker": {"enabled": False},
            "filters": {"doc_type": None, "rule": None, "section": None, "in_force_only": False},
            "cross_refs": {"enabled": False},
        },
    )


@pytest.fixture
def api_config(retrieval_config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Retrieval over the tiny fixture plus throwaway metrics/cache paths, no tracing."""
    return merge(
        retrieval_config,
        {
            "generation": {"k": 4},
            "cross_refs": {"enabled": True},
            "metrics": {"db_path": str(tmp_path / "metrics.db")},
            "api": {"response_cache": {"enabled": True, "dir": str(tmp_path / "cache")}},
            "tracing": {"langfuse_enabled": False},
        },
    )
