from __future__ import annotations

from typing import Any

import pytest

from dpdp_rag.config import load_config, resolve
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
