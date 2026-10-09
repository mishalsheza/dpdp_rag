from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dpdp_rag.config import OVERRIDES_ENV, load_config, merge, resolve
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.ingest.pipeline import IngestResult, build
from dpdp_rag.retrieval.boost import SIGNALS


@pytest.fixture(autouse=True)
def _no_local_config_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    # A developer's .env may set DPDP_CONFIG_OVERRIDES (e.g. groq.yaml); create_app's
    # load_dotenv would then leak it into later tests. Set but empty, it is never loaded.
    monkeypatch.setenv(OVERRIDES_ENV, "")


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
            "query_rewrite": {"strategy": "off"},  # no LLM calls in tests
            # Boosts reorder results; tests of other behaviour keep the plain order.
            # tests/test_boost.py turns them on.
            "boost": {"weights": dict.fromkeys(SIGNALS, 0)},
        },
    )


# A paid price list for tests, so cost accounting and budgets are exercised even though
# the configured Groq free tier costs $0. USD per 1M tokens.
TEST_PRICING: dict[str, Any] = {
    "input": 1.00,
    "output": 5.00,
    "long_context_threshold": 100000,
    "long_input": 2.00,
    "long_output": 10.00,
    "cache_write_multiplier": 1.25,
    "cache_read_multiplier": 0.10,
}


@pytest.fixture
def api_config(retrieval_config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Retrieval over the tiny fixture plus throwaway metrics/cache paths, no tracing."""
    return merge(
        retrieval_config,
        {
            "llm": {"pricing": TEST_PRICING},
            "generation": {"k": 4},
            "cross_refs": {"enabled": True},
            "metrics": {"db_path": str(tmp_path / "metrics.db")},
            "api": {"response_cache": {"enabled": True, "dir": str(tmp_path / "cache")}},
            "tracing": {"langfuse_enabled": False},
        },
    )


GOLDEN_TINY = FIXTURES / "golden_tiny.jsonl"


@pytest.fixture
def eval_config(tmp_path: Path) -> dict[str, Any]:
    return merge(
        load_config("eval.yaml"),
        {
            "golden_file": str(GOLDEN_TINY),
            "output_dir": str(tmp_path / "runs"),
            "judge": {"pricing": TEST_PRICING},
        },
    )
