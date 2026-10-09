"""Qdrant collection management: connect, (re)build from chunks, dense search."""

from __future__ import annotations

import logging
import os
import uuid
import warnings
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException

from dpdp_rag.config import REPO_ROOT
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval.embeddings import Embedder
from dpdp_rag.retrieval.filters import Filters, date_int
from dpdp_rag.retrieval.store import render

log = logging.getLogger(__name__)
_ID_NAMESPACE = uuid.UUID("5f1d7f0e-8a43-4f6b-9c1e-2d7a3b9e6c11")


def point_id(chunk_id: str) -> str:
    """Stable Qdrant point id for a chunk, so re-indexing overwrites rather than duplicates."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def make_client(cfg: dict[str, Any]) -> QdrantClient:
    """Qdrant Cloud/remote if the URL env var is set, else `location`, else the local URL."""
    load_dotenv(REPO_ROOT / ".env")
    url = os.environ.get(cfg["url_env"], "").strip()
    if url:
        return QdrantClient(url=url, api_key=os.environ.get(cfg["api_key_env"]) or None)
    if cfg.get("location"):
        return QdrantClient(location=cfg["location"])
    return QdrantClient(url=cfg["local_url"])


def payload(chunk: Chunk) -> dict[str, Any]:
    data = chunk.model_dump(mode="json")
    data["in_force_int"] = date_int(chunk.in_force_date) if chunk.in_force_date else None
    return data


class QdrantIndex:
    def __init__(
        self, client: QdrantClient, cfg: dict[str, Any], embedder: Embedder, template: str
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.collection = cfg["collection"]
        self.embedder = embedder
        self.template = template

    def is_current(self, chunks: list[Chunk]) -> bool:
        """True if the collection exists with the right vector size and point count."""
        try:
            exists = self.client.collection_exists(self.collection)
        except ResponseHandlingException as exc:
            raise ConnectionError(
                f"Cannot reach Qdrant ({exc}). Start it with `docker compose up -d`, or set "
                f"{self.cfg['url_env']} / {self.cfg['api_key_env']} for Qdrant Cloud."
            ) from exc
        if not exists:
            return False
        info = self.client.get_collection(self.collection)
        params = info.config.params.vectors
        size = params.size if isinstance(params, qm.VectorParams) else None
        count = self.client.count(self.collection, exact=True).count
        return size == self.embedder.dim and count == len(chunks)

    def build(self, chunks: list[Chunk]) -> int:
        """Recreate the collection and upsert every chunk with its metadata payload."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=self.embedder.dim, distance=qm.Distance.COSINE),
        )
        for name in self.cfg.get("keyword_indexes", []):
            self._index_field(name, qm.PayloadSchemaType.KEYWORD)
        for name in self.cfg.get("integer_indexes", []):
            self._index_field(name, qm.PayloadSchemaType.INTEGER)
        batch = int(self.cfg["upsert_batch_size"])
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            vectors = self.embedder.embed_documents([render(self.template, c) for c in part])
            self.client.upsert(
                self.collection,
                points=[
                    qm.PointStruct(id=point_id(c.chunk_id), vector=v, payload=payload(c))
                    for c, v in zip(part, vectors, strict=True)
                ],
            )
        log.info("Indexed %d chunks into %s", len(chunks), self.collection)
        return len(chunks)

    def _index_field(self, name: str, schema: qm.PayloadSchemaType) -> None:
        # In-process (":memory:"/path) mode has no payload indexes and warns; that is fine.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.client.create_payload_index(self.collection, name, field_schema=schema)

    def search(self, query: str, limit: int, filters: Filters) -> list[tuple[str, float]]:
        """(chunk_id, cosine score) pairs, best first."""
        result = self.client.query_points(
            self.collection,
            query=self.embedder.embed_query(query),
            query_filter=filters.to_qdrant(),
            limit=limit,
            with_payload=["chunk_id"],
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in result.points if p.payload]
