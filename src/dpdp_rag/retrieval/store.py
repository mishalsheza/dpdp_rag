"""In-memory store of the chunks in chunks.jsonl, with lookups by provision."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from dpdp_rag.ingest.models import Chunk

_PROVISION = re.compile(r"^section (\w+)(?:\((\w+)\))?(?:\((\w+)\))?$")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def render(template: str, chunk: Chunk) -> str:
    """Fill a template such as "{title}\\n{text}" from chunk fields (None -> "")."""
    values = {k: ("" if v is None else v) for k, v in chunk.model_dump().items()}
    return template.format(**values).strip()


class ChunkStore:
    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.by_id: dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        self._act = [c for c in self.chunks if c.doc_type == "act" and c.section]

    @classmethod
    def from_file(cls, path: Path) -> ChunkStore:
        return cls(load_chunks(path))

    def provision(self, label: str) -> list[Chunk]:
        """Act chunks for "section 9", "section 9(1)" or "section 27(1)(d)".

        A clause reference falls back to its sub-section when the sub-section is a single
        chunk; a sub-section reference also returns the clause chunks it was split into.
        """
        m = _PROVISION.match(label)
        if not m:
            return []
        sec, sub, clause = m.groups()
        hits = [c for c in self._act if c.section == sec]
        if sub:
            hits = [c for c in hits if c.sub_section == sub]
        if clause:
            exact = [c for c in hits if c.clause == clause]
            hits = exact or [c for c in hits if c.clause is None]
        return hits
