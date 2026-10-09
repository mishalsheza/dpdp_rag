"""Rendering retrieved chunks into the prompt's <documents> block.

Retrieved text and the user's question are untrusted: their markup is escaped so they
cannot close the surrounding tags, and the in-force status of every chunk is computed
here (deterministically) rather than left to the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from html import escape

from dpdp_rag.generation.pinpoint import pinpoint
from dpdp_rag.ingest.models import Chunk


def untrusted(text: str) -> str:
    """Escape &, < and > so untrusted text cannot open or close prompt tags."""
    return escape(text, quote=False)


def in_force(chunk: Chunk, as_of: date) -> bool:
    return chunk.in_force_date is not None and chunk.in_force_date <= as_of


def status(chunk: Chunk, as_of: date) -> str:
    if chunk.in_force_date is None:
        return "commencement date unknown"
    if in_force(chunk, as_of):
        return f"IN FORCE on {as_of.isoformat()} (since {chunk.in_force_date.isoformat()})"
    return (
        f"NOT YET IN FORCE on {as_of.isoformat()} "
        f"(comes into force on {chunk.in_force_date.isoformat()})"
    )


def _attr(value: str) -> str:
    return escape(value, quote=True)


def render_chunk(
    index: int,
    chunk: Chunk,
    as_of: date,
    entry_labels: Mapping[str, str],
    doc_labels: Mapping[str, str],
) -> str:
    attrs = {
        "index": str(index),
        "chunk_id": chunk.chunk_id,
        "pinpoint": pinpoint(chunk, entry_labels),
        "source": doc_labels.get(chunk.doc_type, chunk.doc_type),
        "in_force_date": chunk.in_force_date.isoformat() if chunk.in_force_date else "unknown",
        "status": status(chunk, as_of),
    }
    if chunk.corrected_by:
        attrs["corrected_by"] = chunk.corrected_by
    head = " ".join(f'{k}="{_attr(v)}"' for k, v in attrs.items())
    lines = [f"<document {head}>"]
    if chunk.title:
        lines.append(f"<title>{untrusted(chunk.title)}</title>")
    if chunk.corrected_by:
        lines.append(
            f"<note>This text includes the corrections made by {chunk.corrected_by}.</note>"
        )
    lines.append(f"<text>{untrusted(chunk.text)}</text>")
    lines.append("</document>")
    return "\n".join(lines)


def render_context(
    chunks: list[Chunk], as_of: date, entry_labels: Mapping[str, str], doc_labels: Mapping[str, str]
) -> str:
    return "\n\n".join(
        render_chunk(i, c, as_of, entry_labels, doc_labels) for i, c in enumerate(chunks, start=1)
    )
