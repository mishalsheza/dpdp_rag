"""Pinpoint labels for chunks: "Section 9(1)", "Rule 8(3)", "Third Schedule, row 1"."""

from __future__ import annotations

from collections.abc import Mapping

from dpdp_rag.ingest.models import Chunk


def _paren(*parts: str | None) -> str:
    return "".join(f"({p})" for p in parts if p)


def pinpoint(chunk: Chunk, entry_labels: Mapping[str, str]) -> str:
    if chunk.doc_type == "corrigendum":
        item = chunk.chunk_id.split(":", 1)[1]
        return f"{chunk.gsr_no}, item {item}" if item != "preamble" else f"{chunk.gsr_no}"
    if chunk.doc_type == "notification":
        return f"{chunk.gsr_no}" + (f", clause ({chunk.clause})" if chunk.clause else "")
    if chunk.schedule:
        out = chunk.schedule
        if chunk.part:
            out += f", Part {chunk.part}"
        if chunk.item and chunk.item.startswith("Note"):
            return f"{out}, {chunk.item}"
        if chunk.item:
            label = entry_labels.get(chunk.schedule, "item")
            item = f"({chunk.item})" if label == "clause" else chunk.item
            out += f", {label} {item}"
        return out + (", Illustration" if chunk.illustration else "")
    if chunk.section:
        return f"Section {chunk.section}{_paren(chunk.sub_section, chunk.clause)}"
    if chunk.rule:
        out = f"Rule {chunk.rule}{_paren(chunk.sub_rule)}"
        return out + (", Illustration" if chunk.illustration else "")
    if chunk.doc_type == "act":
        return "Act, long title"
    return "Rules, preamble" if chunk.doc_type == "rules" else chunk.chunk_id
