"""Parsing of the corrigendum G.S.R. 892(E) and application of its patches to Rule chunks.

Patches are located by their quoted old_text, searched within the printed page the
corrigendum names. The cited line number is used only to choose between several
occurrences of the same old_text on that page (e.g. "of this Gazette" on lines 22
and 24 of page 24, or "." on page 38).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dpdp_rag.ingest.models import Chunk, LineSpan
from dpdp_rag.ingest.notifications import body_lines
from dpdp_rag.ingest.pdf import PageSettings, read_pdf
from dpdp_rag.ingest.segment import LETTER, make_chunk
from dpdp_rag.ingest.text import join_lines, normalize_quotes

log = logging.getLogger(__name__)

_PAGE_BLOCK = re.compile(
    r"\((?P<roman>[ivx]+)\)\s*in page\s+(?P<page>\d+)\s*,?\s*[–—-]*\s*"
    r"(?P<body>.*?)(?=\([ivx]+\)\s*in page|$)",
    re.S,
)
_ITEM = re.compile(
    r"(?:\((?P<sub>[a-z])\)\s*)?lines?\s+(?P<line>\d+)(?:\s+to\s+(?P<line_end>\d+))?\s*,\s*"
    r'for\s+"(?P<old>[^"]*)"\s*,\s*read\s+"(?P<new>[^"]*)"'
)
_LETTER_RANGE = re.compile(r"^\(([a-z])\)\s+to\s+\(([a-z])\)$")


class Patch(BaseModel):
    """One correction from the corrigendum."""

    item: str  # e.g. "(i)(a)"
    page: int  # printed page of G.S.R. 846(E)
    line: int
    line_end: int | None = None
    old_text: str
    new_text: str
    kind: str = "replace"  # replace | reletter
    source_text: str
    target_rule_or_schedule: str | None = None
    target_chunk_ids: list[str] = []
    match_strategy: str | None = None


def parse_patches(text: str) -> list[Patch]:
    """Structured patches from the corrigendum's running text."""
    text = normalize_quotes(text)
    patches: list[Patch] = []
    for block in _PAGE_BLOCK.finditer(text):
        for m in _ITEM.finditer(block.group("body")):
            sub = f"({m.group('sub')})" if m.group("sub") else ""
            old, new = m.group("old"), m.group("new")
            kind = (
                "reletter" if _LETTER_RANGE.match(old) and _LETTER_RANGE.match(new) else "replace"
            )
            patches.append(
                Patch(
                    item=f"({block.group('roman')}){sub}",
                    page=int(block.group("page")),
                    line=int(m.group("line")),
                    line_end=int(m.group("line_end")) if m.group("line_end") else None,
                    old_text=old,
                    new_text=new,
                    kind=kind,
                    source_text=f"in page {block.group('page')}, {m.group(0)[len(sub) :].strip()}",
                )
            )
    return patches


@dataclass
class _Hit:
    chunk: Chunk
    pos: int
    page: int | None
    line_no: int | None


def _locate(chunk: Chunk, pos: int) -> tuple[int | None, int | None]:
    span: LineSpan | None = None
    for s in chunk.lines:
        if s.offset <= pos:
            span = s
    return (span.page, span.line_no) if span else (chunk.page, None)


def _pattern(old: str) -> re.Pattern[str]:
    body = re.escape(old)
    if old[:1].isalnum():
        body = r"(?<!\w)" + body
    if old[-1:].isalnum():
        body += r"(?!\w)"
    return re.compile(body)


def _on_page(chunk: Chunk, page: int) -> bool:
    return chunk.page is not None and chunk.page <= page <= (chunk.page_end or chunk.page)


def _describe(chunk: Chunk) -> str:
    if chunk.rule:
        return f"rule {chunk.rule}" + (f"({chunk.sub_rule})" if chunk.sub_rule else "")
    out = chunk.schedule or chunk.chunk_id
    if chunk.part:
        out += f", Part {chunk.part}"
    if chunk.item:
        out += f", {chunk.item}" if chunk.item.startswith("Note") else f", item {chunk.item}"
    return out


def _replace(chunk: Chunk, pos: int, old: str, new: str, gsr_no: str) -> None:
    if chunk.text_original is None:
        chunk.text_original = chunk.text
    chunk.text = chunk.text[:pos] + new + chunk.text[pos + len(old) :]
    chunk.text_corrected = chunk.text
    chunk.corrected_by = gsr_no
    delta = len(new) - len(old)
    chunk.set_lines(
        [
            LineSpan(s.page, s.line_no, s.offset + delta if s.offset > pos else s.offset)
            for s in chunk.lines
        ]
    )


def _apply_replace(patch: Patch, chunks: list[Chunk], gsr_no: str) -> None:
    pat = _pattern(patch.old_text)
    hits: list[_Hit] = []
    for chunk in chunks:
        if not _on_page(chunk, patch.page):
            continue
        for m in pat.finditer(chunk.text):
            page, line_no = _locate(chunk, m.start())
            if page == patch.page:
                hits.append(_Hit(chunk, m.start(), page, line_no))
    if not hits:
        raise ValueError(
            f"Corrigendum {patch.item}: {patch.old_text!r} not found on page {patch.page}"
        )
    if len(hits) == 1:
        hit, patch.match_strategy = hits[0], "old_text"
    else:
        hit = min(hits, key=lambda h: abs((h.line_no or 10**6) - patch.line))
        patch.match_strategy = "old_text+line_tiebreak"
        log.info(
            "Corrigendum %s: %d matches of %r on page %d; line %d chosen",
            patch.item,
            len(hits),
            patch.old_text,
            patch.page,
            hit.line_no,
        )
    _replace(hit.chunk, hit.pos, patch.old_text, patch.new_text, gsr_no)
    patch.target_chunk_ids = [hit.chunk.chunk_id]
    patch.target_rule_or_schedule = _describe(hit.chunk)


def _letters(rng: str) -> list[str]:
    m = _LETTER_RANGE.match(rng)
    assert m
    return [chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)]


def _apply_reletter(patch: Patch, chunks: list[Chunk], gsr_no: str) -> None:
    """Re-letter a run of clauses printed as "(a) to (f)" so that they read "(a) to (g)"."""
    last = patch.line_end or patch.line
    run = [
        c
        for c in chunks
        if c.lines
        and c.lines[0].page == patch.page
        and patch.line <= c.lines[0].line_no <= last
        and LETTER.match(c.text)
    ]
    printed = [LETTER.match(c.text).group(1) for c in run]  # type: ignore[union-attr]
    wanted = _letters(patch.new_text)
    if sorted(set(printed)) != _letters(patch.old_text) or len(run) != len(wanted):
        raise ValueError(
            f"Corrigendum {patch.item}: clauses {printed} on page {patch.page} "
            f"do not match {patch.old_text!r} -> {patch.new_text!r}"
        )
    for chunk, old_label, new_label in zip(run, printed, wanted, strict=True):
        if old_label != new_label:
            _replace(
                chunk,
                chunk.text.index(f"({old_label})"),
                f"({old_label})",
                f"({new_label})",
                gsr_no,
            )
        if chunk.item and chunk.item.startswith("Note"):
            chunk.item = f"Note ({new_label})"
    patch.match_strategy = "old_text_label_run"
    patch.target_chunk_ids = [c.chunk_id for c in run]
    patch.target_rule_or_schedule = (
        f"{run[0].schedule}, Note" if run[0].schedule else _describe(run[0])
    )


def apply_patches(patches: list[Patch], chunks: list[Chunk], gsr_no: str) -> None:
    for patch in patches:
        if patch.kind == "reletter":
            _apply_reletter(patch, chunks, gsr_no)
        else:
            _apply_replace(patch, chunks, gsr_no)


@dataclass
class CorrigendumParse:
    patches: list[Patch]
    preamble: Chunk
    published: date


def parse_corrigendum(
    path: Path, source: dict[str, Any], settings: PageSettings
) -> CorrigendumParse:
    from dpdp_rag.ingest.in_force import parse_gazette_date

    lines = [ln for p in read_pdf(path, settings) for ln in p.lines]
    published = parse_gazette_date(next(ln.text for ln in lines if "New Delhi" in ln.text))
    body = body_lines(lines)
    first_item = next(
        i for i, ln in enumerate(body) if re.match(r"^\(i\)\s*in page", ln.text.strip())
    )
    preamble = make_chunk(
        body[:first_item],
        chunk_id=f"{source['doc_id']}:preamble",
        doc_id=source["doc_id"],
        doc_type=source["doc_type"],
        gsr_no=source["gsr_no"],
        title=source["title"],
        in_force_date=published,
        source_file=source["file"],
    )
    patches = parse_patches(join_lines(body[first_item:])[0])
    return CorrigendumParse(patches=patches, preamble=preamble, published=published)


def patch_chunks(
    result: CorrigendumParse, source: dict[str, Any], targets: dict[str, Chunk]
) -> list[Chunk]:
    """One corrigendum chunk per patch, carrying the rule/schedule it corrects."""
    out = [result.preamble]
    for p in result.patches:
        target = targets[p.target_chunk_ids[0]]
        out.append(
            Chunk(
                chunk_id=f"{source['doc_id']}:{p.item}",
                doc_id=source["doc_id"],
                doc_type=source["doc_type"],
                gsr_no=source["gsr_no"],
                title=f"Correction to {p.target_rule_or_schedule}",
                rule=target.rule,
                sub_rule=target.sub_rule,
                schedule=target.schedule,
                part=target.part,
                item=target.item,
                page=None,
                text=f"{p.item} {p.source_text}",
                corrects=list(p.target_chunk_ids),
                in_force_date=result.published,
                source_file=source["file"],
            )
        )
    return out
