"""Chunking of the DPDP Act (India Code edition): one chunk per section / sub-section."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dpdp_rag.ingest.in_force import ActCommencement
from dpdp_rag.ingest.models import Chunk, Line, PageContent
from dpdp_rag.ingest.pdf import PageSettings, read_pdf
from dpdp_rag.ingest.segment import (
    NUMBERED,
    SUB_NUMBER,
    Unit,
    ends_sentence,
    heading_at,
    make_chunk,
    split_clauses,
)
from dpdp_rag.ingest.text import join_lines, normalize_ws

_CHAPTER = re.compile(r"^\s*CHAPTER\s+([IVXL]+)\s*$")


def _body_lines(pages: list[PageContent], start_marker: str, end_marker: str) -> list[Line]:
    """Lines from the long title to the end of the Schedule.

    Everything before the start marker (cover, abbreviations, the arrangement of
    sections, i.e. the table of contents) and from the end marker on is dropped.
    """
    lines = [ln for p in pages for ln in p.lines]
    start = next(i for i, ln in enumerate(lines) if start_marker in ln.text)
    end = next((i for i, ln in enumerate(lines) if end_marker in ln.text), len(lines))
    return lines[start:end]


def _segment(lines: list[Line], schedule_marker: str) -> tuple[list[Unit], list[Line]]:
    """Split body lines into preamble / sub-section units, plus the Schedule lines."""
    units: list[Unit] = [Unit(labels={"section": None, "sub_section": None}, title="Long title")]
    chapter: tuple[str, str] | None = None
    section, sub = 0, 0
    title = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        text = line.text.strip()
        if text == schedule_marker and section > 0:
            return units, lines[i + 1 :]
        m_ch = _CHAPTER.match(text)
        if m_ch:
            chapter = (m_ch.group(1), normalize_ws(lines[i + 1].text))
            i += 2
            continue
        prev = units[-1].lines[-1] if units[-1].lines else None
        m_sec = NUMBERED.match(text)
        if m_sec and int(m_sec.group(1)) == section + 1 and ends_sentence(prev):
            found = heading_at(lines, i, m_sec.group(2))
            if found:
                (title, rest), consumed = found
                section, sub = section + 1, 0
                units.append(Unit(labels=_labels(section, None, chapter), title=title))
                if rest is not None:
                    m_sub = SUB_NUMBER.match(rest.text)
                    if m_sub and m_sub.group(1) == "1":
                        sub = 1
                        units[-1].labels["sub_section"] = "1"
                    units[-1].lines.append(rest)
                i = consumed + 1
                continue
        m_sub = SUB_NUMBER.match(text)
        if section and m_sub and int(m_sub.group(1)) == sub + 1 and ends_sentence(prev):
            sub += 1
            units.append(Unit(labels=_labels(section, str(sub), chapter), title=title))
        units[-1].lines.append(line)
        i += 1
    return units, []


def _labels(
    section: int, sub: str | None, chapter: tuple[str, str] | None
) -> dict[str, str | None]:
    return {
        "section": str(section),
        "sub_section": sub,
        "chapter": chapter[0] if chapter else None,
        "chapter_title": chapter[1] if chapter else None,
    }


def _schedule_rows(
    path: Path,
    settings: PageSettings,
    pdf_page: int,
    column_x: float,
) -> list[tuple[str, list[Line], list[Line]]]:
    """Rows of the penalty Schedule as (number, breach lines, penalty lines).

    The table has no ruling that pymupdf detects, so the page is read twice,
    clipped to the left and right columns, and rows are anchored on "N." lines.
    """
    import pymupdf

    with pymupdf.open(path) as doc:
        page = doc[pdf_page - 1]
        full = page.rect
        left = _clipped(page, settings, pymupdf.Rect(full.x0, full.y0, column_x, full.y1))
        right = _clipped(page, settings, pymupdf.Rect(column_x, full.y0, full.x1, full.y1))
    anchors = [(ln.y0, m.group(1)) for ln in left if (m := NUMBERED.match(ln.text))]
    rows: list[tuple[str, list[Line], list[Line]]] = []
    for idx, (y0, num) in enumerate(anchors):
        y_next = anchors[idx + 1][0] if idx + 1 < len(anchors) else float("inf")
        in_row = [ln for ln in left if y0 - 2 <= ln.y0 < y_next - 2]
        pen = [ln for ln in right if y0 - 2 <= ln.y0 < y_next - 2]
        rows.append((num, in_row, pen))
    return rows


def _clipped(page: Any, settings: PageSettings, clip: Any) -> list[Line]:
    from dpdp_rag.ingest.pdf import read_page

    return read_page(page, settings, clip=clip).lines


def parse_act(
    path: Path,
    source: dict[str, Any],
    settings: PageSettings,
    commencement: ActCommencement,
    max_chunk_chars: int,
) -> list[Chunk]:
    pages = read_pdf(path, settings)
    body = _body_lines(pages, source["body_start_marker"], source["body_end_marker"])
    units, schedule_lines = _segment(body, source["schedule_marker"])
    common = {
        "doc_id": source["doc_id"],
        "doc_type": source["doc_type"],
        "gsr_no": source["gsr_no"],
        "source_file": source["file"],
    }
    chunks: list[Chunk] = []
    for unit in units:
        if not unit.lines:
            continue
        sec, sub = unit.labels["section"], unit.labels["sub_section"]
        base_id = f"{source['doc_id']}:" + (
            "long_title" if sec is None else f"s{sec}" + (f"({sub})" if sub else "")
        )
        text, _ = join_lines(unit.lines)
        needs_split = sec is not None and (
            len(text) > max_chunk_chars or commencement.has_clause_level(sec, sub)
        )
        lead, clauses = split_clauses(unit.lines) if needs_split else (unit.lines, [])
        if not clauses:
            chunks.append(
                make_chunk(unit.lines, chunk_id=base_id, title=unit.title, **_meta(unit), **common)
            )
            continue
        lead_in = join_lines(lead)[0] or None
        for label, clause_lines in clauses:
            chunks.append(
                make_chunk(
                    clause_lines,
                    chunk_id=f"{base_id}({label})",
                    title=unit.title,
                    clause=label,
                    lead_in=lead_in,
                    **_meta(unit),
                    **common,
                )
            )

    if schedule_lines:
        sched_page = schedule_lines[0].pdf_page
        heading = next(
            (ln.text.strip() for ln in schedule_lines if ln.text.strip().startswith("[See")), None
        )
        for num, breach, penalty in _schedule_rows(
            path, settings, sched_page, float(source["schedule_penalty_column_x"])
        ):
            breach_text = re.sub(r"^\s*\d+\.\s*", "", join_lines(breach)[0])
            penalty_text = join_lines(penalty)[0]
            chunk = make_chunk(
                breach + penalty,
                chunk_id=f"{source['doc_id']}:schedule:{num}",
                schedule="The Schedule",
                item=num,
                penalty=penalty_text,
                title=f"The Schedule {heading or ''}".strip(),
                **common,
            )
            chunk.text = f"{num}. {breach_text} Penalty: {penalty_text}"
            chunks.append(chunk)
    return chunks


def _meta(unit: Unit) -> dict[str, Any]:
    return {
        "section": unit.labels.get("section"),
        "sub_section": unit.labels.get("sub_section"),
        "chapter": unit.labels.get("chapter"),
        "chapter_title": unit.labels.get("chapter_title"),
    }
