"""Chunking of the DPDP Rules, 2025 (G.S.R. 846(E)).

One chunk per rule / sub-rule, per Illustration (its Cases kept together), per
Schedule row (numbered paragraph, lettered clause, or table row) and per clause of
a Schedule's Note. Page numbers are the printed Gazette pages (24-41), which the
corrigendum G.S.R. 892(E) refers to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dpdp_rag.ingest.models import Chunk, Line, PageContent, Table
from dpdp_rag.ingest.pdf import PageSettings, read_pdf
from dpdp_rag.ingest.segment import (
    ILLUSTRATION,
    LETTER,
    NUMBERED,
    SUB_NUMBER,
    Unit,
    cases_in,
    ends_sentence,
    heading_at,
    make_chunk,
    succ_label,
)
from dpdp_rag.ingest.text import join_lines, normalize_ws

_ORDINALS = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH"]
_SCHEDULE = re.compile(rf"^({'|'.join(_ORDINALS)})\s+SCHEDULE$")
_PART = re.compile(r"^PART\s+([A-Z])$")
_SEE = re.compile(r"^\[See\s+(.*)\]$")
_NOTE = re.compile(r"^Note\s*:\s*(.*)$")
_SIGNATURE = re.compile(r"^\[F\.\s*No\.")
_ITEM_TITLE = re.compile(r"^([A-Z][^.—–]{0,60})\.\s*[—–]+\s*(.*)$")
_HEADER_CELL = re.compile(r"^S\.\s*no\.?$", re.I)
_COLUMN_NO = re.compile(r"^\(\d+\)$")


@dataclass
class SchedUnit:
    """A Schedule row, illustration or note clause."""

    schedule: str
    number: int
    part: str | None
    item: str | None
    kind: str  # item | clause | row | illustration | note
    see_rules: list[str]
    context: list[str]  # schedule/part titles, used for title and refers_to
    lines: list[Line] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    title: str | None = None


def _segment_rules(lines: list[Line]) -> list[Unit]:
    units: list[Unit] = [Unit(labels={"rule": None, "sub_rule": None}, title="Preamble")]
    rule, sub = 0, 0
    title: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        text = line.text.strip()
        prev = units[-1].lines[-1] if units[-1].lines else None
        m = NUMBERED.match(text)
        if m and int(m.group(1)) == rule + 1 and ends_sentence(prev):
            found = heading_at(lines, i, m.group(2))
            if found:
                (title, rest), consumed = found
                rule, sub = rule + 1, 0
                units.append(Unit(labels={"rule": str(rule), "sub_rule": None}, title=title))
                if rest is not None:
                    m_sub = SUB_NUMBER.match(rest.text)
                    if m_sub and m_sub.group(1) == "1":
                        sub = 1
                        units[-1].labels["sub_rule"] = "1"
                    units[-1].lines.append(rest)
                i = consumed + 1
                continue
        if rule and ILLUSTRATION.match(text):
            units.append(
                Unit(labels={"rule": str(rule), "sub_rule": None}, title=title, illustration=True)
            )
        else:
            m_sub = SUB_NUMBER.match(text)
            if rule and m_sub and int(m_sub.group(1)) == sub + 1 and ends_sentence(prev):
                sub += 1
                units.append(Unit(labels={"rule": str(rule), "sub_rule": str(sub)}, title=title))
        units[-1].lines.append(line)
        i += 1
    return units


def _stream(pages: list[PageContent], start: Line) -> list[Line | Table]:
    """Lines and tables from `start` onwards, in reading order."""
    items: list[tuple[int, float, Line | Table]] = []
    for p in pages:
        items.extend((p.pdf_page, ln.y0, ln) for ln in p.lines)
        items.extend((p.pdf_page, t.y0, t) for t in p.tables)
    items.sort(key=lambda x: (x[0], x[1]))
    return [it for pg, y, it in items if (pg, y) >= (start.pdf_page, start.y0)]


def _segment_schedules(stream: list[Line | Table]) -> list[SchedUnit]:
    units: list[SchedUnit] = []
    name, number, part = "", 0, None
    see: list[str] = []
    context: list[str] = []
    headers: list[str] = []
    item, in_note = 0, False
    clause_expected = "a"
    note_last: str | None = None

    def current() -> SchedUnit | None:
        return (
            units[-1] if units and units[-1].schedule == name and units[-1].part == part else None
        )

    def new(kind: str, item_label: str | None, first: Line | None = None) -> SchedUnit:
        u = SchedUnit(
            schedule=name,
            number=number,
            part=part,
            item=item_label,
            kind=kind,
            see_rules=list(see),
            context=list(context),
            headers=list(headers),
        )
        if first is not None:
            u.lines.append(first)
        units.append(u)
        return u

    for el in stream:
        if isinstance(el, Table):
            for row in el.rows:
                if _HEADER_CELL.match(row[0]):
                    headers = row
                    continue
                if all(_COLUMN_NO.match(c) for c in row if c):
                    continue
                cur = current()
                if not row[0] and cur is not None and cur.kind == "row":
                    cur.cells = [f"{a} {b}".strip() for a, b in zip(cur.cells, row, strict=False)]
                    cur.pages.append(el.page or 0)
                    continue
                u = new("row", row[0].rstrip("."))
                u.cells, u.headers, u.pages = row, list(headers), [el.page or 0]
            continue

        text = el.text.strip()
        if _SIGNATURE.match(text):
            break
        if m := _SCHEDULE.match(text):
            name = f"{m.group(1).title()} Schedule"
            number = _ORDINALS.index(m.group(1)) + 1
            part, see, context, headers = None, [], [""], []
            item, in_note, clause_expected, note_last = 0, False, "a", None
            continue
        if m := _SEE.match(text):
            see = re.findall(r"(\d+)(?:\(\d+\))?", m.group(1))
            continue
        if m := _PART.match(text):
            part, item, headers = m.group(1), 0, []
            context = [context[0], ""]
            continue
        if m := _NOTE.match(text):
            # A Note applies to the whole Schedule ("In this Schedule, ..."), not a Part.
            in_note, note_last, part = True, None, None
            context = context[:1]
            continue
        cur = current()
        prev = cur.lines[-1] if cur and cur.lines else None
        if in_note:
            m = LETTER.match(text)
            if m and (
                note_last is None
                and m.group(1) == "a"
                or note_last is not None
                and m.group(1) in (note_last, succ_label(note_last))
            ):
                note_last = m.group(1)
                new("note", f"Note ({note_last})", el)
            elif cur is not None and cur.kind == "note":
                cur.lines.append(el)
            continue
        if ILLUSTRATION.match(text) and cur is not None:
            new("illustration", cur.item, el)
            continue
        m = NUMBERED.match(text)
        if m and int(m.group(1)) == item + 1 and (cur is None or ends_sentence(prev)):
            item += 1
            u = new("item", str(item), el)
            if t := _ITEM_TITLE.match(m.group(2)):
                u.title = t.group(1)
            continue
        m = LETTER.match(text)
        if (
            item == 0
            and m
            and m.group(1) == clause_expected
            and (cur is None or ends_sentence(prev))
        ):
            new("clause", m.group(1), el)
            clause_expected = succ_label(clause_expected)
            continue
        if cur is None:
            context[-1] = f"{context[-1]} {text}".strip()
        else:
            cur.lines.append(el)
    return units


def _sched_text(u: SchedUnit) -> str:
    if u.kind != "row":
        return join_lines(u.lines)[0]
    pairs = (
        zip(u.headers[1:], u.cells[1:], strict=False)
        if u.headers
        else ((None, c) for c in u.cells[1:])
    )
    body = " | ".join(f"{h}: {c}" if h else c for h, c in pairs)
    return f"{u.item}. {body}"


def _sched_chunk(u: SchedUnit, doc_id: str, note_index: int, common: dict[str, Any]) -> Chunk:
    cid = f"{doc_id}:sch{u.number}" + (f":{u.part}" if u.part else "")
    if u.kind == "note":
        cid += f":note{note_index}"
    else:
        cid += f":{u.item}" + (":ill" if u.kind == "illustration" else "")
    title = " — ".join(c for c in [u.schedule, *u.context] if c)
    fields: dict[str, Any] = dict(
        chunk_id=cid,
        schedule=u.schedule,
        part=u.part,
        item=u.item,
        title=title,
        illustration=u.kind == "illustration",
        **common,
    )
    if u.kind == "row":
        pages = [p for p in u.pages if p]
        return Chunk(
            text=_sched_text(u),
            page=pages[0] if pages else None,
            page_end=pages[-1] if pages else None,
            **fields,
        )
    chunk = make_chunk(u.lines, **fields)
    if u.kind == "illustration":
        chunk.cases = cases_in(chunk.text)
    return chunk


@dataclass
class RulesParse:
    chunks: list[Chunk]
    schedule_rules: dict[str, list[str]]  # chunk_id -> rules named in "[See rule ...]"
    schedule_context: dict[str, str]  # chunk_id -> schedule/part titles (for refers_to)
    published_line: str


def parse_rules(path: Path, source: dict[str, Any], settings: PageSettings) -> RulesParse:
    pages = read_pdf(path, settings)
    lines = [ln for p in pages for ln in p.lines]
    start = next(i for i, ln in enumerate(lines) if ln.text.strip().startswith("G.S.R."))
    sched = next(i for i, ln in enumerate(lines) if ln.text.strip() == source["schedules_marker"])
    published_line = next(ln.text for ln in lines[:start] if "New Delhi" in ln.text)
    common = {
        "doc_id": source["doc_id"],
        "doc_type": source["doc_type"],
        "gsr_no": source["gsr_no"],
        "source_file": source["file"],
    }

    chunks: list[Chunk] = []
    for unit in _segment_rules(lines[start:sched]):
        if not unit.lines:  # a heading whose text starts in the next unit, e.g. rule 19
            continue
        rule, sub = unit.labels["rule"], unit.labels["sub_rule"]
        if rule is None:
            cid = f"{source['doc_id']}:preamble"
        else:
            cid = f"{source['doc_id']}:r{rule}" + (f"({sub})" if sub else "")
            cid += ":ill" if unit.illustration else ""
        chunk = make_chunk(
            unit.lines,
            chunk_id=cid,
            rule=rule,
            sub_rule=sub,
            title=unit.title,
            illustration=unit.illustration,
            **common,
        )
        if unit.illustration:
            chunk.cases = cases_in(chunk.text)
        chunks.append(chunk)

    schedule_rules: dict[str, list[str]] = {}
    schedule_context: dict[str, str] = {}
    note_counter: dict[tuple[int, str | None], int] = {}
    for u in _segment_schedules(_stream(pages, lines[sched])):
        key = (u.number, u.part)
        if u.kind == "note":
            note_counter[key] = note_counter.get(key, 0) + 1
        chunk = _sched_chunk(u, source["doc_id"], note_counter.get(key, 0), common)
        schedule_rules[chunk.chunk_id] = u.see_rules
        schedule_context[chunk.chunk_id] = normalize_ws(" ".join(u.context))
        chunks.append(chunk)
    return RulesParse(chunks, schedule_rules, schedule_context, published_line)
