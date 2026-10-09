"""Shared helpers for splitting a stream of lines into numbered legal units."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from dpdp_rag.ingest.models import Chunk, Line
from dpdp_rag.ingest.text import join_lines

NUMBERED = re.compile(r"^\s*(\d+)\s*\.\s*(.*)$")
SUB_NUMBER = re.compile(r"^\s*\((\d+)\)\s*(.*)$")
LETTER = re.compile(r"^\s*\(([a-z]{1,2})\)\s*(.*)$")
HEADING_SPLIT = re.compile(r"^(?P<title>.*?)\s*[—–]+\s*(?P<rest>.*)$")
ILLUSTRATION = re.compile(r"^\s*Illustrations?\.?\s*$")
CASE = re.compile(r"\bCase\s+(\d+)\s*:")
_TERMINAL = (".", ";", ":", "—", "–", "-")


def succ_label(label: str) -> str:
    """Next clause letter: a -> b, ..., z -> za, za -> zb."""
    if label == "z":
        return "za"
    if len(label) == 2:
        return label[0] + chr(ord(label[1]) + 1)
    return chr(ord(label) + 1)


def ends_sentence(line: Line | None) -> bool:
    """True if a new numbered unit may start after `line` (end of a clause or list item)."""
    if line is None:
        return True
    text = line.text.rstrip()
    return text.endswith(_TERMINAL) or text.endswith(("; and", "; or", ", and", ", or"))


def with_text(line: Line, text: str) -> Line:
    return replace(line, text=text)


@dataclass
class Unit:
    """A run of lines forming one chunk candidate, with its structural labels."""

    labels: dict[str, str | None]
    lines: list[Line] = field(default_factory=list)
    title: str | None = None
    lead_in: str | None = None
    illustration: bool = False

    def text(self) -> str:
        return join_lines(self.lines)[0]


def split_heading(lines: list[Line]) -> tuple[str, Line | None] | None:
    """Find "Title.—rest" across the first lines of a numbered unit.

    Returns the title and the remainder as a Line (or None if the remainder is empty),
    or None when no dash appears in the given lines.
    """
    buf = ""
    for line in lines:
        buf = f"{buf} {line.text.strip()}".strip()
        m = HEADING_SPLIT.match(buf)
        if m:
            title = m.group("title").strip().rstrip(".").strip()
            rest = m.group("rest").strip()
            return title, (with_text(line, rest) if rest else None)
    return None


def heading_at(
    lines: list[Line], i: int, first_rest: str, max_lines: int = 3
) -> tuple[tuple[str, Line | None], int] | None:
    """Heading of a numbered unit starting at lines[i], which may wrap over lines.

    `first_rest` is lines[i] with its "N." number removed. Returns the
    (title, remainder) pair and the index of the last line consumed.
    """
    first = with_text(lines[i], first_rest)
    for k in range(max_lines):
        if i + k >= len(lines):
            break
        head = split_heading([first, *lines[i + 1 : i + 1 + k]])
        if head:
            return head, i + k
    return None


def split_clauses(lines: list[Line]) -> tuple[list[Line], list[tuple[str, list[Line]]]]:
    """Split lines at top-level lettered clauses (a), (b), ... in sequence.

    Returns the lead-in lines and a list of (label, lines). Nested sub-clauses such
    as (i), (ii) are not split because they break the letter sequence.
    """
    lead: list[Line] = []
    clauses: list[tuple[str, list[Line]]] = []
    expected = "a"
    for line in lines:
        m = LETTER.match(line.text)
        if m and m.group(1) == expected:
            clauses.append((expected, [line]))
            expected = succ_label(expected)
        elif clauses:
            clauses[-1][1].append(line)
        else:
            lead.append(line)
    return lead, clauses


def make_chunk(unit_lines: list[Line], **fields: object) -> Chunk:
    text, spans = join_lines(unit_lines)
    pages = [ln.page for ln in unit_lines if ln.page is not None]
    chunk = Chunk(
        text=text, page=pages[0] if pages else None, page_end=pages[-1] if pages else None, **fields
    )  # type: ignore[arg-type]
    chunk.set_lines(spans)
    return chunk


def cases_in(text: str) -> list[str]:
    return [f"Case {n}" for n in dict.fromkeys(CASE.findall(text))]
