"""Chunking of single-page Gazette notifications (G.S.R. 843/844/845(E)): one chunk per clause."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from dpdp_rag.ingest.in_force import parse_gazette_date
from dpdp_rag.ingest.models import Chunk, Line
from dpdp_rag.ingest.pdf import PageSettings, read_pdf
from dpdp_rag.ingest.segment import LETTER, NUMBERED, ends_sentence, make_chunk, succ_label

_SIGNATURE = re.compile(r"^\[F\.\s*No\.")


@dataclass
class NotificationParse:
    chunks: list[Chunk]
    published: date


def body_lines(lines: list[Line]) -> list[Line]:
    """Lines from the "G.S.R. ..." opening to just before the "[F. No. ...]" file reference."""
    start = next(i for i, ln in enumerate(lines) if ln.text.strip().startswith("G.S.R."))
    end = next((i for i, ln in enumerate(lines) if _SIGNATURE.match(ln.text.strip())), len(lines))
    return lines[start:end]


def split_paragraphs(lines: list[Line]) -> list[tuple[str, list[Line]]]:
    """Split into (label, lines): "preamble"/"1" for the opening, then (a), (b)... or 2., 3. ..."""
    parts: list[tuple[str, list[Line]]] = [("1", [])]
    letter, number = "a", 1
    for line in lines:
        prev = parts[-1][1][-1] if parts[-1][1] else None
        m_letter = LETTER.match(line.text)
        m_num = NUMBERED.match(line.text)
        if m_letter and m_letter.group(1) == letter and ends_sentence(prev):
            if parts[0][0] == "1" and letter == "a":
                parts[0] = ("preamble", parts[0][1])
            parts.append((letter, []))
            letter = succ_label(letter)
        elif m_num and int(m_num.group(1)) == number + 1 and ends_sentence(prev):
            number += 1
            parts.append((str(number), []))
        parts[-1][1].append(line)
    return [(label, ls) for label, ls in parts if ls]


def parse_notification(
    path: Path, source: dict[str, Any], settings: PageSettings
) -> NotificationParse:
    lines = [ln for p in read_pdf(path, settings) for ln in p.lines]
    published = parse_gazette_date(next(ln.text for ln in lines if "New Delhi" in ln.text))
    common = {
        "doc_id": source["doc_id"],
        "doc_type": source["doc_type"],
        "gsr_no": source["gsr_no"],
        "source_file": source["file"],
        "title": source["title"],
        "in_force_date": published,
    }
    chunks = [
        make_chunk(
            ls,
            chunk_id=f"{source['doc_id']}:{label}",
            clause=None if label == "preamble" else label,
            **common,
        )
        for label, ls in split_paragraphs(body_lines(lines))
    ]
    return NotificationParse(chunks=chunks, published=published)
