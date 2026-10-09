"""PDF reading with pymupdf: visual lines, printed page numbers, tables, noise removal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from dpdp_rag.ingest.models import Line, PageContent, Table
from dpdp_rag.ingest.text import is_watermark_noise, join_cell, watermark_fragments

_DIGITS = re.compile(r"^\d{1,3}$")
_MARKER = re.compile(r"^[\d*]+$")


@dataclass(frozen=True)
class PageSettings:
    """Per-document layout settings (see configs/ingest.yaml)."""

    header_max_y: float
    footer_min_y: float
    watermark_words: tuple[str, ...]
    watermark_colors: tuple[int, ...]
    fragment_max_len: int
    min_font_size: float
    footer_patterns: tuple[re.Pattern[str], ...]
    editorial_patterns: tuple[re.Pattern[str], ...]
    drop_footnotes: bool = False
    footnote_rule_max_width: float = 200
    detect_tables: bool = False

    @classmethod
    def from_config(cls, noise: dict[str, Any], source: dict[str, Any]) -> PageSettings:
        return cls(
            header_max_y=float(source["header_max_y"]),
            footer_min_y=float(source["footer_min_y"]),
            watermark_words=tuple(noise["watermark_words"]),
            watermark_colors=tuple(noise["watermark_colors"]),
            fragment_max_len=int(noise["watermark_fragment_max_len"]),
            min_font_size=float(noise["min_font_size"]),
            footer_patterns=tuple(re.compile(p) for p in noise["footer_patterns"]),
            editorial_patterns=tuple(re.compile(p) for p in noise["editorial_patterns"]),
            drop_footnotes=bool(source.get("drop_footnotes", False)),
            footnote_rule_max_width=float(noise["footnote_rule_max_width"]),
            detect_tables=bool(source.get("detect_tables", False)),
        )


@dataclass
class _Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def _is_watermark_span(
    span: dict[str, Any], direction: tuple[float, float], s: PageSettings
) -> bool:
    if abs(direction[1]) > 0.01:  # rotated text: the diagonal watermark
        return True
    if span["color"] in s.watermark_colors:
        return True
    return span["text"].strip() in s.watermark_words


def _footnote_top(page: pymupdf.Page, s: PageSettings) -> float | None:
    """y of a short footnote separator rule in the lower half of the page, if any."""
    if not s.drop_footnotes:
        return None
    half = page.rect.height / 2
    for drawing in page.get_drawings():
        r = drawing["rect"]
        if r.height < 2 and 50 < r.width < s.footnote_rule_max_width and r.y0 > half and r.x0 < 80:
            return float(r.y0)
    return None


def _group_lines(spans: list[_Span]) -> list[list[_Span]]:
    """Group spans whose vertical centres coincide into visual lines."""
    groups: list[list[_Span]] = []
    for span in sorted(spans, key=lambda sp: ((sp.y0 + sp.y1) / 2, sp.x0)):
        mid = (span.y0 + span.y1) / 2
        if groups:
            last = groups[-1]
            last_mid = sum((sp.y0 + sp.y1) / 2 for sp in last) / len(last)
            if abs(mid - last_mid) <= 3.0:
                last.append(span)
                continue
        groups.append([span])
    return [sorted(g, key=lambda sp: sp.x0) for g in groups]


def _line_text(spans: list[_Span]) -> str:
    out = ""
    prev: _Span | None = None
    for sp in spans:
        if (
            prev is not None
            and sp.x0 - prev.x1 > 1.5
            and not out.endswith(" ")
            and not sp.text.startswith(" ")
        ):
            out += " "
        out += sp.text
        prev = sp
    return out


def _inside(sp: _Span, bboxes: list[tuple[float, float, float, float]]) -> bool:
    cx, cy = (sp.x0 + sp.x1) / 2, (sp.y0 + sp.y1) / 2
    return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in bboxes)


def read_page(page: pymupdf.Page, s: PageSettings, clip: pymupdf.Rect | None = None) -> PageContent:
    """Body lines and tables of one page; `clip` keeps only spans centred inside it."""
    fragments = watermark_fragments(s.watermark_words, s.fragment_max_len)
    footnote_y = _footnote_top(page, s)

    tables: list[tuple[tuple[float, float, float, float], list[list[str]]]] = []
    if s.detect_tables:
        for tab in page.find_tables().tables:
            rows = [[join_cell(c or "") for c in row] for row in tab.extract()]
            tables.append((tuple(tab.bbox), rows))
    table_boxes = [bbox for bbox, _ in tables]

    printed: int | None = None
    body: list[_Span] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                # Whitespace-only spans are kept: they carry the space in "(a) the".
                if not span["text"] or _is_watermark_span(span, line["dir"], s):
                    continue
                x0, y0, x1, y1 = span["bbox"]
                sp = _Span(span["text"], x0, y0, x1, y1)
                in_band = y1 <= s.header_max_y or y0 >= s.footer_min_y
                if in_band:
                    if _DIGITS.match(span["text"].strip()):
                        printed = int(span["text"].strip())
                    continue
                if span["size"] < s.min_font_size and _MARKER.match(span["text"].strip()):
                    continue  # superscript footnote marker; ordinal suffixes ("25th") are kept
                if footnote_y is not None and y0 >= footnote_y:
                    continue
                if _inside(sp, table_boxes):
                    continue
                if clip is not None and not _inside(sp, [tuple(clip)]):
                    continue
                body.append(sp)

    content = PageContent(pdf_page=page.number + 1, page=printed)
    line_no = 0
    for group in _group_lines(body):
        text = _line_text(group)
        for pat in s.editorial_patterns:
            text = pat.sub("", text)
        if not text.strip() or is_watermark_noise(text, fragments):
            continue
        if any(p.search(text.strip()) for p in s.footer_patterns):
            continue
        line_no += 1
        content.lines.append(
            Line(
                text=text.rstrip(),
                pdf_page=content.pdf_page,
                page=printed,
                line_no=line_no,
                x0=group[0].x0,
                y0=group[0].y0,
            )
        )
    for bbox, rows in tables:
        content.tables.append(Table(rows=rows, pdf_page=content.pdf_page, page=printed, y0=bbox[1]))
    return content


def read_pdf(path: Path, s: PageSettings) -> list[PageContent]:
    with pymupdf.open(path) as doc:
        return [read_page(page, s) for page in doc]
