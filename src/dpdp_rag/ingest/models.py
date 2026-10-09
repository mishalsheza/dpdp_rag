"""Data models shared by the ingestion modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

DocType = Literal["act", "rules", "notification", "corrigendum"]


@dataclass(frozen=True)
class Line:
    """One visual line of body text on a PDF page."""

    text: str
    pdf_page: int  # 1-based page index in the PDF
    page: int | None  # printed page number, when the page carries one
    line_no: int  # 1-based line number in the page body (headers excluded)
    x0: float
    y0: float


@dataclass(frozen=True)
class Table:
    """A table detected on a page; rows are lists of normalised cell texts."""

    rows: list[list[str]]
    pdf_page: int
    page: int | None
    y0: float


@dataclass
class PageContent:
    pdf_page: int
    page: int | None
    lines: list[Line] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)


@dataclass(frozen=True)
class LineSpan:
    """Where a source line starts inside a chunk's text."""

    page: int | None
    line_no: int
    offset: int


class Chunk(BaseModel):
    """A retrievable unit of legal text plus its metadata."""

    chunk_id: str
    doc_id: str
    doc_type: DocType
    gsr_no: str | None = None
    title: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    sub_section: str | None = None
    clause: str | None = None
    rule: str | None = None
    sub_rule: str | None = None
    schedule: str | None = None
    part: str | None = None
    item: str | None = None
    illustration: bool = False
    cases: list[str] = Field(default_factory=list)
    lead_in: str | None = None
    page: int | None = None
    page_end: int | None = None
    text: str
    in_force_date: date | None = None
    refers_to: list[str] = Field(default_factory=list)
    refers_to_provisions: list[str] = Field(default_factory=list)
    penalty: str | None = None
    text_original: str | None = None
    text_corrected: str | None = None
    corrected_by: str | None = None
    source_file: str | None = None

    # Line positions inside `text`, used to place corrigendum patches. Not serialised.
    _lines: list[LineSpan] = PrivateAttr(default_factory=list)

    @property
    def lines(self) -> list[LineSpan]:
        return self._lines

    def set_lines(self, lines: list[LineSpan]) -> None:
        self._lines = lines
