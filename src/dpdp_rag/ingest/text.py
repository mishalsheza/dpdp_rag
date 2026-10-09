"""Text clean-up helpers: noise filtering and line joining."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from dpdp_rag.ingest.models import Line, LineSpan

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize_quotes(text: str) -> str:
    """Map curly quotes to straight ones (for matching, never for stored text)."""
    return text.translate(_QUOTES)


def watermark_fragments(words: Iterable[str], max_len: int) -> set[str]:
    """All substrings of the watermark words up to `max_len` characters.

    For "IndiaCode" and max_len=2 this yields {"I", "In", "nd", "di", "aC", ...}.
    """
    frags: set[str] = set()
    for word in words:
        frags.add(word)
        for size in range(1, max_len + 1):
            for i in range(len(word) - size + 1):
                frags.add(word[i : i + size])
    return frags


def is_watermark_noise(text: str, fragments: set[str]) -> bool:
    """True for lines that are nothing but watermark residue ("aC", "di", "IndiaCode").

    Real one-word lines such as "or" or "and" are kept because they are not
    substrings of the watermark.
    """
    stripped = text.strip()
    return bool(stripped) and stripped.isalpha() and stripped in fragments


def join_lines(lines: Sequence[Line]) -> tuple[str, list[LineSpan]]:
    """Join visual lines into running text, recording where each line starts.

    A line ending in "-" (as in "sub-" / "section") is joined without a space,
    keeping the hyphen, which is correct for the compound words used in these texts.
    """
    parts: list[str] = []
    spans: list[LineSpan] = []
    length = 0
    for line in lines:
        piece = normalize_ws(line.text)
        if not piece:
            continue
        if parts:
            sep = "" if parts[-1].endswith("-") else " "
            parts.append(sep)
            length += len(sep)
        spans.append(LineSpan(page=line.page, line_no=line.line_no, offset=length))
        parts.append(piece)
        length += len(piece)
    return "".join(parts), spans


def join_cell(text: str) -> str:
    """Normalise a table cell whose content wraps over several lines."""
    out = ""
    for raw in text.splitlines():
        piece = normalize_ws(raw)
        if not piece:
            continue
        if out and not out.endswith("-"):
            out += " "
        out += piece
    return out
