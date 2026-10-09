"""Provisions named in an answer's prose ("Section 9(1)", "Rule 10(2)"), checked against
the cited chunks. Citations are validated separately; this catches the answer text naming
a provision the model never cited, which is how unsupported claims slip through."""

from __future__ import annotations

import re
from dataclasses import dataclass

from dpdp_rag.ingest.models import Chunk

_MENTION = re.compile(r"\b(section|rule)s?\s+(\d+[A-Z]?)(?:\s*\(\s*(\w+)\s*\))?", re.IGNORECASE)


@dataclass(frozen=True)
class Mention:
    kind: str  # "section" (the Act) or "rule" (the Rules)
    number: str
    sub: str | None  # sub-section / clause / sub-rule, without brackets

    @property
    def label(self) -> str:
        sub = f"({self.sub})" if self.sub else ""
        return f"{self.kind.capitalize()} {self.number}{sub}"


def find_mentions(text: str) -> list[Mention]:
    """Section and rule mentions in `text`, in order of first appearance, deduplicated."""
    seen: dict[Mention, None] = {}
    for kind, number, sub in _MENTION.findall(text):
        seen.setdefault(Mention(kind.lower(), number.upper(), sub.lower() or None), None)
    return list(seen)


def _identity(chunk: Chunk) -> Mention | None:
    if chunk.section:
        return Mention("section", chunk.section.upper(), chunk.sub_section or chunk.clause)
    if chunk.rule:
        return Mention("rule", chunk.rule.upper(), chunk.sub_rule)
    return None


def is_chunk_of(mention: Mention, chunk: Chunk) -> bool:
    """True if `chunk` is the provision `mention` names, or part or all of it."""
    ident = _identity(chunk)
    if ident is None or (ident.kind, ident.number) != (mention.kind, mention.number):
        return False
    return mention.sub is None or ident.sub is None or ident.sub == mention.sub


def supports(mention: Mention, chunk: Chunk) -> bool:
    """True if citing `chunk` backs naming `mention`: it is that provision, it refers to it
    (e.g. a Schedule row that disapplies Section 9(1)), or its text names it."""
    if is_chunk_of(mention, chunk):
        return True
    label = mention.label.lower()
    if label in (p.lower() for p in chunk.refers_to_provisions):
        return True
    if mention.sub is None and label in (r.lower() for r in chunk.refers_to):
        return True
    return Mention(mention.kind, mention.number, mention.sub) in find_mentions(chunk.text)
