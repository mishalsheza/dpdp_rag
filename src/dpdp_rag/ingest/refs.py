"""Extraction of references to provisions of the DPDP Act from legal text.

Handles forms such as "section 9", "sections 18 to 26", "sub-sections (1) and (3) of
section 9", "clause (b) of sub-section (2) of section 17". References to sections of
other enactments ("section 15 of the Rights of Persons with Disabilities Act, 2016")
are dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LIST_SEP = r"\s*(?:,\s*and|,|and|or|to)\s*"
_PAREN = r"\([a-z0-9]+\)"
_PAREN_LIST = rf"{_PAREN}(?:{_LIST_SEP}{_PAREN})*"
_NUM = r"\d+[A-Z]?"
_NUM_LIST = rf"{_NUM}\b(?:{_LIST_SEP}{_NUM}\b)*"

_REF = re.compile(
    rf"(?:sub-clauses?\s+(?P<subcl>{_PAREN_LIST})\s+of\s+)?"
    rf"(?:clauses?\s+(?P<cl>{_PAREN_LIST})\s+of\s+)?"
    rf"(?:sub-\s?sections?\s+(?P<ss>{_PAREN_LIST})\s+of\s+)?"
    rf"\bsections?\s+(?P<secs>{_NUM_LIST})",
)
_CHAIN = re.compile(r"^\s*(?:,\s*and|,|and|or)?\s*$")
_INTERNAL_OF = re.compile(
    r"^\s*of\s+(?:this Act|the Act|the said Act|the Digital Personal Data Protection Act)"
)
_EXTERNAL_OF = re.compile(
    r"^\s*of\s+(?:the\s+)?(?!Act\b)[A-Z][A-Za-z()'’,&\s-]{0,200}?\b(?:Act|Code)\b"
)
_EXCEPT = re.compile(r"\bexcept\b.*?\bof the said section\b", re.S)


@dataclass(frozen=True, order=True)
class ProvisionRef:
    """A reference to a section, optionally narrowed to a sub-section and clause."""

    section: str
    sub_section: str | None = None
    clause: str | None = None

    def label(self) -> str:
        out = f"section {self.section}"
        if self.sub_section:
            out += f"({self.sub_section})"
        if self.clause:
            out += f"({self.clause})"
        return out


def _expand(items: str, numeric: bool) -> list[str]:
    """Expand "(1) to (8) and (10)" or "18 to 26, 35" into individual labels."""
    tokens = re.findall(r"(\bto\b)|\(([a-z0-9]+)\)|\b(\d+[A-Z]?)\b", items)
    out: list[str] = []
    pending_range = False
    for to_kw, paren, bare in tokens:
        value = paren or bare
        if to_kw:
            pending_range = True
            continue
        if pending_range and out and numeric and value.isdigit() and out[-1].isdigit():
            out.extend(str(n) for n in range(int(out[-1]) + 1, int(value) + 1))
        else:
            out.append(value)
        pending_range = False
    return out


def _refs_from_match(m: re.Match[str]) -> list[ProvisionRef]:
    sections = _expand(m.group("secs"), numeric=True)
    subs = _expand(m.group("ss"), numeric=True) if m.group("ss") else [None]
    clauses = _expand(m.group("cl"), numeric=False) if m.group("cl") else [None]
    if len(sections) > 1:  # "sub-section (2) of sections 3 and 4" does not occur; keep it simple
        return [ProvisionRef(sec) for sec in sections]
    return [ProvisionRef(sections[0], ss, cl) for ss in subs for cl in clauses]


def extract_refs(text: str, max_section: int | None = None) -> list[ProvisionRef]:
    """All references to provisions of the DPDP Act made in `text`, in order of appearance."""
    text = _EXCEPT.sub("", text)
    matches = list(_REF.finditer(text))
    out: list[ProvisionRef] = []
    group: list[re.Match[str]] = []

    def flush(after: str) -> None:
        if not group:
            return
        external = bool(_EXTERNAL_OF.match(after)) and not _INTERNAL_OF.match(after)
        if not external:
            for m in group:
                out.extend(_refs_from_match(m))
        group.clear()

    for i, m in enumerate(matches):
        group.append(m)
        nxt = matches[i + 1] if i + 1 < len(matches) else None
        between = text[m.end() : nxt.start()] if nxt else None
        if between is None or not _CHAIN.match(between):
            flush(text[m.end() : m.end() + 250])

    if max_section is not None:
        out = [r for r in out if r.section.isdigit() and 1 <= int(r.section) <= max_section]
    seen: set[ProvisionRef] = set()
    return [r for r in out if not (r in seen or seen.add(r))]


def refers_to(refs: list[ProvisionRef]) -> tuple[list[str], list[str]]:
    """Section-level labels ("section 9") and full labels ("section 9(1)")."""
    sections = sorted({f"section {r.section}" for r in refs}, key=_section_key)
    provisions = sorted({r.label() for r in refs}, key=_section_key)
    return sections, provisions


def _section_key(label: str) -> tuple[int, str]:
    m = re.match(r"section (\d+)(.*)", label)
    return (int(m.group(1)), m.group(2)) if m else (10**6, label)


_RULE_REF = re.compile(r"\b[Rr]ules?\s+(\d+(?:\s*(?:,|and|to)\s*\d+)*)")


def extract_rule_numbers(text: str) -> list[str]:
    """Rule numbers in "Rules 1, 2 and 17 to 21" or "Rule 4"."""
    out: list[str] = []
    for m in _RULE_REF.finditer(text):
        out.extend(_expand(m.group(1), numeric=True))
    return out
