"""Commencement dates, parsed from G.S.R. 843(E) (Act) and rule 1 (Rules).

Nothing here is hardcoded: the publication date, the offsets ("one year",
"eighteen months") and the provision lists are all read from the notification text.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

from dpdp_rag.ingest.refs import ProvisionRef, extract_refs, extract_rule_numbers

_MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "eighteen": 18,
    "twenty-four": 24,
}  # fmt: skip
_DATED = re.compile(r"the\s+(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+),?\s+(\d{4})")
_OFFSET = re.compile(
    r"\b([a-z-]+|\d+)\s+(years?|months?)\s+(?:from|after)\s+the\s+date\s+of\s+publication"
)
_ON_PUBLICATION = re.compile(r"\bdate of (?:their )?publication\b")


def parse_gazette_date(text: str) -> date:
    """First "the 13th November, 2025" style date in the text."""
    m = _DATED.search(text)
    if not m:
        raise ValueError("No gazette date found")
    return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def offset_months(text: str) -> int:
    """Months after publication at which a clause brings provisions into force."""
    m = _OFFSET.search(text)
    if m:
        qty = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]
        return qty * 12 if m.group(2).startswith("year") else qty
    if _ON_PUBLICATION.search(text):
        return 0
    raise ValueError(f"No commencement offset in: {text[:80]!r}")


@dataclass
class ActCommencement:
    """Commencement dates for provisions of the Act, keyed by the most specific reference."""

    published: date
    entries: dict[ProvisionRef, date] = field(default_factory=dict)
    fallback: date | None = None

    def date_for(
        self, section: str | None, sub_section: str | None = None, clause: str | None = None
    ) -> date | None:
        if section is None:
            return self.fallback
        for key in (
            ProvisionRef(section, sub_section, clause),
            ProvisionRef(section, sub_section),
            ProvisionRef(section),
        ):
            if key in self.entries:
                return self.entries[key]
        if sub_section is None:
            # A whole-section chunk whose sub-sections share one date.
            dates = {d for k, d in self.entries.items() if k.section == section}
            if len(dates) == 1:
                return dates.pop()
        return self.fallback

    def has_clause_level(self, section: str, sub_section: str | None) -> bool:
        return any(
            k.section == section and k.sub_section == sub_section and k.clause for k in self.entries
        )


def parse_act_commencement(
    clauses: list[str], published: date, fallback: date | None, max_section: int | None = None
) -> ActCommencement:
    """Build the Act schedule from the lettered clauses of G.S.R. 843(E)."""
    result = ActCommencement(published=published, fallback=fallback)
    for clause in clauses:
        when = add_months(published, offset_months(clause))
        for ref in extract_refs(clause, max_section=max_section):
            result.entries[ref] = when
    return result


def parse_rules_commencement(sub_rules: list[str], published: date) -> dict[str, date]:
    """Rule number -> commencement date, from sub-rules (2)-(4) of rule 1."""
    out: dict[str, date] = {}
    for text in sub_rules:
        if "come into force" not in text:
            continue
        when = add_months(published, offset_months(text))
        for rule in extract_rule_numbers(text):
            out[rule] = when
    return out
