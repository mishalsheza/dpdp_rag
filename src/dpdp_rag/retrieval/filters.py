"""Metadata filters, applied both in Qdrant and to in-process (BM25, cross-ref) results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from qdrant_client import models as qm

from dpdp_rag.ingest.models import Chunk


def date_int(d: date) -> int:
    """2027-05-13 -> 20270513, stored in the Qdrant payload as `in_force_int`."""
    return d.year * 10000 + d.month * 100 + d.day


def _as_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


@dataclass(frozen=True)
class Filters:
    doc_type: list[str] | None = None
    rule: list[str] | None = None
    section: list[str] | None = None
    in_force_on: date | None = None  # keep chunks with in_force_date <= this date

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Filters:
        on: date | None = None
        if cfg.get("in_force_only"):
            today = cfg.get("today")
            on = date.fromisoformat(str(today)) if today else date.today()
        return cls(
            doc_type=_as_list(cfg.get("doc_type")),
            rule=_as_list(cfg.get("rule")),
            section=_as_list(cfg.get("section")),
            in_force_on=on,
        )

    def date_only(self) -> Filters:
        return replace(Filters(), in_force_on=self.in_force_on)

    def is_empty(self) -> bool:
        return self == Filters()

    def matches(self, chunk: Chunk) -> bool:
        if self.doc_type and chunk.doc_type not in self.doc_type:
            return False
        if self.rule and chunk.rule not in self.rule:
            return False
        if self.section and chunk.section not in self.section:
            return False
        if self.in_force_on is not None:
            return chunk.in_force_date is not None and chunk.in_force_date <= self.in_force_on
        return True

    def to_qdrant(self) -> qm.Filter | None:
        must: list[qm.Condition] = []
        for key, values in (
            ("doc_type", self.doc_type),
            ("rule", self.rule),
            ("section", self.section),
        ):
            if values:
                must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=values)))
        if self.in_force_on is not None:
            must.append(
                qm.FieldCondition(
                    key="in_force_int", range=qm.Range(lte=date_int(self.in_force_on))
                )
            )
        return qm.Filter(must=must) if must else None
