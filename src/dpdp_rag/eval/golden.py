"""Golden question set (data/golden.jsonl): schema, loading and validation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: str
    gold_chunk_ids: list[str]
    reference_answer: str = Field(min_length=1)
    answerable: bool
    as_of_date: date | None = None
    author: str = Field(min_length=1)


class GoldenSetError(ValueError):
    """The golden set is malformed or refers to chunks that do not exist."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("Invalid golden set:\n  - " + "\n  - ".join(problems))


def validate(
    items: list[GoldenItem],
    chunk_ids: set[str],
    categories: Iterable[str],
    unanswerable_category: str = "unanswerable",
) -> list[str]:
    """Every problem found, as messages; empty when the set is valid."""
    allowed = set(categories)
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        where = f"{item.id}:"
        if item.id in seen:
            problems.append(f"{where} duplicate id")
        seen.add(item.id)
        if item.category not in allowed:
            problems.append(f"{where} unknown category {item.category!r}")
        missing = [c for c in item.gold_chunk_ids if c not in chunk_ids]
        if missing:
            problems.append(f"{where} gold_chunk_ids not in chunks.jsonl: {missing}")
        if len(set(item.gold_chunk_ids)) != len(item.gold_chunk_ids):
            problems.append(f"{where} duplicate gold_chunk_ids")
        if item.answerable and not item.gold_chunk_ids:
            problems.append(f"{where} answerable question has no gold_chunk_ids")
        if item.category == unanswerable_category and item.answerable:
            problems.append(f"{where} category {unanswerable_category!r} must be answerable=false")
    return problems


def load_golden(path: Path, chunk_ids: set[str], categories: Iterable[str]) -> list[GoldenItem]:
    """Parse and validate the golden set; raises GoldenSetError listing every problem."""
    items: list[GoldenItem] = []
    problems: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                items.append(GoldenItem.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError) as exc:
                problems.append(f"line {lineno}: {exc}")
    problems += validate(items, chunk_ids, categories)
    if problems:
        raise GoldenSetError(problems)
    return items
