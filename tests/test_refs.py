from __future__ import annotations

import pytest

from dpdp_rag.ingest.refs import extract_refs, extract_rule_numbers, refers_to


def labels(text: str) -> list[str]:
    return [r.label() for r in extract_refs(text, max_section=44)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sub-sections (1) and (3) of section 9 of the Act", ["section 9(1)", "section 9(3)"]),
        ("clause (a) of sub-section (2) of section 17 of the Act", ["section 17(2)(a)"]),
        ("sections 18 to 20", ["section 18", "section 19", "section 20"]),
        (
            "sub-sections (1) to (3) and (10) of section 6",
            ["section 6(1)", "section 6(2)", "section 6(3)", "section 6(10)"],
        ),
        ("under section 15 of the Rights of Persons with Disabilities Act, 2016", []),
        ("section 14A and section 16 of the Telecom Regulatory Authority of India Act, 1997", []),
        ("section 21 of the Indian Penal Code", []),
        ("in the Gazette of India, Part II, Section 3, Sub-section (i)", []),
        ("section 27 except clause (d) of sub-section (1) of the said section", ["section 27"]),
    ],
)
def test_extract_refs(text: str, expected: list[str]) -> None:
    assert labels(text) == expected


def test_refers_to_levels() -> None:
    sections, provisions = refers_to(
        extract_refs("sub-sections (1) and (3) of section 9; section 10")
    )
    assert sections == ["section 9", "section 10"]
    assert provisions == ["section 9(1)", "section 9(3)", "section 10"]


def test_rule_numbers() -> None:
    assert extract_rule_numbers("Rules 3, 5 to 7, 22 and 23 shall") == [
        "3",
        "5",
        "6",
        "7",
        "22",
        "23",
    ]
