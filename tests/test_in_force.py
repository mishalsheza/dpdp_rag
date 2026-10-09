"""in_force_date mapping from G.S.R. 843(E) (Act) and rule 1 (Rules)."""

from __future__ import annotations

from datetime import date

import pytest

from dpdp_rag.ingest.in_force import add_months, offset_months, parse_gazette_date

PUBLISHED = date(2025, 11, 13)
ONE_YEAR = date(2026, 11, 13)
EIGHTEEN_MONTHS = date(2027, 5, 13)


def test_date_arithmetic() -> None:
    assert parse_gazette_date("New Delhi, the 13th November, 2025") == PUBLISHED
    assert add_months(PUBLISHED, 12) == ONE_YEAR
    assert add_months(PUBLISHED, 18) == EIGHTEEN_MONTHS
    assert add_months(date(2025, 8, 31), 6) == date(2026, 2, 28)


@pytest.mark.parametrize(
    ("text", "months"),
    [
        ("shall come into force on the date of their publication in the Official Gazette", 0),
        ("shall come into force one year after the date of publication of this Gazette", 12),
        ("eighteen months from the date of publication of this gazette", 18),
    ],
)
def test_offsets(text: str, months: int) -> None:
    assert offset_months(text) == months


def test_rules_mapping(ingest) -> None:
    expected = {str(r): PUBLISHED for r in (1, 2, 17, 18, 19, 20, 21)}
    expected["4"] = ONE_YEAR
    expected |= {str(r): EIGHTEEN_MONTHS for r in (3, *range(5, 17), 22, 23)}
    assert ingest.rules_commencement == expected


def test_rule_chunks_carry_dates(ingest, by_id) -> None:
    for c in ingest.chunks:
        if c.doc_id == "dpdp_rules_2025" and c.rule:
            assert c.in_force_date == ingest.rules_commencement[c.rule], c.chunk_id
    assert by_id["dpdp_rules_2025:r8:ill"].in_force_date == EIGHTEEN_MONTHS
    assert by_id["dpdp_rules_2025:preamble"].in_force_date == PUBLISHED


@pytest.mark.parametrize(
    ("schedule_chunk", "expected"),
    [
        ("dpdp_rules_2025:sch1:A:1", ONE_YEAR),  # [See rule 4]
        ("dpdp_rules_2025:sch2:a", EIGHTEEN_MONTHS),  # [See rules 5(1) and 16]
        ("dpdp_rules_2025:sch3:1", EIGHTEEN_MONTHS),  # [See rule 8(1)]
        ("dpdp_rules_2025:sch5:1", PUBLISHED),  # [See rule 18]
        ("dpdp_rules_2025:sch6:1", PUBLISHED),  # [See rule 21(2)]
        ("dpdp_rules_2025:sch7:1", EIGHTEEN_MONTHS),  # [See rule 23(1) and 8(3)]
    ],
)
def test_schedules_follow_their_rules(by_id, schedule_chunk: str, expected: date) -> None:
    assert by_id[schedule_chunk].in_force_date == expected


ACT_EXPECTED = [
    ("1", "2", None, PUBLISHED),
    ("2", None, "a", PUBLISHED),
    ("18", "1", None, PUBLISHED),
    ("26", None, None, PUBLISHED),
    ("35", None, None, PUBLISHED),
    ("38", "1", None, PUBLISHED),
    ("43", "2", None, PUBLISHED),
    ("44", "1", None, PUBLISHED),
    ("44", "3", None, PUBLISHED),
    ("6", "9", None, ONE_YEAR),
    ("27", "1", "d", ONE_YEAR),
    ("3", None, None, EIGHTEEN_MONTHS),
    ("5", "1", None, EIGHTEEN_MONTHS),
    ("6", "8", None, EIGHTEEN_MONTHS),
    ("6", "10", None, EIGHTEEN_MONTHS),
    ("10", "2", None, EIGHTEEN_MONTHS),
    ("17", "5", None, EIGHTEEN_MONTHS),
    ("27", "1", "a", EIGHTEEN_MONTHS),
    ("27", "1", "e", EIGHTEEN_MONTHS),
    ("27", "2", None, EIGHTEEN_MONTHS),
    ("34", None, None, EIGHTEEN_MONTHS),
    (
        "36",
        None,
        None,
        EIGHTEEN_MONTHS,
    ),  # 843 lists 36 and 37 at 18 months (India Code footnote differs)
    ("37", "1", None, EIGHTEEN_MONTHS),
    ("44", "2", None, EIGHTEEN_MONTHS),
]


@pytest.mark.parametrize(("section", "sub", "clause", "expected"), ACT_EXPECTED)
def test_act_mapping(ingest, section, sub, clause, expected) -> None:
    assert ingest.act_commencement.date_for(section, sub, clause) == expected


def test_act_chunks_carry_dates(ingest, by_id) -> None:
    for section, sub, clause, expected in ACT_EXPECTED:
        cid = (
            f"dpdp_act_2023:s{section}"
            + (f"({sub})" if sub else "")
            + (f"({clause})" if clause else "")
        )
        assert by_id[cid].in_force_date == expected, cid
    # Mixed dates inside 27(1) force a split into clause chunks.
    assert "dpdp_act_2023:s27(1)" not in by_id
    # The penalty Schedule is applied through section 33.
    assert by_id["dpdp_act_2023:schedule:1"].in_force_date == EIGHTEEN_MONTHS
    # Section 1(1) is not listed in G.S.R. 843(E); it takes the date of assent.
    assert by_id["dpdp_act_2023:s1(1)"].in_force_date == date(2023, 8, 11)


def test_every_chunk_has_a_date(ingest) -> None:
    assert all(c.in_force_date is not None for c in ingest.chunks)
