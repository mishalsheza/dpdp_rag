"""Chunk boundaries and metadata for the Rules and the notifications."""

from __future__ import annotations

from collections import Counter


def _rules(ingest):
    return [c for c in ingest.chunks if c.doc_id == "dpdp_rules_2025"]


def test_every_rule_present(ingest) -> None:
    assert {c.rule for c in _rules(ingest) if c.rule} == {str(n) for n in range(1, 24)}


def test_sub_rule_boundaries(by_id) -> None:
    assert [by_id[f"dpdp_rules_2025:r1({n})"].text[:3] for n in range(1, 5)] == [
        "(1)",
        "(2)",
        "(3)",
        "(4)",
    ]
    r4_5 = by_id["dpdp_rules_2025:r4(5)"]
    assert r4_5.text.startswith("(5) The Board may") and "(6)" not in r4_5.text
    # Heading on one page, sub-rule (1) on the next.
    assert by_id["dpdp_rules_2025:r19(1)"].text.startswith("(1) The Chairperson shall fix")
    assert "dpdp_rules_2025:r19" not in by_id


def test_rule_without_sub_rules(by_id) -> None:
    r9 = by_id["dpdp_rules_2025:r9"]
    assert r9.sub_rule is None and r9.text.startswith(
        "Every Data Fiduciary shall prominently publish"
    )


def test_printed_page_numbers(by_id) -> None:
    assert by_id["dpdp_rules_2025:r1(3)"].page == 24
    assert by_id["dpdp_rules_2025:r13(5)"].page == 29
    assert by_id["dpdp_rules_2025:sch7:3"].page == 41
    assert by_id["dpdp_rules_2025:r3"].page == 24 and by_id["dpdp_rules_2025:r3"].page_end == 25


def test_illustrations_are_chunks_with_their_cases(by_id) -> None:
    ill8 = by_id["dpdp_rules_2025:r8:ill"]
    assert ill8.illustration and ill8.rule == "8"
    assert ill8.cases == ["Case 1", "Case 2"]
    assert "Case 1" not in by_id["dpdp_rules_2025:r8(3)"].text
    ill10 = by_id["dpdp_rules_2025:r10:ill"]
    assert ill10.cases == ["Case 1", "Case 2", "Case 3", "Case 4"]
    assert ill10.text.startswith("Illustration. C is a child")
    sched = by_id["dpdp_rules_2025:sch1:B:1:ill"]
    assert sched.schedule == "First Schedule" and sched.part == "B" and sched.item == "1"
    assert sched.cases == ["Case 1", "Case 2"]
    assert by_id["dpdp_rules_2025:sch1:B:2"].text.startswith("2. The Consent Manager shall ensure")


def test_schedule_rows(ingest) -> None:
    rows = Counter(
        (c.schedule, c.part)
        for c in _rules(ingest)
        if c.schedule and not c.illustration and not (c.item or "").startswith("Note")
    )
    assert rows[("First Schedule", "A")] == 9
    assert rows[("First Schedule", "B")] == 13
    assert rows[("Second Schedule", None)] == 8
    assert rows[("Third Schedule", None)] == 3
    assert rows[("Fourth Schedule", "A")] == 5
    assert rows[("Fourth Schedule", "B")] == 6
    assert rows[("Fifth Schedule", None)] == 9
    assert rows[("Sixth Schedule", None)] == 7
    assert rows[("Seventh Schedule", None)] == 3


def test_table_rows_merge_page_continuations(by_id) -> None:
    row = by_id["dpdp_rules_2025:sch3:2"]  # continues from page 35 onto 36
    assert row.page == 35 and row.page_end == 36
    assert "may be used to get money, goods or services" in row.text
    assert row.text.startswith(
        "2. Class of Data Fiduciaries: Data Fiduciary who is an online gaming"
    )
    assert "Fiduciary, and may be used" not in by_id["dpdp_rules_2025:sch3:3"].text.split("|")[0]


def test_refers_to(by_id) -> None:
    assert by_id["dpdp_rules_2025:r12(1)"].refers_to == ["section 9"]
    assert by_id["dpdp_rules_2025:r12(1)"].refers_to_provisions == ["section 9(1)", "section 9(3)"]
    assert by_id["dpdp_rules_2025:r19(9)"].refers_to == ["section 27"]
    assert by_id["dpdp_rules_2025:sch7:1"].refers_to == ["section 17"]
    # Schedule rows inherit references made in the Schedule heading.
    assert by_id["dpdp_rules_2025:sch4:A:3"].refers_to == ["section 9"]
    # Sections of other Acts are not references to the DPDP Act.
    assert by_id["dpdp_rules_2025:r11(2)"].refers_to == []
    assert by_id["dpdp_rules_2025:preamble"].refers_to == ["section 40"]


def test_notifications_one_chunk_per_clause(ingest) -> None:
    ids = sorted(c.chunk_id for c in ingest.chunks if c.doc_type == "notification")
    assert ids == [
        "gsr_843e:a",
        "gsr_843e:b",
        "gsr_843e:c",
        "gsr_843e:preamble",
        "gsr_844e:1",
        "gsr_844e:2",
        "gsr_845e:1",
    ]


def test_gazette_duplicate_not_indexed(ingest, config) -> None:
    assert "act_2023_gazette.pdf" in {e["file"] for e in config["excluded"]}
    assert all(c.source_file != "act_2023_gazette.pdf" for c in ingest.chunks)
