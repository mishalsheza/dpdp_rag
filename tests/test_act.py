"""Chunk boundaries and metadata for the Act."""

from __future__ import annotations

from collections import Counter


def _act(ingest):
    return [c for c in ingest.chunks if c.doc_type == "act"]


def test_toc_pages_dropped(ingest) -> None:
    for c in _act(ingest):
        assert "ARRANGEMENT OF SECTIONS" not in c.text
        assert "LIST OF ABBREVIATIONS" not in c.text
        assert c.page is not None and c.page >= 5  # printed pages 1-4: cover, abbreviations, TOC


def test_statement_of_objects_not_indexed(ingest) -> None:
    assert all("STATEMENT OF OBJECTS" not in c.text for c in _act(ingest))


def test_every_section_present(ingest) -> None:
    sections = {c.section for c in _act(ingest) if c.section}
    assert sections == {str(n) for n in range(1, 45)}


def test_sub_section_counts(ingest) -> None:
    counts = Counter(c.section for c in _act(ingest) if c.sub_section and not c.clause)
    assert counts["6"] == 10
    assert counts["8"] == 11
    assert counts["28"] == 12
    assert counts["29"] == 10


def test_sub_section_boundaries(by_id) -> None:
    s6_1 = by_id["dpdp_act_2023:s6(1)"]
    assert s6_1.text.startswith("(1) The consent given by the Data Principal shall be free")
    # The Illustration stays with its sub-section; the next sub-section does not leak in.
    assert "telemedicine" in s6_1.text
    assert "(2) Any part of consent" not in s6_1.text
    assert by_id["dpdp_act_2023:s6(2)"].text.startswith("(2) Any part of consent")
    assert by_id["dpdp_act_2023:s6(10)"].text.endswith("the rules made thereunder.")


def test_section_heading_wrapped_over_two_lines(by_id) -> None:
    s21 = by_id["dpdp_act_2023:s21(1)"]
    assert (
        s21.title
        == "Disqualifications for appointment and continuation as Chairperson and Members of Board"
    )
    assert s21.text.startswith("(1) A person shall be disqualified")


def test_section_without_sub_sections_is_one_chunk(by_id) -> None:
    s24 = by_id["dpdp_act_2023:s24"]
    assert s24.sub_section is None and s24.text.startswith("The Board may")


def test_long_section_split_into_clauses(by_id) -> None:
    child = by_id["dpdp_act_2023:s2(f)"]
    assert child.text.startswith("(f) “child” means an individual")
    assert child.lead_in and child.lead_in.startswith("In this Act")
    assert by_id["dpdp_act_2023:s2(zb)"].text.startswith("(zb) “State”")
    # Nested sub-clauses stay inside their clause.
    assert "(ii) a person with disability" in by_id["dpdp_act_2023:s2(j)"].text


def test_chapter_metadata(by_id) -> None:
    assert by_id["dpdp_act_2023:s9(1)"].chapter == "II"
    assert by_id["dpdp_act_2023:s9(1)"].chapter_title == "OBLIGATIONS OF DATA FIDUCIARY"
    assert by_id["dpdp_act_2023:s33(1)"].chapter == "VIII"
    assert by_id["dpdp_act_2023:s44(3)"].chapter == "IX"


def test_penalty_schedule(ingest, by_id) -> None:
    rows = [c for c in ingest.chunks if c.doc_id == "dpdp_act_2023" and c.schedule]
    assert [r.item for r in rows] == [str(n) for n in range(1, 8)]
    row1 = by_id["dpdp_act_2023:schedule:1"]
    assert row1.penalty == "May extend to two hundred and fifty crore rupees"
    assert row1.refers_to == ["section 8"]
    # The penalty is also carried as metadata by the provision it punishes.
    assert by_id["dpdp_act_2023:s8(5)"].penalty == row1.penalty
    assert by_id["dpdp_act_2023:s8(6)"].penalty == "May extend to two hundred crore rupees."
    assert by_id["dpdp_act_2023:s8(4)"].penalty is None
    assert by_id["dpdp_act_2023:s28(1)"].penalty is None  # cited only in the penalty column


def test_refers_to_excludes_other_acts(by_id) -> None:
    assert by_id["dpdp_act_2023:s29(8)"].refers_to == []  # sections of the TRAI Act
    assert by_id["dpdp_act_2023:s25"].refers_to == []  # section 21 of the IPC
    assert by_id["dpdp_act_2023:s5(1)"].refers_to == ["section 6", "section 13"]
