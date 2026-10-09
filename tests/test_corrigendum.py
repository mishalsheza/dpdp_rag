"""Parsing of G.S.R. 892(E) and application of its patches."""

from __future__ import annotations

import pytest

from dpdp_rag.ingest.corrigendum import Patch, apply_patches, parse_patches
from dpdp_rag.ingest.models import Chunk, LineSpan

CORRIGENDUM_TEXT = (
    '(i) in page 24,– (a) line 22, for “of this Gazette”, read “in the Official Gazette"; '
    '(b) line 24, for "of this Gazette", read "in the Official Gazette"; '
    "(ii) in page 29, line 44, for “Department”, read “Departments”; "
    "(v) in page 38,– (a) line 2, for “.”, read “;” and "
    "(b) lines 1 to 15, for “(a) to (f)”, read “(a) to (g)”."
)


def test_parse_patches_structure() -> None:
    patches = parse_patches(CORRIGENDUM_TEXT)
    got = [(p.item, p.page, p.line, p.line_end, p.old_text, p.new_text, p.kind) for p in patches]
    assert got == [
        ("(i)(a)", 24, 22, None, "of this Gazette", "in the Official Gazette", "replace"),
        ("(i)(b)", 24, 24, None, "of this Gazette", "in the Official Gazette", "replace"),
        ("(ii)", 29, 44, None, "Department", "Departments", "replace"),
        ("(v)(a)", 38, 2, None, ".", ";", "replace"),
        ("(v)(b)", 38, 1, 15, "(a) to (f)", "(a) to (g)", "reletter"),
    ]


def test_real_corrigendum_parsed(ingest) -> None:
    patches = ingest.patches
    assert len(patches) == 8
    assert {p.page for p in patches} == {24, 29, 32, 34, 38}
    assert all(p.target_rule_or_schedule and p.target_chunk_ids for p in patches)
    targets = {p.item: p.target_rule_or_schedule for p in patches}
    assert targets == {
        "(i)(a)": "rule 1(3)",
        "(i)(b)": "rule 1(4)",
        "(ii)": "rule 13(5)",
        "(iii)": "rule 23(1)",
        "(iv)(a)": "First Schedule, Part B, item 11",
        "(iv)(b)": "First Schedule, Note (d)",
        "(v)(a)": "Fourth Schedule, Note (a)",
        "(v)(b)": "Fourth Schedule, Note",
    }


@pytest.mark.parametrize(
    ("chunk_id", "old", "new"),
    [
        (
            "dpdp_rules_2025:r1(3)",
            "publication of this Gazette.",
            "publication in the Official Gazette.",
        ),
        (
            "dpdp_rules_2025:r1(4)",
            "publication of this Gazette.",
            "publication in the Official Gazette.",
        ),
        ("dpdp_rules_2025:r13(5)", "or Department of the Central", "or Departments of the Central"),
        ("dpdp_rules_2025:r23(1)", "as may be given in such.", "as may be given in such order."),
        ("dpdp_rules_2025:sch1:B:11", "(c) everybody corporate", "(c) every body corporate"),
        ("dpdp_rules_2025:sch1:note4", "(18 or 2013).", "(18 of 2013)."),
        ("dpdp_rules_2025:sch4:note1", "(35 of 2019).", "(35 of 2019);"),
    ],
)
def test_patch_applied_with_original_kept(by_id, chunk_id: str, old: str, new: str) -> None:
    chunk = by_id[chunk_id]
    assert chunk.corrected_by == "G.S.R. 892(E)"
    assert old in chunk.text_original and old not in chunk.text_corrected
    assert new in chunk.text_corrected
    assert chunk.text == chunk.text_corrected


def test_fourth_schedule_note_relettered(by_id) -> None:
    notes = [by_id[f"dpdp_rules_2025:sch4:note{n}"] for n in range(1, 8)]
    assert [n.text_original[:3] for n in notes] == ["(a)", "(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    assert [n.text[:3] for n in notes] == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
    assert [n.item for n in notes] == [f"Note ({x})" for x in "abcdefg"]
    assert notes[1].text.startswith("(b) “allied healthcare professional”")


def test_unrelated_text_untouched(by_id, ingest) -> None:
    # "Department of Legal Affairs" (rule 17, page 30) is not on page 29.
    assert "Department of Legal Affairs" in by_id["dpdp_rules_2025:r17(1)"].text
    assert by_id["dpdp_rules_2025:r17(1)"].corrected_by is None
    corrected = {c.chunk_id for c in ingest.chunks if c.corrected_by}
    assert corrected == {
        "dpdp_rules_2025:r1(3)",
        "dpdp_rules_2025:r1(4)",
        "dpdp_rules_2025:r13(5)",
        "dpdp_rules_2025:r23(1)",
        "dpdp_rules_2025:sch1:B:11",
        "dpdp_rules_2025:sch1:note4",
        *(f"dpdp_rules_2025:sch4:note{n}" for n in range(1, 8)),
    }
    assert all(c.text_original is None for c in ingest.chunks if not c.corrected_by)


def test_corrigendum_chunks_point_at_targets(by_id) -> None:
    c = by_id["gsr_892e:(ii)"]
    assert c.doc_type == "corrigendum" and c.rule == "13" and c.sub_rule == "5"
    assert str(c.in_force_date) == "2025-12-10"
    # Version links used by the corrections boost: every corrigendum patch names the chunk
    # it corrected, and that chunk carries corrected_by.
    assert c.corrects == ["dpdp_rules_2025:r13(5)"]
    assert by_id["gsr_892e:(iv)(a)"].corrects == ["dpdp_rules_2025:sch1:B:11"]
    for chunk in by_id.values():
        for target in chunk.corrects:
            assert by_id[target].corrected_by == chunk.gsr_no


def test_schedule_rows_carry_their_see_rules(by_id) -> None:
    assert by_id["dpdp_rules_2025:sch3:1"].see_rules == ["8"]
    assert by_id["dpdp_rules_2025:sch1:A:1"].see_rules == ["4"]
    assert by_id["dpdp_rules_2025:r8(1)"].see_rules == []


def _chunk(cid: str, text: str, page: int, lines: list[int]) -> Chunk:
    """A chunk whose i-th line starts at the i-th sentence of `text`."""
    chunk = Chunk(chunk_id=cid, doc_id="t", doc_type="rules", text=text, page=page, page_end=page)
    offsets, pos = [], 0
    for _ in lines:
        offsets.append(pos)
        pos = text.find(". ", pos) + 2
    chunk.set_lines([LineSpan(page, ln, off) for ln, off in zip(lines, offsets, strict=True)])
    return chunk


def test_match_by_old_text_not_line_number() -> None:
    chunk = _chunk("c", "Alpha beta. Gamma everybody delta.", 34, [10, 11])
    patch = Patch(
        item="(x)", page=34, line=1, old_text="everybody", new_text="every body", source_text=""
    )
    apply_patches([patch], [chunk], "G")
    assert chunk.text == "Alpha beta. Gamma every body delta."
    assert patch.match_strategy == "old_text"


def test_line_number_breaks_ties_between_identical_matches() -> None:
    a = _chunk("a", "Rule four of this Gazette. Next.", 24, [22, 23])
    b = _chunk("b", "Rule five of this Gazette. Next.", 24, [24, 25])
    patch = Patch(
        item="(x)",
        page=24,
        line=24,
        old_text="of this Gazette",
        new_text="in the Official Gazette",
        source_text="",
    )
    apply_patches([patch], [a, b], "G")
    assert b.text.startswith("Rule five in the Official Gazette")
    assert a.corrected_by is None
    assert patch.match_strategy == "old_text+line_tiebreak"


def test_old_text_on_other_page_is_not_matched() -> None:
    chunk = _chunk("c", "Department of Legal Affairs.", 30, [1])
    patch = Patch(
        item="(x)", page=29, line=44, old_text="Department", new_text="Departments", source_text=""
    )
    with pytest.raises(ValueError, match="not found"):
        apply_patches([patch], [chunk], "G")


def test_word_boundaries_respected() -> None:
    chunk = _chunk("c", "the Departments and the Department.", 29, [44])
    patch = Patch(
        item="(x)", page=29, line=44, old_text="Department", new_text="Departments", source_text=""
    )
    apply_patches([patch], [chunk], "G")
    assert chunk.text == "the Departments and the Departments."
