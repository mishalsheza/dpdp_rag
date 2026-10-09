"""Metadata boosts: each signal on hand-made chunks, fusion, filters, and the retriever."""

from __future__ import annotations

from typing import Any

import pytest

from dpdp_rag.config import load_config, merge
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.retrieval import Retriever
from dpdp_rag.retrieval.boost import MetadataBooster
from dpdp_rag.retrieval.filters import Filters
from dpdp_rag.retrieval.store import ChunkStore


def _c(cid: str, doc_type: str = "act", text: str = "x", **kw: Any) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=cid.split(":")[0], doc_type=doc_type, text=text, **kw)


CHUNKS = [
    _c("act:s1(2)", section="1", sub_section="2", title="Short title and commencement"),
    _c("act:s2(t)", section="2", clause="t", text="(t) “personal data” means any data about"),
    _c("act:s2(i)", section="2", clause="i", text="(i) “Data Fiduciary” means any person"),
    *[_c(f"act:s8({n})", section="8", sub_section=str(n)) for n in range(1, 8)],
    _c("act:s9(1)", section="9", sub_section="1"),
    _c("act:s9(3)", section="9", sub_section="3"),
    _c("act:s15", section="15"),
    _c(
        "act:schedule:5",
        schedule="The Schedule",
        item="5",
        refers_to_provisions=["section 15"],
    ),
    _c("rules:r1(2)", "rules", rule="1", sub_rule="2", title="Short title and commencement"),
    _c(
        "rules:r1(3)",
        "rules",
        rule="1",
        sub_rule="3",
        title="Short title and commencement",
        corrected_by="G.S.R. 892(E)",
    ),
    _c("rules:r8(1)", "rules", rule="8", sub_rule="1", text="as in the Third Schedule"),
    _c("rules:r8(2)", "rules", rule="8", sub_rule="2", text="other text"),
    *[
        _c(
            f"rules:sch1:B:{n}",
            "rules",
            schedule="First Schedule",
            part="B",
            item=str(n),
            see_rules=["4"],
        )
        for n in range(1, 12)
    ],
    _c("rules:sch3:1", "rules", schedule="Third Schedule", item="1", see_rules=["8"]),
    _c("corr:(i)(a)", "corrigendum", rule="1", sub_rule="3", corrects=["rules:r1(3)"]),
    _c(
        "corr:(iv)(a)",
        "corrigendum",
        schedule="First Schedule",
        part="B",
        item="11",
        corrects=["rules:sch1:B:11"],
    ),
    _c("notif:a", "notification", title="Commencement of provisions"),
]


@pytest.fixture
def booster() -> MetadataBooster:
    return MetadataBooster(ChunkStore(CHUNKS), load_config("default.yaml")["boost"])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What does Section 8(5) require?", ["act:s8(5)"]),
        ("sub-sections (1) and (3) of section 9", ["act:s9(1)", "act:s9(3)"]),
        ("Which parts of section 9 apply?", ["act:s9(1)", "act:s9(3)"]),
        ("What does rule 1(3) say?", ["rules:r1(3)"]),  # not the corrigendum chunk
        ("item 11 of Part B of the First Schedule", ["rules:sch1:B:11"]),  # not all of Part B
        ("Is section 8 relevant?", []),  # 7 chunks: broader than citations.max_chunks
        ("section 15 of the Rights of Persons with Disabilities Act, 2016", []),
        ("How does consent work?", []),
    ],
)
def test_citations(booster, query, expected) -> None:
    assert list(dict.fromkeys(booster.citations(query))) == expected


def test_definitions_need_a_quoted_term_or_a_meaning_question(booster) -> None:
    assert booster.definitions('What counts as "personal data"?') == ["act:s2(t)"]
    assert booster.definitions("What is the meaning of Data Fiduciary?") == ["act:s2(i)"]
    assert booster.definitions("Must a Data Fiduciary protect personal data?") == []


def test_commencement_targets_the_named_document(booster) -> None:
    assert booster.commencement_ids("When do the Rules come into force?") == [
        "rules:r1(2)",
        "rules:r1(3)",
    ]
    assert booster.commencement_ids("When does the Act commence?") == ["act:s1(2)", "notif:a"]
    assert booster.commencement_ids("Is rule 7 in force today?") == []  # not a commencement Q


def test_corrections_pair_both_ways_only_for_correction_questions(booster) -> None:
    q = "What did the corrigendum correct?"
    assert booster.corrections(q, ["corr:(iv)(a)"]) == ["rules:sch1:B:11"]
    assert booster.corrections(q, ["rules:r1(3)"]) == ["corr:(i)(a)"]
    assert booster.corrections("What does rule 1(3) say?", ["rules:r1(3)"]) == []


def test_links_schedule_rows_and_parent_provisions(booster) -> None:
    assert booster.links(["rules:sch3:1"]) == ["rules:r8(1)"]  # only the sub-rule naming it
    assert booster.links(["act:schedule:5"]) == ["act:s15"]
    assert booster.links(["act:s15"]) == ["act:schedule:5"]


def test_apply_fuses_with_weights_and_respects_filters(booster) -> None:
    ranked = [(f"rules:sch1:B:{n}", 1.0) for n in range(1, 12)]
    q = "item 11 of Part B of the First Schedule"
    off = booster.apply(q, ranked, Filters(), dict.fromkeys(booster.cfg["weights"], 0))
    assert off is ranked
    on = booster.apply(q, ranked, Filters(), {"citations": 1})
    assert on[0][0] == "rules:sch1:B:11"
    acts_only = Filters(doc_type=["act"])
    assert booster.apply(q, ranked, acts_only, {"citations": 1}) == ranked  # filtered out


def test_retriever_applies_boosts(retrieval_config) -> None:
    question = "Which children's data obligations does Rule 12 exempt?"
    plain = Retriever(retrieval_config).retrieve(question, 5, mode="bm25")
    cfg = merge(retrieval_config, {"boost": {"weights": {"citations": 1}}})
    boosted = Retriever(cfg).retrieve(question, 5, mode="bm25")
    assert boosted[0].chunk.rule == "12" and "boost" in boosted[0].scores
    assert "boost" not in plain[0].scores
