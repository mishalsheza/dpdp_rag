"""Retriever behaviour on the tiny fixture: Qdrant (in-memory), BM25, hybrid, filters,
reranking and cross-reference expansion. Uses the offline hash embedder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from dpdp_rag.config import merge
from dpdp_rag.retrieval import Filters, Retriever, retrieve
from dpdp_rag.retrieval import retriever as retriever_module
from dpdp_rag.retrieval.filters import date_int
from dpdp_rag.retrieval.qdrant_index import point_id

R8_3 = "dpdp_rules_2025:r8(3)"
CONSENT_Q = "verifiable consent of the parent before processing personal data of a child"


def ids(results) -> list[str]:
    return [r.chunk.chunk_id for r in results]


@pytest.fixture
def retriever(retrieval_config: dict[str, Any]) -> Retriever:
    return Retriever(retrieval_config)


def test_index_upserts_every_chunk_with_metadata_payload(retriever: Retriever) -> None:
    index = retriever.index  # auto_index builds the in-memory collection
    client, name = index.client, index.collection
    assert client.count(name, exact=True).count == len(retriever.store.chunks) == 15
    point = client.retrieve(name, [point_id("dpdp_rules_2025:r12(1)")], with_payload=True)[0]
    assert point.payload["rule"] == "12"
    assert point.payload["doc_type"] == "rules"
    assert point.payload["refers_to"] == ["section 9"]
    assert point.payload["in_force_date"] == "2027-05-13"
    assert point.payload["in_force_int"] == date_int(date(2027, 5, 13))
    assert point.payload["text"].startswith("(1) The provisions of sub-sections (1) and (3)")
    # Re-indexing overwrites by stable id instead of duplicating.
    index.build(retriever.store.chunks)
    assert client.count(name, exact=True).count == 15


def test_dense_finds_exact_text(retriever: Retriever) -> None:
    text = retriever.store.by_id[R8_3].text
    results = retriever.retrieve(text, 3, mode="dense")
    assert ids(results)[0] == R8_3
    assert set(results[0].scores) == {"dense"}


def test_bm25(retriever: Retriever) -> None:
    results = retriever.retrieve(CONSENT_Q, 3, mode="bm25")
    assert ids(results)[0] in {"dpdp_act_2023:s9(1)", "dpdp_rules_2025:r10(1)"}
    assert all(r.score > 0 and set(r.scores) == {"bm25"} for r in results)
    assert retriever.retrieve("the of and", 3, mode="bm25") == []  # stopwords only


def test_hybrid_fuses_both_rankings(retriever: Retriever) -> None:
    results = retriever.retrieve(CONSENT_Q, 4, mode="hybrid")
    assert len(results) == 4
    top = results[0]
    assert {"dense", "bm25", "rrf"} <= set(top.scores)
    assert top.score == top.scores["rrf"]
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert {"dpdp_act_2023:s9(1)", "dpdp_rules_2025:r10(1)"} <= set(ids(results))


def test_k_and_mode_validation(retriever: Retriever) -> None:
    assert len(retriever.retrieve(CONSENT_Q, 2)) == 2
    with pytest.raises(ValueError, match="mode"):
        retriever.retrieve(CONSENT_Q, 2, mode="sparse")


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize(
    ("filters", "check"),
    [
        (Filters(doc_type=["rules"]), lambda c: c.doc_type == "rules"),
        (Filters(rule=["12"]), lambda c: c.rule == "12"),
        (Filters(section=["9"]), lambda c: c.section == "9"),
        (
            Filters(doc_type=["act", "notification"]),
            lambda c: c.doc_type in {"act", "notification"},
        ),
        (Filters(in_force_on=date(2026, 12, 1)), lambda c: c.in_force_date <= date(2026, 12, 1)),
    ],
)
def test_filters_apply_in_every_mode(retriever: Retriever, mode, filters, check) -> None:
    results = retriever.retrieve(
        CONSENT_Q + " data breach retention", 15, mode=mode, filters=filters
    )
    assert results, "expected at least one match"
    assert all(check(r.chunk) for r in results)


def test_in_force_filter_excludes_future_provisions(retriever: Retriever) -> None:
    results = retriever.retrieve(
        "Consent Manager registration Board",
        15,
        mode="dense",
        filters=Filters(in_force_on=date(2026, 12, 1)),
    )
    got = set(ids(results))
    assert "dpdp_rules_2025:r4(1)" in got  # 2026-11-13
    assert "dpdp_act_2023:s6(9)" in got  # 2026-11-13
    assert not any(c.endswith(("s9(1)", "r12(1)", "r8(3)")) for c in got)  # 2027-05-13


def test_in_force_only_from_config(retrieval_config: dict[str, Any]) -> None:
    cfg = merge(retrieval_config, {"filters": {"in_force_only": True, "today": "2025-12-31"}})
    results = Retriever(cfg).retrieve("notification corrigendum Department", 15, mode="bm25")
    assert results and all(r.chunk.in_force_date <= date(2025, 12, 31) for r in results)


def test_cross_reference_expansion_rule_to_section(retriever: Retriever) -> None:
    results = retriever.retrieve(
        "exemption from section 9 for classes of Data Fiduciaries "
        "specified in Part A of Fourth Schedule",
        1,
        filters=Filters(rule=["12"]),
        cross_refs=True,
    )
    assert ids(results) == ["dpdp_rules_2025:r12(1)", "dpdp_act_2023:s9(1)", "dpdp_act_2023:s9(3)"]
    parent, *expanded = results
    for r in expanded:
        assert r.expanded_from == "dpdp_rules_2025:r12(1)"
        assert r.score == pytest.approx(parent.score * 0.5)
    assert [r.via for r in expanded] == ["section 9(1)", "section 9(3)"]


def test_cross_refs_disabled(retriever: Retriever) -> None:
    results = retriever.retrieve(
        "exemption section 9", 1, filters=Filters(rule=["12"]), cross_refs=False
    )
    assert ids(results) == ["dpdp_rules_2025:r12(1)"]


def test_cross_refs_respect_date_filter_and_budget(retrieval_config: dict[str, Any]) -> None:
    query = "one year from the date of publication sub-section (9) of section 6"
    only_843 = Filters(doc_type=["notification"], in_force_on=date(2026, 1, 1))
    r = Retriever(retrieval_config)
    # G.S.R. 843(E)(b) is in force, but s. 6(9) and s. 27(1)(d) only from 2026-11-13.
    assert ids(r.retrieve(query, 1, filters=only_843, cross_refs=True)) == ["gsr_843e:b"]
    later = Filters(doc_type=["notification"], in_force_on=date(2026, 12, 1))
    assert ids(r.retrieve(query, 1, filters=later, cross_refs=True)) == [
        "gsr_843e:b",
        "dpdp_act_2023:s27(1)(d)",
        "dpdp_act_2023:s6(9)",
    ]
    capped = Retriever(merge(retrieval_config, {"cross_refs": {"max_per_chunk": 1}}))
    assert len(capped.retrieve(query, 1, filters=later, cross_refs=True)) == 2


def test_cross_refs_do_not_duplicate_retrieved_chunks(retriever: Retriever) -> None:
    results = retriever.retrieve(CONSENT_Q + " Fourth Schedule exemption", 15, cross_refs=True)
    assert len(ids(results)) == len(set(ids(results)))


class ReverseReranker:
    """Scores documents so that the original order is reversed."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        return [float(i) for i in range(len(documents))]


def test_reranker_reorders_top_n(retrieval_config: dict[str, Any]) -> None:
    cfg = merge(retrieval_config, {"reranker": {"top_n": 3}})
    plain = ids(Retriever(cfg).retrieve(CONSENT_Q, 3, mode="bm25"))
    fake = ReverseReranker()
    reranked = Retriever(cfg, reranker=fake).retrieve(CONSENT_Q, 3, mode="bm25")
    assert ids(reranked) == plain[::-1]
    assert reranked[0].scores["rerank"] == 2.0 and "bm25" in reranked[0].scores
    assert fake.calls == [(CONSENT_Q, 3)]
    # rerank=False skips a configured reranker.
    no = Retriever(cfg, reranker=ReverseReranker()).retrieve(
        CONSENT_Q, 3, mode="bm25", rerank=False
    )
    assert ids(no) == plain


def test_missing_collection_without_auto_index(retrieval_config: dict[str, Any]) -> None:
    cfg = merge(retrieval_config, {"qdrant": {"auto_index": False}})
    with pytest.raises(RuntimeError, match="dpdp-index"):
        Retriever(cfg).retrieve(CONSENT_Q, 3, mode="dense")
    # BM25-only retrieval never touches Qdrant.
    assert Retriever(cfg).retrieve(CONSENT_Q, 3, mode="bm25")


def test_module_level_retrieve(monkeypatch, retriever: Retriever) -> None:
    monkeypatch.setattr(retriever_module, "get_retriever", lambda: retriever)
    results = retrieve(CONSENT_Q, 2)
    assert len(results) == 2
    payload = results[0].to_dict()
    assert payload["chunk_id"] == payload["chunk"]["chunk_id"]
    assert isinstance(payload["score"], float)
    json.dumps(payload)  # serialisable
