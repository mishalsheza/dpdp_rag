"""Unit tests for retrieval building blocks: fusion, filters, chunk store, BM25 tokenizer."""

from __future__ import annotations

from datetime import date

from qdrant_client import models as qm

from conftest import TINY_CHUNKS
from dpdp_rag.retrieval.bm25 import Tokenizer
from dpdp_rag.retrieval.filters import Filters, date_int
from dpdp_rag.retrieval.fusion import rrf
from dpdp_rag.retrieval.qdrant_index import point_id
from dpdp_rag.retrieval.store import ChunkStore, render


def test_rrf_scores_and_order() -> None:
    fused = dict(rrf([["a", "b", "c"], ["b", "c", "d"]], k=60))
    assert fused["b"] == 1 / 62 + 1 / 61  # rank 2 in list 1, rank 1 in list 2
    assert fused["d"] == 1 / 63
    assert [cid for cid, _ in rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)] == ["b", "c", "a", "d"]


def test_rrf_single_list_keeps_order() -> None:
    assert [cid for cid, _ in rrf([["x", "y", "z"]], k=60)] == ["x", "y", "z"]


def test_filters_from_config() -> None:
    f = Filters.from_config(
        {"doc_type": "rules", "rule": 12, "in_force_only": True, "today": "2026-12-01"}
    )
    assert f == Filters(doc_type=["rules"], rule=["12"], in_force_on=date(2026, 12, 1))
    assert Filters.from_config({"in_force_only": False}).is_empty()


def test_filters_match_and_qdrant_form() -> None:
    store = ChunkStore.from_file(TINY_CHUNKS)
    r12 = store.by_id["dpdp_rules_2025:r12(1)"]
    assert Filters(doc_type=["rules"], rule=["12"]).matches(r12)
    assert not Filters(section=["9"]).matches(r12)
    assert not Filters(in_force_on=date(2027, 5, 12)).matches(r12)  # in force 2027-05-13
    assert Filters(in_force_on=date(2027, 5, 13)).matches(r12)

    q = Filters(doc_type=["act"], in_force_on=date(2026, 11, 13)).to_qdrant()
    assert isinstance(q, qm.Filter) and len(q.must) == 2
    assert q.must[1].range.lte == 20261113
    assert Filters().to_qdrant() is None
    assert date_int(date(2027, 5, 13)) == 20270513


def test_store_provision_lookup() -> None:
    store = ChunkStore.from_file(TINY_CHUNKS)
    ids = lambda label: [c.chunk_id for c in store.provision(label)]  # noqa: E731
    assert ids("section 9(1)") == ["dpdp_act_2023:s9(1)"]
    assert ids("section 9") == ["dpdp_act_2023:s9(1)", "dpdp_act_2023:s9(3)"]
    assert ids("section 27(1)(d)") == ["dpdp_act_2023:s27(1)(d)"]
    assert ids("section 8(5)(a)") == ["dpdp_act_2023:s8(5)"]  # clause falls back to sub-section
    assert ids("section 40") == []
    assert ids("rule 4") == []


def test_render_template() -> None:
    store = ChunkStore.from_file(TINY_CHUNKS)
    out = render("{title}\n{text}", store.by_id["gsr_892e:(ii)"])
    assert out.startswith("Correction to rule 13(5)\n(ii) in page 29")


def test_point_ids_are_stable_uuids() -> None:
    assert point_id("dpdp_rules_2025:r12(1)") == point_id("dpdp_rules_2025:r12(1)")
    assert point_id("dpdp_rules_2025:r12(1)") != point_id("dpdp_rules_2025:r12(2)")


def test_tokenizer_drops_stopwords() -> None:
    assert Tokenizer(["the", "of"])("The consent of the Parent, 9(1)") == [
        "consent",
        "parent",
        "9",
        "1",
    ]
