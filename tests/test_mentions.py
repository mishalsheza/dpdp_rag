"""Provisions named in the answer text are checked against the cited documents."""

from __future__ import annotations

from datetime import date
from typing import Any

from conftest import TINY_CHUNKS
from dpdp_rag.generation.answer import Answerer
from dpdp_rag.generation.mentions import Mention, find_mentions, supports
from dpdp_rag.retrieval import Retriever
from dpdp_rag.retrieval.store import ChunkStore
from fakes import CHUNK_ID, FakeLLM

STORE = ChunkStore.from_file(TINY_CHUNKS)
QUESTION = "Which children's data obligations does Rule 12 exempt?"
# Labels for fixture chunks, to name an uncited one in a fake answer.
LABELS = {
    "dpdp_act_2023:s9(1)": "Section 9(1)",
    "dpdp_act_2023:s9(3)": "Section 9(3)",
    "dpdp_rules_2025:r10(1)": "Rule 10(1)",
    "dpdp_rules_2025:r12(1)": "Rule 12(1)",
    "dpdp_act_2023:s8(5)": "Section 8(5)",
}


def test_find_mentions_normalises_and_deduplicates() -> None:
    text = "Under Section 9(1) and rule 10 (2), and again section 9(1); Sections 6 apply."
    assert [m.label for m in find_mentions(text)] == [
        "Section 9(1)",
        "Rule 10(2)",
        "Section 6",
    ]
    assert find_mentions("Section 2(t) defines it") == [Mention("section", "2", "t")]
    assert find_mentions("the First Schedule, item 4") == []


def test_supports_own_provision_and_its_parent_only() -> None:
    s9_1 = STORE.by_id["dpdp_act_2023:s9(1)"]
    assert supports(Mention("section", "9", "1"), s9_1)
    assert supports(Mention("section", "9", None), s9_1)  # "Section 9" covers 9(1)
    assert not supports(Mention("section", "9", "3"), s9_1)
    assert not supports(Mention("rule", "9", "1"), s9_1)


def test_supports_provisions_a_chunk_refers_to() -> None:
    # The Fourth Schedule row disapplies Section 9(1) and 9(3), so citing it backs both.
    row = STORE.by_id["dpdp_rules_2025:sch4:A:3"]
    assert supports(Mention("section", "9", "1"), row)
    assert supports(Mention("section", "9", "3"), row)
    assert not supports(Mention("section", "9", "2"), row)


def _answer_naming(extra: str) -> Any:
    """Cite only the first document; name another supplied one and `extra` in the prose."""

    def reply(user: str) -> dict[str, Any]:
        ids = CHUNK_ID.findall(user)
        other = next(LABELS[i] for i in ids[1:] if i in LABELS and i != ids[0])
        return {
            "answer": f"{LABELS.get(ids[0], 'It')} applies, read with {other}. See {extra}.",
            "citations": [{"chunk_id": ids[0], "pinpoint": ""}],
            "refused": False,
            "refusal_reason": "none",
        }

    return reply


def test_uncited_supplied_provision_is_cited_and_unknown_one_flagged(api_config) -> None:
    llm = FakeLLM(_answer_naming("Rule 99(2)"))
    res = Answerer(api_config, Retriever(api_config), llm).answer(QUESTION, date(2027, 6, 1))
    supplied = CHUNK_ID.findall(llm.calls[0]["user"])
    cited = [c.chunk_id for c in res.citations]
    assert cited[0] == supplied[0] and len(cited) == 2  # the named one was added
    assert cited[1] in supplied and cited[1] in LABELS
    assert res.unverified_mentions == ["Rule 99(2)"]


def test_fully_cited_answer_has_no_unverified_mentions(api_config) -> None:
    def reply(user: str) -> dict[str, Any]:
        first = CHUNK_ID.findall(user)[0]
        return {
            "answer": f"{LABELS.get(first, 'The provision')} says so.",
            "citations": [{"chunk_id": first, "pinpoint": ""}],
            "refused": False,
            "refusal_reason": "none",
        }

    answerer = Answerer(api_config, Retriever(api_config), FakeLLM(reply))
    res = answerer.answer(QUESTION, date(2027, 6, 1))
    assert res.unverified_mentions == [] and len(res.citations) == 1
