"""Metadata boosts: extra rankings built from the query and chunk metadata, fused with the
retrieved candidates by weighted reciprocal rank fusion.

Each signal yields a ranked list of chunk ids; a signal with weight 0 is off.

- citations     provisions the query cites ("Section 8(5)", "rule 1(3)", "item 11 of
                Part B of the First Schedule") resolved to their exact chunks
- definitions   a "what counts as / meaning of" question naming a defined term -> the
                clause defining it (terms parsed from the documents' "“x” means" clauses)
- commencement  a "when does it come into force" question -> the commencement provisions
                of the document it names (Act / Rules)
- corrections   a question about corrections -> each retrieved corrigendum chunk's
                corrected chunk and vice versa (Chunk.corrects / corrected_by)
- links         Schedule rows <-> their parent provisions: "[See rule N]" (Chunk.see_rules)
                and the Act Schedule's refers_to sections, for the top candidates
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from dpdp_rag.ingest.models import Chunk
from dpdp_rag.ingest.refs import _EXTERNAL_OF, _INTERNAL_OF
from dpdp_rag.retrieval.filters import Filters
from dpdp_rag.retrieval.fusion import rrf
from dpdp_rag.retrieval.store import ChunkStore

SIGNALS = ("citations", "definitions", "commencement", "corrections", "links")

_VERBOSE_SECTION = re.compile(
    r"\bsub-?\s?sections?\s+((?:\([a-z0-9]+\)(?:\s*(?:,|and|or)\s*)?)+)\s*of\s+section\s+(\d+[A-Z]?)",
    re.I,
)
_SECTION = re.compile(r"\bsections?\s+(\d+[A-Z]?)((?:\s?\([a-z0-9]+\))*)", re.I)
_RULE = re.compile(r"\brules?\s+(\d+)((?:\s?\([a-z0-9]+\))*)", re.I)
_ORDINAL = r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh)"
_SCHEDULE_ITEM = re.compile(
    rf"\bitem\s+(\d+)(?:\s+of\s+Part\s+([A-Z]))?\s+of\s+the\s+{_ORDINAL}\s+Schedule", re.I
)
_PART_OF_SCHEDULE = re.compile(rf"\bPart\s+([A-Z])\s+of\s+the\s+{_ORDINAL}\s+Schedule", re.I)
_SCHEDULE = re.compile(rf"\b{_ORDINAL}\s+Schedule\b", re.I)
_DEFINED = re.compile(r"^\s*(?:\([a-z0-9]+\)\s*)*[“\"]([^”\"]+)[”\"]\s+(?:means|includes)\b")
_QUOTED = re.compile(r"[“\"']([^”\"']{2,60})[”\"']")


def _parens(s: str) -> list[str]:
    return re.findall(r"\(([a-z0-9]+)\)", s)


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


class MetadataBooster:
    def __init__(self, store: ChunkStore, cfg: dict[str, Any]) -> None:
        self.store = store
        self.cfg = cfg
        self.rules: dict[tuple[str, str | None], list[Chunk]] = defaultdict(list)
        self.schedule_rows: dict[tuple[str, str | None, str | None], list[Chunk]] = defaultdict(
            list
        )
        self.defined: dict[str, list[str]] = defaultdict(list)
        self.corrected_by_chunk: dict[str, list[str]] = defaultdict(list)
        self.schedule_by_section: dict[str, list[str]] = defaultdict(list)
        title_pat = cfg["commencement"]["title_patterns"]
        self.commencement = [
            c for c in store.chunks if c.title and _matches_any(title_pat, c.title)
        ]
        for c in store.chunks:
            if c.doc_type == "rules" and c.rule:
                self.rules[(c.rule, None)].append(c)
                self.rules[(c.rule, c.sub_rule)].append(c)
            if c.schedule and c.doc_type != "corrigendum":
                for key in [(c.schedule, None, None), (c.schedule, c.part, None)]:
                    self.schedule_rows[key].append(c)
                self.schedule_rows[(c.schedule, c.part, c.item)].append(c)
                for label in c.refers_to_provisions:
                    self.schedule_by_section[label].append(c.chunk_id)
            if m := _DEFINED.match(c.text):
                self.defined[m.group(1).lower()].append(c.chunk_id)
            for target in c.corrects:
                self.corrected_by_chunk[target].append(c.chunk_id)

    # -- signals -----------------------------------------------------------------------

    def citations(self, query: str) -> list[str]:
        """Most specific patterns first; a match overlapping an earlier one is skipped, so
        "item 11 of Part B of the First Schedule" does not also cite the whole Schedule."""
        taken: list[tuple[int, int]] = []

        def free(m: re.Match[str]) -> bool:
            if any(m.start() < e and s < m.end() for s, e in taken):
                return False
            taken.append(m.span())
            return True

        out: list[Chunk] = []
        cap = int(self.cfg["citations"]["max_chunks"])

        def add(chunks: list[Chunk]) -> None:
            if len(chunks) <= cap:  # "the Third Schedule" (7 chunks) is too broad to boost
                out.extend(chunks)

        for m in _VERBOSE_SECTION.finditer(query):
            if free(m) and not self._external(query, m):
                add(
                    [
                        c
                        for sub in _parens(m.group(1))
                        for c in self.store.provision(f"section {m.group(2)}({sub})")
                    ]
                )
        for m in _SECTION.finditer(query):
            if free(m) and not self._external(query, m):
                add(self.store.provision(f"section {m[1]}" + m[2].replace(" ", "")))
        for m in _SCHEDULE_ITEM.finditer(query):
            if free(m):
                add(self.schedule_rows.get((f"{m[3].title()} Schedule", m[2], m[1]), []))
        for m in _PART_OF_SCHEDULE.finditer(query):
            if free(m):
                add(self.schedule_rows.get((f"{m[2].title()} Schedule", m[1].upper(), None), []))
        for m in _SCHEDULE.finditer(query):
            if free(m):
                add(self.schedule_rows.get((f"{m[1].title()} Schedule", None, None), []))
        for m in _RULE.finditer(query):
            if free(m):
                parts = _parens(m.group(2))
                add(self.rules.get((m.group(1), parts[0] if parts else None), []))
        return [c.chunk_id for c in out]

    @staticmethod
    def _external(query: str, m: re.Match[str]) -> bool:
        """True for a section of another Act ("section 15 of the ... Act, 2016")."""
        after = query[m.end() : m.end() + 250]
        return bool(_EXTERNAL_OF.match(after)) and not _INTERNAL_OF.match(after)

    def definitions(self, query: str) -> list[str]:
        quoted = [q.lower() for q in _QUOTED.findall(query)]
        terms = [q for q in quoted if q in self.defined]
        if not terms and _matches_any(self.cfg["definitions"]["intent_patterns"], query):
            low = query.lower()
            terms = sorted((t for t in self.defined if t in low), key=len, reverse=True)[:1]
        return [cid for t in terms for cid in self.defined[t]]

    def commencement_ids(self, query: str) -> list[str]:
        ccfg = self.cfg["commencement"]
        if not _matches_any(ccfg["intent_patterns"], query):
            return []
        named = [n for n in ccfg["doc_targets"] if re.search(rf"\b{n}\b", query, re.I)]
        # Only narrow when the query names exactly one document ("the Rules", "the Act").
        allowed = set(ccfg["doc_targets"][named[0]]) if len(named) == 1 else None
        return [c.chunk_id for c in self.commencement if allowed is None or c.doc_type in allowed]

    def corrections(self, query: str, candidates: list[str]) -> list[str]:
        if not _matches_any(self.cfg["corrections"]["intent_patterns"], query):
            return []
        out: list[str] = []  # partners only, so they rise to meet the retrieved chunk
        for cid in candidates:
            out += [*self.store.by_id[cid].corrects, *self.corrected_by_chunk.get(cid, [])]
        return out

    def links(self, candidates: list[str]) -> list[str]:
        out: list[str] = []
        for cid in candidates[: int(self.cfg["links"]["depth"])]:
            c = self.store.by_id[cid]
            if c.schedule and c.doc_type != "corrigendum":
                for rule in c.see_rules:  # sub-rules that name this Schedule
                    out += [
                        r.chunk_id
                        for r in self.rules.get((rule, None), [])
                        if c.schedule.lower() in r.text.lower()
                    ]
                for label in c.refers_to_provisions:  # Act Schedule -> section
                    out += [s.chunk_id for s in self.store.provision(label)]
            if c.doc_type == "act" and c.section:
                label = f"section {c.section}" + (f"({c.sub_section})" if c.sub_section else "")
                out += self.schedule_by_section.get(label, [])
        return out

    # -- fusion ------------------------------------------------------------------------

    def apply(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        filters: Filters,
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """`ranked` fused with every signal whose weight is > 0. Signal chunks must pass
        `filters`; they follow the retrieval order unless the signal keeps its own."""
        weights = weights if weights is not None else self.cfg["weights"]
        ids = [cid for cid, _ in ranked]
        # (builder, keep_order): pair-wise signals keep their own order so a chunk and its
        # partner stay adjacent; the others follow the retrieval order.
        signals = {
            "citations": (lambda: self.citations(query), False),
            "definitions": (lambda: self.definitions(query), False),
            "commencement": (lambda: self.commencement_ids(query), False),
            "corrections": (
                lambda: self.corrections(query, ids[: int(self.cfg["depth"])]),
                True,
            ),
            "links": (lambda: self.links(ids), True),
        }
        lists, ws = [ids], [1.0]
        for name, (build, keep_order) in signals.items():
            w = float(weights.get(name, 0))
            if w <= 0:
                continue
            hits = self._order(build(), ids, filters, keep_order)
            if hits:
                lists.append(hits)
                ws.append(w)
        if len(lists) == 1:
            return ranked
        return rrf(lists, int(self.cfg["rrf_k"]), ws)

    def _order(
        self, hits: Iterable[str], ids: list[str], filters: Filters, keep_order: bool
    ) -> list[str]:
        uniq = [c for c in dict.fromkeys(hits) if filters.matches(self.store.by_id[c])]
        if keep_order:
            return uniq
        pos = {cid: i for i, cid in enumerate(ids)}
        return sorted(uniq, key=lambda c: pos.get(c, len(pos)))
