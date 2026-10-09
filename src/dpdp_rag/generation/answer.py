"""Question answering: retrieve, build the prompt, call the LLM, validate citations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from dpdp_rag.generation.context import in_force, render_context, untrusted
from dpdp_rag.generation.cost import Usage, cost_usd
from dpdp_rag.generation.llm import LLMClient, LLMError
from dpdp_rag.generation.mentions import find_mentions, is_chunk_of, supports
from dpdp_rag.generation.pinpoint import pinpoint
from dpdp_rag.generation.prompts import Prompts
from dpdp_rag.ingest.models import Chunk
from dpdp_rag.observability.tracing import NOOP_TRACE, Trace
from dpdp_rag.retrieval.retriever import Retriever

log = logging.getLogger(__name__)

RefusalReason = Literal["none", "insufficient_context", "legal_advice", "model_refusal"]


class _LLMCitation(BaseModel):
    chunk_id: str
    pinpoint: str


class LLMAnswer(BaseModel):
    """What the model returns; mirrors prompts/answer_schema.json."""

    answer: str
    citations: list[_LLMCitation]
    refused: bool
    refusal_reason: Literal["none", "insufficient_context", "legal_advice"]


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    pinpoint: str
    doc_type: str
    gsr_no: str | None
    page: int | None
    in_force_date: date | None
    in_force: bool
    corrected_by: str | None
    title: str | None = None
    text: str = ""  # the provision as answered from (corrected text where applicable)


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: RefusalReason | None
    as_of_date: date
    model: str
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    retrieved: list[str] = field(default_factory=list)
    # Provisions the answer text names that no cited document backs (shown as a warning).
    unverified_mentions: list[str] = field(default_factory=list)


class Answerer:
    def __init__(
        self,
        config: dict[str, Any],
        retriever: Retriever,
        llm: LLMClient,
        prompts: Prompts | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.llm = llm
        self.prompts = prompts or Prompts.load(config)
        gen = config["generation"]
        self._entry_labels: dict[str, str] = gen["schedule_entry_label"]
        self._doc_labels: dict[str, str] = gen["doc_labels"]

    def _citation(self, chunk: Chunk, as_of: date) -> Citation:
        return Citation(
            chunk_id=chunk.chunk_id,
            pinpoint=pinpoint(chunk, self._entry_labels),
            doc_type=chunk.doc_type,
            gsr_no=chunk.gsr_no,
            page=chunk.page,
            in_force_date=chunk.in_force_date,
            in_force=in_force(chunk, as_of),
            corrected_by=chunk.corrected_by,
            title=chunk.title,
            text=chunk.text,
        )

    def _check_mentions(
        self, text: str, chunks: list[Chunk], citations: list[Citation], as_of: date
    ) -> tuple[list[Citation], list[str]]:
        """Cite supplied chunks the prose names but the model left out; return the names
        no supplied chunk backs."""
        cited_ids = {c.chunk_id for c in citations}
        unverified: list[str] = []
        for mention in find_mentions(text):
            cited = [c for c in chunks if c.chunk_id in cited_ids]
            if any(supports(mention, c) for c in cited):
                continue
            named = [c for c in chunks if c.chunk_id not in cited_ids and is_chunk_of(mention, c)]
            if not named:
                log.warning("Answer names %s, which no supplied document backs", mention.label)
                unverified.append(mention.label)
            elif mention.sub:  # a specific provision: cite it (a bare "Section 9" is too broad)
                cited_ids.update(c.chunk_id for c in named)
                citations = citations + [self._citation(c, as_of) for c in named]
        return citations, unverified

    def _refusal(
        self, as_of: date, retrieved: list[str], reason: RefusalReason, usage: Usage | None = None
    ) -> AnswerResult:
        usage = usage or Usage()
        return AnswerResult(
            answer=self.prompts.insufficient_context,
            citations=[],
            refused=True,
            refusal_reason=reason,
            as_of_date=as_of,
            model=self.llm.model,
            usage=usage,
            cost_usd=cost_usd(usage, self.config["llm"]["pricing"]),
            retrieved=retrieved,
        )

    def answer(self, question: str, as_of: date, trace: Trace | None = None) -> AnswerResult:
        trace = trace or NOOP_TRACE
        with trace.step("retrieve", as_type="retriever", input=question) as step:
            hits = self.retriever.retrieve(question, int(self.config["generation"]["k"]))
            chunks = [h.chunk for h in hits]
            step.update(
                output=[
                    {
                        "chunk_id": h.chunk.chunk_id,
                        "score": h.score,
                        "expanded_from": h.expanded_from,
                    }
                    for h in hits
                ]
            )
        retrieved = [c.chunk_id for c in chunks]
        if not chunks:
            return self._refusal(as_of, retrieved, "insufficient_context")

        user = self.prompts.user.substitute(
            as_of_date=as_of.isoformat(),
            context=render_context(chunks, as_of, self._entry_labels, self._doc_labels),
            question=untrusted(question),
        )
        pricing = self.config["llm"]["pricing"]
        with trace.step("generate", as_type="generation", input=user, model=self.llm.model) as step:
            started = time.perf_counter()
            result = self.llm.generate(self.prompts.system, user, self.prompts.schema)
            cost = cost_usd(result.usage, pricing)
            step.update(
                output=result.text,
                model=result.model,
                usage=result.usage.as_dict(),
                cost_usd=cost,
                metadata={
                    "stop_reason": result.stop_reason,
                    "llm_ms": (time.perf_counter() - started) * 1000,
                },
            )

        if result.stop_reason == "refusal":
            return self._refusal(as_of, retrieved, "model_refusal", result.usage)
        if result.stop_reason == "max_tokens":
            raise LLMError("The model ran out of output tokens before finishing the answer")
        try:
            parsed = LLMAnswer.model_validate_json(result.text)
        except ValidationError as exc:
            raise LLMError(
                f"The model returned output that does not match the schema: {exc}"
            ) from exc

        # Keep only citations to chunks actually supplied; pinpoints come from metadata.
        by_id = {c.chunk_id: c for c in chunks}
        seen: set[str] = set()
        citations: list[Citation] = []
        for cit in parsed.citations:
            chunk = by_id.get(cit.chunk_id)
            if chunk is None:
                log.warning("Dropping citation to unknown chunk %r", cit.chunk_id)
            elif cit.chunk_id not in seen:
                seen.add(cit.chunk_id)
                citations.append(self._citation(chunk, as_of))

        if not parsed.refused and not citations:
            # An answer that cites nothing it was given is not supported by the context.
            return self._refusal(as_of, retrieved, "insufficient_context", result.usage)
        citations, unverified = self._check_mentions(parsed.answer, chunks, citations, as_of)
        return AnswerResult(
            answer=parsed.answer,
            citations=citations,
            refused=parsed.refused,
            refusal_reason=parsed.refusal_reason if parsed.refused else None,
            as_of_date=as_of,
            model=result.model,
            usage=result.usage,
            cost_usd=cost,
            retrieved=retrieved,
            unverified_mentions=unverified,
        )
