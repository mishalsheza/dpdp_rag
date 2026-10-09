"""LLM-as-judge for faithfulness, relevance and refusal-correctness (rubrics in prompts/)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from pydantic import BaseModel, ValidationError

from dpdp_rag.config import resolve
from dpdp_rag.generation.context import untrusted
from dpdp_rag.generation.cost import Usage, cost_usd
from dpdp_rag.generation.llm import LLMClient, LLMError
from dpdp_rag.observability.tracing import NOOP_TRACE, Trace


class _Score(BaseModel):
    reasoning: str
    score: int


class _Verdict(BaseModel):
    reasoning: str
    correct: bool


@dataclass(frozen=True)
class Judgement:
    metric: str
    value: float  # 1-5 score, or 1.0 / 0.0 for refusal correctness
    reasoning: str
    usage: Usage
    cost_usd: float


@dataclass(frozen=True)
class JudgeInput:
    question: str
    as_of_date: str
    answerable: bool
    reference_answer: str
    context: str  # rendered <document> blocks the system answered from (already escaped)
    answer: str
    refused: bool
    refusal_reason: str


def judge_prompt_paths(judge_cfg: dict[str, Any]) -> dict[str, Path]:
    return {name: resolve(p) for name, p in judge_cfg["prompt_files"].items()}


class Judge:
    METRICS = ("faithfulness", "relevance", "refusal")

    def __init__(self, judge_cfg: dict[str, Any], llm: LLMClient) -> None:
        self.cfg = judge_cfg
        self.llm = llm
        files = {n: p.read_text(encoding="utf-8") for n, p in judge_prompt_paths(judge_cfg).items()}
        self.rubrics = {m: files[m].strip() for m in self.METRICS}
        self.input = Template(files["input"])
        self.score_schema = json.loads(files["score_schema"])
        self.refusal_schema = json.loads(files["refusal_schema"])

    def _user(self, x: JudgeInput) -> str:
        return self.input.substitute(
            question=untrusted(x.question),
            as_of_date=x.as_of_date,
            answerable=str(x.answerable).lower(),
            reference_answer=untrusted(x.reference_answer),
            context=x.context,
            answer=untrusted(x.answer),
            refused=str(x.refused).lower(),
            refusal_reason=x.refusal_reason or "none",
        )

    def judge(self, metric: str, x: JudgeInput, trace: Trace = NOOP_TRACE) -> Judgement:
        schema = self.refusal_schema if metric == "refusal" else self.score_schema
        user = self._user(x)
        with trace.step(
            f"judge:{metric}", as_type="generation", input=user, model=self.llm.model
        ) as step:
            result = self.llm.generate(self.rubrics[metric], user, schema)
            cost = cost_usd(result.usage, self.cfg["pricing"])
            step.update(
                output=result.text, model=result.model, usage=result.usage.as_dict(), cost_usd=cost
            )
        if result.stop_reason != "end_turn":
            raise LLMError(f"judge {metric}: stop_reason {result.stop_reason!r}")
        try:
            if metric == "refusal":
                verdict = _Verdict.model_validate_json(result.text)
                value, reasoning = (1.0 if verdict.correct else 0.0), verdict.reasoning
            else:
                scored = _Score.model_validate_json(result.text)
                if not 1 <= scored.score <= 5:
                    raise LLMError(f"judge {metric}: score {scored.score} outside 1-5")
                value, reasoning = float(scored.score), scored.reasoning
        except ValidationError as exc:
            raise LLMError(f"judge {metric}: output does not match schema: {exc}") from exc
        return Judgement(metric, value, reasoning, result.usage, cost)
