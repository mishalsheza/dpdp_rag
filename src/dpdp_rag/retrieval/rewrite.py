"""Query rewriting for retrieval: restate a plain-language question in the Act's
terminology and split a multi-issue situation into sub-issues (one LLM call).

Strategies (`query_rewrite.strategy`), each fused with reciprocal rank fusion:

- off             the original query only
- replace         the statutory restatement only
- fuse            original + restatement
- decompose       original + each sub-issue
- fuse_decompose  original + restatement + each sub-issue
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any

from pydantic import BaseModel, ValidationError

from dpdp_rag.config import resolve
from dpdp_rag.generation.context import untrusted
from dpdp_rag.generation.llm import LLMClient, LLMError

log = logging.getLogger(__name__)

STRATEGIES = ("off", "replace", "fuse", "decompose", "fuse_decompose")


class _Plan(BaseModel):
    statutory_query: str
    sub_issues: list[str]


@dataclass(frozen=True)
class QueryPlan:
    statutory_query: str
    sub_issues: list[str] = field(default_factory=list)

    def queries(
        self, original: str, strategy: str, sub_issue_weight: float = 1.0
    ) -> list[tuple[str, float]]:
        """(query, RRF weight) pairs, original first (it breaks RRF ties). The sub-issues
        share `sub_issue_weight` between them, so several of them cannot outvote the
        original question."""
        if strategy not in STRATEGIES:
            raise ValueError(f"query_rewrite.strategy must be one of {STRATEGIES}")
        restated = [self.statutory_query] if self.statutory_query.strip() else []
        subs = list(dict.fromkeys(q for q in self.sub_issues if q.strip()))
        sub_w = [(q, sub_issue_weight / len(subs)) for q in subs]
        out = {
            "off": [(original, 1.0)],
            "replace": [(q, 1.0) for q in restated] or [(original, 1.0)],
            "fuse": [(original, 1.0), *((q, 1.0) for q in restated)],
            "decompose": [(original, 1.0), *sub_w],
            "fuse_decompose": [(original, 1.0), *((q, 1.0) for q in restated), *sub_w],
        }[strategy]
        seen: set[str] = set()
        return [(q, w) for q, w in out if not (q in seen or seen.add(q))]


class QueryRewriter:
    def __init__(self, cfg: dict[str, Any], llm: LLMClient) -> None:
        self.cfg = cfg
        self.llm = llm
        files = {n: resolve(p).read_text(encoding="utf-8") for n, p in cfg["prompt_files"].items()}
        self.system = Template(files["system"]).substitute(
            max_sub_issues=int(cfg["max_sub_issues"])
        )
        self.user = Template(files["user"])
        self.schema = json.loads(files["schema"])

    def plan(self, query: str) -> QueryPlan:
        """Raises LLMError if the model call fails or its output is unusable."""
        result = self.llm.generate(
            self.system, self.user.substitute(question=untrusted(query)), self.schema
        )
        if result.stop_reason != "end_turn":
            raise LLMError(f"query rewrite: stop_reason {result.stop_reason!r}")
        try:
            plan = _Plan.model_validate_json(result.text)
        except ValidationError as exc:
            raise LLMError(f"query rewrite: output does not match schema: {exc}") from exc
        subs = plan.sub_issues[: int(self.cfg["max_sub_issues"])]
        return QueryPlan(plan.statutory_query, subs)


def make_rewriter(cfg: dict[str, Any]) -> QueryRewriter | None:
    """The configured rewriter, or None when the strategy is "off"."""
    if (cfg or {}).get("strategy", "off") == "off":
        return None
    from dpdp_rag.generation.llm_cache import make_llm  # imports the Groq SDK

    return QueryRewriter(cfg, make_llm(cfg["llm"], cfg.get("cache")))
