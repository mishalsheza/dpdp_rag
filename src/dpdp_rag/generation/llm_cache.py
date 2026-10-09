"""Caching and spend-capping wrapper for LLM calls (used by the eval suite and CI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dpdp_rag.generation.cost import Usage, cost_usd
from dpdp_rag.generation.llm import LLMClient, LLMResult, make_answer_llm
from dpdp_rag.kvcache import SqliteKV, digest


class BudgetExceeded(RuntimeError):
    """Uncached (billed) LLM spend reached the configured cap."""


@dataclass
class Budget:
    """Shared cap on fresh (non-cached) spend across several LLM clients."""

    max_usd: float | None
    spent_usd: float = 0.0

    def check(self) -> None:
        if self.max_usd is not None and self.spent_usd >= self.max_usd:
            raise BudgetExceeded(
                f"LLM budget of ${self.max_usd:.2f} reached (spent ${self.spent_usd:.4f}); "
                "raise budget_usd deliberately or warm the cache"
            )


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    fresh_usage: Usage = field(default_factory=Usage)
    fresh_cost_usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "fresh_tokens": self.fresh_usage.as_dict(),
            "fresh_cost_usd": self.fresh_cost_usd,
        }


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        a.input_tokens + b.input_tokens,
        a.output_tokens + b.output_tokens,
        a.cache_read_tokens + b.cache_read_tokens,
        a.cache_write_tokens + b.cache_write_tokens,
    )


class CachingLLM:
    """Replays identical requests from disk; only cache misses are billed and budgeted.

    The key covers the model, every request parameter in `params`, the system prompt,
    the user message and the output schema, so any change to a prompt, the retrieved
    context or a setting is a miss.
    """

    def __init__(
        self,
        inner: LLMClient,
        kv: SqliteKV | None,
        params: dict[str, Any],
        pricing: dict[str, Any],
        budget: Budget | None = None,
    ) -> None:
        self.inner = inner
        self.kv = kv
        self.params = params
        self.pricing = pricing
        self.budget = budget or Budget(None)
        self.stats = CacheStats()
        self.model = inner.model

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        key = digest(self.model, self.params, system, user, schema)
        if self.kv is not None and (hit := self.kv.get(key)) is not None:
            self.stats.hits += 1
            return LLMResult(
                text=hit["text"],
                usage=Usage(**hit["usage"]),
                model=hit["model"],
                stop_reason=hit["stop_reason"],
            )
        self.budget.check()
        result = self.inner.generate(system, user, schema)
        cost = cost_usd(result.usage, self.pricing)
        self.stats.misses += 1
        self.stats.fresh_usage = _add(self.stats.fresh_usage, result.usage)
        self.stats.fresh_cost_usd += cost
        self.budget.spent_usd += cost
        if self.kv is not None and result.stop_reason in ("end_turn", "refusal"):
            self.kv.put(
                key,
                {
                    "text": result.text,
                    "model": result.model,
                    "stop_reason": result.stop_reason,
                    "usage": vars(result.usage),
                },
            )
        return result


_PARAM_KEYS = ("model", "max_tokens", "effort", "temperature", "cache_system_prompt")


def make_llm(
    llm_cfg: dict[str, Any],
    cache_cfg: dict[str, Any] | None = None,
    budget: Budget | None = None,
    inner: LLMClient | None = None,
) -> CachingLLM:
    """An Anthropic client wrapped with the disk cache (if enabled) and the budget."""
    kv = None
    if cache_cfg and cache_cfg.get("enabled"):
        from dpdp_rag.config import resolve

        kv = SqliteKV(resolve(Path(cache_cfg["path"])), "llm_calls")
    params = {k: llm_cfg.get(k) for k in _PARAM_KEYS}
    return CachingLLM(inner or make_answer_llm(llm_cfg), kv, params, llm_cfg["pricing"], budget)
