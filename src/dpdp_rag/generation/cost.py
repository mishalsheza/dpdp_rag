"""Token usage and USD cost from the `llm.pricing` config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # uncached input
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "cache_read": self.cache_read_tokens,
            "cache_write": self.cache_write_tokens,
            "total": self.total_input + self.output_tokens,
        }


def cost_usd(usage: Usage, pricing: dict[str, Any]) -> float:
    """Cost of one request; prompts over the threshold use the long-context rates."""
    long = usage.total_input > int(pricing["long_context_threshold"])
    rate_in = float(pricing["long_input" if long else "input"]) / 1e6
    rate_out = float(pricing["long_output" if long else "output"]) / 1e6
    return (
        usage.input_tokens * rate_in
        + usage.cache_write_tokens * rate_in * float(pricing["cache_write_multiplier"])
        + usage.cache_read_tokens * rate_in * float(pricing["cache_read_multiplier"])
        + usage.output_tokens * rate_out
    )
