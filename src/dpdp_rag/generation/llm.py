"""LLM client: Claude via the Anthropic SDK, behind a small protocol so tests can mock it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from dpdp_rag.generation.cost import Usage


@dataclass(frozen=True)
class LLMResult:
    text: str  # JSON matching the answer schema (structured output)
    usage: Usage
    model: str
    stop_reason: str | None


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class LLMClient(Protocol):
    model: str

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult: ...


class AnthropicLLM:
    def __init__(self, cfg: dict[str, Any], client: Any = None) -> None:
        if client is None:
            client = anthropic.Anthropic(
                timeout=float(cfg["timeout_s"]), max_retries=int(cfg["max_retries"])
            )
        self._client = client
        self._cfg = cfg
        self.model: str = cfg["model"]

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        system_block: dict[str, Any] = {"type": "text", "text": system}
        if self._cfg.get("cache_system_prompt"):
            system_block["cache_control"] = {"type": "ephemeral"}
        try:
            response = self._create(system_block, user, schema)
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return LLMResult(
            text=text, usage=usage, model=response.model, stop_reason=response.stop_reason
        )

    def _create(self, system_block: dict[str, Any], user: str, schema: dict[str, Any]) -> Any:
        extra: dict[str, Any] = {}
        if self._cfg.get("temperature") is not None:  # only for models that accept it
            extra["temperature"] = float(self._cfg["temperature"])
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self._cfg.get("effort"):  # e.g. claude-haiku-4-5 has no effort parameter
            output_config["effort"] = self._cfg["effort"]
        return self._client.messages.create(
            **extra,
            model=self.model,
            max_tokens=int(self._cfg["max_tokens"]),
            system=[system_block],
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
