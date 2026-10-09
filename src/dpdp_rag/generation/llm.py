"""LLM client: Groq or Claude (Anthropic SDK), behind a small protocol so tests can mock it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import groq

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


# OpenAI-style finish_reason -> the stop_reason names callers check (judge, answer, cache).
_GROQ_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


class GroqLLM:
    """Groq's OpenAI-compatible chat API. Reads GROQ_API_KEY from the environment.

    The answer schema goes in as a strict `json_schema` response format, so `llm.model`
    must be one Groq supports structured outputs for (e.g. openai/gpt-oss-120b).
    """

    def __init__(self, cfg: dict[str, Any], client: Any = None) -> None:
        if client is None:
            client = groq.Groq(timeout=float(cfg["timeout_s"]), max_retries=int(cfg["max_retries"]))
        self._client = client
        self._cfg = cfg
        self.model: str = cfg["model"]

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        try:
            response = self._create(system, user, schema)
        except groq.APIConnectionError as exc:  # includes timeouts
            raise LLMError(f"Could not reach the Groq API: {exc}") from exc
        except groq.APIStatusError as exc:
            raise LLMError(f"Groq API error {exc.status_code}: {exc.message}") from exc
        u = response.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) or 0
        usage = Usage(
            input_tokens=(u.prompt_tokens or 0) - cached,
            output_tokens=u.completion_tokens or 0,
            cache_read_tokens=cached,
        )
        choice = response.choices[0]
        return LLMResult(
            text=choice.message.content or "",
            usage=usage,
            model=response.model,
            stop_reason=_GROQ_STOP_REASONS.get(choice.finish_reason, choice.finish_reason),
        )

    def _create(self, system: str, user: str, schema: dict[str, Any]) -> Any:
        extra: dict[str, Any] = {}
        if self._cfg.get("temperature") is not None:
            extra["temperature"] = float(self._cfg["temperature"])
        if self._cfg.get("reasoning_effort"):  # gpt-oss models: low / medium / high
            extra["reasoning_effort"] = self._cfg["reasoning_effort"]
        return self._client.chat.completions.create(
            **extra,
            model=self.model,
            max_completion_tokens=int(self._cfg["max_tokens"]),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            },
        )


class StubLLM:
    """Deterministic stand-in for load tests and offline demos (llm.provider: stub).

    It cites the first supplied document and says plainly that no model was called, so
    a stub answer can never be mistaken for a real one. Optional `stub_latency_ms`
    simulates model latency.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = "stub"
        self._latency_s = float(cfg.get("stub_latency_ms") or 0) / 1000

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        import json
        import re
        import time

        if self._latency_s:
            time.sleep(self._latency_s)
        ids = re.findall(r'chunk_id="([^"]+)"', user)
        body = {
            "answer": "[stub answer: no language model was called] See the cited provision.",
            "citations": [{"chunk_id": ids[0], "pinpoint": ""}] if ids else [],
            "refused": not ids,
            "refusal_reason": "none" if ids else "insufficient_context",
        }
        return LLMResult(
            text=json.dumps(body), usage=Usage(), model=self.model, stop_reason="end_turn"
        )


def make_answer_llm(cfg: dict[str, Any]) -> LLMClient:
    """The client for `llm.provider`: "groq" (default), "anthropic" or "stub" (no API calls)."""
    provider = cfg.get("provider", "groq")
    if provider == "anthropic":
        return AnthropicLLM(cfg)
    if provider == "groq":
        return GroqLLM(cfg)
    if provider == "stub":
        return StubLLM(cfg)
    raise ValueError(f"Unknown llm.provider {provider!r}")
