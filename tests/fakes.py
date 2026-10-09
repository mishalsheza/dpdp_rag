"""Test doubles: a mocked LLM and a recording tracer."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from dpdp_rag.generation.cost import Usage
from dpdp_rag.generation.llm import LLMResult

CHUNK_ID = re.compile(r'chunk_id="([^"]+)"')
DEFAULT_USAGE = Usage(input_tokens=1000, output_tokens=200)


def cite_first(user: str) -> dict[str, Any]:
    """Default reply: answer from, and cite, the first document in the prompt."""
    ids = CHUNK_ID.findall(user)
    return {
        "answer": "Per the cited provision, ...",
        "citations": [{"chunk_id": ids[0], "pinpoint": "whatever the model says"}] if ids else [],
        "refused": False,
        "refusal_reason": "none",
    }


class FakeLLM:
    model = "fake-haiku"

    def __init__(
        self,
        reply: dict[str, Any] | Callable[[str], dict[str, Any]] = cite_first,
        usage: Usage = DEFAULT_USAGE,
        stop_reason: str = "end_turn",
        raw_text: str | None = None,
    ) -> None:
        self.reply = reply
        self.usage = usage
        self.stop_reason = stop_reason
        self.raw_text = raw_text
        self.calls: list[dict[str, Any]] = []

    def generate(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        self.calls.append({"system": system, "user": user, "schema": schema})
        payload = self.reply(user) if callable(self.reply) else self.reply
        text = self.raw_text if self.raw_text is not None else json.dumps(payload)
        return LLMResult(
            text=text, usage=self.usage, model=self.model, stop_reason=self.stop_reason
        )


class RecordingTracer:
    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.flushed = 0

    @contextmanager
    def trace(
        self, name: str, *, config_hash: str, input: Any, metadata: dict[str, Any] | None = None
    ) -> Iterator[Any]:
        record: dict[str, Any] = {
            "name": name,
            "config_hash": config_hash,
            "input": input,
            "metadata": metadata,
            "steps": [],
            "output": None,
        }
        self.traces.append(record)
        yield _RecordingTrace(record)

    def flush(self) -> None:
        self.flushed += 1


class _RecordingTrace:
    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record

    @contextmanager
    def step(
        self, name: str, *, as_type: str = "span", input: Any = None, model: str | None = None
    ) -> Iterator[Any]:
        step: dict[str, Any] = {"name": name, "as_type": as_type, "model": model}
        self.record["steps"].append(step)
        yield _RecordingStep(step)

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        self.record["output"] = output
        self.record["end_metadata"] = metadata

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        self.record.setdefault("scores", {})[name] = value


class _RecordingStep:
    def __init__(self, step: dict[str, Any]) -> None:
        self.step = step

    def update(self, **kwargs: Any) -> None:
        self.step.update({k: v for k, v in kwargs.items() if v is not None})
