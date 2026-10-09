"""Tracing of /ask requests. Langfuse when configured, otherwise a no-op.

A request is one trace (tagged with the config_hash) containing a "retrieve" step and a
"generate" generation step with model, token usage and cost.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

log = logging.getLogger(__name__)


class Step(Protocol):
    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


class Trace(Protocol):
    def step(
        self, name: str, *, as_type: str = "span", input: Any = None, model: str | None = None
    ) -> AbstractContextManager[Step]: ...

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None: ...


class Tracer(Protocol):
    def trace(
        self, name: str, *, config_hash: str, input: Any, metadata: dict[str, Any] | None = None
    ) -> AbstractContextManager[Trace]: ...

    def flush(self) -> None: ...


class _NoopStep:
    def update(self, **_: Any) -> None:
        return None


class _NoopTrace:
    @contextmanager
    def step(self, name: str, **_: Any) -> Iterator[Step]:
        yield _NoopStep()

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        return None


NOOP_TRACE: Trace = _NoopTrace()


class NoopTracer:
    @contextmanager
    def trace(self, name: str, **_: Any) -> Iterator[Trace]:
        yield NOOP_TRACE

    def flush(self) -> None:
        return None


class _LangfuseStep:
    def __init__(self, observation: Any) -> None:
        self._obs = observation

    def update(
        self,
        *,
        output: Any = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"output": output, "metadata": metadata, "model": model}
        if usage is not None:
            kwargs["usage_details"] = usage
        if cost_usd is not None:
            kwargs["cost_details"] = {"total": cost_usd}
        self._obs.update(**{k: v for k, v in kwargs.items() if v is not None})


class _LangfuseTrace:
    def __init__(self, client: Any, root: Any) -> None:
        self._client = client
        self._root = root
        self.output: Any = None

    @contextmanager
    def step(
        self, name: str, *, as_type: str = "span", input: Any = None, model: str | None = None
    ) -> Iterator[Step]:
        with self._client.start_as_current_observation(
            name=name, as_type=as_type, input=input, model=model
        ) as obs:
            yield _LangfuseStep(obs)

    def end(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        self.output = output
        self._root.update(output=output, metadata=metadata)


class LangfuseTracer:
    def __init__(self, client: Any = None) -> None:
        if client is None:
            from langfuse import Langfuse

            client = Langfuse()  # reads LANGFUSE_PUBLIC_KEY / _SECRET_KEY / _HOST
        self._client = client

    @contextmanager
    def trace(
        self, name: str, *, config_hash: str, input: Any, metadata: dict[str, Any] | None = None
    ) -> Iterator[Trace]:
        from langfuse import propagate_attributes

        attrs = {"config_hash": config_hash, **{k: str(v) for k, v in (metadata or {}).items()}}
        with (
            propagate_attributes(
                trace_name=name,
                metadata=attrs,
                version=config_hash,
                tags=[f"config:{config_hash[:12]}"],
            ),
            self._client.start_as_current_observation(
                name=name, as_type="span", input=input, metadata=attrs
            ) as root,
        ):
            trace = _LangfuseTrace(self._client, root)
            yield trace
            self._client.set_current_trace_io(input=input, output=trace.output)

    def flush(self) -> None:
        self._client.flush()


def make_tracer(cfg: dict[str, Any]) -> Tracer:
    """Langfuse if enabled in config and its keys are in the environment, else a no-op."""
    keys = os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    if cfg.get("langfuse_enabled") and keys:
        try:
            return LangfuseTracer()
        except Exception:  # tracing must never take the API down
            log.exception("Langfuse unavailable; tracing disabled")
    return NoopTracer()
