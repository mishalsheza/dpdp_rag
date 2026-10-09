"""FastAPI service: POST /ask, GET /healthz, GET /version."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dpdp_rag.api.cache import ResponseCache, cache_key
from dpdp_rag.api.versioning import config_hash as compute_config_hash
from dpdp_rag.api.versioning import git_sha
from dpdp_rag.config import REPO_ROOT, load_config, resolve
from dpdp_rag.generation.answer import Answerer, AnswerResult
from dpdp_rag.generation.llm import AnthropicLLM, LLMClient, LLMError
from dpdp_rag.metrics.store import MetricsStore, RequestMetric
from dpdp_rag.observability.tracing import Tracer, make_tracer
from dpdp_rag.retrieval.retriever import RetrievalUnavailable, Retriever

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    as_of_date: date | None = None


class CitationOut(BaseModel):
    chunk_id: str
    pinpoint: str
    doc_type: str
    gsr_no: str | None
    page: int | None
    in_force_date: date | None
    in_force: bool
    corrected_by: str | None


class Tokens(BaseModel):
    input: int
    output: int
    cache_read: int
    cache_write: int
    total: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    refused: bool
    refusal_reason: str | None
    as_of_date: date
    latency_ms: float
    tokens: Tokens
    cost_usd: float
    config_hash: str
    model: str
    cached: bool


def _payload(result: AnswerResult) -> dict[str, Any]:
    """The cacheable part of a response (no latency / cache flag)."""
    return {
        "answer": result.answer,
        "citations": [asdict(c) for c in result.citations],
        "refused": result.refused,
        "refusal_reason": result.refusal_reason,
        "as_of_date": result.as_of_date,
        "tokens": result.usage.as_dict(),
        "cost_usd": result.cost_usd,
        "model": result.model,
    }


_ZERO_TOKENS = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0}


def create_app(
    config: dict[str, Any] | None = None,
    *,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    tracer: Tracer | None = None,
    metrics: MetricsStore | None = None,
    cache: ResponseCache | None = None,
) -> FastAPI:
    """Build the app. Arguments left as None are created from configs/default.yaml."""
    load_dotenv(REPO_ROOT / ".env")
    config = config or load_config("default.yaml")
    chash = compute_config_hash(config)
    tracer = tracer or make_tracer(config.get("tracing", {}))
    metrics = metrics or MetricsStore(resolve(config["metrics"]["db_path"]))
    cache_cfg = config["api"]["response_cache"]
    if cache is None and cache_cfg.get("enabled"):
        cache = ResponseCache(resolve(cache_cfg["dir"]))
    state: dict[str, Answerer] = {}

    def answerer() -> Answerer:
        # Built on first use so the app starts without Qdrant or an API key at hand.
        if "answerer" not in state:
            state["answerer"] = Answerer(
                config, retriever or Retriever(config), llm or AnthropicLLM(config["llm"])
            )
        return state["answerer"]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        tracer.flush()

    app = FastAPI(title="dpdp-rag", lifespan=lifespan)
    app.state.config_hash = chash
    app.state.metrics = metrics

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> dict[str, str]:
        return {"git_sha": git_sha(), "config_hash": chash, "model": config["llm"]["model"]}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        started = time.perf_counter()
        if len(req.question) > int(config["api"]["max_question_chars"]):
            raise HTTPException(
                422, f"question longer than {config['api']['max_question_chars']} characters"
            )
        as_of = req.as_of_date or date.today()
        key = cache_key(chash, req.question, as_of)
        with tracer.trace(
            config["tracing"].get("trace_name", "ask"),
            config_hash=chash,
            input={"question": req.question, "as_of_date": as_of.isoformat()},
            metadata={"as_of_date": as_of.isoformat()},
        ) as trace:
            hit = cache.get(key) if cache else None
            if hit is not None:
                body = hit | {"tokens": _ZERO_TOKENS, "cost_usd": 0.0}
                cached = True
            else:
                try:
                    result = answerer().answer(req.question, as_of, trace)
                except LLMError as exc:
                    log.error("LLM error: %s", exc)
                    raise HTTPException(502, str(exc)) from exc
                except (ConnectionError, RetrievalUnavailable) as exc:
                    log.error("Retrieval unavailable: %s", exc)
                    raise HTTPException(503, str(exc)) from exc
                body = _payload(result)
                cached = False
                if cache:
                    cache.put(
                        key,
                        AskResponse.model_validate(
                            body | {"latency_ms": 0, "config_hash": chash, "cached": False}
                        ).model_dump(mode="json", exclude={"latency_ms", "config_hash", "cached"}),
                    )
            response = AskResponse.model_validate(
                body
                | {
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "config_hash": chash,
                    "cached": cached,
                }
            )
            trace.end(
                response.model_dump(mode="json"),
                metadata={"cached": cached, "refused": response.refused},
            )

        tokens = response.tokens
        metrics.record(
            RequestMetric(
                config_hash=chash,
                latency_ms=response.latency_ms,
                tokens_in=tokens.input + tokens.cache_read + tokens.cache_write,
                tokens_out=tokens.output,
                cost_usd=response.cost_usd,
                cached=cached,
                refused=response.refused,
                source="api",
            )
        )
        return response

    return app
