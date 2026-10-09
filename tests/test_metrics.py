"""SQLite request metrics and the latency / cost / count helpers."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from dpdp_rag.metrics import MetricsStore, RequestMetric, Where, percentile


def metric(
    latency: float,
    *,
    cost: float = 0.001,
    cached: bool = False,
    refused: bool = False,
    source: str = "api",
    chash: str = "h1",
    ts: datetime | None = None,
) -> RequestMetric:
    return RequestMetric(
        config_hash=chash,
        latency_ms=latency,
        tokens_in=100,
        tokens_out=20,
        cost_usd=cost,
        cached=cached,
        refused=refused,
        source=source,
        timestamp=ts or datetime.now(UTC),
    )


@pytest.fixture
def store(tmp_path) -> MetricsStore:
    return MetricsStore(tmp_path / "nested" / "metrics.db")


def test_schema_and_row(store: MetricsStore) -> None:
    store.record(metric(12.5, cost=0.002, refused=True))
    conn = sqlite3.connect(store.db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(requests)")]
    assert cols == [
        "id",
        "timestamp",
        "config_hash",
        "latency_ms",
        "tokens_in",
        "tokens_out",
        "cost_usd",
        "cached",
        "refused",
        "source",
    ]
    row = conn.execute(
        "SELECT config_hash, latency_ms, tokens_in, tokens_out, cost_usd, cached,"
        " refused, source FROM requests"
    ).fetchone()
    assert row == ("h1", 12.5, 100, 20, 0.002, 0, 1, "api")


def test_source_is_constrained(store: MetricsStore) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.record(metric(1.0, source="batch"))


def test_percentiles() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0
    assert percentile([7.0], 99) == 7.0
    assert percentile([], 50) is None
    with pytest.raises(ValueError):
        percentile(values, 0)


def test_latency_cost_and_counts(store: MetricsStore) -> None:
    for i in range(1, 101):
        store.record(metric(float(i), cost=0.01 if i % 2 else 0.03, cached=i <= 10, refused=i > 95))
    assert store.p50_latency() == 50.0
    assert store.p99_latency() == 99.0
    assert store.mean_cost() == pytest.approx(0.02)
    assert store.request_counts() == {
        "total": 100,
        "cached": 10,
        "refused": 5,
        "api": 100,
        "eval": 0,
    }
    # Excluding cached requests shifts the latency distribution.
    assert store.p50_latency(Where(include_cached=False)) == 55.0


def test_filters_by_config_source_and_time(store: MetricsStore) -> None:
    old = datetime.now(UTC) - timedelta(days=2)
    store.record(metric(10, chash="h1", source="api", ts=old))
    store.record(metric(20, chash="h1", source="eval"))
    store.record(metric(30, chash="h2", source="eval", cost=0.5))
    assert store.request_counts(Where(config_hash="h1"))["total"] == 2
    assert store.request_counts(Where(source="eval")) == {
        "total": 2,
        "cached": 0,
        "refused": 0,
        "api": 0,
        "eval": 2,
    }
    assert store.p50_latency(Where(since=datetime.now(UTC) - timedelta(hours=1))) == 20.0
    assert store.mean_cost(Where(config_hash="h2")) == pytest.approx(0.5)
    assert store.mean_cost(Where(config_hash="nope")) is None
    summary = store.summary(Where(config_hash="h1"))
    assert summary["counts"]["total"] == 2 and summary["p99_latency_ms"] == 20.0
