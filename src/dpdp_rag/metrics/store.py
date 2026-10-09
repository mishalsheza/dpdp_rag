"""Per-request metrics persisted to SQLite (data/metrics.db), plus summary helpers."""

from __future__ import annotations

import math
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Source = Literal["api", "eval"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    config_hash TEXT    NOT NULL,
    latency_ms  REAL    NOT NULL,
    tokens_in   INTEGER NOT NULL,
    tokens_out  INTEGER NOT NULL,
    cost_usd    REAL    NOT NULL,
    cached      INTEGER NOT NULL CHECK (cached IN (0, 1)),
    refused     INTEGER NOT NULL CHECK (refused IN (0, 1)),
    source      TEXT    NOT NULL CHECK (source IN ('api', 'eval'))
);
CREATE INDEX IF NOT EXISTS idx_requests_config ON requests (config_hash, source);
CREATE INDEX IF NOT EXISTS idx_requests_time ON requests (timestamp);
"""


@dataclass(frozen=True)
class RequestMetric:
    config_hash: str
    latency_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float
    cached: bool
    refused: bool
    source: Source
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class Where:
    """Optional filters shared by the summary helpers."""

    config_hash: str | None = None
    source: Source | None = None
    since: datetime | None = None
    include_cached: bool = True

    def sql(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if self.config_hash:
            clauses.append("config_hash = ?")
            args.append(self.config_hash)
        if self.source:
            clauses.append("source = ?")
            args.append(self.source)
        if self.since:
            clauses.append("timestamp >= ?")
            args.append(_iso(self.since))
        if not self.include_cached:
            clauses.append("cached = 0")
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", args


def _iso(ts: datetime) -> str:
    return (ts if ts.tzinfo else ts.replace(tzinfo=UTC)).astimezone(UTC).isoformat()


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (p in 0-100) of `values`; None when empty."""
    if not values:
        return None
    if not 0 < p <= 100:
        raise ValueError("p must be in (0, 100]")
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100 * len(ordered))) - 1]


class MetricsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, m: RequestMetric) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO requests (timestamp, config_hash, latency_ms, tokens_in, tokens_out,"
                " cost_usd, cached, refused, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _iso(m.timestamp),
                    m.config_hash,
                    m.latency_ms,
                    m.tokens_in,
                    m.tokens_out,
                    m.cost_usd,
                    int(m.cached),
                    int(m.refused),
                    m.source,
                ),
            )

    def _column(self, column: str, where: Where) -> list[float]:
        clause, args = where.sql()
        with self._connect() as conn:
            rows = conn.execute(f"SELECT {column} FROM requests{clause}", args).fetchall()
        return [float(r[0]) for r in rows]

    def latency_percentile(self, p: float, where: Where | None = None) -> float | None:
        return percentile(self._column("latency_ms", where or Where()), p)

    def p50_latency(self, where: Where | None = None) -> float | None:
        return self.latency_percentile(50, where)

    def p99_latency(self, where: Where | None = None) -> float | None:
        return self.latency_percentile(99, where)

    def mean_cost(self, where: Where | None = None) -> float | None:
        costs = self._column("cost_usd", where or Where())
        return sum(costs) / len(costs) if costs else None

    def request_counts(self, where: Where | None = None) -> dict[str, int]:
        clause, args = (where or Where()).sql()
        with self._connect() as conn:
            total, cached, refused = conn.execute(
                f"SELECT COUNT(*), COALESCE(SUM(cached), 0), COALESCE(SUM(refused), 0)"
                f" FROM requests{clause}",
                args,
            ).fetchone()
            by_source = dict(
                conn.execute(
                    f"SELECT source, COUNT(*) FROM requests{clause} GROUP BY source",
                    args,
                ).fetchall()
            )
        return {
            "total": total,
            "cached": cached,
            "refused": refused,
            "api": by_source.get("api", 0),
            "eval": by_source.get("eval", 0),
        }

    def summary(self, where: Where | None = None) -> dict[str, Any]:
        return {
            "counts": self.request_counts(where),
            "p50_latency_ms": self.p50_latency(where),
            "p99_latency_ms": self.p99_latency(where),
            "mean_cost_usd": self.mean_cost(where),
        }
